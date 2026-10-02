"""Evaluate frozen detectors with validation-selected background cuts."""
import argparse
import os
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import uproot
import yaml
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.m1_autoencoder import Autoencoder
from src.models.m2_isolation_forest import IsolationForestAnomalyDetector
from src.models.m3_deep_svdd import DeepSVDD
from src.models.m4_deep_sad import DeepSAD
from src.models.m5_supervised import SupervisedMLP
from src.evaluation.metrics import get_core_metrics, evaluate_mass_sculpting, calculate_score_mjj_dependence
from src.evaluation.bump_hunt import perform_bump_hunt, bootstrap_null_significances
from src.evaluation.working_points import score_cut, score_cut_weights


def read_rows(dset, indices, cols):
    indices=np.asarray(indices,dtype=np.int64)
    if not len(indices): return np.empty((0,len(cols)),dtype=np.float32)
    matrix=dset[:] if isinstance(dset,h5py.Dataset) else np.asarray(dset)
    return matrix[indices][:,cols]


def load_eval_arrays(split_dir, features_file, config, scaler, split):
    bidx=np.load(os.path.join(split_dir,f'background_{split}.npy'))
    sidx=np.load(os.path.join(split_dir,f'signal2_{split}.npy'))
    cols=[0,1,2,3,4,5] if config['data'].get('use_mjj',False) else [0,1,2,3,4]
    
    if config['data'].get('use_extended', False):
        cols=[0,1,2,3,4,5,6,7] if config['data'].get('use_mjj',False) else [0,1,2,3,4,5,6]
        features_file = 'data/processed/events_v2_extended_features.h5'
        
    with h5py.File(features_file,'r') as f: matrix=f['features'][:]
    xb=read_rows(matrix,bidx,cols); xs=read_rows(matrix,sidx,cols)
    mjj_idx = 7 if config['data'].get('use_extended', False) else 5
    mb=matrix[bidx,mjj_idx]; ms=matrix[sidx,mjj_idx]
    xb=scaler.transform(xb).astype(np.float32); xs=scaler.transform(xs).astype(np.float32)
    return xb,xs,np.concatenate([mb,ms]),np.r_[np.zeros(len(xb)),np.ones(len(xs))]


def load_model(name, variant, seed, input_dim, config, device):
    path=os.path.join('models',f'{name}_{variant}_{seed}.pt')
    if not os.path.exists(path): raise FileNotFoundError(f'Trained model checkpoint missing: {path}')
    if name in ('M2_IsolationForest', 'M0_TauCut', 'M8_ANODE', 'M12_CWoLaTrees', 'M11_CATHODE_CVAE'):
        with open(path,'rb') as f: return pickle.load(f)
    ckpt=torch.load(path,map_location=device,weights_only=False)
    if name=='M1_Autoencoder': model=Autoencoder(input_dim)
    elif name in ('M3_DeepSVDD','M6_MassAware'):
        model=DeepSVDD(input_dim); model.c=ckpt['center']
    elif name=='M4_DeepSAD':
        model=DeepSAD(input_dim,eta=config['model'].get('eta',1.0)); model.c=ckpt['center']
    elif name=='M5_Supervised': model=SupervisedMLP(input_dim)
    elif name=='M7_CWoLa':
        from src.models.m7_cwola import CWoLaClassifier
        model=CWoLaClassifier(5)
    else: raise ValueError(f'Unknown model: {name}')
    model.load_state_dict(ckpt['model_state_dict']); return model.to(device).eval()


def score(model, X, name, device):
    if name in ('M2_IsolationForest', 'M0_TauCut', 'M8_ANODE', 'M12_CWoLaTrees', 'M11_CATHODE_CVAE'): return model.get_anomaly_score(X)
    out=[]
    with torch.no_grad():
        for start in range(0,len(X),10000):
            x=torch.as_tensor(X[start:start+10000],dtype=torch.float32,device=device)
            out.append(model.get_anomaly_score(x).detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def threshold_at_efficiency(scores, efficiency):
    return score_cut(scores, efficiency)


def save_mass_diagnostics(mjj,scores,thresholds,path):
    edges=np.linspace(2500,4500,21); centers=.5*(edges[:-1]+edges[1:])
    total,_=np.histogram(mjj,bins=edges)
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    axes[0].step(centers,total/max(total.sum(),1),where='mid',color='#203B53',label='Inclusive background')
    for key,color in (('10pct','#158B8B'),('1pct','#C7832D')):
        selected=score_cut_weights(scores,*thresholds[key])
        passing,_=np.histogram(mjj,bins=edges,weights=selected)
        norm=passing.sum()
        if norm: axes[0].step(centers,passing/norm,where='mid',color=color,label=f"Score cut, validation εB={key}")
        eff=np.divide(passing,total,out=np.full(len(total),np.nan,dtype=float),where=total>0)
        err=np.sqrt(np.divide(eff*(1-eff),total,out=np.zeros_like(eff),where=total>0))
        axes[1].errorbar(centers,eff,yerr=err,fmt='o-',ms=3,color=color,label=f'Validation εB={key}')
    # Binned mean anomaly score reveals residual score-mass dependence.
    means=[]; errors=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        vals=scores[(mjj>=lo)&(mjj<hi)]
        means.append(float(np.mean(vals)) if len(vals) else np.nan)
        errors.append(float(np.std(vals)/np.sqrt(len(vals))) if len(vals)>1 else np.nan)
    axes[2].errorbar(centers,means,yerr=errors,fmt='o-',ms=3,color='#70469B',label='Background mean score')
    axes[0].set(title='Background mass spectrum',xlabel='$m_{JJ}$ [GeV]',ylabel='Normalized events')
    axes[1].set(title='Binned background acceptance',xlabel='$m_{JJ}$ [GeV]',ylabel='Score acceptance')
    axes[2].set(title='Mean score by mass bin',xlabel='$m_{JJ}$ [GeV]',ylabel='Mean anomaly score')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); os.makedirs(os.path.dirname(path),exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)


def evaluate_model(config):
    mlflow.set_tracking_uri('sqlite:///mlflow_reproduction.db')
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run():
        _evaluate_model(config)


def _evaluate_model(config):
    name=config['model']['name']; seed=int(config['training']['seed'])
    variant=f"f{float(config['data'].get('f_train',0)):g}_k{int(config['data'].get('k_labels',0))}"
    scaler_path=os.path.join('models',f'scaler_{name}_{variant}_{seed}.pkl')
    with open(scaler_path,'rb') as f: scaler=pickle.load(f)
    split_dir='data/splits'; f2='data/processed/events_v2_features.h5'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xb,Xs,mjj_test,y_test=load_eval_arrays(split_dir,f2,config,scaler,'test')
    Xbv,_,mjj_val,_=load_eval_arrays(split_dir,f2,config,scaler,'val')
    model=load_model(name,variant,seed,Xb.shape[1],config,device)
    sb,ss=score(model,Xb,name,device),score(model,Xs,name,device)
    sbv=score(model,Xbv,name,device)
    scores=np.r_[sb,ss]
    metrics=get_core_metrics(y_test,scores)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, scores)
    valid = fpr > 0
    sic = np.zeros_like(tpr)
    sic[valid] = tpr[valid] / np.sqrt(fpr[valid])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(tpr, 1.0 / np.where(fpr > 0, fpr, np.nan), label=f'AUC={metrics["roc_auc"]:.3f}')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Signal Efficiency (TPR)')
    ax[0].set_ylabel('Background Rejection (1/FPR)')
    ax[0].set_title('ROC Curve')
    ax[0].legend()
    ax[0].grid(True)
    
    ax[1].plot(tpr, sic)
    ax[1].set_xlabel('Signal Efficiency (TPR)')
    ax[1].set_ylabel('Significance (SIC)')
    ax[1].set_title('SIC Curve')
    ax[1].grid(True)
    
    fig.tight_layout()
    curve_path = os.path.join('reports', 'figures', f'{name}_{variant}_{seed}_roc_sic.png')
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    mlflow.log_artifact(curve_path)

    metrics['n_test_background']=int(len(sb)); metrics['n_test_signal_2prong']=int(len(ss))
    metrics['test_signal_prevalence']=float(len(ss)/(len(sb)+len(ss)))
    thresholds={'10pct':threshold_at_efficiency(sbv,.10),'1pct':threshold_at_efficiency(sbv,.01)}
    metrics.update(evaluate_mass_sculpting(np.asarray(mjj_test[:len(sb)]),sb,thresholds))
    metrics['score_mjj_dependence']=calculate_score_mjj_dependence(np.asarray(mjj_test[:len(sb)]),sb)
    for key,(th,tie_probability) in thresholds.items():
        metrics[f'validation_threshold_{key}']=th
        metrics[f'validation_tie_acceptance_{key}']=tie_probability
    metrics['validation_bkg_eff_at_10pct']=float(score_cut_weights(sbv,*thresholds['10pct']).mean())
    metrics['validation_bkg_eff_at_1pct']=float(score_cut_weights(sbv,*thresholds['1pct']).mean())
    variant=f"f{float(config['data'].get('f_train',0)):g}_k{int(config['data'].get('k_labels',0))}"
    diagnostic_path=os.path.join('reports','figures',f'{name}_{variant}_{seed}_mass_diagnostics.png')
    save_mass_diagnostics(np.asarray(mjj_test[:len(sb)]),sb,thresholds,diagnostic_path)

    rng=np.random.default_rng(seed)
    n_inj=min(int(len(sb)*.005),len(ss))
    chosen=rng.choice(len(ss),n_inj,replace=False) if n_inj else np.empty(0,dtype=int)
    mjj_bkg=np.asarray(mjj_test[:len(sb)])
    mjj_mix=np.r_[mjj_bkg,np.asarray(mjj_test[len(sb):])[chosen]]
    score_mix=np.r_[sb,ss[chosen]]
    metrics.update(perform_bump_hunt(mjj_mix,prefix='pre_cut_'))
    keep10=score_cut_weights(score_mix,*thresholds['10pct'])
    keep1=score_cut_weights(score_mix,*thresholds['1pct'])
    post10=mjj_mix[keep10>0]
    post1=mjj_mix[keep1>0]
    metrics.update(perform_bump_hunt(mjj_mix,prefix='at_10pct_bkg_',weights=keep10))
    metrics.update(perform_bump_hunt(mjj_mix,prefix='at_1pct_bkg_',weights=keep1))
    null_pre=perform_bump_hunt(mjj_bkg,prefix='null_pre_cut_')
    metrics.update(null_pre)
    z_pre=null_pre['null_pre_cut_local_significance']
    pre_boot=bootstrap_null_significances(mjj_bkg,n_trials=50,seed=seed,observed_z=z_pre)
    for key,value in pre_boot.items(): metrics[f'null_pre_bootstrap_{key}']=value
    null_keep=score_cut_weights(sb,*thresholds['1pct'])
    post_null=mjj_bkg[null_keep>0]
    null_post=perform_bump_hunt(mjj_bkg,prefix='null_at_1pct_bkg_',weights=null_keep)
    metrics.update(null_post)
    z_post=null_post['null_at_1pct_bkg_local_significance']
    post_boot=bootstrap_null_significances(mjj_bkg,n_trials=50,seed=seed+1,observed_z=z_post,base_weights=null_keep)
    for key,value in post_boot.items(): metrics[f'null_at_1pct_bootstrap_{key}']=value

    # Held-out 3-prong signals are scored against the unchanged 2-prong background test sample.
    f3='data/processed/events_Z_XY_qqq_features.h5'; i3=os.path.join(split_dir,'signal3_test.npy')
    if os.path.exists(f3) and os.path.exists(i3):
        with h5py.File(f3,'r') as f: matrix3=f['features'][:]
        idx=np.load(i3); cols=[0,1,2,3,4,5] if config['data'].get('use_mjj',False) else [0,1,2,3,4]
        x3=matrix3[idx][:,cols]; m3=matrix3[idx,5]
        x3=scaler.transform(x3).astype(np.float32); s3=score(model,x3,name,device)
        metrics['roc_auc_3prong']=float(roc_auc_score(np.r_[np.zeros(len(sb)),np.ones(len(s3))],np.r_[sb,s3]))
        metrics['n_test_signal_3prong']=int(len(s3))
        for e in (.01,.05,.10,.30,.50):
            cut,tie_probability=score_cut(s3,e) if len(s3) else (np.inf,0.)
            eb=float(score_cut_weights(sb,cut,tie_probability).mean())
            metrics[f'rej_{int(e*100)}_3prong']=float(1/eb) if eb else float('inf')
        metrics['3prong_mjj_median']=float(np.nanmedian(m3))

    os.makedirs('reports',exist_ok=True)
    root_path=os.path.join('reports',f'{name}_{seed}_spectra.root')
    with uproot.recreate(root_path) as fout:
        for key,values,weights in [('inclusive',mjj_mix,None),('at_10pct_bkg',mjj_mix,keep10),('at_1pct_bkg',mjj_mix,keep1)]:
            fout[key]=np.histogram(values,bins=40,range=(2500,4500),weights=weights)
    mlflow.log_artifact(root_path)
    mlflow.log_artifact(diagnostic_path)
    results_path='reports/tables/results.csv'; os.makedirs(os.path.dirname(results_path),exist_ok=True)
    record={'model':name,'f_train':config['data']['f_train'],'k_labels':config['data'].get('k_labels',0),'seed':seed,**metrics}
    df=pd.DataFrame([record])
    if os.path.exists(results_path):
        old=pd.read_csv(results_path)
        old=old[~((old.model==name)&(old.seed==seed)&(old.f_train==record['f_train'])&(old.k_labels==record['k_labels']))]
        df=pd.concat([old,df],ignore_index=True,sort=False)
    df.to_csv(results_path,index=False)
    mlflow.log_metrics({k:v for k,v in metrics.items() if np.isfinite(v)})
    print(f"Saved validation-thresholded test metrics for {name} seed={seed}")


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    args=parser.parse_args()
    with open(args.config,encoding='utf-8') as f: config=yaml.safe_load(f)
    evaluate_model(config)

