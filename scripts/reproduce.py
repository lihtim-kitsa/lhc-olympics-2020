"""Run the seed-42 grid and five-seed core headline comparisons."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
import pandas as pd

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import train_model
from scripts.evaluate import evaluate_model


def run_grid():
    started = time.perf_counter()
    verified=False
    manifest_path='data/manifest.yaml'
    if os.path.exists(manifest_path):
        with open(manifest_path,encoding='utf-8') as f: manifest=yaml.safe_load(f) or {}
        files=manifest.get('files',[])
        if len(files)==4:
            verified=True
            for item in files:
                path=os.path.join('data','raw',item['filename'])
                if not os.path.exists(path) or os.path.getsize(path)!=item['byte_size']:
                    verified=False; break
                digest=hashlib.md5()
                with open(path,'rb') as source:
                    for chunk in iter(lambda:source.read(4*1024*1024),b''): digest.update(chunk)
                if 'md5:'+digest.hexdigest()!=item['checksum']:
                    verified=False; break
    if not verified: subprocess.run([sys.executable,'scripts/download_data.py'],check=True)
    else: print('Using all four checksum-verified files recorded in data/manifest.yaml.',flush=True)
    subprocess.run([sys.executable,'scripts/build_features.py'],check=True)
    subprocess.run([sys.executable,'scripts/build_extended_features.py'],check=True)
    subprocess.run([sys.executable,'src/data/make_dataset.py'],check=True)
    grid=[]
    for seed in [42]:
        for model_id in ('m1_autoencoder','m2_isolation_forest','m3_deep_svdd'):
            with open(f'configs/{model_id}.yaml',encoding='utf-8') as f: base=yaml.safe_load(f)
            for f_train in (0.0,0.1,0.5,1.0):
                cfg=copy.deepcopy(base)
                cfg['data']['f_train']=f_train; cfg['data']['k_labels']=0; cfg['training']['seed']=seed
                grid.append(cfg)

        with open('configs/m4_deep_sad.yaml',encoding='utf-8') as f: sad_base=yaml.safe_load(f)
        for f_train in (0.0,0.5,1.0):
            for k in (10,100,1000):
                cfg=copy.deepcopy(sad_base)
                cfg['data']['f_train']=f_train; cfg['data']['k_labels']=k; cfg['training']['seed']=seed
                grid.append(cfg)
        for model_id in ('m5_supervised','m6_mass_aware'):
            with open(f'configs/{model_id}.yaml',encoding='utf-8') as f:
                cfg=yaml.safe_load(f)
                cfg['training']['seed']=seed
                grid.append(cfg)

    for seed in (43,44,45,46):
        for model_id in ('m1_autoencoder','m2_isolation_forest','m3_deep_svdd',
                         'm4_deep_sad','m5_supervised','m6_mass_aware'):
            with open(f'configs/{model_id}.yaml',encoding='utf-8') as f:
                cfg=yaml.safe_load(f)
            cfg['training']['seed']=seed
            grid.append(cfg)

    manifest=[]
    for i,cfg in enumerate(grid,1):
        entry={'index':i,'model':cfg['model']['name'],'f_train_percent':cfg['data']['f_train'],
               'k_labels':cfg['data'].get('k_labels',0),'seed':cfg['training']['seed']}
        print(f"[{i}/{len(grid)}] {entry}",flush=True)
        variant=f"f{float(entry['f_train_percent']):g}_k{int(entry['k_labels'])}"
        stem=f"{entry['model']}_{variant}_{entry['seed']}"
        result_path='reports/tables/results.csv'
        if os.path.exists(result_path) and os.path.exists(os.path.join('models',stem+'.pt')):
            previous=pd.read_csv(result_path)
            if {'n_test_background','test_signal_prevalence'}.issubset(previous.columns):
                found=previous[(previous.model==entry['model'])&(previous.f_train==entry['f_train_percent'])&
                    (previous.k_labels==entry['k_labels'])&(previous.seed==entry['seed'])]
                if len(found):
                    entry['status']='already completed; retained'
                    manifest.append(entry)
                    os.makedirs('reports/tables',exist_ok=True)
                    with open('reports/tables/run_manifest.json','w',encoding='utf-8') as f:
                        json.dump(manifest,f,indent=2)
                    continue
        train_model(cfg)
        evaluate_model(cfg)
        entry['status']='completed'
        manifest.append(entry)
        os.makedirs('reports/tables',exist_ok=True)
        with open('reports/tables/run_manifest.json','w',encoding='utf-8') as f: json.dump(manifest,f,indent=2)

    results=pd.read_csv('reports/tables/results.csv')
    baseline=results[results.seed.isin([42,43,44,45,46]) & results.model.isin([
        'M1_Autoencoder','M2_IsolationForest','M3_DeepSVDD','M4_DeepSAD','M5_Supervised','M6_MassAware'])]
    numeric=[c for c in ('roc_auc','roc_auc_3prong','rej_10','rej_10_3prong','mass_sculpting_jsd_at_10pct_bkg') if c in baseline]
    summary=baseline.groupby(['model','f_train','k_labels'])[numeric].agg(['mean','std','count'])
    summary.to_csv('reports/tables/replicate_summary.csv')
    with open('reports/tables/reproduction_timing.json','w',encoding='utf-8') as f:
        json.dump({'elapsed_seconds':time.perf_counter()-started,
                   'configured_runs':len(grid),
                   'retained_runs':sum(item['status'].startswith('already') for item in manifest),
                   'completed_runs':sum(item['status']=='completed' for item in manifest)},f,indent=2)
    print(f"Completed {len(manifest)} configured and replicate runs.")


if __name__=='__main__': run_grid()
