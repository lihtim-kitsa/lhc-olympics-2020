"""Build compact detector features from the official LHCO files.

The released high-level file is the canonical full-sample source because it
uses the challenge authors' FastJet-contrib definitions. A deterministic raw
event subsample is independently reclustered to validate the kinematic path.
Use --from-raw to compute the entire sample with this project's explicitly
documented exclusive-kT tau_N definition (much slower and not numerically
identical to the released contrib observables).
"""
import argparse
import json
import os
import sys

import h5py
import hdf5plugin  # noqa: F401; registers the raw files' Blosc compression filter
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features.clustering import extract_event_features

FEATURES=['mJ1','dmJ','tau21_J1','tau21_J2','dRJJ','mJJ','label']
RAW_DATASET='df/block0_values'


def find_raw_dataset(h5):
    for key in (RAW_DATASET,'table','events'):
        if key in h5 and isinstance(h5[key],h5py.Dataset): return h5[key]
    raise ValueError(f'Could not find raw event matrix in HDF5; keys={list(h5.keys())}')


def from_high_level(raw_path, high_path, output_path, validation_path, sample_size=128):
    frame=pd.read_hdf(high_path,key='df')
    required=['pxj1','pyj1','pzj1','mj1','tau1j1','tau2j1','pxj2','pyj2','pzj2','mj2','tau1j2','tau2j2']
    missing=set(required)-set(frame.columns)
    if missing: raise ValueError(f'Missing columns in supplied LHCO features: {sorted(missing)}')
    with h5py.File(raw_path,'r') as raw:
        raw_events=find_raw_dataset(raw)
        if raw_events.shape[0]!=len(frame): raise ValueError(f'Raw/high-level event counts differ: {raw_events.shape[0]} vs {len(frame)}')
        if raw_events.shape[1]!=2101: raise ValueError(f'Expected raw event width 2101, got {raw_events.shape[1]}')
        labels=np.empty(len(frame),dtype=np.float32)
        for start in range(0,len(frame),50000):
            end=min(start+50000,len(frame)); labels[start:end]=raw_events[start:end,-1]
        n=min(sample_size,len(frame)); sample=np.unique(np.linspace(0,len(frame)-1,n,dtype=np.int64))
        raw_sample=raw_events[sample,:-1]
    a=frame
    def eta(prefix):
        pt=np.hypot(a[f'px{prefix}'].to_numpy(),a[f'py{prefix}'].to_numpy())
        return np.arcsinh(np.divide(a[f'pz{prefix}'].to_numpy(),pt,out=np.zeros_like(pt),where=pt>0))
    pt1=np.hypot(a.pxj1,a.pyj1).to_numpy(); pt2=np.hypot(a.pxj2,a.pyj2).to_numpy()
    phi1=np.arctan2(a.pyj1,a.pxj1).to_numpy(); phi2=np.arctan2(a.pyj2,a.pxj2).to_numpy()
    deta=eta('j1')-eta('j2'); dphi=(phi1-phi2+np.pi)%(2*np.pi)-np.pi
    e1=np.sqrt(a.pxj1.to_numpy()**2+a.pyj1.to_numpy()**2+a.pzj1.to_numpy()**2+a.mj1.to_numpy()**2)
    e2=np.sqrt(a.pxj2.to_numpy()**2+a.pyj2.to_numpy()**2+a.pzj2.to_numpy()**2+a.mj2.to_numpy()**2)
    mjj2=(e1+e2)**2-(a.pxj1.to_numpy()+a.pxj2.to_numpy())**2-(a.pyj1.to_numpy()+a.pyj2.to_numpy())**2-(a.pzj1.to_numpy()+a.pzj2.to_numpy())**2
    tau21_1=np.divide(a.tau2j1.to_numpy(),a.tau1j1.to_numpy(),out=np.full(len(a),np.nan),where=a.tau1j1.to_numpy()>0)
    tau21_2=np.divide(a.tau2j2.to_numpy(),a.tau1j2.to_numpy(),out=np.full(len(a),np.nan),where=a.tau1j2.to_numpy()>0)
    x=np.column_stack([a.mj1.to_numpy(),np.abs(a.mj1.to_numpy()-a.mj2.to_numpy()),
        tau21_1,tau21_2,
        np.hypot(deta,dphi),np.sqrt(np.maximum(mjj2,0)),labels]).astype(np.float32)
    bad=(~np.isfinite(x[:,:6]).all(axis=1))|(x[:,5]<=0)
    x[bad]=np.nan

    # Independent low-level FastJet cross-check on deterministic rows.
    raw_features=[]
    for row in tqdm(raw_sample,desc='Validating raw-event sample'):
        raw_features.append(extract_event_features(row))
    checked=[(int(i),v) for i,v in zip(sample,raw_features) if v is not None]
    deltas={k:[] for k in ('mJ1','dmJ','tau21_J1','tau21_J2','dRJJ','mJJ')}
    for i,v in checked:
        for k in deltas: deltas[k].append(float(v[k]-x[i,FEATURES.index(k)]))
    audit={'canonical_source':os.path.basename(high_path),'raw_source':os.path.basename(raw_path),
        'rows':len(frame),'raw_validation_sample_requested':len(sample),'raw_validation_sample_valid':len(checked),
        'low_level_definition':'FastJet anti-kT R=1; tau_N uses exclusive-kT axes, beta=1, R0=1',
        'released_tau_definition':'Values supplied by LHCO authors; their FastJet-contrib configuration is not fully encoded in this repository.',
        'raw_minus_released_kinematic_delta':{k:{'mean':float(np.mean(v)) if v else None,'mae':float(np.mean(np.abs(v))) if v else None,'max_abs':float(np.max(np.abs(v))) if v else None} for k,v in deltas.items()},
        'invalid_rows':int(bad.sum()),'invalid_fraction':float(bad.mean())}
    temporary_path=output_path+'.tmp'
    with h5py.File(temporary_path,'w') as out:
        ds=out.create_dataset('features',data=x,dtype='f4',chunks=(min(10000,len(x)),len(FEATURES)),compression='gzip',compression_opts=1)
        ds.attrs['feature_names']=','.join(FEATURES)
        ds.attrs['feature_schema_version']='1.1.0'
        ds.attrs['source']='official LHCO supplied high-level kinematics and subjettiness; truth bit aligned from raw event file'
        ds.attrs['tau21_definition']='released LHCO tau2/tau1 values'
    os.replace(temporary_path,output_path)
    os.makedirs(os.path.dirname(validation_path),exist_ok=True)
    with open(validation_path,'w',encoding='utf-8') as out: json.dump(audit,out,indent=2)
    return audit


def from_raw(raw_path, output_path, chunk_size=10000):
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    temporary_path=output_path+'.tmp'
    with h5py.File(raw_path,'r') as source:
        dset=find_raw_dataset(source)
        if dset.ndim!=2 or dset.shape[1]!=2101: raise ValueError(f'Expected N x 2101 raw event matrix, got {dset.shape}')
        with h5py.File(temporary_path,'w') as out:
            target=out.create_dataset('features',shape=(dset.shape[0],len(FEATURES)),dtype='f4',chunks=(min(chunk_size,dset.shape[0]),len(FEATURES)),compression='gzip',compression_opts=1)
            target.attrs['feature_names']=','.join(FEATURES); target.attrs['feature_schema_version']='1.1.0'
            target.attrs['source']='raw event matrix reclustered with FastJet; exclusive-kT tau_N axes'
            for start in tqdm(range(0,dset.shape[0],chunk_size),desc=f'Extracting {os.path.basename(raw_path)}'):
                end=min(start+chunk_size,dset.shape[0]); rows=dset[start:end]
                result=np.full((len(rows),len(FEATURES)),np.nan,dtype=np.float32)
                for i,row in enumerate(rows):
                    label=row[-1]; values=extract_event_features(row[:-1])
                    if values is not None:
                        result[i,:6]=[values[k] for k in FEATURES[:6]]; result[i,6]=label
                target[start:end]=result
    os.replace(temporary_path,output_path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--from-raw',action='store_true'); parser.add_argument('--sample-size',type=int,default=128)
    args=parser.parse_args(); raw_dir='data/raw'; proc_dir='data/processed'; os.makedirs(proc_dir,exist_ok=True)
    files=[('events_anomalydetection_v2.h5','events_anomalydetection_v2.features.h5','events_v2_features.h5'),('events_anomalydetection_Z_XY_qqq.h5','events_anomalydetection_Z_XY_qqq.features.h5','events_Z_XY_qqq_features.h5')]
    for raw_name,high_name,out_name in files:
        raw=os.path.join(raw_dir,raw_name); high=os.path.join(raw_dir,high_name); output=os.path.join(proc_dir,out_name)
        if args.from_raw: from_raw(raw,output)
        elif os.path.exists(high):
            audit=from_high_level(raw,high,output,'reports/feature_validation_'+out_name.replace('.h5','.json'),args.sample_size)
            print(json.dumps(audit,indent=2))
        else: raise FileNotFoundError(f'Official high-level feature file required: {high}')


if __name__=='__main__': main()
