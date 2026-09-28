"""Create deterministic, class-stratified event-level train/validation/test splits."""
import json
import os

import h5py
import numpy as np


def _valid_rows(features):
    return np.isfinite(features[:, :6]).all(axis=1) & (features[:, 5] > 0)


def create_splits(features_file, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    rng=np.random.default_rng(seed)
    with h5py.File(features_file,'r') as f:
        data=f['features'][:]
    valid=_valid_rows(data)
    labels=data[:,6]
    if not np.isin(labels[valid],[0,1]).all(): raise ValueError('Expected binary background/signal labels')
    summary={'seed':int(seed),'fractions':{'train':.6,'validation':.2,'test':.2},'classes':{}}
    for cls,prefix in ((0,'background'),(1,'signal2')):
        idx=np.flatnonzero(valid & (labels==cls)); rng.shuffle(idx)
        ntrain=int(.6*len(idx)); nval=int(.2*len(idx))
        parts={'train':idx[:ntrain],'val':idx[ntrain:ntrain+nval],'test':idx[ntrain+nval:]}
        for split,rows in parts.items(): np.save(os.path.join(output_dir,f'{prefix}_{split}.npy'),rows)
        summary['classes'][prefix]={k:int(len(v)) for k,v in parts.items()}
    with open(os.path.join(output_dir,'split_manifest.json'),'w',encoding='utf-8') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))


def create_3prong_split(features_file, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    with h5py.File(features_file,'r') as f: data=f['features'][:]
    idx=np.flatnonzero(_valid_rows(data))
    np.save(os.path.join(output_dir,'signal3_test.npy'),idx)
    print(f'3-prong held-out signal: {len(idx)} valid events')


if __name__=='__main__':
    create_splits('data/processed/events_v2_features.h5','data/splits',seed=42)
    f3='data/processed/events_Z_XY_qqq_features.h5'
    if os.path.exists(f3): create_3prong_split(f3,'data/splits')
