import os
import h5py
import json
import numpy as np
import pandas as pd

def build_extended_features():
    high_path = 'data/raw/events_anomalydetection_v2.features.h5'
    output_path = 'data/processed/events_v2_extended_features.h5'
    
    if not os.path.exists(high_path):
        print(f"Skipping extended features: {high_path} not found.")
        return

    print("Building extended 7-feature dataset...")
    a = pd.read_hdf(high_path, key='df')
    
    def eta(prefix):
        pt = np.hypot(a[f'px{prefix}'].to_numpy(), a[f'py{prefix}'].to_numpy())
        return np.arcsinh(np.divide(a[f'pz{prefix}'].to_numpy(), pt, out=np.zeros_like(pt), where=pt>0))
        
    pt1=np.hypot(a.pxj1,a.pyj1).to_numpy(); pt2=np.hypot(a.pxj2,a.pyj2).to_numpy()
    phi1=np.arctan2(a.pyj1,a.pxj1).to_numpy(); phi2=np.arctan2(a.pyj2,a.pxj2).to_numpy()
    deta = eta('j1') - eta('j2')
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    
    e1=np.sqrt(a.pxj1.to_numpy()**2+a.pyj1.to_numpy()**2+a.pzj1.to_numpy()**2+a.mj1.to_numpy()**2)
    e2=np.sqrt(a.pxj2.to_numpy()**2+a.pyj2.to_numpy()**2+a.pzj2.to_numpy()**2+a.mj2.to_numpy()**2)
    mjj2=(e1+e2)**2-(a.pxj1.to_numpy()+a.pxj2.to_numpy())**2-(a.pyj1.to_numpy()+a.pyj2.to_numpy())**2-(a.pzj1.to_numpy()+a.pzj2.to_numpy())**2
    
    tau21_1=np.divide(a.tau2j1.to_numpy(),a.tau1j1.to_numpy(),out=np.full(len(a),np.nan),where=a.tau1j1.to_numpy()>0)
    tau21_2=np.divide(a.tau2j2.to_numpy(),a.tau1j2.to_numpy(),out=np.full(len(a),np.nan),where=a.tau1j2.to_numpy()>0)
    tau32_1=np.divide(a.tau3j1.to_numpy(),a.tau2j1.to_numpy(),out=np.full(len(a),np.nan),where=a.tau2j1.to_numpy()>0)
    tau32_2=np.divide(a.tau3j2.to_numpy(),a.tau2j2.to_numpy(),out=np.full(len(a),np.nan),where=a.tau2j2.to_numpy()>0)
    
    # We load labels from the 5-feature dataset
    with h5py.File('data/processed/events_v2_features.h5', 'r') as old_f:
        labels = old_f['features'][:, 6]

    features = np.column_stack([
        a.mj1.to_numpy(), np.abs(a.mj1.to_numpy()-a.mj2.to_numpy()),
        tau21_1, tau21_2, tau32_1, tau32_2, np.hypot(deta,dphi),
        np.sqrt(np.maximum(mjj2,0)), labels
    ]).astype(np.float32)
    
    feat_names = ['mJ1','dmJ','tau21_J1','tau21_J2','tau32_J1','tau32_J2','dRJJ','mJJ','label']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('features', data=features, compression='gzip', compression_opts=4)
        f.attrs['columns'] = json.dumps(feat_names)
        
    print(f"Saved {output_path} with features {feat_names}")

if __name__ == '__main__':
    build_extended_features()
