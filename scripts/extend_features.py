import os
import h5py
import numpy as np
import pandas as pd

def build_extended_features(high_path, output_path):
    print(f"Reading {high_path}...")
    frame = pd.read_hdf(high_path, key='df')
    
    # Calculate tau21
    tau21_j1 = frame['tau2j1'] / np.maximum(frame['tau1j1'], 1e-8)
    tau21_j2 = frame['tau2j2'] / np.maximum(frame['tau1j2'], 1e-8)
    
    # Calculate tau32
    tau32_j1 = frame['tau3j1'] / np.maximum(frame['tau2j1'], 1e-8)
    tau32_j2 = frame['tau3j2'] / np.maximum(frame['tau2j2'], 1e-8)
    
    # Calculate dijet mass
    px1, py1, pz1, m1 = frame['pxj1'], frame['pyj1'], frame['pzj1'], frame['mj1']
    px2, py2, pz2, m2 = frame['pxj2'], frame['pyj2'], frame['pzj2'], frame['mj2']
    
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)
    
    px_tot = px1 + px2
    py_tot = py1 + py2
    pz_tot = pz1 + pz2
    e_tot = e1 + e2
    
    mjj = np.sqrt(np.maximum(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2, 0.0))
    
    # Calculate dRJJ (approximated from px, py, pz or extracted)
    # We can just extract dRJJ if we use the old features file as base.
    pass

def main():
    pass

if __name__ == '__main__':
    main()
