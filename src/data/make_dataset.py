import os
import h5py
import numpy as np

def create_splits(features_file, output_dir, seed=42):
    """
    Reads the features file, separates background (label=0) and signal (label=1),
    and creates 60/20/20 train/val/test splits, saving their row indices.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(features_file, 'r') as f:
        dset = f['features']
        # Read only the labels and mJJ to filter invalid events
        labels = dset[:, 6]
        mJJs = dset[:, 5]
        
        # Valid events are those without NaN (e.g. mJJ > 0)
        valid_mask = ~np.isnan(mJJs)
        
        # Background
        bkg_indices = np.where((labels == 0) & valid_mask)[0]
        np.random.shuffle(bkg_indices)
        
        n_bkg = len(bkg_indices)
        b_train_end = int(0.6 * n_bkg)
        b_val_end = int(0.8 * n_bkg)
        
        bkg_train = bkg_indices[:b_train_end]
        bkg_val = bkg_indices[b_train_end:b_val_end]
        bkg_test = bkg_indices[b_val_end:]
        
        # Signal
        sig_indices = np.where((labels == 1) & valid_mask)[0]
        np.random.shuffle(sig_indices)
        
        n_sig = len(sig_indices)
        s_train_end = int(0.6 * n_sig)
        s_val_end = int(0.8 * n_sig)
        
        sig_train = sig_indices[:s_train_end]
        sig_val = sig_indices[s_train_end:s_val_end]
        sig_test = sig_indices[s_val_end:]
        
        print(f"Background: {n_bkg} valid events. Train: {len(bkg_train)}, Val: {len(bkg_val)}, Test: {len(bkg_test)}")
        print(f"Signal: {n_sig} valid events. Train: {len(sig_train)}, Val: {len(sig_val)}, Test: {len(sig_test)}")
        
        np.save(os.path.join(output_dir, 'background_train.npy'), bkg_train)
        np.save(os.path.join(output_dir, 'background_val.npy'), bkg_val)
        np.save(os.path.join(output_dir, 'background_test.npy'), bkg_test)
        
        np.save(os.path.join(output_dir, 'signal2_train.npy'), sig_train)
        np.save(os.path.join(output_dir, 'signal2_val.npy'), sig_val)
        np.save(os.path.join(output_dir, 'signal2_test.npy'), sig_test)

def create_3prong_split(features_file, output_dir):
    """
    For 3-prong, the whole file is the test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(features_file, 'r') as f:
        dset = f['features']
        mJJs = dset[:, 5]
        valid_indices = np.where(~np.isnan(mJJs))[0]
        
        np.save(os.path.join(output_dir, 'signal3_test.npy'), valid_indices)
        print(f"3-prong Signal: {len(valid_indices)} valid events.")

if __name__ == '__main__':
    proc_dir = os.path.join('data', 'processed')
    splits_dir = os.path.join('data', 'splits')
    
    f2 = os.path.join(proc_dir, 'events_v2_features.h5')
    f3 = os.path.join(proc_dir, 'events_Z_XY_qqq_features.h5')
    
    if os.path.exists(f2):
        create_splits(f2, splits_dir)
    else:
        print(f"Features file {f2} not found. Run build_features.py first.")
        
    if os.path.exists(f3):
        create_3prong_split(f3, splits_dir)
