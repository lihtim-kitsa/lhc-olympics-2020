import os
import torch
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append(os.path.abspath('src'))
from models.m3_deep_svdd import DeepSVDD

# Setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = 'cpu'

# Load model
model = DeepSVDD(input_dim=5)
checkpoint = torch.load('models/M3_DeepSVDD_f0_k0_42.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.c = checkpoint['center']
model.eval()

# Check weights
print("Weight stats:")
for name, param in model.named_parameters():
    print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")

print("\nModel center:", model.c)

# Let's generate some random inputs and see the output
x = torch.randn(100, 5)
with torch.no_grad():
    out = model.net(x)

print("\nOutput on random normal inputs:")
print("Mean:", out.mean(dim=0))
print("Std:", out.std(dim=0))

# Now try with actual data
features_file = 'data/processed/events_v2_features.h5'
split_dir = 'data/splits'

bkg_idx = np.load(os.path.join(split_dir, 'background_train.npy'))
cols = [0, 1, 2, 3, 4]

with h5py.File(features_file, 'r') as f:
    X_bkg_unscaled = f['features'][bkg_idx[:1000], cols]

scaler = StandardScaler().fit(X_bkg_unscaled)
X_bkg = scaler.transform(X_bkg_unscaled).astype(np.float32)

x_real = torch.from_numpy(X_bkg)
with torch.no_grad():
    out_real = model.net(x_real)

print("\nOutput on real background inputs (1000 samples):")
print("Mean:", out_real.mean(dim=0))
print("Std:", out_real.std(dim=0))
print("Loss (MSE to center):", ((out_real - model.c)**2).sum(dim=1).mean().item())
