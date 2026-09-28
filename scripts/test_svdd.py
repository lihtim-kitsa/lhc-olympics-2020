import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from src.models.m3_deep_svdd import DeepSVDD
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
x = torch.randn(1000, 5)
y = torch.zeros(1000)
loader = DataLoader(TensorDataset(x, y), batch_size=256)

model = DeepSVDD()
model.init_center(loader)
print("Center:", model.c)
print("Center mean:", model.c.mean().item())
