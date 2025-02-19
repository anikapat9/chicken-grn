import scanpy as sc
import torch
from torch.utils.data import Dataset

class RNADataset(Dataset):
    def __init__(self, adata_path):
        self.adata = sc.read(adata_path)  # Load .h5ad or .csv
        sc.pp.normalize_total(self.adata)  # Normalize
        sc.pp.log1p(self.adata)  # Log-transform
        
    def __len__(self):
        return self.adata.n_obs  # Number of samples
    
    def __getitem__(self, idx):
        # Convert to PyTorch tensors (genes × time)
        return torch.tensor(self.adata.X[idx], dtype=torch.float32)
