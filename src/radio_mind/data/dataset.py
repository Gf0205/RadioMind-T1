from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from .load import DatasetStore

class RMLDataset(Dataset):
    def __init__(self, store: DatasetStore, indices, normalize: str = "none", mean=None, std=None):
        self.store, self.indices, self.normalize, self.mean, self.std = store, indices, normalize, mean, std
        if normalize == "global" and (mean is None or std is None):
            # Statistics are computed from the training indices only.  The (2, 1)
            # shape broadcasts over one I/Q sample without adding a batch axis.
            x = np.asarray(store.X[indices], dtype=np.float32)
            self.mean = torch.from_numpy(x.mean(axis=(0, 2), keepdims=False)[:, None])
            self.std = torch.from_numpy((x.std(axis=(0, 2), keepdims=False) + 1e-6)[:, None])
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = int(self.indices[i]); x = torch.from_numpy(self.store.X[j].copy())
        if self.normalize == "global": x = (x - self.mean) / self.std
        elif self.normalize == "rms": x = x / (torch.sqrt(torch.mean(x * x)) + 1e-8)
        return x, int(self.store.mod_ids[j]), int(self.store.snr_db[j])
