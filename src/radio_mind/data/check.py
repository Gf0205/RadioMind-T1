from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from .load import MODULATIONS, SNRS, load_store
from .split import load_or_create_splits

def run_check(data_root: str | None = None) -> None:
    root = Path(data_root or os.environ.get("DATA_ROOT", "./data")); store = load_store(root); splits = load_or_create_splits(store)
    print(f"samples={store.size}, shape={tuple(store.X.shape[1:])}, dtype={store.X.dtype}")
    print(f"nan={int(np.isnan(store.X).sum())}, inf={int(np.isinf(store.X).sum())}, abs_max={float(np.abs(store.X).max()):.6f}")
    expected = {"train": 700, "val": 150, "test": 150}
    for name, idx in splits.items():
        expected_size = expected[name] * len(MODULATIONS) * len(SNRS)
        if len(idx) != expected_size:
            raise ValueError(f"{name} has {len(idx)} samples, expected {expected_size}")
        for m in range(len(MODULATIONS)):
            for snr in SNRS:
                n = int(((store.mod_ids[idx] == m) & (store.snr_db[idx] == snr)).sum())
                if n != expected[name]:
                    raise ValueError(f"invalid cell count: {(name, m, snr, n)}")
        mods = set(store.mod_ids[idx].tolist()); snrs = set(store.snr_db[idx].tolist())
        if mods != set(range(len(MODULATIONS))):
            raise ValueError(f"{name} does not cover every modulation")
        if snrs != set(SNRS):
            raise ValueError(f"{name} does not cover every SNR")
        print(f"{name}: {len(idx)} samples; all 220 cells = {expected[name]}; classes=11; snrs=20")
    all_idx = np.concatenate(list(splits.values()))
    if len(all_idx) != store.size:
        raise ValueError("combined split size does not match the cache")
    if len(np.unique(all_idx)) != store.size:
        raise ValueError("combined splits contain duplicates or missing samples")
    print("check: PASS (all cells balanced; full class/SNR coverage; no overlap, duplicates, or missing samples)")
