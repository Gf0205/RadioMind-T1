from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

from .load import DatasetStore, MODULATIONS, SNRS

SPLIT_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")
EXPECTED_PER_CELL = {"train": 700, "val": 150, "test": 150}
PROVENANCE_FIELDS = (
    "source_cache_schema_version",
    "source_pickle_size",
    "source_pickle_mtime_ns",
)
REQUIRED_SPLIT_FIELDS = set(SPLIT_NAMES) | {
    "split_seed",
    "split_schema_version",
    *PROVENANCE_FIELDS,
}


class SplitCacheValidationError(ValueError):
    """Raised when a split cache does not satisfy the T1 protocol."""


def build_splits(store: DatasetStore, seed: int = 20260907) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, list[np.ndarray]] = {name: [] for name in SPLIT_NAMES}
    for mod_id, _ in enumerate(MODULATIONS):
        for snr in SNRS:
            indices = np.flatnonzero(
                (store.mod_ids == mod_id) & (store.snr_db == snr)
            )
            if len(indices) != 1000:
                raise ValueError(f"cell {(mod_id, snr)} has {len(indices)} samples")
            indices = indices.copy()
            rng.shuffle(indices)
            result["train"].append(indices[:700])
            result["val"].append(indices[700:850])
            result["test"].append(indices[850:])
    return {
        name: np.concatenate(parts).astype(np.int64)
        for name, parts in result.items()
    }


def _cache_provenance(store: DatasetStore) -> dict[str, int]:
    metadata_path = store.root / "metadata.npz"
    try:
        with np.load(metadata_path, allow_pickle=False) as metadata:
            source_fields = {
                "source_cache_schema_version": "cache_schema_version",
                "source_pickle_size": "source_pickle_size",
                "source_pickle_mtime_ns": "source_pickle_mtime_ns",
            }
            missing = set(source_fields.values()) - set(metadata.files)
            if missing:
                raise SplitCacheValidationError(
                    "metadata cache lacks split provenance fields: "
                    + ", ".join(sorted(missing))
                )
            provenance = {
                split_name: int(np.asarray(metadata[metadata_name]).item())
                for split_name, metadata_name in source_fields.items()
            }
    except SplitCacheValidationError:
        raise
    except Exception as error:
        raise SplitCacheValidationError(
            f"cannot read cache provenance from {metadata_path}: {error}"
        ) from error
    return provenance


def _validate_splits(
    store: DatasetStore,
    splits: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    missing = set(SPLIT_NAMES) - set(splits)
    if missing:
        raise SplitCacheValidationError(
            "split cache missing arrays: " + ", ".join(sorted(missing))
        )

    validated: dict[str, np.ndarray] = {}
    all_indices: list[np.ndarray] = []
    for name in SPLIT_NAMES:
        indices = splits[name]
        if not isinstance(indices, np.ndarray):
            raise SplitCacheValidationError(f"cached {name} split must be an ndarray")
        if indices.ndim != 1:
            raise SplitCacheValidationError(
                f"cached {name} split must be one-dimensional, got {indices.shape}"
            )
        if indices.dtype != np.dtype(np.int64):
            raise SplitCacheValidationError(
                f"cached {name} split must have dtype int64, got {indices.dtype}"
            )
        per_cell = EXPECTED_PER_CELL[name]
        expected_size = per_cell * len(MODULATIONS) * len(SNRS)
        if len(indices) != expected_size:
            raise SplitCacheValidationError(
                f"invalid cached {name} split size: {len(indices)}"
            )
        if len(np.unique(indices)) != len(indices):
            raise SplitCacheValidationError(
                f"duplicate indices inside cached {name} split"
            )
        if indices.min(initial=0) < 0 or indices.max(initial=-1) >= store.size:
            raise SplitCacheValidationError(
                f"cached {name} split contains an out-of-range index"
            )
        for mod_id in range(len(MODULATIONS)):
            for snr in SNRS:
                count = int(np.count_nonzero(
                    (store.mod_ids[indices] == mod_id)
                    & (store.snr_db[indices] == snr)
                ))
                if count != per_cell:
                    raise SplitCacheValidationError(
                        f"invalid cached {name} cell {(mod_id, snr)}: {count}"
                    )
        validated[name] = indices
        all_indices.append(indices)

    joined = np.concatenate(all_indices)
    if len(joined) != store.size or len(np.unique(joined)) != store.size:
        raise SplitCacheValidationError(
            "cached splits overlap or do not cover the dataset"
        )
    return validated


def _scalar_int(cache: np.lib.npyio.NpzFile, field: str) -> int:
    value = np.asarray(cache[field])
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise SplitCacheValidationError(
            f"split cache field {field} must be a scalar int64"
        )
    return int(value.item())


def _load_split_cache(
    path: Path,
    store: DatasetStore,
    requested_seed: int,
    provenance: dict[str, int],
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as cache:
            missing = REQUIRED_SPLIT_FIELDS - set(cache.files)
            if missing:
                raise SplitCacheValidationError(
                    "split cache missing fields: " + ", ".join(sorted(missing))
                )
            cached_seed = _scalar_int(cache, "split_seed")
            cached_schema = _scalar_int(cache, "split_schema_version")
            cached_provenance = {
                field: _scalar_int(cache, field) for field in PROVENANCE_FIELDS
            }
            splits = {name: np.asarray(cache[name]).copy() for name in SPLIT_NAMES}
    except SplitCacheValidationError:
        raise
    except Exception as error:
        raise SplitCacheValidationError(
            f"cannot load split cache {path}: {error}"
        ) from error

    if cached_seed != requested_seed:
        raise SplitCacheValidationError(
            f"split seed mismatch: requested {requested_seed}, cache has {cached_seed}"
        )
    if cached_schema != SPLIT_SCHEMA_VERSION:
        raise SplitCacheValidationError(
            f"unsupported split schema version: {cached_schema}"
        )
    for field in PROVENANCE_FIELDS:
        if cached_provenance[field] != provenance[field]:
            raise SplitCacheValidationError(
                f"split provenance mismatch for {field}: "
                f"current {provenance[field]}, cache has {cached_provenance[field]}"
            )
    return _validate_splits(store, splits)


def _write_split_cache_atomically(
    path: Path,
    store: DatasetStore,
    splits: dict[str, np.ndarray],
    seed: int,
    provenance: dict[str, int],
) -> None:
    validated = _validate_splits(store, splits)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".radiomind-split-",
        suffix=".tmp.npz",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez(
            temporary_path,
            **validated,
            split_seed=np.int64(seed),
            split_schema_version=np.int64(SPLIT_SCHEMA_VERSION),
            **{field: np.int64(value) for field, value in provenance.items()},
        )
        _load_split_cache(temporary_path, store, seed, provenance)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_manifest(store: DatasetStore, splits: dict[str, np.ndarray]) -> None:
    split_labels = np.empty(store.size, dtype="U5")
    for name, indices in splits.items():
        split_labels[indices] = name
    path = store.root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_id", "iq_path", "array_index", "modulation_id",
            "mod_str", "snr_db", "split",
        ])
        for index in range(store.size):
            mod_id = int(store.mod_ids[index])
            writer.writerow([
                str(store.sample_id[index]),
                "X.npy",
                index,
                mod_id,
                MODULATIONS[mod_id],
                int(store.snr_db[index]),
                str(split_labels[index]),
            ])


def load_or_create_splits(
    store: DatasetStore,
    seed: int = 20260907,
    force: bool = False,
) -> dict[str, np.ndarray]:
    path = store.root / f"splits_{seed}.npz"
    provenance = _cache_provenance(store)
    if path.exists() and not force:
        try:
            splits = _load_split_cache(path, store, seed, provenance)
        except SplitCacheValidationError as error:
            raise SplitCacheValidationError(
                f"existing split cache is invalid: {error}; "
                "rerun with force=True to rebuild"
            ) from error
    else:
        splits = build_splits(store, seed)
        _write_split_cache_atomically(path, store, splits, seed, provenance)
        splits = _load_split_cache(path, store, seed, provenance)
    _write_manifest(store, splits)
    return splits
