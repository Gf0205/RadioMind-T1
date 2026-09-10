from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
SNRS = list(range(-20, 20, 2))
CACHE_SCHEMA_VERSION = 1
CELL_SHAPE = (1000, 2, 128)
SAMPLE_SHAPE = (2, 128)
EXPECTED_SAMPLES = len(MODULATIONS) * len(SNRS) * CELL_SHAPE[0]
EXPECTED_X_SHAPE = (EXPECTED_SAMPLES, *SAMPLE_SHAPE)
REQUIRED_METADATA_FIELDS = {
    "modulation_id",
    "snr_db",
    "sample_id",
    "cache_schema_version",
    "source_pickle_size",
    "source_pickle_mtime_ns",
}


class CacheValidationError(ValueError):
    """Raised when a source dataset or converted cache violates its contract."""


@dataclass
class DatasetStore:
    root: Path
    X: np.ndarray
    mod_ids: np.ndarray
    snr_db: np.ndarray
    sample_id: np.ndarray

    @property
    def size(self) -> int:
        return int(self.X.shape[0])


def _find_source_path(data_root: Path) -> Path | None:
    for candidate in (
        data_root / "RML2016.10a_dict.pkl",
        data_root.parent / "RML2016.10a_dict.pkl",
    ):
        if candidate.is_file():
            return candidate
    return None


def _source_path(data_root: Path) -> Path:
    source = _find_source_path(data_root)
    if source is None:
        raise FileNotFoundError(f"RML2016.10a_dict.pkl not found under {data_root}")
    return source


def _expected_keys() -> set[tuple[str, int]]:
    return {(modulation, snr) for modulation in MODULATIONS for snr in SNRS}


def _validate_pickle_keys(raw: object) -> None:
    if not isinstance(raw, dict):
        raise CacheValidationError(
            f"source pickle must contain a dict, got {type(raw).__name__}"
        )
    actual_keys = set(raw.keys())
    expected_keys = _expected_keys()
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        missing_text = ", ".join(sorted(map(repr, missing))) or "none"
        extra_text = ", ".join(sorted(map(repr, extra))) or "none"
        raise CacheValidationError(
            f"source pickle key mismatch; missing=[{missing_text}]; "
            f"extra=[{extra_text}]"
        )


def _expected_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modulation_ids = np.repeat(
        np.arange(len(MODULATIONS), dtype=np.int64),
        len(SNRS) * CELL_SHAPE[0],
    )
    snr_db = np.tile(
        np.repeat(np.asarray(SNRS, dtype=np.int64), CELL_SHAPE[0]),
        len(MODULATIONS),
    )
    sample_ids = np.asarray(
        [
            f"{mod_id:02d}_{snr:+03d}_{cell_index:04d}"
            for mod_id in range(len(MODULATIONS))
            for snr in SNRS
            for cell_index in range(CELL_SHAPE[0])
        ],
        dtype="U16",
    )
    return modulation_ids, snr_db, sample_ids


def _validate_cache_files(
    x_path: Path,
    metadata_path: Path,
    source_path: Path | None = None,
) -> DatasetStore:
    try:
        x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise CacheValidationError(f"cannot load X cache {x_path}: {error}") from error
    if not isinstance(x, np.memmap) or x.mode != "r" or x.flags.writeable:
        raise CacheValidationError("X.npy must be a read-only memory map")
    if x.shape != EXPECTED_X_SHAPE:
        raise CacheValidationError(
            f"invalid X.npy shape: expected {EXPECTED_X_SHAPE}, got {x.shape}"
        )
    if x.dtype != np.dtype(np.float32):
        raise CacheValidationError(
            f"invalid X.npy dtype: expected float32, got {x.dtype}"
        )
    if not np.isfinite(x).all():
        raise CacheValidationError("X.npy contains NaN or Inf")

    try:
        with np.load(metadata_path, allow_pickle=False) as metadata:
            fields = set(metadata.files)
            missing_fields = REQUIRED_METADATA_FIELDS - fields
            if missing_fields:
                raise CacheValidationError(
                    "metadata.npz missing fields: "
                    + ", ".join(sorted(missing_fields))
                )
            modulation_ids = np.asarray(metadata["modulation_id"]).copy()
            snr_db = np.asarray(metadata["snr_db"]).copy()
            sample_ids = np.asarray(metadata["sample_id"]).copy()
            schema_version = int(np.asarray(metadata["cache_schema_version"]).item())
            source_size = int(np.asarray(metadata["source_pickle_size"]).item())
            source_mtime_ns = int(
                np.asarray(metadata["source_pickle_mtime_ns"]).item()
            )
    except CacheValidationError:
        raise
    except Exception as error:
        raise CacheValidationError(
            f"cannot load metadata cache {metadata_path}: {error}"
        ) from error

    if schema_version != CACHE_SCHEMA_VERSION:
        raise CacheValidationError(
            f"unsupported cache schema version: {schema_version}"
        )
    arrays = {
        "modulation_id": modulation_ids,
        "snr_db": snr_db,
        "sample_id": sample_ids,
    }
    for name, values in arrays.items():
        if values.ndim != 1 or len(values) != EXPECTED_SAMPLES:
            raise CacheValidationError(
                f"invalid metadata {name} length/shape: {values.shape}"
            )
    if modulation_ids.dtype != np.dtype(np.int64):
        raise CacheValidationError("metadata modulation_id must be int64")
    if snr_db.dtype != np.dtype(np.int64):
        raise CacheValidationError("metadata snr_db must be int64")
    if sample_ids.dtype.kind != "U":
        raise CacheValidationError("metadata sample_id must be a Unicode array")
    if len(np.unique(sample_ids)) != EXPECTED_SAMPLES:
        raise CacheValidationError("metadata sample_id values must be globally unique")

    expected_modulation_ids, expected_snr_db, expected_sample_ids = _expected_metadata()
    if not np.array_equal(modulation_ids, expected_modulation_ids):
        raise CacheValidationError(
            "metadata modulation_id does not match the fixed array_index order"
        )
    if not np.array_equal(snr_db, expected_snr_db):
        raise CacheValidationError(
            "metadata snr_db does not match the fixed array_index order"
        )
    if not np.array_equal(sample_ids, expected_sample_ids):
        raise CacheValidationError(
            "metadata sample_id does not match the fixed array_index order"
        )

    if source_path is not None:
        source_stat = source_path.stat()
        if source_stat.st_size != source_size:
            raise CacheValidationError(
                "source pickle size differs from cache provenance"
            )
        if source_stat.st_mtime_ns != source_mtime_ns:
            raise CacheValidationError(
                "source pickle mtime differs from cache provenance"
            )

    return DatasetStore(
        root=x_path.parent,
        X=x,
        mod_ids=modulation_ids,
        snr_db=snr_db,
        sample_id=sample_ids,
    )


def validate_cache(
    data_root: str | Path,
    source_path: str | Path | None = None,
) -> DatasetStore:
    """Validate and open an existing cache without modifying it."""
    root = Path(data_root)
    source = Path(source_path) if source_path is not None else _find_source_path(root)
    return _validate_cache_files(root / "X.npy", root / "metadata.npz", source)


def _commit_cache_files(
    temporary_x: Path,
    temporary_metadata: Path,
    final_x: Path,
    final_metadata: Path,
) -> None:
    backup_x = final_x.with_name(f".{final_x.name}.radiomind-backup")
    backup_metadata = final_metadata.with_name(
        f".{final_metadata.name}.radiomind-backup"
    )
    if backup_x.exists() or backup_metadata.exists():
        raise CacheValidationError(
            "unfinished cache backup exists; inspect it before rebuilding"
        )
    had_x = final_x.exists()
    had_metadata = final_metadata.exists()
    try:
        if had_x:
            os.replace(final_x, backup_x)
        if had_metadata:
            os.replace(final_metadata, backup_metadata)
        os.replace(temporary_x, final_x)
        os.replace(temporary_metadata, final_metadata)
    except BaseException:
        if backup_x.exists():
            os.replace(backup_x, final_x)
        elif not had_x:
            final_x.unlink(missing_ok=True)
        if backup_metadata.exists():
            os.replace(backup_metadata, final_metadata)
        elif not had_metadata:
            final_metadata.unlink(missing_ok=True)
        raise
    else:
        backup_x.unlink(missing_ok=True)
        backup_metadata.unlink(missing_ok=True)


def _write_cache_atomically(
    root: Path,
    x: np.ndarray,
    modulation_ids: np.ndarray,
    snr_db: np.ndarray,
    sample_ids: np.ndarray,
    source_path: Path,
    source_size: int,
    source_mtime_ns: int,
) -> None:
    x_fd, x_name = tempfile.mkstemp(
        dir=root, prefix=".radiomind-", suffix=".tmp.npy"
    )
    metadata_fd, metadata_name = tempfile.mkstemp(
        dir=root, prefix=".radiomind-", suffix=".tmp.npz"
    )
    os.close(x_fd)
    os.close(metadata_fd)
    temporary_x = Path(x_name)
    temporary_metadata = Path(metadata_name)
    try:
        np.save(temporary_x, x, allow_pickle=False)
        np.savez(
            temporary_metadata,
            modulation_id=modulation_ids,
            snr_db=snr_db,
            sample_id=sample_ids,
            cache_schema_version=np.int64(CACHE_SCHEMA_VERSION),
            source_pickle_size=np.int64(source_size),
            source_pickle_mtime_ns=np.int64(source_mtime_ns),
        )
        _validate_cache_files(temporary_x, temporary_metadata, source_path)
        _commit_cache_files(
            temporary_x,
            temporary_metadata,
            root / "X.npy",
            root / "metadata.npz",
        )
    finally:
        temporary_x.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def load_store(data_root: str | Path, force: bool = False) -> DatasetStore:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    x_path = root / "X.npy"
    metadata_path = root / "metadata.npz"
    source = _find_source_path(root)

    if not force and x_path.exists() and metadata_path.exists():
        try:
            return _validate_cache_files(x_path, metadata_path, source)
        except CacheValidationError as error:
            raise CacheValidationError(
                f"existing cache is invalid: {error}; rerun with force=True to rebuild"
            ) from error
    if not force and (x_path.exists() or metadata_path.exists()):
        raise CacheValidationError(
            "cache is incomplete; rerun with force=True to rebuild"
        )

    source_path = _source_path(root)
    source_stat_before = source_path.stat()
    with source_path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    _validate_pickle_keys(raw)

    arrays: list[np.ndarray] = []
    modulation_ids: list[int] = []
    snr_values: list[int] = []
    sample_ids: list[str] = []
    for mod_id, modulation in enumerate(MODULATIONS):
        for snr in SNRS:
            key = (modulation, snr)
            array = np.asarray(raw[key])
            if array.shape != CELL_SHAPE:
                raise CacheValidationError(
                    f"unexpected shape for {key}: expected {CELL_SHAPE}, got {array.shape}"
                )
            if array.dtype != np.dtype(np.float32):
                raise CacheValidationError(
                    f"unexpected dtype for {key}: expected float32, got {array.dtype}"
                )
            if not np.isfinite(array).all():
                raise CacheValidationError(f"source cell {key} contains NaN or Inf")
            arrays.append(array)
            modulation_ids.extend([mod_id] * CELL_SHAPE[0])
            snr_values.extend([snr] * CELL_SHAPE[0])
            sample_ids.extend(
                f"{mod_id:02d}_{snr:+03d}_{cell_index:04d}"
                for cell_index in range(CELL_SHAPE[0])
            )

    x = np.concatenate(arrays, axis=0)
    modulation_id_array = np.asarray(modulation_ids, dtype=np.int64)
    snr_array = np.asarray(snr_values, dtype=np.int64)
    sample_id_array = np.asarray(sample_ids, dtype="U16")
    source_stat_after = source_path.stat()
    if (
        source_stat_after.st_size != source_stat_before.st_size
        or source_stat_after.st_mtime_ns != source_stat_before.st_mtime_ns
    ):
        raise CacheValidationError("source pickle changed while cache was generated")
    _write_cache_atomically(
        root,
        x,
        modulation_id_array,
        snr_array,
        sample_id_array,
        source_path,
        source_stat_before.st_size,
        source_stat_before.st_mtime_ns,
    )
    return _validate_cache_files(x_path, metadata_path, source_path)
