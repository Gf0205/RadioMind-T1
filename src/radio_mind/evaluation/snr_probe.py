"""Metrics and scalar baselines for T2 SNR regression experiments."""
from __future__ import annotations

import numpy as np


def compute_snr_metrics(
    predicted_db,
    true_db,
    expected_snrs: list[int],
) -> dict:
    predicted = np.asarray(predicted_db, dtype=np.float64).reshape(-1)
    target = np.asarray(true_db, dtype=np.float64).reshape(-1)
    if predicted.shape != target.shape or len(target) == 0:
        raise ValueError(
            "predicted and true SNR must have the same non-zero length"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("predicted and true SNR must be finite")
    unexpected = sorted(set(target.tolist()) - set(expected_snrs))
    if unexpected:
        raise ValueError(f"unexpected true SNR values: {unexpected}")

    error = predicted - target
    prediction_std = float(predicted.std())
    target_std = float(target.std())
    correlation = (
        float(np.corrcoef(predicted, target)[0, 1])
        if prediction_std > 0.0 and target_std > 0.0
        else None
    )
    per_snr = {}
    for snr in expected_snrs:
        mask = target == snr
        if not mask.any():
            raise ValueError(f"true SNR {snr} is absent")
        group_prediction = predicted[mask]
        per_snr[str(snr)] = {
            "count": int(mask.sum()),
            "mean_predicted_snr_db": float(group_prediction.mean()),
            "prediction_std_db": float(group_prediction.std()),
            "mae_db": float(np.abs(group_prediction - snr).mean()),
        }
    return {
        "mae_db": float(np.abs(error).mean()),
        "rmse_db": float(np.sqrt(np.mean(error * error))),
        "pearson_correlation": correlation,
        "per_snr": per_snr,
    }


def log_rms_feature(samples, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(samples)
    if x.ndim != 3 or tuple(x.shape[1:]) != (2, 128):
        raise ValueError(f"expected samples shaped (N, 2, 128), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("samples must be finite")
    x64 = x.astype(np.float64, copy=False)
    rms = np.sqrt(np.mean(x64[:, 0] ** 2 + x64[:, 1] ** 2, axis=1))
    return np.log10(rms + eps)


def fit_scalar_linear_regression(feature, target) -> tuple[float, float]:
    x = np.asarray(feature, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or len(y) == 0:
        raise ValueError("feature and target must have the same non-zero length")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("feature and target must be finite")
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)
