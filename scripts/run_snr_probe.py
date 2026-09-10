from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.data.dataset import RMLDataset
from radio_mind.data.load import SNRS, load_store
from radio_mind.data.split import load_or_create_splits
from radio_mind.evaluation.snr_probe import (
    compute_snr_metrics,
    fit_scalar_linear_regression,
    log_rms_feature,
)
from radio_mind.models import RFNet, load_t1_modulation_weights
from radio_mind.training.trainer import atomic_torch_save, seed_everything


EXPECTED_SPLIT_SIZES = {"train": 154000, "val": 33000, "test": 33000}


def _resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device: {name}")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _new_result_dir() -> tuple[str, Path]:
    while True:
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_t2_snr_probe"
        out = ROOT / "results" / run_id
        try:
            out.mkdir(parents=True, exist_ok=False)
            return run_id, out
        except FileExistsError:
            time.sleep(1.0)


def _extract_features(
    encoder: nn.Module,
    dataset: RMLDataset,
    batch_size: int,
    device: torch.device,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = torch.empty((len(dataset), 256), dtype=torch.float32)
    targets_db = torch.empty(len(dataset), dtype=torch.float32)
    offset = 0
    with torch.inference_mode():
        for batch_index, (x, _modulation, snr_db) in enumerate(loader):
            z = encoder(x.to(device)).cpu()
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError(
                    f"phase={phase} batch_index={batch_index}: non-finite features"
                )
            end = offset + len(x)
            features[offset:end] = z
            targets_db[offset:end] = snr_db.to(torch.float32)
            offset = end
            if (batch_index + 1) % 50 == 0 or offset == len(dataset):
                print(f"feature_extract phase={phase} samples={offset}/{len(dataset)}")
    if offset != len(dataset):
        raise RuntimeError(f"phase={phase}: feature extraction length mismatch")
    return features, targets_db


def _evaluate_head(
    head: nn.Module,
    features: torch.Tensor,
    target_db: torch.Tensor,
    batch_size: int,
    target_scale_db: float,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    head.eval()
    loader = DataLoader(
        TensorDataset(features, target_db),
        batch_size=batch_size,
        shuffle=False,
    )
    squared_error_scaled = 0.0
    absolute_error_db = 0.0
    seen = 0
    predictions = []
    with torch.inference_mode():
        for batch_index, (z, target) in enumerate(loader):
            prediction_scaled = head(z.to(device))
            target = target.to(device)
            if not bool(torch.isfinite(prediction_scaled).all()):
                raise FloatingPointError(
                    f"phase=evaluate batch_index={batch_index}: non-finite prediction"
                )
            target_scaled = target / target_scale_db
            difference_scaled = prediction_scaled - target_scaled
            prediction_db = prediction_scaled * target_scale_db
            squared_error_scaled += float((difference_scaled**2).sum())
            absolute_error_db += float(torch.abs(prediction_db - target).sum())
            seen += len(target)
            predictions.append(prediction_db.cpu())
    if seen == 0:
        raise RuntimeError("cannot evaluate an empty feature set")
    return (
        squared_error_scaled / seen,
        absolute_error_db / seen,
        torch.cat(predictions).numpy(),
    )


def _raw_log_rms(store, indices: np.ndarray, batch_size: int) -> np.ndarray:
    result = np.empty(len(indices), dtype=np.float64)
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset : offset + batch_size]
        result[offset : offset + len(selected)] = log_rms_feature(store.X[selected])
    return result


def _save_plots(metrics: dict, out: Path) -> None:
    per_snr = metrics["per_snr"]
    x = np.asarray(SNRS)
    means = np.asarray([per_snr[str(s)]["mean_predicted_snr_db"] for s in SNRS])
    spreads = np.asarray([per_snr[str(s)]["prediction_std_db"] for s in SNRS])
    mae = np.asarray([per_snr[str(s)]["mae_db"] for s in SNRS])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, means, yerr=spreads, marker="o", capsize=3, label="probe mean ± std")
    ax.plot(x, x, linestyle="--", color="black", label="ideal")
    ax.set(xlabel="True SNR (dB)", ylabel="Predicted SNR (dB)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_true_snr.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, mae, marker="o")
    ax.set(xlabel="True SNR (dB)", ylabel="MAE (dB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "snr_mae_vs_true.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen RFEncoder SNR probe.")
    parser.add_argument("--config", default="configs/t2_snr_probe.yaml")
    args = parser.parse_args()
    config_path = _resolve_from_root(args.config)
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("normalization") != "none":
        raise ValueError("T2.2 requires normalization=none")
    if int(cfg["split_seed"]) != 20260907:
        raise ValueError("T2.2 requires split_seed=20260907")
    if float(cfg["target_scale_db"]) <= 0:
        raise ValueError("target_scale_db must be positive")

    seed_everything(int(cfg["seed"]), deterministic=True)
    device = _device(str(cfg["device"]))
    store = load_store(_resolve_from_root(cfg["data_root"]))
    splits = load_or_create_splits(store, int(cfg["split_seed"]))
    for split_name, expected_size in EXPECTED_SPLIT_SIZES.items():
        if len(splits[split_name]) != expected_size:
            raise ValueError(
                f"{split_name} split has {len(splits[split_name])}, expected {expected_size}"
            )

    checkpoint_path = _resolve_from_root(cfg["t1_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("trusted T1 checkpoint must contain model weights")
    model = RFNet(dropout=0.6)
    load_t1_modulation_weights(model, checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.snr_head.parameters():
        parameter.requires_grad_(True)
    model.encoder.eval()
    model.modulation_head.eval()
    model.snr_head.train()
    model.to(device)

    encoder_before = {
        key: value.detach().cpu().clone()
        for key, value in model.encoder.state_dict().items()
    }
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_params != 257:
        raise RuntimeError(f"expected 257 trainable parameters, got {trainable_params}")

    run_id, out = _new_result_dir()
    saved_config = dict(cfg)
    saved_config.update(
        {
            "run_id": run_id,
            "device_resolved": str(device),
            "git_commit": _git_commit(),
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
    )
    (out / "config.yaml").write_text(
        yaml.safe_dump(saved_config, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"run_id={run_id} device={device} total_params={total_params} "
        f"trainable_params={trainable_params}"
    )

    train_dataset = RMLDataset(store, splits["train"], normalize="none")
    val_dataset = RMLDataset(store, splits["val"], normalize="none")
    train_features, train_target_db = _extract_features(
        model.encoder, train_dataset, int(cfg["batch_size"]), device, "train"
    )
    val_features, val_target_db = _extract_features(
        model.encoder, val_dataset, int(cfg["batch_size"]), device, "validation"
    )

    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        TensorDataset(train_features, train_target_db / float(cfg["target_scale_db"])),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.snr_head.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.MSELoss()
    best_val_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    encoder_grad_none = True
    head_grad_nonzero_finite = True
    checkpoint_out = out / "best_snr_head.pt"

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.encoder.eval()
        model.modulation_head.eval()
        model.snr_head.train()
        train_squared_error = 0.0
        train_seen = 0
        for batch_index, (z, target_scaled) in enumerate(train_loader):
            z = z.to(device)
            target_scaled = target_scaled.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction_scaled = model.snr_head(z)
            loss = loss_fn(prediction_scaled, target_scaled)
            if not bool(torch.isfinite(prediction_scaled).all()) or not bool(
                torch.isfinite(loss)
            ):
                raise FloatingPointError(
                    f"phase=train epoch={epoch} batch_index={batch_index}: non-finite result"
                )
            loss.backward()
            encoder_grad_none = encoder_grad_none and all(
                parameter.grad is None for parameter in model.encoder.parameters()
            )
            head_gradient_norm = 0.0
            for parameter in model.snr_head.parameters():
                gradient = parameter.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    head_grad_nonzero_finite = False
                else:
                    head_gradient_norm += float(gradient.norm())
            if head_gradient_norm == 0.0:
                head_grad_nonzero_finite = False
            if not encoder_grad_none or not head_grad_nonzero_finite:
                raise RuntimeError("frozen encoder or SNR-head gradient contract failed")
            optimizer.step()
            train_squared_error += float(loss) * len(z)
            train_seen += len(z)
        if train_seen == 0:
            raise RuntimeError("empty train feature loader")

        val_mse, val_mae_db, _ = _evaluate_head(
            model.snr_head,
            val_features,
            val_target_db,
            int(cfg["batch_size"]),
            float(cfg["target_scale_db"]),
            device,
        )
        row = {
            "epoch": epoch,
            "train_mse": train_squared_error / train_seen,
            "val_mse": val_mse,
            "val_mae_db": val_mae_db,
        }
        history.append(row)
        print(
            f"epoch {epoch}/{cfg['max_epochs']}: train_mse={row['train_mse']:.6f} "
            f"val_mse={val_mse:.6f} val_mae_db={val_mae_db:.4f}"
        )
        if val_mae_db < best_val_mae:
            best_val_mae = val_mae_db
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                {
                    "snr_head": model.snr_head.state_dict(),
                    "epoch": epoch,
                    "val_mae_db": val_mae_db,
                    "val_mse": val_mse,
                },
                checkpoint_out,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(cfg["patience"]):
            break

    best = torch.load(checkpoint_out, map_location=device, weights_only=False)
    model.snr_head.load_state_dict(best["snr_head"], strict=True)
    model.snr_head.eval()

    test_dataset = RMLDataset(store, splits["test"], normalize="none")
    test_features, test_target_db = _extract_features(
        model.encoder, test_dataset, int(cfg["batch_size"]), device, "test"
    )
    _test_mse, _test_mae, probe_prediction_db = _evaluate_head(
        model.snr_head,
        test_features,
        test_target_db,
        int(cfg["batch_size"]),
        float(cfg["target_scale_db"]),
        device,
    )
    true_test_db = test_target_db.numpy().astype(np.float64)
    probe_metrics = compute_snr_metrics(probe_prediction_db, true_test_db, SNRS)

    train_true_db = train_target_db.numpy().astype(np.float64)
    constant_prediction_db = np.full_like(true_test_db, train_true_db.mean())
    constant_metrics = compute_snr_metrics(constant_prediction_db, true_test_db, SNRS)

    train_log_rms = _raw_log_rms(store, splits["train"], int(cfg["batch_size"]))
    test_log_rms = _raw_log_rms(store, splits["test"], int(cfg["batch_size"]))
    rms_slope, rms_intercept = fit_scalar_linear_regression(
        train_log_rms,
        train_true_db,
    )
    rms_prediction_db = rms_slope * test_log_rms + rms_intercept
    rms_metrics = compute_snr_metrics(rms_prediction_db, true_test_db, SNRS)

    encoder_after = model.encoder.state_dict()
    encoder_weights_unchanged = all(
        torch.equal(encoder_before[key], encoder_after[key].detach().cpu())
        for key in encoder_before
    )
    if not encoder_weights_unchanged:
        raise RuntimeError("frozen encoder weights changed")

    metrics = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "t1_checkpoint": str(checkpoint_path),
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "total_params": total_params,
        "trainable_params": trainable_params,
        "encoder_grad_none": encoder_grad_none,
        "snr_head_grad_nonzero_finite": head_grad_nonzero_finite,
        "encoder_weights_unchanged": encoder_weights_unchanged,
        "best_epoch": best_epoch,
        "best_val_mae_db": best_val_mae,
        "history": history,
        "constant_mean_db": float(train_true_db.mean()),
        "raw_rms_linear": {"slope": rms_slope, "intercept": rms_intercept},
        "constant_mean_baseline": constant_metrics,
        "raw_rms_linear_baseline": rms_metrics,
        "frozen_rf_linear_probe": probe_metrics,
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _save_plots(probe_metrics, out)

    print(f"trainable_params={trainable_params}")
    print(f"encoder_grad_none={encoder_grad_none}")
    print(f"encoder_weights_unchanged={encoder_weights_unchanged}")
    print(f"snr_head_grad_nonzero_finite={head_grad_nonzero_finite}")
    print(f"best_epoch={best_epoch}")
    print(f"best_val_mae_db={best_val_mae:.6f}")
    for name, result in (
        ("constant_mean", constant_metrics),
        ("raw_rms_linear", rms_metrics),
        ("frozen_rf_linear_probe", probe_metrics),
    ):
        print(
            f"{name}: mae_db={result['mae_db']:.6f} "
            f"rmse_db={result['rmse_db']:.6f} "
            f"correlation={result['pearson_correlation']}"
        )


if __name__ == "__main__":
    main()
