from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.data.dataset import RMLDataset
from radio_mind.data.load import MODULATIONS, SNRS, load_store
from radio_mind.data.split import load_or_create_splits
from radio_mind.evaluation.metrics import evaluate, save_loss_curve, save_metrics
from radio_mind.evaluation.snr_probe import compute_snr_metrics
from radio_mind.models import RFNet
from radio_mind.training.rfnet_trainer import fit_rfnet_control
from radio_mind.training.trainer import atomic_torch_save, seed_everything


EXPECTED_SPLIT_SIZES = {"train": 154000, "val": 33000, "test": 33000}


class IndexedDataset(Dataset):
    def __init__(self, base: RMLDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, position: int):
        x, modulation, snr_db = self.base[position]
        return x, modulation, snr_db, int(self.base.indices[position])


def _resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
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
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def _new_result_dir() -> tuple[str, Path]:
    while True:
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_t2_multitask_control"
        out = ROOT / "results" / run_id
        try:
            out.mkdir(parents=True, exist_ok=False)
            return run_id, out
        except FileExistsError:
            time.sleep(1.0)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _substate_equal(a: dict, b: dict, prefix: str) -> bool:
    keys = [key for key in a if key.startswith(prefix)]
    return bool(keys) and all(torch.equal(a[key], b[key]) for key in keys)


def _gradient_audit(
    initial_state: dict[str, torch.Tensor],
    first_batch,
    target_scale_db: float,
    snr_lambda: float,
    seed: int,
    device: torch.device,
) -> dict:
    seed_everything(seed, deterministic=True)
    model = RFNet()
    model.load_state_dict(initial_state, strict=True)
    model.to(device)
    model.train()
    x, modulation, snr_db, array_index = first_batch
    x = x.to(device)
    modulation = modulation.to(device)
    snr_db = snr_db.to(device, dtype=torch.float32)
    modulation_logits, predicted_snr_scaled = model.forward_multitask(x)
    mod_loss = nn.CrossEntropyLoss()(modulation_logits, modulation)
    snr_loss = nn.MSELoss()(predicted_snr_scaled, snr_db / target_scale_db)
    encoder_parameters = tuple(model.encoder.parameters())
    mod_gradients = torch.autograd.grad(
        mod_loss, encoder_parameters, retain_graph=True, allow_unused=False
    )
    snr_gradients = torch.autograd.grad(
        snr_loss, encoder_parameters, allow_unused=False
    )
    mod_squared = torch.zeros((), device=device, dtype=torch.float64)
    snr_squared = torch.zeros((), device=device, dtype=torch.float64)
    dot = torch.zeros((), device=device, dtype=torch.float64)
    for mod_gradient, snr_gradient in zip(mod_gradients, snr_gradients):
        if not bool(torch.isfinite(mod_gradient).all()) or not bool(
            torch.isfinite(snr_gradient).all()
        ):
            raise FloatingPointError("gradient audit produced non-finite gradients")
        mod64 = mod_gradient.to(torch.float64)
        snr64 = snr_gradient.to(torch.float64)
        mod_squared += torch.sum(mod64 * mod64)
        snr_squared += torch.sum(snr64 * snr64)
        dot += torch.sum(mod64 * snr64)
    mod_norm = torch.sqrt(mod_squared)
    snr_norm = torch.sqrt(snr_squared)
    cosine = dot / (mod_norm * snr_norm)
    result = {
        "batch_size": len(x),
        "first_batch_array_index_sha256": hashlib.sha256(
            array_index.numpy().astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "mod_loss": mod_loss.item(),
        "snr_loss": snr_loss.item(),
        "mod_grad_norm": mod_norm.item(),
        "snr_grad_norm": snr_norm.item(),
        "weighted_snr_grad_norm": (snr_lambda * snr_norm).item(),
        "gradient_cosine": cosine.item(),
        "weighted_to_mod_norm_ratio": (snr_lambda * snr_norm / mod_norm).item(),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _make_loaders(store, splits, batch_size: int, seed: int):
    train = IndexedDataset(RMLDataset(store, splits["train"], normalize="none"))
    val = RMLDataset(store, splits["val"], normalize="none")
    test = RMLDataset(store, splits["test"], normalize="none")
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def _evaluate_snr(model, loader, device, target_scale_db: float) -> dict:
    model.eval()
    predicted = []
    target = []
    with torch.inference_mode():
        for batch_index, (x, _modulation, snr_db) in enumerate(loader):
            _modulation_logits, prediction_scaled = model.forward_multitask(x.to(device))
            if not bool(torch.isfinite(prediction_scaled).all()):
                raise FloatingPointError(
                    f"SNR test batch_index={batch_index}: non-finite prediction"
                )
            predicted.append((prediction_scaled * target_scale_db).cpu().numpy())
            target.append(snr_db.numpy())
    if not predicted:
        raise RuntimeError("SNR test loader is empty")
    return compute_snr_metrics(np.concatenate(predicted), np.concatenate(target), SNRS)


def _save_snr_plots(metrics: dict, out: Path) -> None:
    per_snr = metrics["per_snr"]
    means = [per_snr[str(snr)]["mean_predicted_snr_db"] for snr in SNRS]
    spreads = [per_snr[str(snr)]["prediction_std_db"] for snr in SNRS]
    mae = [per_snr[str(snr)]["mae_db"] for snr in SNRS]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(SNRS, means, yerr=spreads, marker="o", capsize=3, label="mean ± std")
    ax.plot(SNRS, SNRS, "k--", label="ideal")
    ax.set(xlabel="True SNR (dB)", ylabel="Predicted SNR (dB)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_true_snr.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(SNRS, mae, marker="o")
    ax.set(xlabel="True SNR (dB)", ylabel="MAE (dB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "snr_mae_vs_true.png", dpi=160)
    plt.close(fig)


def _save_arm_metrics(name, model, test_loader, device, train_info, out, snr_metrics=None):
    metrics = evaluate(model, test_loader, str(device), SNRS, MODULATIONS)
    metrics.update(
        {
            "experiment_arm": name,
            "best_epoch": train_info["best_epoch"],
            "best_val_modulation_accuracy": train_info["best_val_modulation_accuracy"],
            "train_time_seconds": train_info["train_time_seconds"],
            "history": train_info["history"],
            "snr_metrics": snr_metrics,
        }
    )
    save_metrics(metrics, out, MODULATIONS, SNRS)
    history_for_plot = [
        {
            "epoch": row["epoch"],
            "train_loss": row["train_modulation_loss"],
            "val_loss": row["val_modulation_loss"],
        }
        for row in train_info["history"]
    ]
    save_loss_curve(history_for_plot, out)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled T2.3 B/C experiment.")
    parser.add_argument("--config", default="configs/t2_multitask_control.yaml")
    args = parser.parse_args()
    config_path = _resolve_from_root(args.config)
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required_contract = {
        "split_seed": 20260907,
        "normalization": "none",
        "batch_size": 512,
        "lr": 0.001,
        "epochs": 50,
        "patience": 8,
        "scheduler": "cosine",
        "dropout": 0.6,
        "snr_lambda": 0.1,
        "target_scale_db": 20.0,
        "seed": 20260907,
    }
    mismatches = {
        key: (cfg.get(key), expected)
        for key, expected in required_contract.items()
        if cfg.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"T2.3 configuration violates the fixed contract: {mismatches}")

    device = _device(str(cfg["device"]))
    store = load_store(_resolve_from_root(cfg["data_root"]))
    splits = load_or_create_splits(store, int(cfg["split_seed"]))
    for name, expected in EXPECTED_SPLIT_SIZES.items():
        if len(splits[name]) != expected:
            raise ValueError(f"{name} split has {len(splits[name])}, expected {expected}")

    seed_everything(int(cfg["seed"]), deterministic=True)
    template = RFNet(dropout=float(cfg["dropout"]))
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in template.state_dict().items()
    }
    initial_hash = _state_hash(initial_state)
    model_b = RFNet(dropout=float(cfg["dropout"]))
    model_c = RFNet(dropout=float(cfg["dropout"]))
    model_b.load_state_dict(initial_state, strict=True)
    model_c.load_state_dict(initial_state, strict=True)
    state_b = model_b.state_dict()
    state_c = model_c.state_dict()
    initial_equivalence = {
        "encoder_initial_equal": _substate_equal(state_b, state_c, "encoder."),
        "modulation_head_initial_equal": _substate_equal(
            state_b, state_c, "modulation_head."
        ),
        "snr_head_initial_equal": _substate_equal(state_b, state_c, "snr_head."),
        "initial_state_hash": initial_hash,
    }
    if not all(
        initial_equivalence[key]
        for key in (
            "encoder_initial_equal",
            "modulation_head_initial_equal",
            "snr_head_initial_equal",
        )
    ):
        raise RuntimeError("B/C initial states are not equal")

    run_id, out = _new_result_dir()
    atomic_torch_save(
        {"model": initial_state, "state_hash": initial_hash, "seed": cfg["seed"]},
        out / "initial_state.pt",
    )
    saved_config = dict(cfg)
    saved_config.update(
        {"run_id": run_id, "device_resolved": str(device), "git_commit": _git_commit()}
    )
    (out / "config.yaml").write_text(
        yaml.safe_dump(saved_config, sort_keys=False), encoding="utf-8"
    )
    print(f"run_id={run_id} device={device} initial_state_hash={initial_hash}")
    for key, value in initial_equivalence.items():
        print(f"{key}={value}")

    audit_train_loader, _audit_val, _audit_test = _make_loaders(
        store, splits, int(cfg["batch_size"]), int(cfg["seed"])
    )
    first_batch = next(iter(audit_train_loader))
    gradient_audit = _gradient_audit(
        initial_state,
        first_batch,
        float(cfg["target_scale_db"]),
        float(cfg["snr_lambda"]),
        int(cfg["seed"]),
        device,
    )
    (out / "gradient_audit.json").write_text(
        json.dumps(gradient_audit, indent=2), encoding="utf-8"
    )
    for key, value in gradient_audit.items():
        print(f"gradient_audit {key}={value}")

    seed_everything(int(cfg["seed"]), deterministic=True)
    train_b, val_b, test_b = _make_loaders(
        store, splits, int(cfg["batch_size"]), int(cfg["seed"])
    )
    info_b = fit_rfnet_control(
        model_b, train_b, val_b, device,
        experiment_name="B_modulation_only",
        use_snr_loss=False,
        snr_lambda=float(cfg["snr_lambda"]),
        target_scale_db=float(cfg["target_scale_db"]),
        lr=float(cfg["lr"]), epochs=int(cfg["epochs"]),
        patience=int(cfg["patience"]), checkpoint_dir=out / "B_modulation_only/checkpoints",
        initial_state_hash=initial_hash,
    )
    best_b = torch.load(
        out / "B_modulation_only/checkpoints/best.pt",
        map_location=device,
        weights_only=False,
    )
    model_b.load_state_dict(best_b["model"], strict=True)
    metrics_b = _save_arm_metrics(
        "B_modulation_only", model_b, test_b, device, info_b,
        out / "B_modulation_only",
    )
    reference_overall = float(cfg["t1_reference"]["overall_accuracy"])
    allowed_difference = float(cfg["t1_reference"]["allowed_overall_abs_difference"])
    b_difference = abs(metrics_b["overall_accuracy"] - reference_overall)
    if b_difference > allowed_difference:
        raise RuntimeError(
            f"B reproduction gate failed: overall abs difference {b_difference:.6f} "
            f"> {allowed_difference:.6f}; C was not started"
        )

    del model_b
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_everything(int(cfg["seed"]), deterministic=True)
    train_c, val_c, test_c = _make_loaders(
        store, splits, int(cfg["batch_size"]), int(cfg["seed"])
    )
    info_c = fit_rfnet_control(
        model_c, train_c, val_c, device,
        experiment_name="C_multitask",
        use_snr_loss=True,
        snr_lambda=float(cfg["snr_lambda"]),
        target_scale_db=float(cfg["target_scale_db"]),
        lr=float(cfg["lr"]), epochs=int(cfg["epochs"]),
        patience=int(cfg["patience"]), checkpoint_dir=out / "C_multitask/checkpoints",
        initial_state_hash=initial_hash,
    )
    best_c = torch.load(
        out / "C_multitask/checkpoints/best.pt",
        map_location=device,
        weights_only=False,
    )
    model_c.load_state_dict(best_c["model"], strict=True)
    snr_metrics_c = _evaluate_snr(
        model_c, test_c, device, float(cfg["target_scale_db"])
    )
    metrics_c = _save_arm_metrics(
        "C_multitask", model_c, test_c, device, info_c,
        out / "C_multitask", snr_metrics=snr_metrics_c,
    )
    _save_snr_plots(snr_metrics_c, out / "C_multitask")

    common_epochs = min(len(info_b["history"]), len(info_c["history"]))
    sample_order_equal = all(
        info_b["history"][index]["sample_order_sha256"]
        == info_c["history"][index]["sample_order_sha256"]
        for index in range(common_epochs)
    )
    lr_equal = all(
        info_b["history"][index]["learning_rate"]
        == info_c["history"][index]["learning_rate"]
        for index in range(common_epochs)
    )
    comparison = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "initial_equivalence": initial_equivalence,
        "gradient_audit": gradient_audit,
        "common_epochs": common_epochs,
        "sample_order_equal_for_common_epochs": sample_order_equal,
        "lr_equal_for_common_epochs": lr_equal,
        "B": {
            "overall_accuracy": metrics_b["overall_accuracy"],
            "accuracy_snr_ge0": metrics_b["accuracy_snr_ge0"],
            "macro_f1": metrics_b["macro_f1"],
            "best_epoch": info_b["best_epoch"],
            "t1_overall_abs_difference": b_difference,
        },
        "C": {
            "overall_accuracy": metrics_c["overall_accuracy"],
            "accuracy_snr_ge0": metrics_c["accuracy_snr_ge0"],
            "macro_f1": metrics_c["macro_f1"],
            "best_epoch": info_c["best_epoch"],
            "snr_mae_db": snr_metrics_c["mae_db"],
            "snr_rmse_db": snr_metrics_c["rmse_db"],
            "snr_pearson": snr_metrics_c["pearson_correlation"],
        },
        "C_minus_B": {
            "overall_accuracy": metrics_c["overall_accuracy"] - metrics_b["overall_accuracy"],
            "accuracy_snr_ge0": metrics_c["accuracy_snr_ge0"] - metrics_b["accuracy_snr_ge0"],
            "macro_f1": metrics_c["macro_f1"] - metrics_b["macro_f1"],
        },
        "frozen_probe_reference": {"mae_db": 5.18930174584461, "rmse_db": 6.876364243549837},
    }
    (out / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
