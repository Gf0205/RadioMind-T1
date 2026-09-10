from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.data.dataset import RMLDataset
from radio_mind.data.load import MODULATIONS, SNRS, load_store
from radio_mind.data.split import load_or_create_splits
from radio_mind.evaluation.metrics import evaluate, save_loss_curve, save_metrics
from radio_mind.models import RFNet
from radio_mind.training.rfnet_trainer import fit_rfnet_control
from radio_mind.training.trainer import seed_everything


EXPECTED_SPLIT_SIZES = {"train": 154000, "val": 33000, "test": 33000}
REPRODUCTION_FIELDS = (
    "sample_order_sha256",
    "learning_rate",
    "train_modulation_loss",
    "val_modulation_accuracy",
)


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


def _new_result_dir() -> tuple[str, Path]:
    while True:
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_t2_b50_control"
        out = ROOT / "results" / run_id
        try:
            out.mkdir(parents=True, exist_ok=False)
            return run_id, out
        except FileExistsError:
            time.sleep(1.0)


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


def _compare_prefix(old_history: list[dict], new_history: list[dict]) -> dict:
    if len(new_history) < len(old_history):
        raise ValueError("B50 history is shorter than the original B history")
    mismatches = []
    for row_index, old_row in enumerate(old_history):
        new_row = new_history[row_index]
        for field in REPRODUCTION_FIELDS:
            if old_row[field] != new_row[field]:
                mismatches.append(
                    {
                        "epoch": row_index + 1,
                        "field": field,
                        "old": old_row[field],
                        "new": new_row[field],
                    }
                )
    return {
        "epochs_compared": len(old_history),
        "fields": list(REPRODUCTION_FIELDS),
        "strict_equal": not mismatches,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the matched-budget B50 control.")
    parser.add_argument("--config", default="configs/t2_b50_control.yaml")
    args = parser.parse_args()
    config_path = _resolve_from_root(args.config)
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    expected = {
        "split_seed": 20260907,
        "normalization": "none",
        "seed": 20260907,
        "device": "cuda",
        "dropout": 0.6,
        "batch_size": 512,
        "lr": 0.001,
        "epochs": 50,
        "patience": 8,
        "scheduler": "cosine",
        "target_scale_db": 20.0,
        "snr_lambda": 0.1,
        "early_stopping": False,
    }
    mismatches = {
        key: (cfg.get(key), value)
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B50 configuration violates the fixed contract: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("B50 control requires CUDA")
    device = torch.device("cuda")

    source = _resolve_from_root(cfg["source_control_run"])
    initial_checkpoint = torch.load(
        source / "initial_state.pt", map_location="cpu", weights_only=False
    )
    old_b_metrics = json.loads(
        (source / "B_modulation_only/metrics.json").read_text(encoding="utf-8")
    )
    old_c_metrics = json.loads(
        (source / "C_multitask/metrics.json").read_text(encoding="utf-8")
    )
    old_comparison = json.loads(
        (source / "comparison.json").read_text(encoding="utf-8")
    )
    if initial_checkpoint.get("state_hash") != old_comparison["initial_equivalence"]["initial_state_hash"]:
        raise ValueError("source initial-state hash does not match comparison.json")

    store = load_store(_resolve_from_root(cfg["data_root"]))
    splits = load_or_create_splits(store, int(cfg["split_seed"]))
    for name, size in EXPECTED_SPLIT_SIZES.items():
        if len(splits[name]) != size:
            raise ValueError(f"{name} split has {len(splits[name])}, expected {size}")

    model = RFNet(dropout=float(cfg["dropout"]))
    model.load_state_dict(initial_checkpoint["model"], strict=True)
    seed_everything(int(cfg["seed"]), deterministic=True)
    train_loader, val_loader, test_loader = _make_loaders(
        store, splits, int(cfg["batch_size"]), int(cfg["seed"])
    )
    run_id, out = _new_result_dir()
    saved_config = dict(cfg)
    saved_config.update(
        {"run_id": run_id, "git_commit": _git_commit(), "device_resolved": str(device)}
    )
    (out / "config.yaml").write_text(
        yaml.safe_dump(saved_config, sort_keys=False), encoding="utf-8"
    )
    print(
        f"run_id={run_id} device={device} "
        f"initial_state_hash={initial_checkpoint['state_hash']}"
    )

    train_info = fit_rfnet_control(
        model, train_loader, val_loader, device,
        experiment_name="B50_modulation_only",
        use_snr_loss=False,
        snr_lambda=float(cfg["snr_lambda"]),
        target_scale_db=float(cfg["target_scale_db"]),
        lr=float(cfg["lr"]), epochs=int(cfg["epochs"]),
        patience=int(cfg["patience"]), checkpoint_dir=out / "checkpoints",
        initial_state_hash=initial_checkpoint["state_hash"],
        enable_early_stopping=False,
    )
    reproduction = _compare_prefix(old_b_metrics["history"], train_info["history"])
    (out / "reproduction_check.json").write_text(
        json.dumps(reproduction, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"reproduction_epochs={reproduction['epochs_compared']} "
        f"strict_equal={reproduction['strict_equal']}"
    )
    if not reproduction["strict_equal"]:
        raise RuntimeError("B50 failed strict reproduction of the original B prefix")

    best = torch.load(out / "checkpoints/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    metrics = evaluate(model, test_loader, str(device), SNRS, MODULATIONS)
    metrics.update(
        {
            "run_id": run_id,
            "git_commit": _git_commit(),
            "experiment_arm": "B50_matched_budget",
            "best_epoch": train_info["best_epoch"],
            "best_val_modulation_accuracy": train_info["best_val_modulation_accuracy"],
            "train_time_seconds": train_info["train_time_seconds"],
            "history": train_info["history"],
            "reproduction_check": reproduction,
        }
    )
    save_metrics(metrics, out, MODULATIONS, SNRS)
    save_loss_curve(
        [
            {
                "epoch": row["epoch"],
                "train_loss": row["train_modulation_loss"],
                "val_loss": row["val_modulation_loss"],
            }
            for row in train_info["history"]
        ],
        out,
    )

    comparison = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "reproduction_check": reproduction,
        "original_B": {
            key: old_b_metrics[key]
            for key in ("overall_accuracy", "accuracy_snr_ge0", "macro_f1", "best_epoch")
        },
        "B50": {
            key: metrics[key]
            for key in ("overall_accuracy", "accuracy_snr_ge0", "macro_f1", "best_epoch")
        },
        "C50": {
            key: old_c_metrics[key]
            for key in ("overall_accuracy", "accuracy_snr_ge0", "macro_f1", "best_epoch")
        },
        "C50_minus_B50": {
            key: old_c_metrics[key] - metrics[key]
            for key in ("overall_accuracy", "accuracy_snr_ge0", "macro_f1")
        },
        "per_snr": {
            snr: {
                "original_B": old_b_metrics["per_snr_accuracy"][snr],
                "B50": metrics["per_snr_accuracy"][snr],
                "C50": old_c_metrics["per_snr_accuracy"][snr],
                "C50_minus_B50": old_c_metrics["per_snr_accuracy"][snr]
                - metrics["per_snr_accuracy"][snr],
            }
            for snr in map(str, SNRS)
        },
    }
    (out / "matched_budget_comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
