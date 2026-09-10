from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


def evaluate(model, loader, device: str, snrs: list[int], class_names: list[str]) -> dict:
    model.eval(); ys, ps, ss = [], [], []
    loss_sum = 0.0; n = 0; criterion = torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch_index, (x, y, snr) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    f"evaluation batch_index={batch_index}: non-finite logits"
                )
            loss = criterion(logits, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"evaluation batch_index={batch_index}: non-finite loss"
                )
            loss_sum += float(loss) * len(y); n += len(y)
            ys.extend(y.cpu().numpy()); ps.extend(logits.argmax(1).cpu().numpy()); ss.extend(snr.numpy())
    ys, ps, ss = np.asarray(ys), np.asarray(ps), np.asarray(ss)
    if n == 0:
        raise ValueError("evaluation loader is empty")
    if not (len(ys) == len(ps) == len(ss) == n):
        raise ValueError(
            "evaluation prediction, label, and SNR lengths must match"
        )
    num_classes = len(class_names)
    if np.any((ys < 0) | (ys >= num_classes)):
        raise ValueError(f"evaluation labels must be in [0, {num_classes - 1}]")
    if np.any((ps < 0) | (ps >= num_classes)):
        raise ValueError(f"evaluation predictions must be in [0, {num_classes - 1}]")
    expected_snrs = set(snrs)
    actual_snrs = set(ss.tolist())
    unexpected_snrs = sorted(actual_snrs - expected_snrs)
    if unexpected_snrs:
        raise ValueError(f"evaluation contains unexpected SNR values: {unexpected_snrs}")
    missing_snrs = [snr for snr in snrs if snr not in actual_snrs]
    if missing_snrs:
        raise ValueError(f"evaluation is missing required SNR values: {missing_snrs}")
    cm = confusion_matrix(ys, ps, labels=np.arange(len(class_names)))
    per_snr = {str(s): float((ps[ss == s] == ys[ss == s]).mean()) for s in snrs}
    high = ps[ss >= 0] == ys[ss >= 0]
    return {
        "test_loss": loss_sum / n,
        "overall_accuracy": float((ps == ys).mean()),
        "accuracy_snr_ge0": float(high.mean()),
        "macro_f1": float(f1_score(
            ys, ps, labels=np.arange(len(class_names)),
            average="macro", zero_division=0,
        )),
        "per_snr_accuracy": per_snr,
        "confusion_matrix": (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist(),
        "per_class_f1": {class_names[i]: float(v) for i, v in enumerate(f1_score(ys, ps, labels=np.arange(len(class_names)), average=None, zero_division=0))},
        "_y": ys, "_p": ps, "_s": ss,
    }


def save_metrics(metrics: dict, out_dir: str | Path, class_names: list[str], snrs: list[int]) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    y, p, s = metrics.pop("_y"), metrics.pop("_p"), metrics.pop("_s")
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plt.figure(figsize=(8, 4)); plt.plot(snrs, [metrics["per_snr_accuracy"][str(x)] for x in snrs], marker="o")
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy"); plt.ylim(0, 1); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(out / "acc_vs_snr.png", dpi=140); plt.close()
    cm = np.asarray(metrics["confusion_matrix"])
    plt.figure(figsize=(8, 7)); plt.imshow(cm, vmin=0, vmax=1, cmap="Blues"); plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right"); plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=140); plt.close()

def save_loss_curve(history: list[dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.grid(alpha=0.25); plt.legend()
    plt.tight_layout(); plt.savefig(out / "loss_curve.png", dpi=140); plt.close()
