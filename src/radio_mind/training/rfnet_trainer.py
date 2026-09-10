"""Controlled RF-Net training used only by the T2.3 paired experiment."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch
from torch import nn

from .trainer import atomic_torch_save


def _finite(value: torch.Tensor, phase: str, epoch: int, batch_index: int, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(
            f"phase={phase} epoch={epoch} batch_index={batch_index}: non-finite {name}"
        )


def _validation(model, loader, device, target_scale_db: float) -> dict:
    model.eval()
    criterion_mod = nn.CrossEntropyLoss()
    criterion_snr = nn.MSELoss()
    mod_loss_sum = 0.0
    snr_loss_sum = 0.0
    snr_absolute_error_db = 0.0
    correct = 0
    seen = 0
    with torch.inference_mode():
        for batch_index, (x, modulation, snr_db) in enumerate(loader):
            x = x.to(device)
            modulation = modulation.to(device)
            snr_db = snr_db.to(device, dtype=torch.float32)
            modulation_logits, predicted_snr_scaled = model.forward_multitask(x)
            _finite(modulation_logits, "validation", 0, batch_index, "modulation logits")
            _finite(predicted_snr_scaled, "validation", 0, batch_index, "SNR prediction")
            mod_loss = criterion_mod(modulation_logits, modulation)
            snr_loss = criterion_snr(predicted_snr_scaled, snr_db / target_scale_db)
            _finite(mod_loss, "validation", 0, batch_index, "modulation loss")
            _finite(snr_loss, "validation", 0, batch_index, "SNR loss")
            batch_size = len(modulation)
            mod_loss_sum += mod_loss.item() * batch_size
            snr_loss_sum += snr_loss.item() * batch_size
            snr_absolute_error_db += float(
                torch.abs(predicted_snr_scaled * target_scale_db - snr_db).sum()
            )
            correct += int((modulation_logits.argmax(dim=1) == modulation).sum())
            seen += batch_size
    if seen == 0:
        raise RuntimeError("validation loader is empty")
    return {
        "modulation_loss": mod_loss_sum / seen,
        "modulation_accuracy": correct / seen,
        "snr_mse": snr_loss_sum / seen,
        "snr_mae_db": snr_absolute_error_db / seen,
    }


def fit_rfnet_control(
    model,
    train_loader,
    val_loader,
    device,
    *,
    experiment_name: str,
    use_snr_loss: bool,
    snr_lambda: float,
    target_scale_db: float,
    lr: float,
    epochs: int,
    patience: int,
    checkpoint_dir: str | Path,
    initial_state_hash: str,
    enable_early_stopping: bool = True,
) -> dict:
    """Train one arm, selecting checkpoints only by validation modulation accuracy."""
    if snr_lambda < 0.0 or target_scale_db <= 0.0:
        raise ValueError("invalid SNR loss configuration")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    criterion_mod = nn.CrossEntropyLoss()
    criterion_snr = nn.MSELoss()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    training_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        model.train()
        order_hash = hashlib.sha256()
        mod_loss_sum = 0.0
        snr_loss_sum = 0.0
        total_loss_sum = 0.0
        correct = 0
        seen = 0
        for batch_index, (x, modulation, snr_db, array_index) in enumerate(train_loader):
            order_hash.update(array_index.numpy().astype("<i8", copy=False).tobytes())
            x = x.to(device)
            modulation = modulation.to(device)
            snr_db = snr_db.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            modulation_logits, predicted_snr_scaled = model.forward_multitask(x)
            _finite(modulation_logits, "train", epoch, batch_index, "modulation logits")
            _finite(predicted_snr_scaled, "train", epoch, batch_index, "SNR prediction")
            mod_loss = criterion_mod(modulation_logits, modulation)
            snr_loss = criterion_snr(predicted_snr_scaled, snr_db / target_scale_db)
            total_loss = mod_loss + snr_lambda * snr_loss if use_snr_loss else mod_loss
            _finite(mod_loss, "train", epoch, batch_index, "modulation loss")
            _finite(snr_loss, "train", epoch, batch_index, "SNR loss")
            _finite(total_loss, "train", epoch, batch_index, "total loss")
            total_loss.backward()
            optimizer.step()

            batch_size = len(modulation)
            mod_loss_sum += mod_loss.detach().item() * batch_size
            snr_loss_sum += snr_loss.detach().item() * batch_size
            total_loss_sum += total_loss.detach().item() * batch_size
            correct += int((modulation_logits.argmax(dim=1) == modulation).sum())
            seen += batch_size
        if seen == 0:
            raise RuntimeError("train loader is empty")
        scheduler.step()

        validation = _validation(model, val_loader, device, target_scale_db)
        row = {
            "epoch": epoch,
            "learning_rate": epoch_lr,
            "train_modulation_loss": mod_loss_sum / seen,
            "train_snr_mse": snr_loss_sum / seen,
            "train_total_loss": total_loss_sum / seen,
            "train_modulation_accuracy": correct / seen,
            "val_modulation_loss": validation["modulation_loss"],
            "val_modulation_accuracy": validation["modulation_accuracy"],
            "val_snr_mse": validation["snr_mse"],
            "val_snr_mae_db": validation["snr_mae_db"],
            "sample_order_sha256": order_hash.hexdigest(),
            "epoch_time_seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"{experiment_name} epoch {epoch}/{epochs}: "
            f"train_mod_loss={row['train_modulation_loss']:.4f} "
            f"train_snr_mse={row['train_snr_mse']:.5f} "
            f"val_mod_loss={row['val_modulation_loss']:.4f} "
            f"val_mod_acc={row['val_modulation_accuracy']:.4f} "
            f"val_snr_mae_db={row['val_snr_mae_db']:.4f} "
            f"lr={epoch_lr:.8g} epoch_time={row['epoch_time_seconds']:.1f}s"
        )

        improved = validation["modulation_accuracy"] > best_accuracy
        if improved:
            best_accuracy = validation["modulation_accuracy"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_modulation_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "experiment_name": experiment_name,
            "use_snr_loss": use_snr_loss,
            "snr_lambda": snr_lambda,
            "initial_state_hash": initial_state_hash,
            "train_time_seconds": time.time() - training_start,
        }
        if improved:
            atomic_torch_save(state, checkpoint_dir / "best.pt")
        atomic_torch_save(state, checkpoint_dir / "last.pt")
        if enable_early_stopping and stale_epochs >= patience:
            break

    return {
        "best_epoch": best_epoch,
        "best_val_modulation_accuracy": best_accuracy,
        "history": history,
        "train_time_seconds": time.time() - training_start,
    }
