from __future__ import annotations

import os
import random
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn


_RESUME_FIELDS = {
    "model",
    "optimizer",
    "scheduler",
    "epoch",
    "best_val_accuracy",
    "best_epoch",
    "stale_epochs",
    "history",
    "train_time_seconds",
    "train_generator_state",
    "python_rng_state",
    "numpy_rng_state",
    "torch_rng_state",
    "cuda_rng_state_all",
}


def atomic_torch_save(state, path: str | Path) -> None:
    """Write a torch checkpoint atomically without damaging an existing file."""
    target = Path(path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _require_finite(
    value: torch.Tensor,
    *,
    phase: str,
    epoch: int,
    batch_index: int,
    name: str,
) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(
            f"phase={phase} epoch={epoch} batch_index={batch_index}: "
            f"non-finite {name}"
        )


def _serialize_numpy_rng_state() -> dict:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": bit_generator,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(state: Mapping) -> None:
    np.random.set_state(_numpy_rng_state_tuple(state))


def _numpy_rng_state_tuple(state: Mapping) -> tuple:
    return (
        state["bit_generator"],
        np.asarray(state["keys"], dtype=np.uint32),
        state["position"],
        state["has_gauss"],
        state["cached_gaussian"],
    )


def validate_resume_state(
    state,
    scheduler_name: str,
    device: str | torch.device | None = None,
) -> None:
    """Reject incomplete or configuration-incompatible resume checkpoints."""
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint must contain a mapping")
    missing = sorted(_RESUME_FIELDS - set(state))
    if missing:
        raise ValueError(f"resume checkpoint missing required fields: {missing}")

    if not isinstance(state["model"], Mapping) or not state["model"]:
        raise ValueError("resume checkpoint field model must be a non-empty mapping")
    optimizer_state = state["optimizer"]
    if (
        not isinstance(optimizer_state, Mapping)
        or set(optimizer_state) != {"state", "param_groups"}
        or not isinstance(optimizer_state["state"], Mapping)
        or not isinstance(optimizer_state["param_groups"], list)
        or not optimizer_state["param_groups"]
    ):
        raise ValueError("resume checkpoint field optimizer is invalid")
    if scheduler_name == "cosine":
        scheduler_fields = {
            "T_max", "eta_min", "base_lrs", "last_epoch", "_step_count",
            "_last_lr",
        }
        if (
            not isinstance(state["scheduler"], Mapping)
            or not scheduler_fields <= set(state["scheduler"])
        ):
            raise ValueError(
                "resume checkpoint requires valid scheduler state for scheduler=cosine"
            )
    elif scheduler_name == "none":
        if state["scheduler"] is not None:
            raise ValueError(
                "resume checkpoint scheduler must be None for scheduler=none"
            )
    else:
        raise ValueError(f"unsupported scheduler: {scheduler_name}")

    for name in ("epoch", "best_epoch", "stale_epochs"):
        if isinstance(state[name], bool) or not isinstance(state[name], int):
            raise ValueError(f"resume checkpoint field {name} must be an integer")
    if state["epoch"] < 0 or not 0 <= state["best_epoch"] <= state["epoch"]:
        raise ValueError("resume checkpoint has invalid epoch/best_epoch")
    if state["stale_epochs"] < 0:
        raise ValueError("resume checkpoint stale_epochs must be non-negative")
    if not isinstance(state["history"], list):
        raise ValueError("resume checkpoint history must be a list")
    if len(state["history"]) != state["epoch"]:
        raise ValueError("resume checkpoint history length must equal epoch")
    if any(
        not isinstance(row, Mapping) or row.get("epoch") != expected_epoch
        for expected_epoch, row in enumerate(state["history"], start=1)
    ):
        raise ValueError("resume checkpoint history epochs must be sequential")
    try:
        best_is_finite = bool(np.isfinite(state["best_val_accuracy"]))
        time_is_finite = bool(np.isfinite(state["train_time_seconds"]))
    except TypeError as error:
        raise ValueError("resume checkpoint numeric fields are invalid") from error
    if not best_is_finite:
        raise ValueError("resume checkpoint best_val_accuracy must be finite")
    if not time_is_finite or state["train_time_seconds"] < 0:
        raise ValueError("resume checkpoint train_time_seconds must be finite and non-negative")

    for name in ("train_generator_state", "torch_rng_state"):
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError(
                f"resume checkpoint field {name} must be a one-dimensional uint8 tensor"
            )
        try:
            torch.Generator().set_state(value)
        except RuntimeError as error:
            raise ValueError(f"resume checkpoint field {name} is invalid") from error
    if not isinstance(state["python_rng_state"], tuple):
        raise ValueError("resume checkpoint python_rng_state must be a tuple")
    try:
        random.Random().setstate(state["python_rng_state"])
    except (TypeError, ValueError) as error:
        raise ValueError("resume checkpoint python_rng_state is invalid") from error
    numpy_state = state["numpy_rng_state"]
    numpy_fields = {
        "bit_generator", "keys", "position", "has_gauss", "cached_gaussian",
    }
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != numpy_fields:
        raise ValueError("resume checkpoint numpy_rng_state is invalid")
    try:
        np.random.RandomState().set_state(_numpy_rng_state_tuple(numpy_state))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("resume checkpoint numpy_rng_state is invalid") from error

    cuda_state = state["cuda_rng_state_all"]
    if cuda_state is not None and (
        not isinstance(cuda_state, list)
        or not all(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.uint8
            and value.ndim == 1
            for value in cuda_state
        )
    ):
        raise ValueError("resume checkpoint cuda_rng_state_all must be a list or None")
    if device is not None and str(device).startswith("cuda") and not cuda_state:
        raise ValueError("resume checkpoint lacks CUDA RNG state for a CUDA run")


def _restore_rng_state(state: Mapping, train_generator: torch.Generator) -> None:
    random.setstate(state["python_rng_state"])
    _restore_numpy_rng_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_state = state["cuda_rng_state_all"]
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    train_generator.set_state(state["train_generator_state"])


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def fit(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    patience,
    scheduler_name,
    ckpt_dir,
    resume_state=None,
):
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1)) if scheduler_name == "cosine" else None
    train_generator = getattr(train_loader, "generator", None)
    if not isinstance(train_generator, torch.Generator):
        raise ValueError("train DataLoader must use an explicit torch.Generator")
    if resume_state is not None:
        validate_resume_state(resume_state, scheduler_name, device)
        opt.load_state_dict(resume_state["optimizer"])
        if sch is not None:
            sch.load_state_dict(resume_state["scheduler"])
        _restore_rng_state(resume_state, train_generator)
    state_source = resume_state if resume_state is not None else {}
    start_epoch = int(state_source.get("epoch", 0))
    loss_fn = nn.CrossEntropyLoss()
    best = float(state_source.get("best_val_accuracy", -1.0))
    best_epoch = int(state_source.get("best_epoch", 0))
    stale = int(state_source.get("stale_epochs", 0))
    history = list(state_source.get("history", []))
    prior_time = float(state_source.get("train_time_seconds", 0.0))
    start = time.time()
    ckpt = Path(ckpt_dir); ckpt.mkdir(parents=True, exist_ok=True)
    if stale >= patience:
        start_epoch = epochs
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        epoch_lr = float(opt.param_groups[0]["lr"])
        model.train(); running = seen = 0
        for batch_index, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
            logits = model(x)
            _require_finite(
                logits, phase="train", epoch=epoch + 1,
                batch_index=batch_index, name="logits",
            )
            loss = loss_fn(logits, y)
            _require_finite(
                loss, phase="train", epoch=epoch + 1,
                batch_index=batch_index, name="loss",
            )
            loss.backward(); opt.step(); running += loss.item() * len(y); seen += len(y)
        if seen == 0:
            raise RuntimeError(f"phase=train epoch={epoch + 1}: empty DataLoader")
        if sch: sch.step()
        model.eval(); val_loss = val_correct = val_seen = 0
        with torch.inference_mode():
            for batch_index, (x, y, _) in enumerate(val_loader):
                x, y = x.to(device), y.to(device); logits = model(x)
                _require_finite(
                    logits, phase="validation", epoch=epoch + 1,
                    batch_index=batch_index, name="logits",
                )
                loss = loss_fn(logits, y)
                _require_finite(
                    loss, phase="validation", epoch=epoch + 1,
                    batch_index=batch_index, name="loss",
                )
                val_loss += loss.item() * len(y); val_correct += int((logits.argmax(1) == y).sum()); val_seen += len(y)
        if val_seen == 0:
            raise RuntimeError(f"phase=validation epoch={epoch + 1}: empty DataLoader")
        val_acc = val_correct / val_seen; row = {"epoch": epoch + 1, "train_loss": running / seen, "val_loss": val_loss / val_seen, "val_accuracy": val_acc, "learning_rate": epoch_lr, "epoch_time_seconds": time.time() - epoch_start}; history.append(row)
        print(
            f"epoch {epoch + 1}/{epochs}: "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={val_acc:.4f} "
            f"lr={epoch_lr:.8g} "
            f"epoch_time={row['epoch_time_seconds']:.1f}s"
        )
        is_best = False
        if val_acc > best:
            best, best_epoch, stale = val_acc, epoch + 1, 0
            is_best = True
        else: stale += 1
        elapsed = prior_time + time.time() - start
        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict() if sch else None,
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_accuracy": best,
            "stale_epochs": stale,
            "history": history,
            "train_time_seconds": elapsed,
            "train_generator_state": train_generator.get_state().clone(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": _serialize_numpy_rng_state(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_state_all": (
                [value.clone() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None
            ),
        }
        if is_best:
            atomic_torch_save(state, ckpt / "best.pt")
        atomic_torch_save(state, ckpt / "last.pt")
        if stale >= patience: break
    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best,
        "history": history,
        "train_time_seconds": prior_time + time.time() - start,
    }
