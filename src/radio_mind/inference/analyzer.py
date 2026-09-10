"""Strict RF-Net v1 modulation and SNR inference interface."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from radio_mind.data.load import MODULATIONS
from radio_mind.models import RFNet


MODEL_TYPE = "RFNet"
ARCHITECTURE_VERSION = "rfnet_v1"
EXPECTED_INPUT_SHAPE = (2, 128)
EXPECTED_CHANNEL_ORDER = ["I", "Q"]
REQUIRED_METADATA_FIELDS = {
    "deployment_schema_version",
    "model_type",
    "architecture_version",
    "modulation_classes",
    "snr_scale_db",
    "normalization",
    "source_git_commit",
    "training_arm",
    "snr_lambda",
    "dropout",
    "input_shape",
    "channel_order",
    "checkpoint_sha256",
}


class CheckpointContractError(ValueError):
    """Raised when a checkpoint or its deployment metadata is incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name not in {"cuda", "cpu"}:
        raise ValueError(f"device must be one of auto/cuda/cpu, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointContractError(f"deployment metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise CheckpointContractError("deployment metadata must be a mapping")
    missing = sorted(REQUIRED_METADATA_FIELDS - value.keys())
    if missing:
        raise CheckpointContractError(f"deployment metadata missing fields: {missing}")
    return value


def _validate_metadata(metadata: dict[str, Any], checkpoint_path: Path) -> None:
    expected = {
        "deployment_schema_version": 1,
        "model_type": MODEL_TYPE,
        "architecture_version": ARCHITECTURE_VERSION,
        "modulation_classes": MODULATIONS,
        "snr_scale_db": 20.0,
        "normalization": "none",
        "training_arm": "multitask",
        "snr_lambda": 0.1,
        "dropout": 0.6,
        "input_shape": list(EXPECTED_INPUT_SHAPE),
        "channel_order": EXPECTED_CHANNEL_ORDER,
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    source_commit = metadata.get("source_git_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        mismatches["source_git_commit"] = {
            "actual": source_commit,
            "expected": "a 40-character Git commit",
        }
    actual_hash = _sha256(checkpoint_path)
    expected_hash = str(metadata.get("checkpoint_sha256", "")).lower()
    if actual_hash != expected_hash:
        mismatches["checkpoint_sha256"] = {
            "actual": actual_hash,
            "expected": expected_hash,
        }
    if mismatches:
        raise CheckpointContractError(f"deployment metadata mismatch: {mismatches}")


def _validate_checkpoint(checkpoint: object, metadata: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise CheckpointContractError("checkpoint must contain a mapping")
    required = {"model", "experiment_name", "use_snr_loss", "snr_lambda"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise CheckpointContractError(f"checkpoint missing required fields: {missing}")
    if checkpoint["experiment_name"] != "C_multitask":
        raise CheckpointContractError(
            f"checkpoint training arm must be C_multitask, got {checkpoint['experiment_name']!r}"
        )
    if checkpoint["use_snr_loss"] is not True:
        raise CheckpointContractError("checkpoint is not a multitask checkpoint")
    if float(checkpoint["snr_lambda"]) != float(metadata["snr_lambda"]):
        raise CheckpointContractError("checkpoint snr_lambda does not match metadata")
    state = checkpoint["model"]
    if not isinstance(state, dict) or not state:
        raise CheckpointContractError("checkpoint model state must be a non-empty mapping")
    return state


class RFAnalyzer:
    """Analyze one or more I/Q samples with the frozen RF-Net v1 deployment."""

    def __init__(
        self,
        model: RFNet,
        *,
        device: torch.device,
        modulation_classes: list[str],
        snr_scale_db: float,
        normalization: str,
        metadata: dict[str, Any],
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.modulation_classes = tuple(modulation_classes)
        self.snr_scale_db = float(snr_scale_db)
        self.normalization = normalization
        self.metadata = dict(metadata)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        metadata: str | Path,
        device: str = "auto",
    ) -> "RFAnalyzer":
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        metadata_path = Path(metadata).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        deployment = _read_metadata(metadata_path)
        _validate_metadata(deployment, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = _validate_checkpoint(loaded, deployment)
        model = RFNet(
            num_classes=len(deployment["modulation_classes"]),
            dropout=float(deployment["dropout"]),
        )
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise CheckpointContractError(
                f"checkpoint model state is incompatible with {ARCHITECTURE_VERSION}: {error}"
            ) from error
        return cls(
            model,
            device=_resolve_device(device),
            modulation_classes=list(deployment["modulation_classes"]),
            snr_scale_db=float(deployment["snr_scale_db"]),
            normalization=str(deployment["normalization"]),
            metadata=deployment,
        )

    @staticmethod
    def _prepare_input(iq: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, bool]:
        if isinstance(iq, np.ndarray):
            if iq.dtype.kind not in {"f", "i", "u"}:
                raise TypeError(f"I/Q input must be real numeric data, got dtype {iq.dtype}")
            tensor = torch.from_numpy(np.ascontiguousarray(iq))
        elif isinstance(iq, torch.Tensor):
            if iq.layout != torch.strided or iq.dtype == torch.bool or iq.is_complex():
                raise TypeError(f"I/Q input must be a dense real numeric tensor, got {iq.dtype}")
            tensor = iq.detach()
        else:
            raise TypeError("I/Q input must be a numpy array or torch Tensor")

        single = tensor.ndim == 2
        if single:
            if tuple(tensor.shape) != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"single I/Q input must have shape {EXPECTED_INPUT_SHAPE} in I/Q order; "
                    f"got {tuple(tensor.shape)}"
                )
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            if tuple(tensor.shape[1:]) != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"batch I/Q input must have shape (B, 2, 128) in I/Q order; "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.shape[0] == 0:
                raise ValueError("I/Q batch must not be empty")
        else:
            raise ValueError(
                f"I/Q input must have shape (2, 128) or (B, 2, 128); got {tuple(tensor.shape)}"
            )

        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("I/Q input contains NaN or Inf")
        if tensor.numel() and float(tensor.abs().max()) > torch.finfo(torch.float32).max:
            raise ValueError("I/Q input contains values outside the float32 range")
        tensor = tensor.to(dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("I/Q input cannot be safely converted to float32")
        return tensor.contiguous(), single

    def _forward(self, iq: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        x, single = self._prepare_input(iq)
        if self.normalization != "none":
            raise CheckpointContractError(
                f"RF-Net v1 analyzer supports its deployed normalization='none', got {self.normalization!r}"
            )
        self.model.eval()
        with torch.inference_mode():
            logits, snr_scaled = self.model.forward_multitask(x.to(self.device))
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("RF-Net produced non-finite modulation logits")
        if not bool(torch.isfinite(snr_scaled).all()):
            raise FloatingPointError("RF-Net produced non-finite SNR output")
        return logits.cpu(), (snr_scaled * self.snr_scale_db).cpu(), single

    def analyze_iq(self, iq: np.ndarray | torch.Tensor) -> dict[str, Any] | list[dict[str, Any]]:
        """Return JSON-serializable modulation and SNR predictions."""
        logits, snr_db, single = self._forward(iq)
        probabilities = torch.softmax(logits, dim=1)
        probability, modulation_id = probabilities.max(dim=1)
        results = [
            {
                "modulation": self.modulation_classes[int(class_id)],
                "modulation_id": int(class_id),
                "modulation_probability": float(probability_value),
                "snr_db": float(snr_value),
            }
            for class_id, probability_value, snr_value in zip(
                modulation_id, probability, snr_db
            )
        ]
        return results[0] if single else results
