from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_root: str = "./data"
    split_seed: int = 20260907
    normalize: str = "none"
    num_workers: int = 0


@dataclass
class ModelConfig:
    name: str = "oshea_cnn2"
    dropout: float = 0.6


@dataclass
class TrainConfig:
    seed: int = 20260907
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 50
    patience: int = 8
    scheduler: str = "cosine"
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    deterministic: bool = True


@dataclass
class Config:
    run_mode: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    defaults = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    cfg_path = Path(path) if path else defaults
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "run_mode" not in raw:
        raise ValueError(f"config must define run_mode: full|smoke|sanity ({cfg_path})")
    run_mode = str(raw["run_mode"])
    data = DataConfig(**{**asdict(DataConfig()), **raw.get("data", {})})
    model = ModelConfig(**{**asdict(ModelConfig()), **raw.get("model", {})})
    train = TrainConfig(**{**asdict(TrainConfig()), **raw.get("train", {})})
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        data.data_root = env_root
    if data.normalize not in {"none", "global", "rms"}:
        raise ValueError("data.normalize must be one of none|global|rms")
    if run_mode not in {"full", "smoke", "sanity"}:
        raise ValueError("run_mode must be one of full|smoke|sanity")
    if train.batch_size <= 0 or train.epochs <= 0 or train.patience <= 0:
        raise ValueError("train.batch_size, train.epochs, and train.patience must be positive")
    if train.lr <= 0:
        raise ValueError("train.lr must be positive")
    if train.scheduler not in {"none", "cosine"}:
        raise ValueError("train.scheduler must be one of none|cosine")
    if train.device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("train.device must be one of auto|cuda|mps|cpu")
    return Config(run_mode=run_mode, data=data, model=model, train=train)


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
