"""Strict, weight-only migration from the frozen T1 CNN2 to RF-Net v1."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from .rfnet import RFNet


T1_TO_RFNET_KEYS = {
    "conv1.weight": "encoder.conv1.weight",
    "conv1.bias": "encoder.conv1.bias",
    "conv2.weight": "encoder.conv2.weight",
    "conv2.bias": "encoder.conv2.bias",
    "dense1.weight": "encoder.dense1.weight",
    "dense1.bias": "encoder.dense1.bias",
    "dense2.weight": "modulation_head.output.weight",
    "dense2.bias": "modulation_head.output.bias",
}


def load_t1_modulation_weights(
    model: RFNet,
    t1_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load all and only frozen T1 model weights into an RF-Net modulation path.

    This is a weight migration operation, not checkpoint resume. The RF-Net SNR
    head retains its independent initialization.
    """
    if not isinstance(model, RFNet):
        raise TypeError(f"model must be RFNet, got {type(model).__name__}")
    if not isinstance(t1_state_dict, Mapping):
        raise TypeError("T1 state_dict must be a mapping")

    actual_keys = set(t1_state_dict)
    expected_keys = set(T1_TO_RFNET_KEYS)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "invalid T1 state_dict keys: "
            f"missing={missing}, unexpected={unexpected}"
        )

    target_state = model.state_dict()
    migrated_state = dict(target_state)
    for source_key, target_key in T1_TO_RFNET_KEYS.items():
        source = t1_state_dict[source_key]
        target = target_state[target_key]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"T1 state_dict value {source_key!r} must be a tensor")
        if source.shape != target.shape:
            raise ValueError(
                f"shape mismatch for {source_key!r} -> {target_key!r}: "
                f"source={tuple(source.shape)}, target={tuple(target.shape)}"
            )
        if source.dtype != target.dtype:
            raise ValueError(
                f"dtype mismatch for {source_key!r} -> {target_key!r}: "
                f"source={source.dtype}, target={target.dtype}"
            )
        migrated_state[target_key] = source

    model.load_state_dict(migrated_state, strict=True)
