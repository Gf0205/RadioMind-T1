from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.data.load import load_store
from radio_mind.data.split import load_or_create_splits
from radio_mind.models import OsheaCNN2, RFNet, load_t1_modulation_weights


def _compare(old: torch.nn.Module, new: torch.nn.Module, x: torch.Tensor) -> dict:
    with torch.inference_mode():
        old_logits = old(x)
        new_logits = new(x)
    difference = (old_logits - new_logits).abs()
    return {
        "old_shape": tuple(old_logits.shape),
        "new_shape": tuple(new_logits.shape),
        "max_abs_diff": float(difference.max()),
        "sum_abs_diff": float(difference.sum()),
        "values": difference.numel(),
        "argmax_equal": int(
            (old_logits.argmax(dim=1) == new_logits.argmax(dim=1)).sum()
        ),
        "samples": len(x),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify frozen T1 CNN2 and RF-Net modulation-logit equivalence."
    )
    parser.add_argument("checkpoint", type=Path, help="trusted T1 best.pt")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--split-seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        parser.error("--samples and --batch-size must be positive")

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("T1 checkpoint must contain a model state_dict")

    old = OsheaCNN2()
    old.load_state_dict(checkpoint["model"], strict=True)
    new = RFNet()
    load_t1_modulation_weights(new, checkpoint["model"])
    old.eval()
    new.eval()

    torch.manual_seed(20260907)
    dummy = torch.randn(8, 2, 128)
    dummy_result = _compare(old, new, dummy)
    print(f"dummy_old_logits_shape={dummy_result['old_shape']}")
    print(f"dummy_new_logits_shape={dummy_result['new_shape']}")
    print(f"dummy_max_abs_diff={dummy_result['max_abs_diff']:.17g}")

    data_root = args.data_root
    if data_root is None:
        data_root = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    store = load_store(data_root)
    splits = load_or_create_splits(store, args.split_seed)
    indices = splits["test"][: min(args.samples, len(splits["test"]))]

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    compared_values = 0
    argmax_equal = 0
    old_shape = new_shape = None
    for offset in range(0, len(indices), args.batch_size):
        batch_indices = indices[offset : offset + args.batch_size]
        x = torch.from_numpy(np.asarray(store.X[batch_indices]))
        result = _compare(old, new, x)
        max_abs_diff = max(max_abs_diff, result["max_abs_diff"])
        sum_abs_diff += result["sum_abs_diff"]
        compared_values += result["values"]
        argmax_equal += result["argmax_equal"]
        old_shape = result["old_shape"]
        new_shape = result["new_shape"]

    samples = len(indices)
    mean_abs_diff = sum_abs_diff / compared_values
    print(f"old_logits_batch_shape={old_shape}")
    print(f"new_logits_batch_shape={new_shape}")
    print(f"samples={samples}")
    print(f"max_abs_diff={max_abs_diff:.17g}")
    print(f"mean_abs_diff={mean_abs_diff:.17g}")
    print(f"argmax_equal={argmax_equal}/{samples} ({argmax_equal / samples:.2%})")

    probe = torch.from_numpy(np.asarray(store.X[indices[: min(16, len(indices))]]))
    with torch.inference_mode():
        z = new.encode(probe)
        modulation_logits = new.classify(z)
        snr_estimate = new.estimate_snr(z)

    encoder_calls = 0

    def count_encoder_calls(_module, _inputs, _output):
        nonlocal encoder_calls
        encoder_calls += 1

    hook = new.encoder.register_forward_hook(count_encoder_calls)
    with torch.inference_mode():
        multitask_modulation, multitask_snr = new.forward_multitask(probe)
    hook.remove()

    print(f"encode_shape={tuple(z.shape)}")
    print(f"classify_shape={tuple(modulation_logits.shape)}")
    print(f"estimate_snr_shape={tuple(snr_estimate.shape)}")
    print(f"forward_multitask_shapes={tuple(multitask_modulation.shape)},{tuple(multitask_snr.shape)}")
    print(f"forward_multitask_encoder_calls={encoder_calls}")

    if dummy_result["max_abs_diff"] != 0.0 or max_abs_diff != 0.0:
        raise RuntimeError("T1 and RF-Net modulation logits are not exactly equal")
    if argmax_equal != samples:
        raise RuntimeError("T1 and RF-Net modulation predictions differ")
    if encoder_calls != 1:
        raise RuntimeError("forward_multitask must invoke the encoder exactly once")


if __name__ == "__main__":
    main()
