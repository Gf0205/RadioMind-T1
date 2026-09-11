#!/usr/bin/env python3
"""Bounded MiniMind M1.2 pretraining benchmark.

This is an external benchmark driver: it imports the pinned MiniMind sources but
does not modify their model, dataset, optimizer, or checkpoint implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch import optim
from torch.utils.data import DataLoader


SEED = 20260907
MAX_SEQ_LEN = 340
MICRO_BATCH = 8
ACCUMULATION_STEPS = 8
OPTIMIZER_STEPS = 20
SAMPLES = MICRO_BATCH * ACCUMULATION_STEPS * OPTIMIZER_STEPS
LEARNING_RATE = 5e-4
GRAD_CLIP = 1.0
EMA_BETA = 0.98
FORMAL_MICRO_STEPS = 31_016
WARMUP_OPTIMIZER_STEPS = 5


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_subset(path: Path, samples: int) -> None:
    if path.exists():
        with path.open("rb") as handle:
            count = sum(1 for line in handle if line.strip())
        if count == samples:
            return
        raise RuntimeError(f"existing subset has {count} records, expected {samples}: {path}")

    import requests

    source = (
        "https://huggingface.co/datasets/jingyaogong/minimind_dataset/"
        "resolve/main/pretrain_t2t_mini.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    written = 0
    try:
        with requests.get(source, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tmp.open("wb") as output:
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    output.write(line + b"\n")
                    written += 1
                    if written == samples:
                        break
        if written != samples:
            raise RuntimeError(f"source ended after {written} records; expected {samples}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class NvidiaSmiSampler:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.peak_device_mib = 0
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            while not self._stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    value = int(result.stdout.strip().splitlines()[0])
                    self.peak_device_mib = max(self.peak_device_mib, value)
                    self.samples += 1
                except Exception as exc:  # best-effort external observation
                    self.error = f"{type(exc).__name__}: {exc}"
                    return
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _load_minimind(repo_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    minimind = repo_root / "third_party" / "minimind"
    sys.path.insert(0, str(minimind))
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, lm_checkpoint, setup_seed

    return PretrainDataset, MiniMindConfig, get_lr, init_model, (lm_checkpoint, setup_seed)


def verify_checkpoint(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    _, MiniMindConfig, get_lr, init_model, helpers = _load_minimind(repo_root)
    lm_checkpoint, setup_seed = helpers
    setup_seed(SEED)
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model, _ = init_model(config, "none", tokenizer_path=str(repo_root / "third_party/minimind/model"), device="cpu")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint = lm_checkpoint(config, weight="m1_2_benchmark", save_dir=str(checkpoint_dir))
    if checkpoint is None:
        raise RuntimeError("resume checkpoint was not found")

    required = {"model", "optimizer", "scaler", "epoch", "step", "world_size"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise RuntimeError(f"resume checkpoint missing keys: {missing}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    expected_lr = get_lr(OPTIMIZER_STEPS * ACCUMULATION_STEPS, FORMAL_MICRO_STEPS, LEARNING_RATE)
    restored_lr = float(optimizer.param_groups[-1]["lr"])
    report = {
        "model_load_strict": True,
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_state_present": len(optimizer.state) > 0,
        "scaler_scale": float(scaler.get_scale()),
        "scaler_restored": bool(checkpoint["scaler"]),
        "epoch": int(checkpoint["epoch"]),
        "step": int(checkpoint["step"]),
        "restored_lr": restored_lr,
        "expected_lr": expected_lr,
        "lr_equal": restored_lr == expected_lr,
    }
    if not report["optimizer_state_present"] or not report["scaler_restored"]:
        raise RuntimeError(f"incomplete optimizer/scaler state: {report}")
    if report["epoch"] != 0 or report["step"] != 160 or not report["lr_equal"]:
        raise RuntimeError(f"epoch/step/LR restoration mismatch: {report}")
    print("CHECKPOINT_VERIFY_JSON=" + json.dumps(report, sort_keys=True))
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires a T4 GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name:
        raise RuntimeError(f"expected a T4 GPU, found {gpu_name!r}")

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path).resolve()
    _prepare_subset(data_path, SAMPLES)

    PretrainDataset, MiniMindConfig, get_lr, init_model, helpers = _load_minimind(repo_root)
    lm_checkpoint, setup_seed = helpers
    setup_seed(SEED)
    device = torch.device("cuda:0")
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model, tokenizer = init_model(
        config,
        "none",
        tokenizer_path=str(repo_root / "third_party/minimind/model"),
        device=str(device),
    )
    model.train()
    dataset = PretrainDataset(str(data_path), tokenizer, max_length=MAX_SEQ_LEN)
    if len(dataset) != SAMPLES:
        raise RuntimeError(f"dataset length {len(dataset)} != {SAMPLES}")
    setup_seed(SEED)
    indices = torch.randperm(len(dataset)).tolist()
    loader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        sampler=indices,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    if len(loader) != OPTIMIZER_STEPS * ACCUMULATION_STEPS:
        raise RuntimeError(f"loader length {len(loader)} != 160")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    sampler = NvidiaSmiSampler(interval=0.1)
    sampler.start()
    micro_records: list[dict[str, Any]] = []
    optimizer_records: list[dict[str, Any]] = []
    ema_loss: float | None = None
    cumulative_supervised = 0
    cumulative_positions = 0
    window_supervised = 0
    window_positions = 0
    window_start = 0.0

    try:
        for micro_step, (input_ids, labels) in enumerate(loader, start=1):
            if (micro_step - 1) % ACCUMULATION_STEPS == 0:
                torch.cuda.synchronize(device)
                window_start = time.perf_counter()
                window_supervised = 0
                window_positions = 0

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            supervised_tokens = int((labels[:, 1:] != -100).sum().item())
            input_positions = int(input_ids.numel())
            cumulative_supervised += supervised_tokens
            cumulative_positions += input_positions
            window_supervised += supervised_tokens
            window_positions += input_positions
            lr = float(get_lr(micro_step, FORMAL_MICRO_STEPS, LEARNING_RATE))
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                result = model(input_ids, labels=labels)
                raw_loss = result.loss + result.aux_loss
                scaled_loss = raw_loss / ACCUMULATION_STEPS
            raw_loss_value = float(raw_loss.detach().item())
            if not math.isfinite(raw_loss_value):
                raise FloatingPointError(f"non-finite loss at micro_step={micro_step}: {raw_loss_value}")
            ema_loss = raw_loss_value if ema_loss is None else EMA_BETA * ema_loss + (1.0 - EMA_BETA) * raw_loss_value
            scale_before = float(scaler.get_scale())
            scaler.scale(scaled_loss).backward()
            micro_record: dict[str, Any] = {
                "micro_step": micro_step,
                "optimizer_step": math.ceil(micro_step / ACCUMULATION_STEPS),
                "loss": raw_loss_value,
                "ema_loss_beta_0_98": ema_loss,
                "supervised_tokens": supervised_tokens,
                "input_positions": input_positions,
                "lr": lr,
                "scaler_scale": scale_before,
            }

            if micro_step % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                grad_norm = float(grad_norm_tensor.detach().item())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite pre-clip grad norm at optimizer_step={micro_step // ACCUMULATION_STEPS}: {grad_norm}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                duration = time.perf_counter() - window_start
                opt_step = micro_step // ACCUMULATION_STEPS
                record = {
                    "optimizer_step": opt_step,
                    "time_seconds": duration,
                    "supervised_tokens": window_supervised,
                    "input_positions": window_positions,
                    "preclip_grad_norm": grad_norm,
                    "lr": lr,
                    "scaler_scale_before": scale_before,
                    "scaler_scale_after": float(scaler.get_scale()),
                }
                optimizer_records.append(record)
                micro_record["preclip_grad_norm"] = grad_norm
                print(
                    f"opt_step={opt_step:02d}/20 loss={raw_loss_value:.6f} "
                    f"ema={ema_loss:.6f} grad_norm={grad_norm:.6f} lr={lr:.10f} "
                    f"scale={scaler.get_scale():.1f} time={duration:.4f}s "
                    f"supervised_tokens={window_supervised}"
                )
            micro_records.append(micro_record)
            del input_ids, labels, result, raw_loss, scaled_loss
    finally:
        sampler.stop()

    if len(micro_records) != 160 or len(optimizer_records) != 20:
        raise RuntimeError("benchmark did not complete exactly 160 micro-batches / 20 optimizer steps")
    if cumulative_positions != SAMPLES * MAX_SEQ_LEN:
        raise RuntimeError("input position count mismatch")

    measured = optimizer_records[WARMUP_OPTIMIZER_STEPS:]
    measured_time = sum(item["time_seconds"] for item in measured)
    measured_supervised = sum(item["supervised_tokens"] for item in measured)
    measured_positions = sum(item["input_positions"] for item in measured)
    durations = [item["time_seconds"] for item in measured]
    throughput = {
        "measured_optimizer_steps": len(measured),
        "mean_optimizer_step_seconds": statistics.fmean(durations),
        "median_optimizer_step_seconds": statistics.median(durations),
        "effective_supervised_tokens_per_second": measured_supervised / measured_time,
        "theoretical_input_positions_per_second": measured_positions / measured_time,
        "measured_supervised_tokens": measured_supervised,
        "measured_input_positions": measured_positions,
        "measured_time_seconds": measured_time,
    }
    memory = {
        "torch_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "torch_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "nvidia_smi_device_peak_mib": sampler.peak_device_mib,
        "nvidia_smi_samples": sampler.samples,
        "nvidia_smi_error": sampler.error,
    }

    save_started = time.perf_counter()
    lm_checkpoint(
        config,
        weight="m1_2_benchmark",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        step=160,
        save_dir=str(checkpoint_dir),
    )
    checkpoint_write_seconds = time.perf_counter() - save_started
    weight_path = checkpoint_dir / "m1_2_benchmark_768.pth"
    resume_path = checkpoint_dir / "m1_2_benchmark_768_resume.pth"
    if not weight_path.is_file() or not resume_path.is_file():
        raise RuntimeError("explicit checkpoint files were not created")

    verify_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--verify-checkpoint",
        "--repo-root",
        str(repo_root),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    verify_started = time.perf_counter()
    verification = subprocess.run(verify_command, check=False, capture_output=True, text=True)
    checkpoint_load_verify_seconds = time.perf_counter() - verify_started
    marker = next(
        (line for line in verification.stdout.splitlines() if line.startswith("CHECKPOINT_VERIFY_JSON=")),
        None,
    )
    if marker is None:
        verify_report = {
            "passed": False,
            "returncode": verification.returncode,
            "error": (verification.stderr or verification.stdout).strip()[-4000:],
        }
    else:
        verify_report = json.loads(marker.split("=", 1)[1])
        verify_report["passed"] = verification.returncode == 0

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    git_submodule_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root / "third_party/minimind",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    estimated_seconds = 50_000_000 / throughput["effective_supervised_tokens_per_second"]
    report = {
        "contract": {
            "model": "MiniMind-3 Dense 64M",
            "fresh_init": True,
            "seed": SEED,
            "max_seq_len": MAX_SEQ_LEN,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": ACCUMULATION_STEPS,
            "effective_batch": MICRO_BATCH * ACCUMULATION_STEPS,
            "optimizer_steps": OPTIMIZER_STEPS,
            "micro_batches": len(micro_records),
            "samples": SAMPLES,
            "dtype": "float16",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "lr_schedule_total_micro_steps": FORMAL_MICRO_STEPS,
            "warmup_optimizer_steps_excluded": WARMUP_OPTIMIZER_STEPS,
        },
        "environment": {
            "os": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": gpu_name,
            "git_commit": git_commit,
            "minimind_commit": git_submodule_commit,
            "dataset_path": str(data_path),
            "dataset_sha256": _sha256(data_path),
        },
        "counts": {
            "supervised_tokens_seen": cumulative_supervised,
            "input_positions_processed": cumulative_positions,
        },
        "numerical_health": {
            "all_losses_finite": all(math.isfinite(item["loss"]) for item in micro_records),
            "all_grad_norms_finite": all(math.isfinite(item["preclip_grad_norm"]) for item in optimizer_records),
            "first_loss": micro_records[0]["loss"],
            "last_loss": micro_records[-1]["loss"],
            "min_loss": min(item["loss"] for item in micro_records),
            "max_loss": max(item["loss"] for item in micro_records),
            "final_ema_loss_beta_0_98": micro_records[-1]["ema_loss_beta_0_98"],
            "scaler_initial": micro_records[0]["scaler_scale"],
            "scaler_final": optimizer_records[-1]["scaler_scale_after"],
        },
        "throughput": throughput,
        "memory": memory,
        "checkpoint": {
            "weight_path": str(weight_path),
            "weight_bytes": weight_path.stat().st_size,
            "resume_path": str(resume_path),
            "resume_bytes": resume_path.stat().st_size,
            "write_seconds": checkpoint_write_seconds,
            "new_process_load_verify_seconds": checkpoint_load_verify_seconds,
            "verification": verify_report,
        },
        "estimated_50m_pure_training_seconds": estimated_seconds,
        "micro_batches": micro_records,
        "optimizer_steps": optimizer_records,
    }
    report_path = output_dir / "m1_2_benchmark.json"
    tmp_report = report_path.with_suffix(".json.tmp")
    tmp_report.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    os.replace(tmp_report, report_path)
    print("M1_2_BENCHMARK_SUMMARY=" + json.dumps({
        "supervised_tokens_seen": cumulative_supervised,
        "input_positions_processed": cumulative_positions,
        **throughput,
        **memory,
        "first_loss": report["numerical_health"]["first_loss"],
        "last_loss": report["numerical_health"]["last_loss"],
        "final_ema_loss": report["numerical_health"]["final_ema_loss_beta_0_98"],
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "checkpoint_resume_bytes": resume_path.stat().st_size,
        "checkpoint_verify": verify_report,
        "estimated_50m_pure_training_seconds": estimated_seconds,
        "report_path": str(report_path),
    }, sort_keys=True))
    return 0 if verify_report.get("passed") else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--data-path", default="/content/RadioMind/third_party/minimind/dataset/pretrain_m1_2_benchmark_1280.jsonl")
    parser.add_argument("--output-dir", default="/content/RadioMind/results/minimind_m1_2_benchmark")
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    if args.verify_checkpoint:
        if not args.checkpoint_dir:
            parser.error("--checkpoint-dir is required with --verify-checkpoint")
        return verify_checkpoint(args)
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
