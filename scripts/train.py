from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from radio_mind.config import load_config, resolve_device
from radio_mind.data.load import MODULATIONS, SNRS, load_store
from radio_mind.data.split import load_or_create_splits
from radio_mind.data.dataset import RMLDataset
from radio_mind.models import OsheaCNN2
from radio_mind.training.trainer import fit, seed_everything, validate_resume_state
from radio_mind.evaluation.metrics import evaluate, save_loss_curve, save_metrics


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _new_run_id(smoke: bool) -> tuple[str, Path]:
    tag = "smoke" if smoke else "oshea"
    while True:
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{tag}"
        out = ROOT / "results" / run_id
        try:
            out.mkdir(parents=True, exist_ok=False)
            return run_id, out
        except FileExistsError:
            time.sleep(1.0)


def _resume_dir(run_id: str) -> Path:
    if (
        not run_id
        or Path(run_id).is_absolute()
        or "/" in run_id
        or "\\" in run_id
        or ".." in run_id
        or Path(run_id).name != run_id
    ):
        raise ValueError("resume run_id must be one safe path component")
    results_root = (ROOT / "results").resolve()
    out = (results_root / run_id).resolve()
    if out.parent != results_root:
        raise ValueError("resume run_id escapes the results directory")
    return out


def _config_path(path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def _save_run_config(out: Path, cfg, commit: str) -> None:
    saved = cfg.to_dict()
    saved["git_commit"] = commit
    (out / "config.yaml").write_text(
        yaml.safe_dump(saved, sort_keys=False), encoding="utf-8"
    )


def _smoke_indices(store, splits):
    result = {"train": [], "val": [], "test": []}
    take = {"train": 14, "val": 3, "test": 3}
    for mod_id in range(len(MODULATIONS)):
        for snr in SNRS:
            for name in result:
                source = splits[name]
                mask = (store.mod_ids[source] == mod_id) & (store.snr_db[source] == snr)
                result[name].extend(source[mask][:take[name]].tolist())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume")
    args = ap.parse_args()
    if args.resume and args.smoke:
        ap.error("--resume already uses the saved run config; do not combine it with --smoke")
    if args.resume and args.config:
        ap.error("--resume must use results/<run_id>/config.yaml; do not combine it with --config")
    if args.smoke and args.config:
        ap.error("--smoke uses configs/smoke.yaml; do not combine it with --config")

    if args.resume:
        try:
            out = _resume_dir(args.resume)
        except ValueError as error:
            ap.error(str(error))
        if not out.is_dir():
            ap.error(f"run not found: {out}")
        config_path = out / "config.yaml"
        if not config_path.exists():
            ap.error(f"resume config not found: {config_path}")
        run_id = args.resume
    else:
        config_path = ROOT / "configs" / "smoke.yaml" if args.smoke else _config_path(args.config)

    cfg = load_config(config_path)
    smoke_mode = cfg.run_mode == "smoke"
    if not args.resume:
        run_id, out = _new_run_id(smoke_mode)
    seed_everything(cfg.train.seed, cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    root = Path(cfg.data.data_root); root = root if root.is_absolute() else ROOT / root
    cfg.data.data_root = str(root.resolve())
    store = load_store(root); splits = load_or_create_splits(store, cfg.data.split_seed)
    if smoke_mode:
        splits = _smoke_indices(store, splits)
    train_ds = RMLDataset(store, splits["train"], cfg.data.normalize); val_ds = RMLDataset(store, splits["val"], cfg.data.normalize, train_ds.mean, train_ds.std); test_ds = RMLDataset(store, splits["test"], cfg.data.normalize, train_ds.mean, train_ds.std)
    kwargs = {"batch_size": cfg.train.batch_size, "num_workers": cfg.train.num_workers, "pin_memory": cfg.train.pin_memory}
    generator = torch.Generator().manual_seed(cfg.train.seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    model = OsheaCNN2(dropout=cfg.model.dropout)
    resume_state = None
    if args.resume:
        resume_state = torch.load(out / "checkpoints" / "last.pt", map_location="cpu")
        validate_resume_state(resume_state, cfg.train.scheduler, device)
        model.load_state_dict(resume_state["model"])

    commit = _git_commit()
    if not args.resume:
        _save_run_config(out, cfg, commit)
    print(f"run_id={run_id} device={device} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_info = fit(
        model, train_loader, val_loader, device, cfg.train.epochs, cfg.train.lr,
        cfg.train.patience, cfg.train.scheduler, out / "checkpoints", resume_state,
    )
    best = torch.load(out / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    metrics = evaluate(model, test_loader, device, SNRS, MODULATIONS)
    metrics.update({
        "run_id": run_id,
        "config": cfg.to_dict(),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": train_info["best_epoch"],
        "train_time_seconds": train_info["train_time_seconds"],
        "device": device,
        "git_commit": commit,
    })
    save_metrics(metrics, out, MODULATIONS, SNRS)
    save_loss_curve(train_info["history"], out)
    print(json.dumps({k: metrics[k] for k in ("overall_accuracy", "accuracy_snr_ge0", "macro_f1", "best_epoch")}, indent=2))

if __name__ == "__main__": main()
