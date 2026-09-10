# RadioMind

RadioMind T1 provides a reproducible RML2016.10a data pipeline and a single-head O'Shea-style CNN modulation-classification baseline. The same entrypoint supports a local CPU smoke run and a full AutoDL GPU run.

## Protocol and reproducibility

- `RML2016.10a_dict.pkl` is read only with `pickle.load(f, encoding="latin1")`. It is never modified, moved, committed, or redistributed.
- The first load converts it to local `X.npy` memory-mapped data, metadata, a split cache, and `manifest.csv` under `DATA_ROOT`.
- Every `(modulation, SNR)` cell is independently shuffled with `data.split_seed` and split 700/150/150 into train/validation/test. There is no global mixed split.
- `DATA_ROOT` precedence is environment variable > YAML config > default `./data`. No machine-specific path is hard-coded.
- Device selection is automatic in the order CUDA, Apple MPS, CPU. Training saves `last.pt` after every epoch and `best.pt` at the best validation accuracy; `--resume` restores model, optimizer, scheduler, epoch, early-stop state, and history.
- This pickle has no waveform seed, capture session, or source-waveform identity, so it cannot support a physical cross-channel split. Results must not be interpreted as evidence of cross-channel generalization.

The default preprocessing is `none`. `global` uses training-only channel statistics and `rms` uses per-sample unit RMS; neither optional mode is an experiment in T1. No other transform or augmentation is applied.

## Layout

- `configs/default.yaml`: full training defaults
- `configs/smoke.yaml`: 20 samples per cell (14/3/3), one epoch
- `src/radio_mind/data/`: conversion, manifest, stratified split, datasets, checks
- `src/radio_mind/models/oshea_cnn.py`: classifier
- `src/radio_mind/training/trainer.py`: deterministic training, early stop, checkpoints, resume
- `src/radio_mind/evaluation/metrics.py`: fixed test metrics and plots
- `scripts/train.py`: training entrypoint
- `scripts/fetch_results.sh`: fetch one remote run to local `results/`

The model is a PyTorch reproduction of the CNN2 architecture for RML2016.10a, not a bit-exact or training-result-exact reproduction of the original Keras implementation. It reshapes `(B,2,128)` to `(B,1,2,128)`, applies valid `Conv2d` layers with 256 `(1,3)` and 80 `(2,3)` filters, flattens the resulting 9,920 features, and uses dense layers of 256 and 11 units. Dropout is fixed at 0.6 and the output is raw logits for cross-entropy loss.

## Dependencies

`requirements.txt` intentionally does not install PyTorch. Install ordinary Python dependencies from that file, but use the PyTorch build supplied by the CUDA image on AutoDL. Do not run a requirements command that installs or replaces Torch on a working CUDA image.

## Local CPU smoke run

Run commands from the repository root. Keep the source pickle at the repository root; the loader will create ignored safe-format cache files under `./data`.

```powershell
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
$env:PYTHONPATH = "src"
.venv\Scripts\python -m radio_mind.data --check
.venv\Scripts\python scripts/train.py --smoke
```

For Linux or macOS, replace the activation-free executable paths and environment assignment:

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
.venv/bin/python -m radio_mind.data --check
.venv/bin/python scripts/train.py --smoke
```

A smoke run is tagged `YYYYMMDD_HHMMSS_smoke` and writes `config.yaml`, `metrics.json`, `acc_vs_snr.png`, `confusion_matrix.png`, `loss_curve.png`, and `checkpoints/{best,last}.pt`. It exercises the complete pipeline; its tiny-sample accuracy is not the full-run baseline score.

## AutoDL GPU full run

```bash
# One-time dataset upload from the local machine
scp RML2016.10a_dict.pkl root@<instance>:/root/autodl-tmp/

# On AutoDL
git clone <repo> && cd RadioMind
# Keep the CUDA-enabled Torch already supplied by the image.
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -r requirements.txt
export DATA_ROOT=/root/autodl-tmp
export PYTHONPATH=src
python scripts/train.py --smoke

# Do not start the 50-epoch run until smoke and a separately identified
# 3–5 epoch full-data sanity run have passed.

# Resume the same run after an interruption
python scripts/train.py --resume <run_id>
```

To fetch the artifacts, run locally from a RadioMind checkout:

```bash
export REMOTE_HOST=root@<instance>
export REMOTE_ROOT=/root/autodl-tmp/RadioMind
bash scripts/fetch_results.sh <run_id>
```

Each formal run reports full-test overall accuracy, accuracy for SNR ≥ 0 dB, per-SNR accuracy, a normalized confusion matrix, macro-F1, and per-class F1. The expected full-SNR literature band is 80.5–87%; a score outside it calls for a split/model audit before tuning. Validation is used only for early stopping and is never reported as test performance.

T1 intentionally excludes RF-Net multitask learning, augmentation, normalization ablations, OOD/rejection, calibration, LoRA/SFT/GRPO, MiniMind integration, strict split, SDR data, and RML2018.01a.
