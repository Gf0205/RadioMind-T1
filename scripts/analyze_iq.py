from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.inference import RFAnalyzer


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze I/Q samples with RF-Net v1.")
    parser.add_argument("--checkpoint", required=True, help="RF-Net C multitask best.pt")
    parser.add_argument("--input", required=True, help=".npy shaped (2,128) or (B,2,128)")
    parser.add_argument(
        "--metadata",
        default="configs/rfnet_v1_deployment.yaml",
        help="versioned deployment metadata YAML",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    analyzer = RFAnalyzer.from_checkpoint(
        _resolve(args.checkpoint),
        metadata=_resolve(args.metadata),
        device=args.device,
    )
    iq = np.load(_resolve(args.input), allow_pickle=False)
    print(json.dumps(analyzer.analyze_iq(iq), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
