from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.data.load import MODULATIONS
from radio_mind.inference.analyzer import CheckpointContractError, RFAnalyzer
from radio_mind.models import RFNet


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RFAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        torch.manual_seed(20260907)
        self.direct = RFNet(dropout=0.6).eval()
        self.checkpoint = root / "best.pt"
        torch.save(
            {
                "model": self.direct.state_dict(),
                "experiment_name": "C_multitask",
                "use_snr_loss": True,
                "snr_lambda": 0.1,
            },
            self.checkpoint,
        )
        self.metadata = root / "deployment.yaml"
        self.metadata_value = {
            "deployment_schema_version": 1,
            "model_type": "RFNet",
            "architecture_version": "rfnet_v1",
            "modulation_classes": MODULATIONS,
            "snr_scale_db": 20.0,
            "normalization": "none",
            "source_git_commit": "0" * 40,
            "training_arm": "multitask",
            "snr_lambda": 0.1,
            "dropout": 0.6,
            "input_shape": [2, 128],
            "channel_order": ["I", "Q"],
            "checkpoint_sha256": _hash(self.checkpoint),
        }
        self._write_metadata()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_metadata(self) -> None:
        self.metadata.write_text(yaml.safe_dump(self.metadata_value), encoding="utf-8")

    def _analyzer(self) -> RFAnalyzer:
        return RFAnalyzer.from_checkpoint(
            self.checkpoint, metadata=self.metadata, device="cpu"
        )

    def test_outputs_match_direct_rfnet(self) -> None:
        analyzer = self._analyzer()
        x = torch.randn(4, 2, 128)
        with torch.inference_mode():
            logits, snr_scaled = self.direct.forward_multitask(x)
            probabilities = torch.softmax(logits, dim=1)
        result = analyzer.analyze_iq(x)
        self.assertIsInstance(result, list)
        for index, item in enumerate(result):
            expected_id = int(logits[index].argmax())
            self.assertEqual(item["modulation_id"], expected_id)
            self.assertEqual(item["modulation"], MODULATIONS[expected_id])
            self.assertEqual(
                item["modulation_probability"], float(probabilities[index].max())
            )
            self.assertEqual(item["snr_db"], float(snr_scaled[index] * 20.0))

    def test_single_and_batch_are_identical_and_json_serializable(self) -> None:
        analyzer = self._analyzer()
        x = np.random.default_rng(7).normal(size=(3, 2, 128)).astype(np.float32)
        batch = analyzer.analyze_iq(x)
        singles = [analyzer.analyze_iq(sample) for sample in x]
        for batched, single in zip(batch, singles):
            self.assertEqual(batched["modulation"], single["modulation"])
            self.assertEqual(batched["modulation_id"], single["modulation_id"])
            self.assertAlmostEqual(
                batched["modulation_probability"],
                single["modulation_probability"],
                places=6,
            )
            self.assertAlmostEqual(batched["snr_db"], single["snr_db"], places=5)
        json.dumps(batch)

    def test_wrong_shape_and_nonfinite_inputs_are_rejected(self) -> None:
        analyzer = self._analyzer()
        with self.assertRaisesRegex(ValueError, "shape"):
            analyzer.analyze_iq(np.zeros((128, 2), dtype=np.float32))
        for value in (np.nan, np.inf):
            x = np.zeros((2, 128), dtype=np.float32)
            x[0, 0] = value
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                analyzer.analyze_iq(x)

    def test_wrong_checkpoint_type_is_rejected(self) -> None:
        self.metadata_value["model_type"] = "OsheaCNN2"
        self._write_metadata()
        with self.assertRaisesRegex(CheckpointContractError, "model_type"):
            self._analyzer()

    def test_checkpoint_hash_and_training_arm_are_enforced(self) -> None:
        self.metadata_value["checkpoint_sha256"] = "f" * 64
        self._write_metadata()
        with self.assertRaisesRegex(CheckpointContractError, "checkpoint_sha256"):
            self._analyzer()

        checkpoint = torch.load(self.checkpoint, weights_only=True)
        checkpoint["experiment_name"] = "B_modulation_only"
        torch.save(checkpoint, self.checkpoint)
        self.metadata_value["checkpoint_sha256"] = _hash(self.checkpoint)
        self._write_metadata()
        with self.assertRaisesRegex(CheckpointContractError, "C_multitask"):
            self._analyzer()


if __name__ == "__main__":
    unittest.main()
