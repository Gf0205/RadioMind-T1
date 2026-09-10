from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_t2_b50_control import _compare_prefix


class B50ReproductionTests(unittest.TestCase):
    def test_strict_prefix_match(self) -> None:
        row = {
            "sample_order_sha256": "abc",
            "learning_rate": 0.001,
            "train_modulation_loss": 1.25,
            "val_modulation_accuracy": 0.5,
        }
        result = _compare_prefix([dict(row)], [dict(row), dict(row)])
        self.assertTrue(result["strict_equal"])
        self.assertEqual(result["epochs_compared"], 1)

    def test_difference_is_rejected(self) -> None:
        old = {
            "sample_order_sha256": "abc",
            "learning_rate": 0.001,
            "train_modulation_loss": 1.25,
            "val_modulation_accuracy": 0.5,
        }
        new = dict(old)
        new["train_modulation_loss"] += 1e-12
        result = _compare_prefix([old], [new])
        self.assertFalse(result["strict_equal"])
        self.assertEqual(result["mismatches"][0]["field"], "train_modulation_loss")


if __name__ == "__main__":
    unittest.main()
