from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.evaluation.snr_probe import (
    compute_snr_metrics,
    fit_scalar_linear_regression,
    log_rms_feature,
)


class SNRProbeTests(unittest.TestCase):
    def test_constant_mean_contract(self) -> None:
        snrs = list(range(-20, 20, 2))
        target = np.repeat(snrs, 7)
        prediction = np.full_like(target, -1)
        metrics = compute_snr_metrics(prediction, target, snrs)
        self.assertEqual(metrics["mae_db"], 10.0)
        self.assertAlmostEqual(metrics["rmse_db"], np.sqrt(133.0))
        self.assertIsNone(metrics["pearson_correlation"])

    def test_scalar_linear_regression(self) -> None:
        feature = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
        target = 3.5 * feature - 1.25
        slope, intercept = fit_scalar_linear_regression(feature, target)
        self.assertAlmostEqual(slope, 3.5)
        self.assertAlmostEqual(intercept, -1.25)

    def test_log_rms_feature(self) -> None:
        samples = np.zeros((2, 2, 128), dtype=np.float32)
        samples[0, 0] = 3.0
        samples[0, 1] = 4.0
        samples[1, 0] = 1.0
        expected = np.log10(np.asarray([5.0, 1.0]) + 1e-12)
        np.testing.assert_allclose(log_rms_feature(samples), expected)


if __name__ == "__main__":
    unittest.main()
