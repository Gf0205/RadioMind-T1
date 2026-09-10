from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.models import (
    T1_TO_RFNET_KEYS,
    OsheaCNN2,
    RFNet,
    load_t1_modulation_weights,
)


class RFNetEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260907)
        self.old = OsheaCNN2()
        self.new = RFNet()
        self.t1_state = self.old.state_dict()

    def test_mapping_is_complete_and_eval_logits_are_exact(self) -> None:
        snr_before = {
            key: value.clone()
            for key, value in self.new.snr_head.state_dict().items()
        }
        load_t1_modulation_weights(self.new, self.t1_state)
        migrated = self.new.state_dict()
        for source_key, target_key in T1_TO_RFNET_KEYS.items():
            self.assertTrue(torch.equal(self.t1_state[source_key], migrated[target_key]))
        for key, value in self.new.snr_head.state_dict().items():
            self.assertTrue(torch.equal(snr_before[key], value))
        self.old.eval()
        self.new.eval()
        x = torch.randn(5, 2, 128)
        with torch.inference_mode():
            old_logits = self.old(x)
            new_logits = self.new(x)
        self.assertTrue(torch.equal(old_logits, new_logits))

    def test_shapes(self) -> None:
        x = torch.randn(4, 2, 128)
        z = self.new.encode(x)
        self.assertEqual(tuple(z.shape), (4, 256))
        self.assertEqual(tuple(self.new.classify(z).shape), (4, 11))
        self.assertEqual(tuple(self.new.estimate_snr(z).shape), (4,))
        self.assertEqual(tuple(self.new(x).shape), (4, 11))

    def test_missing_t1_key_is_rejected(self) -> None:
        state = dict(self.t1_state)
        del state["conv1.weight"]
        with self.assertRaisesRegex(ValueError, "missing=.*conv1.weight"):
            load_t1_modulation_weights(self.new, state)

    def test_wrong_shape_is_rejected(self) -> None:
        state = dict(self.t1_state)
        state["dense2.weight"] = state["dense2.weight"][:10]
        with self.assertRaisesRegex(ValueError, "shape mismatch.*dense2.weight"):
            load_t1_modulation_weights(self.new, state)

    def test_unexpected_t1_key_is_rejected(self) -> None:
        state = dict(self.t1_state)
        state["extra.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected=.*extra.weight"):
            load_t1_modulation_weights(self.new, state)

    def test_forward_multitask_encodes_once(self) -> None:
        calls = 0

        def count_calls(_module, _inputs, _output):
            nonlocal calls
            calls += 1

        hook = self.new.encoder.register_forward_hook(count_calls)
        try:
            modulation, snr = self.new.forward_multitask(torch.randn(3, 2, 128))
        finally:
            hook.remove()
        self.assertEqual(calls, 1)
        self.assertEqual(tuple(modulation.shape), (3, 11))
        self.assertEqual(tuple(snr.shape), (3,))


if __name__ == "__main__":
    unittest.main()
