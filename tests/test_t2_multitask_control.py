from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radio_mind.models import RFNet


class T2MultitaskControlTests(unittest.TestCase):
    def test_fresh_state_can_be_cloned_exactly_between_arms(self) -> None:
        torch.manual_seed(20260907)
        template = RFNet()
        initial = {key: value.clone() for key, value in template.state_dict().items()}
        model_b = RFNet()
        model_c = RFNet()
        model_b.load_state_dict(initial, strict=True)
        model_c.load_state_dict(initial, strict=True)
        for key in initial:
            self.assertTrue(torch.equal(model_b.state_dict()[key], model_c.state_dict()[key]))

    def test_initialization_contract(self) -> None:
        torch.manual_seed(20260907)
        model = RFNet()
        for module in (
            model.encoder.conv1,
            model.encoder.conv2,
            model.encoder.dense1,
            model.modulation_head.output,
            model.snr_head.output,
        ):
            self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))

    def test_modulation_only_does_not_update_snr_head(self) -> None:
        torch.manual_seed(20260907)
        model = RFNet(dropout=0.0)
        before = {
            key: value.clone() for key, value in model.snr_head.state_dict().items()
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(8, 2, 128)
        target = torch.arange(8) % 11
        modulation_logits, _snr = model.forward_multitask(x)
        loss = torch.nn.CrossEntropyLoss()(modulation_logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for key, value in model.snr_head.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))


if __name__ == "__main__":
    unittest.main()
