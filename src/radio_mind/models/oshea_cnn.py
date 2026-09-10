"""O'Shea CNN2 architecture for RML2016.10a.

This is a PyTorch reproduction of the CNN2 architecture, not a bit-exact or
training-result-exact reproduction of the original Keras implementation.
"""
from __future__ import annotations

import torch
from torch import nn


class OsheaCNN2(nn.Module):
    """Narrow-Conv2d CNN2 baseline for inputs shaped ``(B, 2, 128)``."""

    input_shape = (2, 128)
    flatten_dim = 80 * 1 * 124

    def __init__(self, num_classes: int = 11, dropout: float = 0.6):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(self.flatten_dim, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"expected input shape (B, 2, 128), got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)  # (B, 1, 2, 128)
        x = self.dropout1(self.relu1(self.conv1(x)))  # (B, 256, 2, 126)
        x = self.dropout2(self.relu2(self.conv2(x)))  # (B, 80, 1, 124)
        x = self.flatten(x)  # (B, 9920)
        x = self.dropout3(self.relu3(self.dense1(x)))  # (B, 256)
        return self.dense2(x)  # Raw logits: (B, num_classes)
