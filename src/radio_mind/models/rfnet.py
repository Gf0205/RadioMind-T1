"""RF-Net v1 components built from the frozen T1 CNN2 modulation path."""
from __future__ import annotations

import torch
from torch import nn


class RFEncoder(nn.Module):
    """Encode ``(B, 2, 128)`` I/Q samples as 256-dimensional features."""

    input_shape = (2, 128)
    flatten_dim = 80 * 1 * 124
    feature_dim = 256

    def __init__(self, dropout: float = 0.6):
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
        self.dense1 = nn.Linear(self.flatten_dim, self.feature_dim)
        self.relu3 = nn.ReLU()
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for layer in (self.conv1, self.conv2):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(
            self.dense1.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        if self.dense1.bias is not None:
            nn.init.zeros_(self.dense1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"expected input shape (B, 2, 128), got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)
        x = self.dropout1(self.relu1(self.conv1(x)))
        x = self.dropout2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.relu3(self.dense1(x))


class ModulationHead(nn.Module):
    """Map a shared RF feature to raw modulation-class logits."""

    def __init__(
        self,
        feature_dim: int = RFEncoder.feature_dim,
        num_classes: int = 11,
        dropout: float = 0.6,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(feature_dim, num_classes)
        nn.init.kaiming_normal_(
            self.output.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self._validate_features(z)
        return self.output(self.dropout(z))

    def _validate_features(self, z: torch.Tensor) -> None:
        if z.ndim != 2 or z.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected features shaped (B, {self.feature_dim}), got {tuple(z.shape)}"
            )


class SNRHead(nn.Module):
    """Minimal continuous-SNR head; training is intentionally deferred."""

    def __init__(self, feature_dim: int = RFEncoder.feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.output = nn.Linear(feature_dim, 1)
        nn.init.kaiming_normal_(
            self.output.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected features shaped (B, {self.feature_dim}), got {tuple(z.shape)}"
            )
        return self.output(z).squeeze(-1)


class RFNet(nn.Module):
    """RF-Net v1 with a T1-equivalent modulation path and an unused SNR head."""

    def __init__(self, num_classes: int = 11, dropout: float = 0.6):
        super().__init__()
        self.encoder = RFEncoder(dropout=dropout)
        self.modulation_head = ModulationHead(
            feature_dim=self.encoder.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.snr_head = SNRHead(feature_dim=self.encoder.feature_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.modulation_head(z)

    def estimate_snr(self, z: torch.Tensor) -> torch.Tensor:
        return self.snr_head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))

    def forward_multitask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.classify(z), self.estimate_snr(z)
