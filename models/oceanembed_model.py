"""
OceanEmbed Phase 3 model.

SatelliteEncoder:
    (B, C, 9, 9) -> (B, embedding_dim)

TemperatureDecoder:
    (B, embedding_dim) -> (B, 15)

OceanEmbedModel:
    returns both embedding and predicted temperature profile.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SatelliteEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.projection(x)


class TemperatureDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        output_dim: int = 15,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)


class OceanEmbedModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        embedding_dim: int = 64,
        output_dim: int = 15,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.encoder = SatelliteEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )
        self.decoder = TemperatureDecoder(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        embedding = self.encoder(x)
        prediction = self.decoder(embedding)
        return embedding, prediction