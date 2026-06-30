"""位置ごとの全結合 (MLP) building block。"""

from typing import override

import torch
import torch.nn as nn

__all__ = ["Mlp"]


class Mlp(nn.Module):
    """Transformer の位置ごとの全結合 (Linear-GELU-dropout-Linear-dropout)。"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
