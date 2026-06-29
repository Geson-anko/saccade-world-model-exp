"""RMSNorm 正規化 building block。"""

from typing import override

import torch
import torch.nn as nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """RMS 正規化 (最終次元の二乗平均で割り、学習可能な weight でスケールする)。"""

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 任意の leading batch を許す。正規化は最終次元に対して行う。
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
