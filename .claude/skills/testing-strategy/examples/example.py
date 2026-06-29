"""テスト方針の説明用サンプル (縮約コード・実在モジュールではない)。

public な `Standardizer` をテスト対象とし、private helper `_safe_std` は
直接テストしない例。対になるテストは `test_example.py`。
"""

from typing import override

import torch
import torch.nn as nn

__all__ = ["Standardizer"]  # public 面は __all__ で明示


class Standardizer(nn.Module):
    """入力 tensor を平均 0・標準偏差 1 に標準化する (public・テスト対象)。"""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / _safe_std(x, self.eps)


def _safe_std(x: torch.Tensor, eps: float) -> torch.Tensor:
    # private helper: 直接テストしない。Standardizer 越しに振る舞いを検証する。
    return x.std().clamp_min(eps)
