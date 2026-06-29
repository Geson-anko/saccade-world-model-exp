"""minGRU 系列モデル (Feng et al. 2024, arXiv:2410.01201)。

ゲート ``z`` と候補 ``h̃`` が入力 ``x_t`` のみに依存する (``h_{t-1}`` 非依存) ため、 線形漸化式
``h_t = (1−z_t)·h_{t−1} + z_t·h̃_t`` を log-space 並列スキャンで一括計算できる。
固定長の再帰状態が belief になる。``MinGRU`` のみ :class:`SequenceModel` を継承し、
``MinGRULayer`` / ``MinGRUBlock`` は内部部品の素の :class:`~torch.nn.Module`。
"""

from collections.abc import Callable
from functools import partial
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SequenceModel
from .mlp import Mlp
from .norm import RMSNorm
from .weight import init_weights

__all__ = ["MinGRU", "MinGRUBlock", "MinGRULayer"]


def _g(x: torch.Tensor) -> torch.Tensor:
    """連続正値活性 ``g``: ``x≥0`` で ``x+0.5``、``x<0`` で ``sigmoid(x)``。"""
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def _log_g(x: torch.Tensor) -> torch.Tensor:
    """``log g(x)``: ``x≥0`` で ``log(relu(x)+0.5)``、``x<0`` で
    ``−softplus(−x)``。

    ``torch.where`` は両ブランチを評価するため、``(x+0.5).log()`` を直書きすると ``x<−0.5`` で
    ``log(負)=NaN`` が backward に逆流する。``relu`` でクランプして防ぐ。
    """
    return torch.where(
        x >= 0,
        torch.log(F.relu(x) + 0.5),
        -F.softplus(-x),
    )


def _parallel_scan_log(
    log_coeffs: torch.Tensor, log_values: torch.Tensor
) -> torch.Tensor:
    """log-space の線形再帰スキャン (h_0 を含まない純粋系列)。

    ``h_t = coeff_t · h_{t−1} + value_t`` (``h_0=0``) を seq 軸 (``dim=-2``) で一括計算する。
    ``a = cumsum(log_coeffs)``、``b = logcumsumexp(log_values − a)`` として
    ``exp(a + b)`` を返す。

    Args:
        log_coeffs: 係数の対数 ``log(1−z)``。shape ``(*, len, dim)``。
        log_values: 値の対数 ``log z + log g(h̃)``。shape ``(*, len, dim)``。

    Returns:
        h_0 寄与を除いた hidden 系列 ``(*, len, dim)``。
    """
    a = torch.cumsum(log_coeffs, dim=-2)
    b = torch.logcumsumexp(log_values - a, dim=-2)
    return torch.exp(a + b)


class MinGRULayer(nn.Module):
    """純粋 minGRU セル (出力射影なし)。``input_dim != hidden_dim`` を許す。"""

    def __init__(
        self, input_dim: int, hidden_dim: int, *, init_std: float = 0.02
    ) -> None:
        # init_std is accepted for a uniform constructor signature across the
        # minGRU stack; weights are actually initialised by MinGRU.apply.
        super().__init__()
        self.fc_z = nn.Linear(input_dim, hidden_dim)
        self.fc_h = nn.Linear(input_dim, hidden_dim)

    @override
    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-space 並列スキャンで hidden 系列を一括計算する。

        Args:
            x: 入力系列 ``(*, len, input_dim)``。
            hidden: 初期 hidden ``h_0`` ``(*, hidden_dim)``。``None`` なら zeros。

        Returns:
            全ステップの hidden 列 ``out: (*, len, hidden_dim)`` と末尾 hidden
            ``h_last: (*, hidden_dim)``。
        """
        k = self.fc_z(x)
        log_z = -F.softplus(-k)  # log z
        log_one_minus_z = -F.softplus(k)  # log(1−z)
        log_tilde = _log_g(self.fc_h(x))  # log g(h̃)

        log_values = log_z + log_tilde
        out = _parallel_scan_log(log_one_minus_z, log_values)

        if hidden is not None:
            # h_0 別項: decay_t = prod_{i<=t}(1−z_i) を h_0 にかけて足し戻す。
            decay = torch.exp(torch.cumsum(log_one_minus_z, dim=-2))
            out = out + decay * hidden.unsqueeze(-2)
        return out, out[..., -1, :]

    def step(
        self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        """単一ステップの直接漸化式 ``h = (1−z)·h_prev + z·g(h̃)``。

        log-space を経由しない素の式で、並列スキャンの独立参照になる。

        Args:
            x_t: 単一ステップ入力 ``(*, input_dim)``。
            h_prev: 直前 hidden ``(*, hidden_dim)``。``None`` なら zeros。

        Returns:
            更新後 hidden ``h_t: (*, hidden_dim)``。
        """
        z = torch.sigmoid(self.fc_z(x_t))
        h_tilde = _g(self.fc_h(x_t))
        if h_prev is None:
            return z * h_tilde
        return (1 - z) * h_prev + z * h_tilde


class MinGRUBlock(nn.Module):
    """Pre-norm 残差ブロック (minGRU セル + MLP)。in=out=dim。"""

    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.gru = MinGRULayer(dim, dim, init_std=init_std)
        self.norm2 = RMSNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, dropout)

    @override
    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x:(*,len,dim)`` → ``(out:(*,len,dim), h_last:(*,dim))``。"""
        gru_out, h_last = self.gru(self.norm1(x), hidden)
        x = x + gru_out
        x = x + self.mlp(self.norm2(x))
        return x, h_last

    def step(
        self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x_t:(*,dim)`` → ``(out_t:(*,dim), h_t:(*,dim))``。"""
        h_t = self.gru.step(self.norm1(x_t), h_prev)
        x_t = x_t + h_t
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, h_t


class MinGRU(SequenceModel[torch.Tensor]):
    """積層 minGRU 系列モデル (``SequenceModel[torch.Tensor]``)。

    全 ``depth`` ブロックの末尾 hidden を ``depth`` 軸に stack した ``(*, depth, dim)``
    を belief (= 系列モデルの内部状態) として扱う。これを ``forward`` / ``step`` の戻り
    hidden として次チャンクへ渡すと履歴を跨いで連続させられる (基底 :class:`SequenceModel`
    の契約)。入出力は同形 (in=out=dim)。

    Attributes:
        dim: 入出力およびブロック内部の特徴次元。
        depth: 積層ブロック数。belief の ``depth`` 軸長と一致する。
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """積層 minGRU を構築する。

        Raises:
            ValueError: ``dim`` または ``depth`` が正でないとき (特徴次元・層数として
                成立しないため)。
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.dim: int = dim
        self.depth: int = depth
        self.blocks = nn.ModuleList(
            [
                MinGRUBlock(
                    dim, mlp_ratio=mlp_ratio, dropout=dropout, init_std=init_std
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)

        self.apply(partial(init_weights, init_std=init_std))

    def _run_blocks(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None,
        run_block: Callable[
            [MinGRUBlock, torch.Tensor, torch.Tensor | None],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """各ブロックへ hidden を ``depth`` 軸で分配して順に通し、末尾 hidden を stack する。

        ``run_block`` が並列 (``MinGRUBlock.forward``) か単一ステップ
        (``MinGRUBlock.step``) かだけが ``_forward`` / ``_step`` の違いで、
        分配・畳み込み・最終 norm・stack は共通。
        """
        h_last_per_layer: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            assert isinstance(blk, MinGRUBlock)
            h0 = hidden[..., i, :] if hidden is not None else None
            x, h_last = run_block(blk, x, h0)
            h_last_per_layer.append(h_last)
        out = self.norm(x)
        h_stacked = torch.stack(h_last_per_layer, dim=-2)  # (*, depth, dim)
        return out, h_stacked

    @override
    def _forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 全ブロックを並列スキャンで通す。belief hidden は (*, depth, dim)。
        return self._run_blocks(x, hidden, lambda blk, inp, h: blk(inp, h))

    @override
    def _step(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 全ブロックを 1 ステップ漸化式で通す。belief hidden は (*, depth, dim)。
        return self._run_blocks(x, hidden, lambda blk, inp, h: blk.step(inp, h))
