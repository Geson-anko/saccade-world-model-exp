"""系列モデルの抽象基底 ``SequenceModel``。

履歴を内部状態 (belief) に畳み込みながら系列を処理する系列モデルの共通契約。 minGRU / Transformer
など実体は差し替え可能で、いずれも本基底を継承する。 hidden 状態の型はモデル依存のため型引数 ``THidden`` でパラメータ化する
(minGRU は ``torch.Tensor``、再帰状態を持たない Transformer は ``None`` など)。
"""

import abc
from typing import final

import torch
import torch.nn as nn

__all__ = ["SequenceModel"]


class SequenceModel[THidden](nn.Module, abc.ABC):
    """系列モデルの抽象基底 (型引数 ``THidden`` は hidden 状態の型)。

    公開メソッド ``forward`` / ``step`` は基底が所有する ``@final`` な検証ラッパで、 入力 shape
    (``forward`` は ``(*, seq, dim)``、``step`` は ``(*, dim)``) を検証して
    から抽象フック ``_forward`` / ``_step`` へ委譲し、戻り ``out`` が入力と同形である
    ことも検証する。サブクラスはフックのみを実装し、``@final`` のため検証ラッパを override
    してバイパスすることはできない。

    hidden は ``None`` を渡すと初期状態から開始し、戻り値の hidden を次回入力へ渡す
    ことでチャンクを跨いで連続させる。初期 hidden の取得法 (zeros / None / 学習可能)
    はモデル依存のため本契約には含めない。
    """

    def __call__(
        self, x: torch.Tensor, hidden: THidden | None = None
    ) -> tuple[torch.Tensor, THidden]:
        # nn.Module.__call__ は Any を返すため、forward の戻り型を呼び出し側へ伝える
        # 薄い委譲 (型のためだけのメソッド)。runtime は super().__call__ が hooks 経由
        # で forward に dispatch する。
        return super().__call__(x, hidden)

    @final
    def forward(
        self, x: torch.Tensor, hidden: THidden | None = None
    ) -> tuple[torch.Tensor, THidden]:
        """``x: (*, seq, dim)`` の入出力 shape を検証し ``_forward`` へ委譲 (override
        不可)。"""
        if x.ndim < 2:
            raise ValueError(
                f"forward expects (*, seq, dim) with ndim >= 2, got shape {tuple(x.shape)}"
            )
        out, hidden = self._forward(x, hidden)
        if out.shape != x.shape:
            raise ValueError(
                "_forward must return out matching the input shape (*, seq, dim) "
                f"{tuple(x.shape)}, got {tuple(out.shape)}"
            )
        return out, hidden

    @final
    def step(
        self, x: torch.Tensor, hidden: THidden | None = None
    ) -> tuple[torch.Tensor, THidden]:
        """``x: (*, dim)`` の入出力 shape を検証し ``_step`` へ委譲 (override 不可)。"""
        if x.ndim < 1:
            raise ValueError(
                f"step expects (*, dim) with ndim >= 1, got shape {tuple(x.shape)}"
            )
        out, hidden = self._step(x, hidden)
        if out.shape != x.shape:
            raise ValueError(
                "_step must return out matching the input shape (*, dim) "
                f"{tuple(x.shape)}, got {tuple(out.shape)}"
            )
        return out, hidden

    @abc.abstractmethod
    def _forward(
        self, x: torch.Tensor, hidden: THidden | None = None
    ) -> tuple[torch.Tensor, THidden]:
        """``forward`` の実体。``x: (*, seq, dim)`` → ``(out: (*, seq, dim),
        hidden)``。

        Args:
            x: 入力系列 (基底で ndim >= 2 を検証済み)。任意の leading batch 次元を許す。
            hidden: 直前までの内部状態。``None`` なら初期状態から開始する。

        Returns:
            出力系列 (入力と同形 ``(*, seq, dim)``。基底が検証する) と、系列末尾時点の
            更新後 hidden。
        """
        ...

    @abc.abstractmethod
    def _step(
        self, x: torch.Tensor, hidden: THidden | None = None
    ) -> tuple[torch.Tensor, THidden]:
        """``step`` の実体。``x: (*, dim)`` → ``(out: (*, dim), hidden)``。

        Args:
            x: 単一タイムステップの入力 (基底で ndim >= 1 を検証済み)。
            hidden: 直前の内部状態。``None`` なら初期状態から開始する。

        Returns:
            単一ステップ出力 (入力と同形 ``(*, dim)``。基底が検証する) と更新後 hidden。
        """
        ...
