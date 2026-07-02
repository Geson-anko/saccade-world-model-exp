from __future__ import annotations

from typing import ClassVar, Self, final, override

import attrs
import torch

from .base import (
    BatchedElement,
    BatchedElementSequence,
    Element,
    ElementSequence,
)

__all__ = [
    "BatchedLatent",
    "BatchedLatentSequence",
    "Latent",
    "LatentSequence",
]


@final
@attrs.define(slots=True, frozen=True, eq=False)
class Latent(Element):
    """(dim,) の潜在ベクトルを内包する不変な値オブジェクト。"""

    _NDIM: ClassVar[int] = 1
    _SHAPE_DESC: ClassVar[str] = "(dim,)"


@final
@attrs.define(slots=True, frozen=True, eq=False)
class LatentSequence(ElementSequence[Latent]):
    """(seq, dim) の潜在系列を内包する不変な値オブジェクト。要素 0（空系列）も許容する。"""

    _NDIM: ClassVar[int] = 2
    _SHAPE_DESC: ClassVar[str] = "(seq, dim)"

    @classmethod
    @override
    def _item_type(cls) -> type[Latent]:
        return Latent


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedLatent(BatchedElement[Latent]):
    """(batch, dim) の潜在バッチを内包する不変な値オブジェクト。要素 0（空バッチ）も許容する。"""

    _NDIM: ClassVar[int] = 2
    _SHAPE_DESC: ClassVar[str] = "(batch, dim)"

    @classmethod
    @override
    def _item_type(cls) -> type[Latent]:
        return Latent


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedLatentSequence(
    BatchedElementSequence[BatchedLatent, LatentSequence, Latent]
):
    """(batch, seq, dim) の潜在系列バッチを内包する不変な値オブジェクト。要素 0（空バッチ）も許容する。"""

    _NDIM: ClassVar[int] = 3
    _SHAPE_DESC: ClassVar[str] = "(batch, seq, dim)"

    @classmethod
    @override
    def _item_type(cls) -> type[LatentSequence]:
        return LatentSequence

    @classmethod
    @override
    def _batch_type(cls) -> type[BatchedLatent]:
        return BatchedLatent

    def shift(self) -> Self:
        """Seq 軸(dim=1)を後方へ 1 つずらし、先頭を 0 埋めした新インスタンスを返す。

        ``[e_1, …, e_T]`` → ``[0, e_1, …, e_{T-1}]``（末尾 ``e_T`` は押し出す）。
        自己回帰の潜在予測で input / target を作るために使う（in-place ではない）。
        """
        t = self.tensor
        shifted = torch.zeros_like(t)
        shifted[:, 1:] = t[:, :-1]
        return type(self)(shifted)
