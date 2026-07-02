from __future__ import annotations

from typing import ClassVar, final, override

import attrs
import torch
import torchvision.transforms.v2.functional as F

from ..size import Size2d
from .base import (
    BatchedElement,
    BatchedElementSequence,
    Element,
    ElementSequence,
)
from .image import Image, ImageSequence

__all__ = [
    "FOCUS_DIM",
    "BatchedFocus",
    "BatchedFocusSequence",
    "Focus",
    "FocusSequence",
]

FOCUS_DIM = 3


class _FocusValidation:
    """Point∈[-1,1]・zoom∈[0,1] の値域検証を提供する mixin。"""

    __slots__ = ()

    tensor: torch.Tensor

    def is_valid(self) -> bool:
        """全要素が point∈[-1,1]・zoom∈[0,1] に収まっているか返す。"""
        point, zoom = self.tensor[..., :2], self.tensor[..., 2]
        point_ok = (point >= -1.0).all() & (point <= 1.0).all()
        zoom_ok = (zoom >= 0.0).all() & (zoom <= 1.0).all()
        return bool(point_ok & zoom_ok)

    def validate(self) -> None:
        """値域外の要素があれば ValueError を投げる (収まっていれば何もしない)。"""
        if not self.is_valid():
            raise ValueError(
                "Focus values out of range: point must be in [-1, 1], zoom in [0, 1]"
            )


@final
@attrs.define(slots=True, frozen=True, eq=False)
class Focus(_FocusValidation, Element):
    """画像の正方切り取りを指示する行動 a=(point, zoom)。

    tensor は (3,) の [x, y, zoom]。point は [-1, 1] (中心 0, y=+1 が下)、zoom は
    [0, 1]。
    """

    _NDIM: ClassVar[int] = 1
    _SHAPE_DESC: ClassVar[str] = f"({FOCUS_DIM},)"
    _SHAPE: ClassVar[list[int | None] | None] = [FOCUS_DIM]

    @property
    def point(self) -> tuple[float, float]:
        return (float(self.tensor[0]), float(self.tensor[1]))

    @property
    def zoom(self) -> float:
        return float(self.tensor[2])

    @classmethod
    def init(cls, point: tuple[float, float], zoom: float) -> Focus:
        """値域検証のうえ [x, y, zoom] を保持する Focus を構築する。

        point が [-1, 1] 外、または zoom が [0, 1] 外なら ValueError。
        """
        x, y = point
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0):
            raise ValueError(f"Focus point must be within [-1, 1], got {point}")
        if not (0.0 <= zoom <= 1.0):
            raise ValueError(f"Focus zoom must be within [0, 1], got {zoom}")
        return cls(torch.tensor([x, y, zoom], dtype=torch.float32))

    def __call__(self, image: Image) -> Image:
        """この行動で image を正方切り取りした新しい Image を返す (はみ出しは 0 埋め)。"""
        squared = image.square_pad()  # 既に正方なら square_pad が自身を返す
        s = squared.height
        x, y = self.point
        cx = (s / 2) * (1 + x)
        cy = (s / 2) * (1 + y)
        crop = max(1, int(round(self.zoom * s)))
        left = int(round(cx - crop / 2))
        top = int(round(cy - crop / 2))
        return Image(F.crop(squared.tensor, top, left, crop, crop))


@final
@attrs.define(slots=True, frozen=True, eq=False)
class FocusSequence(_FocusValidation, ElementSequence[Focus]):
    """(seq, 3) の行動系列を内包する不変な値オブジェクト。

    各行は [x, y, zoom] の行動ベクトル。要素 0（空系列）も許容する。
    """

    _NDIM: ClassVar[int] = 2
    _SHAPE_DESC: ClassVar[str] = f"(seq, {FOCUS_DIM})"
    _SHAPE: ClassVar[list[int | None] | None] = [None, FOCUS_DIM]

    @classmethod
    @override
    def item_type(cls) -> type[Focus]:
        return Focus

    def apply(self, image: Image, size: Size2d) -> ImageSequence:
        """各 Focus で image を切り取り size へ resize した観測列を返す。

        空列は from_elements が弾く (C を決められないため)。値域不正テンソルは Focus 側で扱う。
        """
        return ImageSequence.from_elements(focus(image).resize(size) for focus in self)


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedFocus(_FocusValidation, BatchedElement[Focus]):
    """(batch, 3) の行動バッチを内包する不変な値オブジェクト。

    各行は [x, y, zoom] の行動ベクトル。要素 0（空バッチ）も許容する。
    """

    _NDIM: ClassVar[int] = 2
    _SHAPE_DESC: ClassVar[str] = f"(batch, {FOCUS_DIM})"
    _SHAPE: ClassVar[list[int | None] | None] = [None, FOCUS_DIM]

    @classmethod
    @override
    def item_type(cls) -> type[Focus]:
        return Focus


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedFocusSequence(
    _FocusValidation, BatchedElementSequence[BatchedFocus, FocusSequence, Focus]
):
    """(batch, seq, 3) の行動系列バッチを内包する不変な値オブジェクト。

    各行は [x, y, zoom] の行動ベクトル。要素 0（空バッチ）も許容する。
    """

    _NDIM: ClassVar[int] = 3
    _SHAPE_DESC: ClassVar[str] = f"(batch, seq, {FOCUS_DIM})"
    _SHAPE: ClassVar[list[int | None] | None] = [None, None, FOCUS_DIM]

    @classmethod
    @override
    def item_type(cls) -> type[FocusSequence]:
        return FocusSequence

    @classmethod
    @override
    def _batch_type(cls) -> type[BatchedFocus]:
        return BatchedFocus
