from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import final, override

import attrs
import torch
import torchvision.transforms.v2.functional as F

from .device import DeviceLike, DeviceTransferMixin
from .image import Image, ImageSequence
from .size import Size2d

__all__ = ["BatchedFocusSequence", "Focus", "FocusSequence"]


def _focus_tensor_is_valid(t: torch.Tensor) -> bool:
    point, zoom = t[..., :2], t[..., 2]
    point_ok = (point >= -1.0).all() & (point <= 1.0).all()
    zoom_ok = (zoom >= 0.0).all() & (zoom <= 1.0).all()
    return bool(point_ok & zoom_ok)


@attrs.define(frozen=True)
class Focus:
    """画像の正方切り取りを指示する行動 a=(point, zoom)。

    point は [-1, 1] (中心 0, y=+1 が下)、zoom は [0, 1]。 point/zoom
    はスカラーで値等価が自然なため eq=True (既定) のまま hashable とする。
    """

    point: tuple[float, float]
    zoom: float

    def __attrs_post_init__(self) -> None:
        x, y = self.point
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0):
            raise ValueError(f"Focus point must be within [-1, 1], got {self.point}")
        if not (0.0 <= self.zoom <= 1.0):
            raise ValueError(f"Focus zoom must be within [0, 1], got {self.zoom}")

    def tensor(self) -> torch.Tensor:
        """行動ベクトル [x, y, zoom] を shape (3,) の float32 tensor で返す (CPU)。"""
        x, y = self.point
        return torch.tensor([x, y, self.zoom], dtype=torch.float32)

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
class FocusSequence(DeviceTransferMixin):
    """(seq, 3) の行動系列を内包する不変な値オブジェクト。

    各行は [x, y, zoom] の行動ベクトル。要素 0（空系列）も許容する。
    """

    tensor: torch.Tensor  # (seq, 3)

    def __attrs_post_init__(self) -> None:
        if self.tensor.ndim != 2 or self.tensor.shape[-1] != 3:
            raise ValueError(
                f"FocusSequence expects a (seq, 3) tensor, got "
                f"ndim={self.tensor.ndim} shape={tuple(self.tensor.shape)}"
            )

    def __iter__(self) -> Iterator[Focus]:
        return (Focus((float(x), float(y)), float(z)) for x, y, z in self.tensor)

    @property
    @override
    def device(self) -> torch.device:
        return self.tensor.device

    @override
    def to(self, device: DeviceLike) -> FocusSequence:
        return FocusSequence(self.tensor.to(device))

    def is_valid(self) -> bool:
        """全要素が point∈[-1,1]・zoom∈[0,1] に収まっているか返す。"""
        return _focus_tensor_is_valid(self.tensor)

    def validate(self) -> None:
        """値域外の要素があれば ValueError を投げる (収まっていれば何もしない)。"""
        if not self.is_valid():
            raise ValueError(
                "Focus values out of range: point must be in [-1, 1], zoom in [0, 1]"
            )

    @classmethod
    def from_focuses(cls, focuses: Iterable[Focus]) -> FocusSequence:
        """複数の Focus を先頭次元に積んだ FocusSequence を構築する。

        空入力は ValueError。各 Focus の tensor() を stack する。
        """
        materialized = list(focuses)
        if not materialized:
            raise ValueError("from_focuses requires at least one Focus")
        stacked = torch.stack([focus.tensor() for focus in materialized], dim=0)
        return cls(stacked)

    def apply(self, image: Image, size: Size2d) -> ImageSequence:
        """各 Focus で image を切り取り size へ resize した観測列を返す。

        空列は from_images が弾く (C を決められないため)。値域不正テンソルは Focus 構築時に
        ValueError。
        """
        return ImageSequence.from_images(focus(image).resize(size) for focus in self)


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedFocusSequence(DeviceTransferMixin):
    """(batch, seq, 3) の行動系列バッチを内包する不変な値オブジェクト。

    各行は [x, y, zoom] の行動ベクトル。要素 0（空バッチ）も許容する。
    """

    tensor: torch.Tensor  # (batch, seq, 3)

    def __attrs_post_init__(self) -> None:
        if self.tensor.ndim != 3 or self.tensor.shape[-1] != 3:
            raise ValueError(
                f"BatchedFocusSequence expects a (batch, seq, 3) tensor, got "
                f"ndim={self.tensor.ndim} shape={tuple(self.tensor.shape)}"
            )

    def __iter__(self) -> Iterator[FocusSequence]:
        return (FocusSequence(seq) for seq in self.tensor)

    @property
    @override
    def device(self) -> torch.device:
        return self.tensor.device

    @override
    def to(self, device: DeviceLike) -> BatchedFocusSequence:
        return BatchedFocusSequence(self.tensor.to(device))

    def is_valid(self) -> bool:
        """全要素が point∈[-1,1]・zoom∈[0,1] に収まっているか返す。"""
        return _focus_tensor_is_valid(self.tensor)

    def validate(self) -> None:
        """値域外の要素があれば ValueError を投げる (収まっていれば何もしない)。"""
        if not self.is_valid():
            raise ValueError(
                "Focus values out of range: point must be in [-1, 1], zoom in [0, 1]"
            )

    @classmethod
    def from_sequences(cls, sequences: Iterable[FocusSequence]) -> BatchedFocusSequence:
        """複数の FocusSequence を先頭次元に積んだ BatchedFocusSequence を構築する。

        空入力は ValueError。各 FocusSequence の tensor を stack する。
        """
        materialized = list(sequences)
        if not materialized:
            raise ValueError("from_sequences requires at least one FocusSequence")
        stacked = torch.stack([sequence.tensor for sequence in materialized], dim=0)
        return cls(stacked)
