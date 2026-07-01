from __future__ import annotations

import enum
from pathlib import Path
from typing import ClassVar, final, override

import attrs
import torch
import torchvision.transforms.v2.functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2.functional import InterpolationMode

from ..size import Size2d, size_2d_to_tuple
from .base import (
    BatchedElement,
    BatchedElementSequence,
    Element,
    ElementSequence,
)

# float メソッドが組み込み float をシャドウするため、型注釈用に別名を保持する。
_float = float

__all__ = [
    "BatchedImage",
    "BatchedImageSequence",
    "ChannelFormat",
    "Image",
    "ImageSequence",
]


class ChannelFormat(enum.IntEnum):
    """チャンネル構成。値はチャンネル数 (1=GRAY, 3=RGB, 4=RGBA)。"""

    GRAY = 1
    RGB = 3
    RGBA = 4


@final
@attrs.define(slots=True, frozen=True, eq=False)
class Image(Element):
    """(C, H, W) の単一画像を内包する不変な値オブジェクト。

    frozen だが内部 tensor の in-place 変更は防げない。変換系メソッドは新しい Image を返す。
    """

    _NDIM: ClassVar[int] = 3
    _SHAPE_DESC: ClassVar[str] = "(C, H, W)"

    @property
    def channels(self) -> int:
        return self.tensor.shape[0]

    @property
    def height(self) -> int:
        return self.tensor.shape[1]

    @property
    def width(self) -> int:
        return self.tensor.shape[2]

    @property
    def size(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def channel_format(self) -> ChannelFormat:
        return ChannelFormat(self.channels)

    @property
    def is_squared(self) -> bool:
        return self.height == self.width

    def float(self) -> Image:
        """浮動小数点 (float32) へキャストした新しい Image を返す。"""
        return Image(self.tensor.float())

    def uint8(self) -> Image:
        """符号なし 8bit 整数 (uint8) へキャストした新しい Image を返す。"""
        return Image(self.tensor.to(torch.uint8))

    def standardize(self, mean: _float = 0.0, std: _float = 1.0) -> Image:
        """画素統計を平均 mean・標準偏差 std に揃えた新しい Image を返す。

        画像全体で統計をとり、定数画像では全画素を mean にする (0 除算回避)。
        """
        t = self.tensor.float()
        centered = t - t.mean()
        spread = t.std()
        if float(spread) == 0.0:
            return Image(centered + mean)
        return Image(centered / spread * std + mean)

    def normalize(self, min: _float = 0.0, max: _float = 1.0) -> Image:
        """画素の min/max を [min, max] へ線形伸張した新しい Image を返す。

        画像全体で min/max をとり、レンジゼロでは全画素を min にする。
        """
        t = self.tensor.float()
        lo = t.min()
        hi = t.max()
        if float(hi) == float(lo):
            return Image(torch.full_like(t, min))
        return Image((t - lo) / (hi - lo) * (max - min) + min)

    def as_channel_format(self, fmt: ChannelFormat) -> Image:
        """指定のチャンネル形式へ変換した新しい Image を返す。

        いったん RGB を経由して目標形式へ移す。同一形式なら自身を返す。
        """
        if self.channel_format is fmt:
            return self
        rgb = self._as_rgb()
        match fmt:
            case ChannelFormat.GRAY:
                out = F.rgb_to_grayscale(rgb, num_output_channels=1)
            case ChannelFormat.RGB:
                out = rgb
            case ChannelFormat.RGBA:
                out = torch.cat([rgb, self._alpha_like(rgb)], dim=0)
        return Image(out)

    def _as_rgb(self) -> torch.Tensor:
        match self.channel_format:
            case ChannelFormat.GRAY:
                return self.tensor.repeat(3, 1, 1)
            case ChannelFormat.RGBA:
                return self.tensor[:3]
            case ChannelFormat.RGB:
                return self.tensor

    @staticmethod
    def _alpha_like(rgb: torch.Tensor) -> torch.Tensor:
        fill = 1.0 if rgb.is_floating_point() else 255
        return torch.full((1, *rgb.shape[1:]), fill, dtype=rgb.dtype, device=rgb.device)

    def square_pad(self, fill_value: _float = 0) -> Image:
        """短辺を長辺に合わせて対称パディングし正方化した新しい Image を返す。

        差が奇数なら前側を 1px 小さくする。既に正方なら自身を返す。
        """
        if self.is_squared:
            return self
        diff = abs(self.height - self.width)
        before = diff // 2
        after = diff - before
        if self.width < self.height:
            padding = [before, 0, after, 0]
        else:
            padding = [0, before, 0, after]
        return Image(F.pad(self.tensor, padding, fill=fill_value))

    def resize(self, size: Size2d) -> Image:
        """指定サイズへリサイズした新しい Image を返す (int は正方、dtype 保存)。"""
        h, w = size_2d_to_tuple(size)
        out = F.resize(
            self.tensor,
            [h, w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return Image(out)

    @classmethod
    def load(cls, path: Path) -> Image:
        """画像ファイルを読み込み Image を返す (uint8・元チャンネルのまま)。"""
        tensor = read_image(str(path), mode=ImageReadMode.UNCHANGED)
        return cls(tensor)

    def save(self, path: Path) -> None:
        """[0, 255] へ正規化し uint8 で画像ファイルに保存する。

        拡張子でフォーマット判定。RGBA は PNG のみ可 (JPEG は alpha 非対応)。
        """
        # JPEG は alpha を持てない。拡張子に反して暗黙に別フォーマットへ
        # 切り替えず、不整合は明示的に弾く。
        if (
            path.suffix.lower() in (".jpg", ".jpeg")
            and self.channel_format is ChannelFormat.RGBA
        ):
            raise ValueError(f"JPEG は alpha 非対応で RGBA を保存できない: {path.name}")
        # torchvision.io.write_png は 1/3ch のみで RGBA を書けないため PIL を使う。
        tensor = self.normalize(0, 255).uint8().tensor.cpu()
        F.to_pil_image(tensor).save(path)


@final
@attrs.define(slots=True, frozen=True, eq=False)
class ImageSequence(ElementSequence[Image]):
    """(len, C, H, W) の画像系列を内包する不変な値オブジェクト。

    観測列をまとめて運ぶための内部表現。indexing で個々の Image を取り出す。要素 0（空系列）も許容する。
    """

    _NDIM: ClassVar[int] = 4
    _SHAPE_DESC: ClassVar[str] = "(len, C, H, W)"

    @classmethod
    @override
    def _item_type(cls) -> type[Image]:
        return Image


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedImage(BatchedElement[Image]):
    """(batch, C, H, W) の画像バッチを内包する不変な値オブジェクト。

    画像のバッチをまとめて運ぶための内部表現。indexing で個々の Image を取り出す。要素 0（空バッチ）も許容する。
    """

    _NDIM: ClassVar[int] = 4
    _SHAPE_DESC: ClassVar[str] = "(batch, C, H, W)"

    @classmethod
    @override
    def _item_type(cls) -> type[Image]:
        return Image


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedImageSequence(BatchedElementSequence[BatchedImage, ImageSequence]):
    """(batch, len, C, H, W) の画像系列バッチを内包する不変な値オブジェクト。

    観測列のバッチをまとめて運ぶための内部表現。batch 軸で個々の ImageSequence を、seq 軸で 個々の
    BatchedImage を取り出す。要素 0（空バッチ）も許容する。
    """

    _NDIM: ClassVar[int] = 5
    _SHAPE_DESC: ClassVar[str] = "(batch, len, C, H, W)"

    @classmethod
    @override
    def _item_type(cls) -> type[ImageSequence]:
        return ImageSequence

    @classmethod
    @override
    def _batch_type(cls) -> type[BatchedImage]:
        return BatchedImage
