from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import final, overload, override

import attrs
import torch
import torchvision.transforms.v2.functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2.functional import InterpolationMode

from .device import DeviceLike, DeviceTransferMixin
from .size import Size2d, size_2d_to_tuple

# float メソッドが組み込み float をシャドウするため、型注釈用に別名を保持する。
_float = float

__all__ = ["BatchedImageSequence", "ChannelFormat", "Image", "ImageSequence"]


class ChannelFormat(enum.IntEnum):
    """チャンネル構成。値はチャンネル数 (1=GRAY, 3=RGB, 4=RGBA)。"""

    GRAY = 1
    RGB = 3
    RGBA = 4


@final
@attrs.define(slots=True, frozen=True, eq=False)
class Image(DeviceTransferMixin):
    """(C, H, W) の単一画像を内包する不変な値オブジェクト。

    frozen だが内部 tensor の in-place 変更は防げない。変換系メソッドは新しい Image を返す。
    """

    tensor: torch.Tensor  # (channels, height, width)

    def __attrs_post_init__(self) -> None:
        if self.tensor.ndim != 3:
            raise ValueError(
                f"Image expects a (C, H, W) tensor, got "
                f"ndim={self.tensor.ndim} shape={tuple(self.tensor.shape)}"
            )

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

    @property
    @override
    def device(self) -> torch.device:
        return self.tensor.device

    @override
    def to(self, device: DeviceLike) -> Image:
        """指定 device に転送した新しい Image を返す (in-place ではない)。"""
        return Image(self.tensor.to(device))

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
class ImageSequence(DeviceTransferMixin):
    """(len, C, H, W) の画像系列を内包する不変な値オブジェクト。

    観測列をまとめて運ぶための内部表現。indexing で個々の Image を取り出す。 要素 0（空系列）も許容する。
    """

    tensor: torch.Tensor  # (len, C, H, W)

    def __attrs_post_init__(self) -> None:
        if self.tensor.ndim != 4:
            raise ValueError(
                f"ImageSequence expects a (len, C, H, W) tensor, got "
                f"ndim={self.tensor.ndim} shape={tuple(self.tensor.shape)}"
            )

    def __len__(self) -> int:
        return self.tensor.shape[0]

    @overload
    def __getitem__(self, index: int) -> Image: ...
    @overload
    def __getitem__(self, index: slice) -> ImageSequence: ...
    def __getitem__(self, index: int | slice) -> Image | ImageSequence:
        if isinstance(index, slice):
            return ImageSequence(self.tensor[index])
        return Image(self.tensor[index])

    def __iter__(self) -> Iterator[Image]:
        return (Image(frame) for frame in self.tensor)

    @property
    @override
    def device(self) -> torch.device:
        return self.tensor.device

    @override
    def to(self, device: DeviceLike) -> ImageSequence:
        return ImageSequence(self.tensor.to(device))

    @classmethod
    def from_images(cls, images: Iterable[Image]) -> ImageSequence:
        """複数の Image を先頭次元に積んだ ImageSequence を構築する。

        各 Image の (C, H, W) が一致している必要がある (呼び出し側責務)。 空入力は
        ValueError。shape 不一致は torch.stack の例外を伝播する。
        """
        materialized = list(images)
        if not materialized:
            raise ValueError("from_images requires at least one Image")
        stacked = torch.stack([image.tensor for image in materialized], dim=0)
        return cls(stacked)


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedImageSequence(DeviceTransferMixin):
    """(batch, len, C, H, W) の画像系列バッチを内包する不変な値オブジェクト。

    観測列のバッチをまとめて運ぶための内部表現。indexing で個々の ImageSequence を取り出す。要素
    0（空バッチ）も許容する。
    """

    tensor: torch.Tensor  # (batch, len, C, H, W)

    def __attrs_post_init__(self) -> None:
        if self.tensor.ndim != 5:
            raise ValueError(
                f"BatchedImageSequence expects a (batch, len, C, H, W) tensor, got "
                f"ndim={self.tensor.ndim} shape={tuple(self.tensor.shape)}"
            )

    def __len__(self) -> int:
        return self.tensor.shape[0]

    @overload
    def __getitem__(self, index: int) -> ImageSequence: ...
    @overload
    def __getitem__(self, index: slice) -> BatchedImageSequence: ...
    def __getitem__(self, index: int | slice) -> ImageSequence | BatchedImageSequence:
        if isinstance(index, slice):
            return BatchedImageSequence(self.tensor[index])
        return ImageSequence(self.tensor[index])

    def __iter__(self) -> Iterator[ImageSequence]:
        return (ImageSequence(seq) for seq in self.tensor)

    @property
    @override
    def device(self) -> torch.device:
        return self.tensor.device

    @override
    def to(self, device: DeviceLike) -> BatchedImageSequence:
        return BatchedImageSequence(self.tensor.to(device))

    @classmethod
    def from_sequences(cls, sequences: Iterable[ImageSequence]) -> BatchedImageSequence:
        """複数の ImageSequence を先頭次元に積んだ BatchedImageSequence を構築する。

        各 ImageSequence の (len, C, H, W) が一致している必要がある (呼び出し側責務)。 空入力は
        ValueError。shape 不一致は torch.stack の例外を伝播する。
        """
        materialized = list(sequences)
        if not materialized:
            raise ValueError("from_sequences requires at least one ImageSequence")
        stacked = torch.stack([sequence.tensor for sequence in materialized], dim=0)
        return cls(stacked)
