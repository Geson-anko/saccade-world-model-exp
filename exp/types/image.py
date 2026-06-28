import enum
from pathlib import Path
from typing import Self, override

import attrs
import torch
import torchvision.transforms.v2.functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2.functional import InterpolationMode

from .mixin import DeviceLike, DeviceTransferMixin
from .size import Size2d, size_2d_to_tuple

# float メソッドが組み込み float をシャドウするため、型注釈用に別名を保持する。
_float = float

__all__ = ["ChannelFormat", "Image"]


class ChannelFormat(enum.IntEnum):
    """チャンネル構成。値はチャンネル数 (1=GRAY, 3=RGB, 4=RGBA)。"""

    GRAY = 1
    RGB = 3
    RGBA = 4


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
    def to(self, device: DeviceLike) -> Self:
        """指定 device に転送した新しい Image を返す (in-place ではない)。"""
        return type(self)(self.tensor.to(device))

    def float(self) -> Self:
        """浮動小数点 (float32) へキャストした新しい Image を返す。"""
        return type(self)(self.tensor.float())

    def uint8(self) -> Self:
        """符号なし 8bit 整数 (uint8) へキャストした新しい Image を返す。"""
        return type(self)(self.tensor.to(torch.uint8))

    def standardize(self, mean: _float = 0.0, std: _float = 1.0) -> Self:
        """画素統計を平均 mean・標準偏差 std に揃えた新しい Image を返す。

        画像全体で統計をとり、定数画像では全画素を mean にする (0 除算回避)。
        """
        t = self.tensor.float()
        centered = t - t.mean()
        spread = t.std()
        if float(spread) == 0.0:
            return type(self)(centered + mean)
        return type(self)(centered / spread * std + mean)

    def normalize(self, min: _float = 0.0, max: _float = 1.0) -> Self:
        """画素の min/max を [min, max] へ線形伸張した新しい Image を返す。

        画像全体で min/max をとり、レンジゼロでは全画素を min にする。
        """
        t = self.tensor.float()
        lo = t.min()
        hi = t.max()
        if float(hi) == float(lo):
            return type(self)(torch.full_like(t, min))
        return type(self)((t - lo) / (hi - lo) * (max - min) + min)

    def as_channel_format(self, fmt: ChannelFormat) -> Self:
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
        return type(self)(out)

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

    def square_pad(self, fill_value: _float = 0) -> Self:
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
        return type(self)(F.pad(self.tensor, padding, fill=fill_value))

    def focus(self, point: tuple[_float, _float], zoom: _float) -> Self:
        """注視点 point・拡大率 zoom で正方領域を切り取った新しい Image を返す。

        point は [-1, 1] (中心 0, y=+1 が下)、zoom は (0, 1]。はみ出しは 0 埋め。
        """
        x, y = point
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0):
            raise ValueError(f"focus point must be within [-1, 1], got {point}")
        if not (0.0 < zoom <= 1.0):
            raise ValueError(f"focus zoom must be within (0, 1], got {zoom}")
        squared = self.square_pad()  # 既に正方なら square_pad が自身を返す
        s = squared.height
        cx = (s / 2) * (1 + x)
        cy = (s / 2) * (1 + y)
        crop = max(1, int(round(zoom * s)))
        left = int(round(cx - crop / 2))
        top = int(round(cy - crop / 2))
        return type(self)(F.crop(squared.tensor, top, left, crop, crop))

    def resize(self, size: Size2d) -> Self:
        """指定サイズへリサイズした新しい Image を返す (int は正方、dtype 保存)。"""
        h, w = size_2d_to_tuple(size)
        out = F.resize(
            self.tensor,
            [h, w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return type(self)(out)

    @classmethod
    def load(cls, path: Path) -> Self:
        """画像ファイルを読み込み Image を返す (uint8・元チャンネルのまま)。"""
        tensor = read_image(str(path), mode=ImageReadMode.UNCHANGED)
        return cls(tensor)

    def save(self, path: Path) -> None:
        """[0, 255] へ正規化し uint8 で画像ファイルに保存する。

        GRAY/RGB/RGBA に対応。拡張子でフォーマット判定 (JPEG は alpha 非対応)。
        """
        # to_pil_image 経由で保存する (torchvision.io.write_png は 1/3ch のみで
        # RGBA を書けないため)。PIL が拡張子からフォーマットを判定する。
        tensor = self.normalize(0, 255).uint8().tensor.cpu()
        F.to_pil_image(tensor).save(path)
