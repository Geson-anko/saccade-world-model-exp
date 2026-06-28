from typing import Self, override

import attrs
import torch

from .mixin import DeviceLike, DeviceTransferMixin

__all__ = ["Image"]


@attrs.define(slots=True, frozen=True, eq=False)
class Image(DeviceTransferMixin):
    """(C, H, W) の単一画像を内包する不変な値オブジェクト。

    frozen だが内部 tensor の in-place 変更 (add_ 等) は防げない。
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
    @override
    def device(self) -> torch.device:
        return self.tensor.device

    @override
    def to(self, device: DeviceLike) -> Self:
        """指定 device に転送した新しい Image を返す (in-place ではない)。"""
        return type(self)(self.tensor.to(device))
