from __future__ import annotations

from typing import final

import attrs
import torch

__all__ = ["ScalarTensor"]


@final
@attrs.define(slots=True, frozen=True, eq=False)
class ScalarTensor:
    """0 次元 (スカラー) tensor を内包する不変な値オブジェクト。損失値の戻り型。

    ``numel() == 1`` であれば ``()`` / ``(1,)`` / ``(1, 1)`` などを受理し、内部では
    shape ``()`` へ正規化して保持する。``reshape(())`` は view を返すため autograd の
    勾配経路を保つ。記録・表示用のスカラー取り出しは ``item()`` で行う。
    """

    tensor: torch.Tensor  # 内部は shape () で保持

    def __attrs_post_init__(self) -> None:
        if self.tensor.numel() != 1:
            raise ValueError(
                f"ScalarTensor expects a single-element tensor, got "
                f"numel={self.tensor.numel()} shape={tuple(self.tensor.shape)}"
            )
        object.__setattr__(self, "tensor", self.tensor.reshape(()))

    def item(self) -> float:
        return float(self.tensor)
