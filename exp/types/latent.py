from __future__ import annotations

from typing import final

import attrs
import torch

__all__ = ["BatchedLatentSequence"]


@final
@attrs.define(slots=True, frozen=True, eq=False)
class BatchedLatentSequence:
    """(batch, seq, dim) の潜在系列バッチを内包する不変な値オブジェクト。

    画像系列の値オブジェクトと異なり、indexing / iteration / device 転送の面は 持たない
    (単体潜在型が未定のため、需要が出るまで投機実装しない)。下流は ``tensor`` で (batch, seq, dim)
    を直接受け取る。
    """

    tensor: torch.Tensor  # (batch, seq, dim)

    def __attrs_post_init__(self) -> None:
        if self.tensor.ndim != 3:
            raise ValueError(
                f"BatchedLatentSequence expects a (batch, seq, dim) tensor, got "
                f"ndim={self.tensor.ndim} shape={tuple(self.tensor.shape)}"
            )
