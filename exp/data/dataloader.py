from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, override

from torch.utils.data import DataLoader, Dataset

from exp.types import (
    BatchedFocusSequence,
    BatchedImageSequence,
    FocusSequence,
    ImageSequence,
)

__all__ = ["GlimpseDataLoader", "collate_glimpses"]


def collate_glimpses(
    batch: Sequence[tuple[FocusSequence, ImageSequence]],
) -> tuple[BatchedFocusSequence, BatchedImageSequence]:
    """GlimpseDataset のサンプル列をバッチへまとめる collate 関数。

    DataLoader の collate_fn= に渡せる。空リストは from_sequences が ValueError
    を出すため追加検証はしない。
    """
    focuses = BatchedFocusSequence.from_sequences(f for f, _ in batch)
    observations = BatchedImageSequence.from_sequences(o for _, o in batch)
    return focuses, observations


class GlimpseDataLoader(DataLoader[tuple[FocusSequence, ImageSequence]]):
    """collate_glimpses を常に適用するグリンプス専用 DataLoader。

    collate_fn は collate_glimpses に固定する (kwargs で渡すと TypeError)。 素の
    DataLoader は __iter__ の要素型を持たないため、バッチ型 (BatchedFocusSequence,
    BatchedImageSequence) をここで型付けする。
    """

    def __init__(
        self,
        dataset: Dataset[tuple[FocusSequence, ImageSequence]],
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset, collate_fn=collate_glimpses, **kwargs)

    @override
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[tuple[BatchedFocusSequence, BatchedImageSequence]]:
        return super().__iter__()
