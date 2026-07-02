from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from exp.types import (
    FOCUS_DIM,
    BatchedFocusSequence,
    BatchedImageSequence,
    ChannelFormat,
    FocusSequence,
    Image,
    ImageSequence,
    Size2d,
)

__all__ = ["GlimpseDataset", "collate_glimpses", "random_focus_sequence"]

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def random_focus_sequence(
    seq_len: int, generator: torch.Generator | None = None
) -> FocusSequence:
    """長さ seq_len のランダム行動列を生成する。

    x, y は [-1, 1]、zoom は [0, 1] の一様分布からサンプルする。seq_len=0 は 空
    FocusSequence として合法 (追加検証はしない)。
    """
    raw = torch.rand(seq_len, FOCUS_DIM, generator=generator)
    scale = torch.tensor([2.0, 2.0, 1.0])
    offset = torch.tensor([-1.0, -1.0, 0.0])
    return FocusSequence((raw * scale + offset).float())


class GlimpseDataset(Dataset[tuple[FocusSequence, ImageSequence]]):
    """1 画像 = 1 エピソードのグリンプス系列を供給する Dataset。

    data_dir 以下から png・jpg・jpeg (大文字拡張子を含む) を再帰的に収集し、
    パスのソート順で並べる。__getitem__ は同じ index でも呼ぶたびに異なる
    ランダム行動列を生成する。再現性は generator の seed で確保する。ソース
    画像はリサイズせず、切り取り後の観測のみ image_size へ揃える。
    """

    def __init__(
        self,
        data_dir: Path,
        image_size: Size2d,
        seq_len: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self._paths = sorted(
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not self._paths:
            raise ValueError(f"no image files found under {data_dir}")
        self._image_size = image_size
        self._seq_len = seq_len
        self._generator = generator

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[FocusSequence, ImageSequence]:
        image = (
            Image.load(self._paths[index]).as_channel_format(ChannelFormat.RGB).float()
        )
        focuses = random_focus_sequence(self._seq_len, self._generator)
        return focuses, focuses.apply(image, self._image_size)


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
