"""Tensor を内包する不変値オブジェクトの基底群。

`Element` / `ElementArray`（+ `ElementSequence` / `BatchedElement` エイリアス）/
`BatchedElementSequence` を提供する。具象値オブジェクト（Image / Focus / Latent と
その系列・バッチ）はここへ集約した共通挙動（shape 検証・device 転送・collection
プロトコル・stack ファクトリ）を継承する。

軸の設計:
  - ``Element``           : 末端 tensor を保持する leaf。
  - ``ElementArray[T]``   : tensor を先頭 1 軸で積んだ収集型。時系列を表す
    ``ElementSequence`` とバッチを表す ``BatchedElement`` は構造が同一なので本クラス
    のエイリアス（意味は命名で区別する）。
  - ``BatchedElementSequence[TBatch, TSeq, TElem]`` : ``(batch, seq, ...)`` の 2 軸。
    ``iter_batch`` は batch 軸(dim=0)を反復して系列 ``TSeq`` を、``iter_sequence`` は
    seq 軸(dim=1)を反復してバッチ ``TBatch`` を返す。

これらの基底は公開面（``exp.types`` / ``exp`` の ``__all__``）には載せない内部土台だが、
実装ロジックを持つため直接テストする（`tests/types/elements/test_base.py`）。
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar, Self, overload, override

import attrs
import torch

from ..device import DeviceLike, DeviceTransferMixin

__all__ = [
    "BatchedElement",
    "BatchedElementSequence",
    "Element",
    "ElementArray",
    "ElementSequence",
]


@attrs.define(slots=True, frozen=True, eq=False)
class Element(DeviceTransferMixin, abc.ABC):
    """Shape が規約化された tensor を 1 つ内包する不変な値オブジェクトの基底。

    サブクラスは ClassVar ``_NDIM``（期待次元数）と ``_SHAPE_DESC``（エラーメッセージ用の 人間可読な
    shape 表記）を必ず定義する。軸ごとにサイズを固定したい場合は ``_SHAPE`` に 各軸の期待サイズ（ワイルドカードは
    None）を与える（不変条件: ``len(_SHAPE) == _NDIM``）。 frozen だが内部 tensor の in-
    place 変更は防げない。
    """

    tensor: torch.Tensor

    _NDIM: ClassVar[int]
    _SHAPE_DESC: ClassVar[str]
    _SHAPE: ClassVar[list[int | None] | None] = None

    def __attrs_post_init__(self) -> None:
        t = self.tensor
        ok = t.ndim == self._NDIM
        if ok and self._SHAPE is not None:
            ok = all(
                expected is None or t.shape[axis] == expected
                for axis, expected in enumerate(self._SHAPE)
            )
        if not ok:
            raise ValueError(
                f"{type(self).__name__} expects a {self._SHAPE_DESC} tensor, got "
                f"ndim={t.ndim} shape={tuple(t.shape)}"
            )

    @property
    @override
    def device(self) -> torch.device:
        return self.tensor.device

    @override
    def to(self, device: DeviceLike) -> Self:
        """指定 device へ転送した新しいインスタンスを返す (in-place ではない)。"""
        return type(self)(self.tensor.to(device))


@attrs.define(slots=True, frozen=True, eq=False)
class ElementArray[T: Element](Element, abc.ABC):
    """要素 tensor を先頭 1 軸で積んだ収集型の基底。

    ``ElementSequence``（時系列）と ``BatchedElement``（バッチ）はこのクラスのエイリアス。 先頭軸で
    indexing / iteration し、要素型 ``T`` を復元する。要素 0（空）も許容する。
    """

    @classmethod
    @abc.abstractmethod
    def item_type(cls) -> type[T]:
        """先頭軸を 1 つ取り出したときに復元する要素型を返す。"""

    def __len__(self) -> int:
        return self.tensor.shape[0]

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    def __getitem__(self, index: int | slice) -> T | Self:
        if isinstance(index, slice):
            return type(self)(self.tensor[index])
        return self.item_type()(self.tensor[index])

    def __iter__(self) -> Iterator[T]:
        item_type = self.item_type()
        return (item_type(row) for row in self.tensor)

    @classmethod
    def from_elements(cls, elements: Iterable[T]) -> Self:
        """要素列を先頭軸に stack して構築する。

        空入力は ValueError。shape 不一致は torch.stack の例外を伝播する。
        """
        materialized = list(elements)
        if not materialized:
            raise ValueError(f"{cls.__name__} requires at least one element")
        return cls(torch.stack([e.tensor for e in materialized], dim=0))


# 構造は共通・意味は命名で区別する。エイリアスは「単純代入」で定義する
# (PEP695 の `type` 文エイリアスは基底クラスとして subscript できないため)。
ElementSequence = ElementArray
BatchedElement = ElementArray


@attrs.define(slots=True, frozen=True, eq=False)
class BatchedElementSequence[
    TBatch: BatchedElement[Any],
    TSeq: ElementSequence[Any],
    TElem: Element,
](ElementArray[TSeq], abc.ABC):
    """``(batch, seq, ...)`` の 2 軸収集型の基底。

    先頭軸(batch)を反復する ``iter_batch``（= ``__iter__``, → ``TSeq``）と、第 2
    軸(seq) を反復する ``iter_sequence``（→ ``TBatch``）の 2
    モードを持つ。``__getitem__`` / ``__len__`` は batch
    軸を対象とする（``ElementArray[TSeq]`` から継承）。

    ``__getitem__`` は単一の batch 軸に加えて 2 要素タプル索引も受ける （``[i,j]→TElem`` /
    ``[i,:]→TSeq`` / ``[:,j]→TBatch`` / ``[:,:]→Self``）。 各軸を int で潰し
    slice で残すことで復元型が決まる。
    """

    @classmethod
    @abc.abstractmethod
    def _batch_type(cls) -> type[TBatch]:
        """Seq 軸を 1 つ取り出したときに復元するバッチ要素型を返す。"""

    @overload
    def __getitem__(self, index: int) -> TSeq: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> TElem: ...
    @overload
    def __getitem__(self, index: tuple[int, slice]) -> TSeq: ...
    @overload
    def __getitem__(self, index: tuple[slice, int]) -> TBatch: ...
    @overload
    def __getitem__(self, index: tuple[slice, slice]) -> Self: ...
    def __getitem__(
        self, index: int | slice | tuple[int | slice, int | slice]
    ) -> TElem | TSeq | TBatch | Self:
        if not isinstance(index, tuple):
            return super().__getitem__(index)
        batch_index, seq_index = index
        selected = self.tensor[batch_index, seq_index]
        match batch_index, seq_index:
            case slice(), slice():  # -> Self
                return type(self)(selected)
            case slice(), _:  # -> TBatch (batch を slice, seq を固定)
                return self._batch_type()(selected)
            case _, slice():  # -> TSeq (batch を固定, seq を slice)
                return self.item_type()(selected)
            case _, _:  # -> TElem (leaf)
                return self.item_type().item_type()(selected)

    def iter_batch(self) -> Iterator[TSeq]:
        """Batch 軸(dim=0)を反復し系列 ``TSeq`` を yield する（``__iter__`` と同義）。"""
        return iter(self)

    def iter_sequence(self) -> Iterator[TBatch]:
        """Seq 軸(dim=1)を反復しバッチ ``TBatch`` を yield する。"""
        batch_type = self._batch_type()
        return (batch_type(col) for col in self.tensor.unbind(dim=1))

    @classmethod
    def from_sequences(cls, sequences: Iterable[TSeq]) -> Self:
        """系列列を batch 軸(dim=0)に stack して構築する (空入力は ValueError)。"""
        materialized = list(sequences)
        if not materialized:
            raise ValueError(f"{cls.__name__} requires at least one sequence")
        return cls(torch.stack([s.tensor for s in materialized], dim=0))

    @classmethod
    def from_batches(cls, batches: Iterable[TBatch]) -> Self:
        """バッチ列を seq 軸(dim=1)に stack して構築する (空入力は ValueError)。"""
        materialized = list(batches)
        if not materialized:
            raise ValueError(f"{cls.__name__} requires at least one batch")
        return cls(torch.stack([b.tensor for b in materialized], dim=1))
