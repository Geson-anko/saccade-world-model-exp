"""Behaviour spec for ``exp.types.elements.base``.

These tests translate the approved spec for the tensor value-object base
classes into executable form. The bases (``Element`` / ``ElementArray``
with its ``ElementSequence`` / ``BatchedElement`` aliases, and
``BatchedElementSequence``) are abstract IF *this project owns*, so per
the project testing strategy they are exercised through purpose-built
concrete fake subclasses (defined here) rather than through any single
concrete type. This isolates the shared logic (shape validation, device
transfer, the collection protocol, and the stack factories) from Image /
Focus / Latent specifics.

The fakes are the "self-owned ABC fakes" the strategy permits; torch
itself is never mocked. Tensors are built deterministically on real CPU
torch.

Only public behaviour is asserted. The type hooks ``_item_type`` and
``_batch_type`` (both private) are never called directly: their contract
is observed through ``__getitem__`` / ``__iter__`` / ``iter_sequence`` /
the ``from_*`` factories.
"""

from typing import ClassVar, override

import attrs
import pytest
import torch

from exp.types.elements.base import (
    BatchedElement,
    BatchedElementSequence,
    Element,
    ElementArray,
    ElementSequence,
)

# --- Fake concrete subclasses ------------------------------------------------
# One fake per base role, so the shared base logic can be driven directly.
# All are frozen identity-equality value objects like the real element types.


@attrs.define(slots=True, frozen=True, eq=False)
class _Leaf(Element):
    # A rank-1 leaf with no per-axis size constraint (any length is valid).
    _NDIM: ClassVar[int] = 1
    _SHAPE_DESC: ClassVar[str] = "(n,)"


@attrs.define(slots=True, frozen=True, eq=False)
class _FixedLeaf(Element):
    # A rank-2 leaf whose last axis is pinned to 3 via _SHAPE ([wildcard, 3]);
    # used to exercise per-axis size validation, not just ndim.
    _NDIM: ClassVar[int] = 2
    _SHAPE_DESC: ClassVar[str] = "(k, 3)"
    _SHAPE: ClassVar[list[int | None] | None] = [None, 3]


@attrs.define(slots=True, frozen=True, eq=False)
class _Arr(ElementArray[_Leaf]):
    # A rank-2 collection over _Leaf rows: leading axis is the collection axis.
    _NDIM: ClassVar[int] = 2
    _SHAPE_DESC: ClassVar[str] = "(m, n)"

    @classmethod
    @override
    def _item_type(cls) -> type[_Leaf]:
        return _Leaf


@attrs.define(slots=True, frozen=True, eq=False)
class _BatchLeaf(BatchedElement[_Leaf]):
    # Structurally identical to _Arr (BatchedElement is an ElementArray alias);
    # named as a "batch of leaves" to fill the TBatch role of _BatchSeq.
    _NDIM: ClassVar[int] = 2
    _SHAPE_DESC: ClassVar[str] = "(b, n)"

    @classmethod
    @override
    def _item_type(cls) -> type[_Leaf]:
        return _Leaf


@attrs.define(slots=True, frozen=True, eq=False)
class _BatchSeq(BatchedElementSequence[_BatchLeaf, _Arr, _Leaf]):
    # A rank-3 (batch, seq, n) collection: batch axis yields _Arr sequences,
    # seq axis yields _BatchLeaf batches.
    _NDIM: ClassVar[int] = 3
    _SHAPE_DESC: ClassVar[str] = "(b, m, n)"

    @classmethod
    @override
    def _item_type(cls) -> type[_Arr]:
        return _Arr

    @classmethod
    @override
    def _batch_type(cls) -> type[_BatchLeaf]:
        return _BatchLeaf


# A per-index value-distinct (m, n) block, so per-row torch.equal checks pin
# both element identity and ordering rather than passing on all-zero coincidence.
def _distinct_arr(index: int) -> _Arr:
    stride = 3 * 4
    block = torch.arange(
        index * stride, (index + 1) * stride, dtype=torch.float32
    ).reshape(3, 4)
    return _Arr(block)


def _distinct_leaf(index: int) -> _Leaf:
    return _Leaf(torch.arange(index * 4, (index + 1) * 4, dtype=torch.float32))


# --- Element: shape validation -----------------------------------------------


class TestElementConstruction:
    def test_accepts_tensor_of_expected_ndim(self):
        # _Leaf requires ndim 1 with no per-axis constraint; any length passes.
        leaf = _Leaf(torch.zeros(4))

        assert leaf.tensor.shape == (4,)

    def test_rejects_tensor_of_wrong_ndim(self):
        # A rank-2 tensor violates _Leaf's ndim=1 contract.
        with pytest.raises(ValueError, match=r"\(n,\)"):
            _Leaf(torch.zeros(2, 4))

    def test_error_reports_shape_desc_and_actual_ndim_and_shape(self):
        # The message must surface SHAPE_DESC and the offending ndim/shape
        # (substring checks only; exact wording is not part of the spec).
        with pytest.raises(ValueError) as exc:
            _Leaf(torch.zeros(2, 5))

        message = str(exc.value)
        assert "(n,)" in message  # SHAPE_DESC
        assert "ndim=2" in message
        assert "(2, 5)" in message  # actual shape

    def test_wildcard_axis_accepts_any_size(self):
        # _FixedLeaf pins only the last axis (_SHAPE=[None, 3]); the wildcard
        # leading axis accepts any size.
        assert _FixedLeaf(torch.zeros(5, 3)).tensor.shape == (5, 3)
        assert _FixedLeaf(torch.zeros(1, 3)).tensor.shape == (1, 3)

    def test_fixed_axis_accepts_matching_size(self):
        # last axis == 3 satisfies the [None, 3] per-axis constraint.
        fixed = _FixedLeaf(torch.zeros(7, 3))

        assert fixed.tensor.shape == (7, 3)

    def test_fixed_axis_rejects_wrong_size(self):
        # Correct ndim but last axis 2 != 3 must be rejected by _SHAPE checking.
        with pytest.raises(ValueError, match=r"\(k, 3\)"):
            _FixedLeaf(torch.zeros(5, 2))

    def test_fixed_axis_rejects_wrong_ndim(self):
        # A rank-1 tensor fails the ndim gate before per-axis checks apply.
        with pytest.raises(ValueError, match=r"\(k, 3\)"):
            _FixedLeaf(torch.zeros(3))


# --- Element: device transfer ------------------------------------------------


class TestElementDeviceTransfer:
    def test_device_reflects_underlying_tensor(self):
        # `device` is a pass-through view of the wrapped tensor's device.
        leaf = _Leaf(torch.zeros(4))

        assert leaf.device == torch.device("cpu")

    def test_to_returns_new_instance_of_same_type(self):
        # `to` is not in-place: it returns a fresh, equal-valued instance of the
        # same concrete type (verified across CPU->CPU, the always-available hop).
        leaf = _Leaf(torch.arange(4, dtype=torch.float32))

        result = leaf.to("cpu")

        assert type(result) is _Leaf
        assert result is not leaf
        assert torch.equal(result.tensor, leaf.tensor)

    def test_to_accepts_torch_device(self):
        # `to` accepts a torch.device object, not only a string alias.
        leaf = _Leaf(torch.arange(4, dtype=torch.float32))

        result = leaf.to(torch.device("cpu"))

        assert type(result) is _Leaf
        assert torch.equal(result.tensor, leaf.tensor)


# --- ElementArray: collection protocol ---------------------------------------


class TestElementArrayCollection:
    def test_len_is_leading_axis_size(self):
        # len() reports the collection (leading) axis, not the element width.
        arr = _Arr(torch.zeros(3, 4))

        assert len(arr) == 3

    def test_len_of_empty_is_zero(self):
        # An empty (0, n) collection is a valid value with length 0.
        arr = _Arr(torch.zeros(0, 4))

        assert len(arr) == 0

    def test_getitem_int_returns_element(self):
        # Integer indexing drops the leading axis and restores the element type.
        arr = _Arr.from_elements([_distinct_leaf(0), _distinct_leaf(1)])

        item = arr[1]

        assert type(item) is _Leaf
        assert torch.equal(item.tensor, arr.tensor[1])

    def test_getitem_slice_returns_same_collection_type(self):
        # Slicing keeps the collection type and selects along the leading axis.
        arr = _Arr.from_elements(
            [_distinct_leaf(0), _distinct_leaf(1), _distinct_leaf(2)]
        )

        sliced = arr[1:]

        assert type(sliced) is _Arr
        assert sliced.tensor.shape == (2, 4)
        assert torch.equal(sliced.tensor, arr.tensor[1:])

    def test_iter_yields_elements_in_order(self):
        # Iteration yields one element per leading-axis row, in order.
        leaves = [_distinct_leaf(0), _distinct_leaf(1), _distinct_leaf(2)]
        arr = _Arr.from_elements(leaves)

        items = list(arr)

        assert len(items) == 3
        assert all(type(item) is _Leaf for item in items)
        for i, item in enumerate(items):
            assert torch.equal(item.tensor, arr.tensor[i])

    def test_iter_empty_yields_nothing(self):
        assert list(_Arr(torch.zeros(0, 4))) == []


# --- ElementArray: from_elements factory -------------------------------------


class TestElementArrayFromElements:
    def test_stacks_multiple_elements_on_leading_axis(self):
        # Value-distinct leaves stack into a (m, n) collection; per-row
        # torch.equal pins both element identity and ordering.
        leaves = [_distinct_leaf(0), _distinct_leaf(1), _distinct_leaf(2)]

        result = _Arr.from_elements(leaves)

        assert type(result) is _Arr
        assert result.tensor.shape == (3, 4)
        assert torch.equal(result.tensor[0], leaves[0].tensor)
        assert torch.equal(result.tensor[1], leaves[1].tensor)
        assert torch.equal(result.tensor[2], leaves[2].tensor)

    def test_single_element_yields_leading_axis_one(self):
        # The minimum non-empty input is one element, yielding leading axis 1.
        leaf = _distinct_leaf(2)

        result = _Arr.from_elements([leaf])

        assert type(result) is _Arr
        assert result.tensor.shape == (1, 4)
        assert torch.equal(result.tensor[0], leaf.tensor)

    def test_accepts_generator(self):
        # A one-shot generator must work: the input is materialized before the
        # empty check and the stack both scan it.
        result = _Arr.from_elements(_distinct_leaf(i) for i in range(2))

        assert type(result) is _Arr
        assert result.tensor.shape == (2, 4)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one element" contract.
        with pytest.raises(ValueError, match="at least one element"):
            _Arr.from_elements([])


# --- BatchedElementSequence: dual-axis collection protocol -------------------


class TestBatchedElementSequenceIteration:
    def test_iter_batch_yields_sequences_over_batch_axis(self):
        # iter_batch reduces the batch axis (dim=0), yielding one sequence (_Arr)
        # per batch entry, in order.
        sequences = [_distinct_arr(0), _distinct_arr(1)]
        batched = _BatchSeq.from_sequences(sequences)

        items = list(batched.iter_batch())

        assert len(items) == 2
        assert all(type(item) is _Arr for item in items)
        for i, item in enumerate(items):
            assert torch.equal(item.tensor, batched.tensor[i])

    def test_iter_batch_matches_default_iter(self):
        # iter_batch is documented as identical to __iter__.
        batched = _BatchSeq.from_sequences([_distinct_arr(0), _distinct_arr(1)])

        via_iter = [item.tensor for item in batched]
        via_iter_batch = [item.tensor for item in batched.iter_batch()]

        assert len(via_iter) == len(via_iter_batch)
        for a, b in zip(via_iter, via_iter_batch, strict=True):
            assert torch.equal(a, b)

    def test_iter_sequence_yields_batches_over_seq_axis(self):
        # iter_sequence reduces the seq axis (dim=1), yielding one batch
        # (_BatchLeaf) per seq position; each has shape (batch, n).
        batched = _BatchSeq(
            torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        )

        items = list(batched.iter_sequence())

        assert len(items) == 3  # one per seq position
        assert all(type(item) is _BatchLeaf for item in items)
        for j, item in enumerate(items):
            assert item.tensor.shape == (2, 4)
            assert torch.equal(item.tensor, batched.tensor[:, j])

    def test_getitem_int_indexes_batch_axis(self):
        # __getitem__/__len__ target the batch axis (inherited from ElementArray),
        # so an int index restores the sequence type _Arr.
        batched = _BatchSeq.from_sequences([_distinct_arr(0), _distinct_arr(1)])

        item = batched[1]

        assert type(item) is _Arr
        assert torch.equal(item.tensor, batched.tensor[1])

    def test_len_is_batch_axis_size(self):
        batched = _BatchSeq(torch.zeros(4, 3, 2))

        assert len(batched) == 4


# --- BatchedElementSequence: 2-tuple (batch, seq) indexing -------------------
# A single value-distinct (batch, seq, n) tensor drives every case below, so
# torch.equal pins BOTH the selected element's identity AND its ordering along
# both the batch and seq axes (an all-zero block could not distinguish e.g.
# t[:, j] from t[j, :]). The 2-tuple contract: each axis is reduced by an int
# and kept by a slice, so the restored type is chosen per (batch-axis kind,
# seq-axis kind): (int,int)->leaf, (int,slice)->sequence, (slice,int)->batch,
# (slice,slice)->batched-sequence.


class TestBatchedElementSequenceGetitem2D:
    def test_int_int_returns_leaf_at_batch_and_seq_position(self):
        # [i, j] reduces both axes to the single leaf at (batch i, seq j).
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[1, 2]

        assert type(x) is _Leaf
        assert torch.equal(x.tensor, t[1, 2])

    def test_int_full_slice_returns_sequence_for_fixed_batch(self):
        # [i, :] fixes the batch and keeps the whole seq axis -> the _Arr sequence.
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[1, :]

        assert type(x) is _Arr
        assert torch.equal(x.tensor, t[1, :])

    def test_int_partial_slice_returns_sequence_subrange(self):
        # [0, 1:3] fixes the batch and sub-slices the seq axis -> _Arr of shape (2, n).
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[0, 1:3]

        assert type(x) is _Arr
        assert x.tensor.shape == (2, 4)
        assert torch.equal(x.tensor, t[0, 1:3])

    def test_full_slice_int_returns_batch_for_fixed_seq_position(self):
        # [:, j] keeps the whole batch and fixes the seq position -> the _BatchLeaf.
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[:, 2]

        assert type(x) is _BatchLeaf
        assert torch.equal(x.tensor, t[:, 2])

    def test_partial_slice_int_returns_batch_subrange(self):
        # [1:, 0] sub-slices the batch and fixes the seq position -> _BatchLeaf (2, n).
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[1:, 0]

        assert type(x) is _BatchLeaf
        assert x.tensor.shape == (2, 4)
        assert torch.equal(x.tensor, t[1:, 0])

    def test_full_slice_full_slice_returns_whole_batched_sequence(self):
        # [:, :] keeps both axes -> a _BatchSeq equal to the entire tensor.
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[:, :]

        assert type(x) is _BatchSeq
        assert torch.equal(x.tensor, t[:, :])

    def test_partial_slice_partial_slice_returns_batched_sequence_subblock(self):
        # [0:2, 1:3] sub-slices both axes -> a _BatchSeq subblock of shape (2, 2, n).
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[0:2, 1:3]

        assert type(x) is _BatchSeq
        assert x.tensor.shape == (2, 2, 4)
        assert torch.equal(x.tensor, t[0:2, 1:3])

    def test_negative_int_int_returns_last_leaf(self):
        # Negative ints address from the end on both axes -> the leaf at (-1, -1).
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[-1, -1]

        assert type(x) is _Leaf
        assert torch.equal(x.tensor, t[-1, -1])

    def test_negative_int_full_slice_returns_last_sequence(self):
        # A negative batch index with a full seq slice -> the _Arr for the last batch.
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[-1, :]

        assert type(x) is _Arr
        assert torch.equal(x.tensor, t[-1, :])

    def test_single_slice_still_targets_batch_axis(self):
        # Backward compat: a lone slice (no tuple) keeps targeting the batch axis
        # and preserves the _BatchSeq type.
        t = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        bes = _BatchSeq(t)

        x = bes[1:]

        assert type(x) is _BatchSeq
        assert torch.equal(x.tensor, t[1:])


# --- BatchedElementSequence: stack factories ---------------------------------


class TestBatchedElementSequenceFromSequences:
    def test_stacks_sequences_on_batch_axis(self):
        # from_sequences stacks along a NEW batch axis (dim=0): (m, n) sequences
        # become (batch, m, n); per-entry torch.equal pins identity and order.
        sequences = [_distinct_arr(0), _distinct_arr(1)]

        result = _BatchSeq.from_sequences(sequences)

        assert type(result) is _BatchSeq
        assert result.tensor.shape == (2, 3, 4)
        assert torch.equal(result.tensor[0], sequences[0].tensor)
        assert torch.equal(result.tensor[1], sequences[1].tensor)

    def test_single_sequence_yields_batch_one(self):
        sequence = _distinct_arr(3)

        result = _BatchSeq.from_sequences([sequence])

        assert type(result) is _BatchSeq
        assert result.tensor.shape == (1, 3, 4)
        assert torch.equal(result.tensor[0], sequence.tensor)

    def test_accepts_generator(self):
        result = _BatchSeq.from_sequences(_distinct_arr(i) for i in range(2))

        assert type(result) is _BatchSeq
        assert result.tensor.shape == (2, 3, 4)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one sequence" contract.
        with pytest.raises(ValueError, match="at least one sequence"):
            _BatchSeq.from_sequences([])


class TestBatchedElementSequenceFromBatches:
    def test_stacks_batches_on_seq_axis(self):
        # from_batches stacks along the seq axis (dim=1): (batch, n) batches
        # become (batch, seq, n). Three batches -> seq length 3, and column j of
        # the result equals batch j.
        batches = [
            _BatchLeaf(torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)),
            _BatchLeaf(torch.arange(2 * 4, 4 * 4, dtype=torch.float32).reshape(2, 4)),
            _BatchLeaf(torch.arange(4 * 4, 6 * 4, dtype=torch.float32).reshape(2, 4)),
        ]

        result = _BatchSeq.from_batches(batches)

        assert type(result) is _BatchSeq
        assert result.tensor.shape == (2, 3, 4)
        assert torch.equal(result.tensor[:, 0], batches[0].tensor)
        assert torch.equal(result.tensor[:, 1], batches[1].tensor)
        assert torch.equal(result.tensor[:, 2], batches[2].tensor)

    def test_single_batch_yields_seq_one(self):
        batch = _BatchLeaf(torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4))

        result = _BatchSeq.from_batches([batch])

        assert type(result) is _BatchSeq
        assert result.tensor.shape == (2, 1, 4)
        assert torch.equal(result.tensor[:, 0], batch.tensor)

    def test_accepts_generator(self):
        result = _BatchSeq.from_batches(_BatchLeaf(torch.zeros(2, 4)) for _ in range(3))

        assert type(result) is _BatchSeq
        assert result.tensor.shape == (2, 3, 4)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one batch" contract.
        with pytest.raises(ValueError, match="at least one batch"):
            _BatchSeq.from_batches([])


# --- Alias identity ----------------------------------------------------------


class TestAliasIdentity:
    def test_element_sequence_is_element_array(self):
        # ElementSequence and BatchedElement are documented as simple-assignment
        # aliases of ElementArray (same object): the sequence/batch distinction
        # is carried by naming only, not by separate classes.
        assert ElementSequence is ElementArray

    def test_batched_element_is_element_array(self):
        assert BatchedElement is ElementArray
