"""Behaviour spec for ``exp.types.elements.latent``.

These tests translate the approved spec for the four immutable latent
value objects (``Latent`` / ``LatentSequence`` / ``BatchedLatent`` /
``BatchedLatentSequence``) into executable form. They are written
against the *spec*, not any implementation: an implementation that
diverges should make these red. Scenarios are grouped into one class per
behaviour area.

Tensors are constructed deterministically on real CPU torch (no mocking
of torch) per the project testing strategy. Latent tensors carry no
shape constraint beyond rank (dim sizes are free), so validation pins
ndim only. Device transfer (``to`` / ``device``) touches real CPU torch
and is classified integration-real.

The three axes use distinct sizes (BATCH=2, SEQ=3, DIM=4) so that a
swapped axis in indexing / iteration / stacking is observable rather
than masked by a symmetric shape.
"""

import pytest
import torch

from exp.types.elements.latent import (
    BatchedLatent,
    BatchedLatentSequence,
    Latent,
    LatentSequence,
)

BATCH = 2
SEQ = 3
DIM = 4


def _distinct_latent(index: int) -> Latent:
    # A (DIM,) vector whose values are unique to `index`, so order-
    # preservation is observable after stacking / indexing.
    return Latent(torch.arange(index * DIM, (index + 1) * DIM))


def _distinct_latent_sequence(index: int) -> LatentSequence:
    # A (SEQ, DIM) block unique to `index`.
    stride = SEQ * DIM
    return LatentSequence(
        torch.arange(index * stride, (index + 1) * stride).reshape(SEQ, DIM)
    )


def _distinct_batched_latent(index: int) -> BatchedLatent:
    # A (BATCH, DIM) block unique to `index`.
    stride = BATCH * DIM
    return BatchedLatent(
        torch.arange(index * stride, (index + 1) * stride).reshape(BATCH, DIM)
    )


# --------------------------------------------------------------------------- #
# Latent: (dim,) leaf value object
# --------------------------------------------------------------------------- #
class TestLatentConstruction:
    def test_accepts_1d_tensor(self):
        # (dim,) is the one valid rank; the tensor passes through unchanged.
        latent = Latent(torch.zeros(DIM))

        assert latent.tensor.shape == (DIM,)

    def test_rejects_0d_tensor(self):
        # A scalar tensor lacks the single dim axis required by (dim,).
        with pytest.raises(ValueError, match=r"\(dim,\)"):
            Latent(torch.tensor(1.0))

    def test_rejects_2d_tensor(self):
        # A 2D tensor carries an extra axis the leaf value object forbids.
        with pytest.raises(ValueError, match=r"\(dim,\)"):
            Latent(torch.zeros(SEQ, DIM))

    def test_error_reports_actual_ndim_and_shape(self):
        # The message must surface the offending ndim and shape (substring only).
        with pytest.raises(ValueError) as exc:
            Latent(torch.zeros(3, 5))

        assert "(dim,)" in str(exc.value)
        assert "2" in str(exc.value)  # ndim
        assert "3" in str(exc.value) and "5" in str(exc.value)  # shape


class TestLatentEquality:
    def test_identity_for_same_instance(self):
        # eq=False contract: an instance compares equal only to itself.
        latent = Latent(torch.zeros(DIM))
        assert latent == latent

    def test_distinct_instances_are_not_equal(self):
        # Equal-valued but distinct instances must NOT be equal, and the
        # comparison must not raise "boolean value of Tensor is ambiguous".
        assert Latent(torch.zeros(DIM)) != Latent(torch.zeros(DIM))


class TestLatentDeviceTransfer:
    # integration-real: real CPU torch device transfer.

    def test_device_reflects_tensor(self):
        assert Latent(torch.zeros(DIM)).device == torch.device("cpu")

    def test_to_returns_new_latent(self):
        # `to` is not in-place: it returns a fresh Latent with equal values.
        latent = Latent(torch.zeros(DIM))

        result = latent.to("cpu")

        assert type(result) is Latent
        assert result is not latent
        assert torch.equal(result.tensor, latent.tensor)

    def test_to_accepts_torch_device(self):
        # DeviceLike is `torch.device | str`; the torch.device form must work.
        latent = Latent(torch.zeros(DIM))

        result = latent.to(torch.device("cpu"))

        assert type(result) is Latent
        assert torch.equal(result.tensor, latent.tensor)


# --------------------------------------------------------------------------- #
# LatentSequence: (seq, dim) time-series collection
# --------------------------------------------------------------------------- #
class TestLatentSequenceConstruction:
    def test_accepts_2d_tensor(self):
        seq = LatentSequence(torch.zeros(SEQ, DIM))
        assert len(seq) == SEQ

    def test_accepts_empty_sequence(self):
        # An empty (0, dim) sequence is a valid value, not an error.
        seq = LatentSequence(torch.zeros(0, DIM))
        assert len(seq) == 0

    def test_rejects_1d_tensor(self):
        # A 1D tensor lacks the leading seq axis required by (seq, dim).
        with pytest.raises(ValueError, match=r"\(seq, dim\)"):
            LatentSequence(torch.zeros(DIM))

    def test_rejects_3d_tensor(self):
        # A 3D tensor carries a batch axis the sequence forbids.
        with pytest.raises(ValueError, match=r"\(seq, dim\)"):
            LatentSequence(torch.zeros(BATCH, SEQ, DIM))

    def test_error_reports_actual_ndim_and_shape(self):
        # The message must surface the offending ndim and shape (substring only).
        with pytest.raises(ValueError) as exc:
            LatentSequence(torch.zeros(DIM))

        assert "(seq, dim)" in str(exc.value)
        assert "1" in str(exc.value)  # ndim


class TestLatentSequenceLen:
    def test_len_is_leading_axis(self):
        assert len(LatentSequence(torch.zeros(SEQ, DIM))) == SEQ

    def test_len_of_empty_is_zero(self):
        assert len(LatentSequence(torch.zeros(0, DIM))) == 0


class TestLatentSequenceIndexing:
    def test_int_index_returns_latent(self):
        seq = LatentSequence(torch.arange(SEQ * DIM).reshape(SEQ, DIM))

        item = seq[1]

        assert type(item) is Latent
        assert torch.equal(item.tensor, seq.tensor[1])

    def test_negative_index_returns_last_latent(self):
        seq = LatentSequence(torch.arange(SEQ * DIM).reshape(SEQ, DIM))

        item = seq[-1]

        assert type(item) is Latent
        assert torch.equal(item.tensor, seq.tensor[SEQ - 1])

    def test_slice_returns_latent_sequence(self):
        seq = LatentSequence(torch.arange(SEQ * DIM).reshape(SEQ, DIM))

        sub = seq[1:3]

        assert type(sub) is LatentSequence
        assert len(sub) == 2
        assert torch.equal(sub.tensor[0], seq.tensor[1])
        assert torch.equal(sub.tensor[1], seq.tensor[2])

    def test_out_of_range_int_raises_index_error(self):
        seq = LatentSequence(torch.zeros(SEQ, DIM))
        with pytest.raises(IndexError):
            _ = seq[SEQ]


class TestLatentSequenceIter:
    def test_iterates_rows_as_latents(self):
        seq = LatentSequence(torch.arange(SEQ * DIM).reshape(SEQ, DIM))

        items = list(seq)

        assert len(items) == SEQ
        assert all(type(item) is Latent for item in items)
        for i, item in enumerate(items):
            assert torch.equal(item.tensor, seq.tensor[i])

    def test_iterates_empty_as_no_latents(self):
        assert list(LatentSequence(torch.zeros(0, DIM))) == []


class TestLatentSequenceDeviceTransfer:
    # integration-real: real CPU torch device transfer.

    def test_device_reflects_tensor(self):
        assert LatentSequence(torch.zeros(SEQ, DIM)).device == torch.device("cpu")

    def test_device_available_for_empty_sequence(self):
        assert LatentSequence(torch.zeros(0, DIM)).device == torch.device("cpu")

    def test_to_returns_new_sequence(self):
        # `to` is not in-place: it returns a fresh LatentSequence.
        seq = LatentSequence(torch.zeros(SEQ, DIM))

        result = seq.to("cpu")

        assert type(result) is LatentSequence
        assert result is not seq
        assert torch.equal(result.tensor, seq.tensor)

    def test_to_accepts_torch_device(self):
        seq = LatentSequence(torch.zeros(SEQ, DIM))

        result = seq.to(torch.device("cpu"))

        assert type(result) is LatentSequence
        assert torch.equal(result.tensor, seq.tensor)


class TestLatentSequenceFromElements:
    def test_stacks_multiple_latents(self):
        # Value-distinct Latents stack into a (seq, dim) sequence; the per-row
        # torch.equal checks pin both element identity and order.
        latents = [_distinct_latent(0), _distinct_latent(1), _distinct_latent(2)]

        result = LatentSequence.from_elements(latents)

        assert type(result) is LatentSequence
        assert len(result) == 3
        assert result.tensor.shape == (3, DIM)
        assert torch.equal(result.tensor[0], latents[0].tensor)
        assert torch.equal(result.tensor[1], latents[1].tensor)
        assert torch.equal(result.tensor[2], latents[2].tensor)

    def test_single_latent(self):
        # The minimum non-empty input is one Latent, yielding leading axis 1.
        latent = _distinct_latent(7)

        result = LatentSequence.from_elements([latent])

        assert type(result) is LatentSequence
        assert len(result) == 1
        assert result.tensor.shape == (1, DIM)
        assert torch.equal(result.tensor[0], latent.tensor)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one" contract (substring only).
        with pytest.raises(ValueError, match="at least one"):
            LatentSequence.from_elements([])

    def test_accepts_generator(self):
        # A one-shot generator (N>=1) must work: the empty check and the stack
        # both succeed, proving the input is materialized before being scanned.
        result = LatentSequence.from_elements(_distinct_latent(i) for i in range(2))

        assert type(result) is LatentSequence
        assert len(result) == 2
        assert result.tensor.shape == (2, DIM)


# --------------------------------------------------------------------------- #
# BatchedLatent: (batch, dim) batch collection
# --------------------------------------------------------------------------- #
class TestBatchedLatentConstruction:
    def test_accepts_2d_tensor(self):
        batch = BatchedLatent(torch.zeros(BATCH, DIM))
        assert len(batch) == BATCH

    def test_accepts_empty_batch(self):
        # An empty (0, dim) batch is a valid value, not an error.
        batch = BatchedLatent(torch.zeros(0, DIM))
        assert len(batch) == 0

    def test_rejects_1d_tensor(self):
        # A 1D tensor lacks the leading batch axis required by (batch, dim).
        with pytest.raises(ValueError, match=r"\(batch, dim\)"):
            BatchedLatent(torch.zeros(DIM))

    def test_rejects_3d_tensor(self):
        # A 3D tensor carries a seq axis the batch forbids.
        with pytest.raises(ValueError, match=r"\(batch, dim\)"):
            BatchedLatent(torch.zeros(BATCH, SEQ, DIM))

    def test_error_reports_actual_ndim_and_shape(self):
        with pytest.raises(ValueError) as exc:
            BatchedLatent(torch.zeros(DIM))

        assert "(batch, dim)" in str(exc.value)
        assert "1" in str(exc.value)  # ndim


class TestBatchedLatentLen:
    def test_len_is_leading_axis(self):
        assert len(BatchedLatent(torch.zeros(BATCH, DIM))) == BATCH

    def test_len_of_empty_is_zero(self):
        assert len(BatchedLatent(torch.zeros(0, DIM))) == 0


class TestBatchedLatentIndexing:
    def test_int_index_returns_latent(self):
        batch = BatchedLatent(torch.arange(BATCH * DIM).reshape(BATCH, DIM))

        item = batch[1]

        assert type(item) is Latent
        assert torch.equal(item.tensor, batch.tensor[1])

    def test_slice_returns_batched_latent(self):
        batch = BatchedLatent(torch.arange(4 * DIM).reshape(4, DIM))

        sub = batch[1:3]

        assert type(sub) is BatchedLatent
        assert len(sub) == 2
        assert torch.equal(sub.tensor[0], batch.tensor[1])
        assert torch.equal(sub.tensor[1], batch.tensor[2])

    def test_out_of_range_int_raises_index_error(self):
        batch = BatchedLatent(torch.zeros(BATCH, DIM))
        with pytest.raises(IndexError):
            _ = batch[BATCH]


class TestBatchedLatentIter:
    def test_iterates_entries_as_latents(self):
        batch = BatchedLatent(torch.arange(BATCH * DIM).reshape(BATCH, DIM))

        items = list(batch)

        assert len(items) == BATCH
        assert all(type(item) is Latent for item in items)
        for i, item in enumerate(items):
            assert torch.equal(item.tensor, batch.tensor[i])

    def test_iterates_empty_as_no_latents(self):
        assert list(BatchedLatent(torch.zeros(0, DIM))) == []


class TestBatchedLatentDeviceTransfer:
    # integration-real: real CPU torch device transfer.

    def test_device_reflects_tensor(self):
        assert BatchedLatent(torch.zeros(BATCH, DIM)).device == torch.device("cpu")

    def test_to_returns_new_batch(self):
        # `to` is not in-place: it returns a fresh BatchedLatent.
        batch = BatchedLatent(torch.zeros(BATCH, DIM))

        result = batch.to("cpu")

        assert type(result) is BatchedLatent
        assert result is not batch
        assert torch.equal(result.tensor, batch.tensor)

    def test_to_accepts_torch_device(self):
        batch = BatchedLatent(torch.zeros(BATCH, DIM))

        result = batch.to(torch.device("cpu"))

        assert type(result) is BatchedLatent
        assert torch.equal(result.tensor, batch.tensor)


class TestBatchedLatentFromElements:
    def test_stacks_multiple_latents(self):
        latents = [_distinct_latent(0), _distinct_latent(1)]

        result = BatchedLatent.from_elements(latents)

        assert type(result) is BatchedLatent
        assert len(result) == 2
        assert result.tensor.shape == (2, DIM)
        assert torch.equal(result.tensor[0], latents[0].tensor)
        assert torch.equal(result.tensor[1], latents[1].tensor)

    def test_single_latent(self):
        latent = _distinct_latent(3)

        result = BatchedLatent.from_elements([latent])

        assert type(result) is BatchedLatent
        assert len(result) == 1
        assert result.tensor.shape == (1, DIM)
        assert torch.equal(result.tensor[0], latent.tensor)

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            BatchedLatent.from_elements([])

    def test_accepts_generator(self):
        result = BatchedLatent.from_elements(_distinct_latent(i) for i in range(2))

        assert type(result) is BatchedLatent
        assert len(result) == 2
        assert result.tensor.shape == (2, DIM)


# --------------------------------------------------------------------------- #
# BatchedLatentSequence: (batch, seq, dim) two-axis collection
# --------------------------------------------------------------------------- #
class TestBatchedLatentSequenceConstruction:
    def test_accepts_3d_tensor(self):
        # (batch, seq, dim) is the one valid rank; the tensor passes through.
        latent = BatchedLatentSequence(torch.zeros(BATCH, SEQ, DIM))

        assert latent.tensor.shape == (BATCH, SEQ, DIM)

    def test_accepts_empty_batch(self):
        # An empty (0, seq, dim) batch is a valid value, not an error.
        latent = BatchedLatentSequence(torch.zeros(0, SEQ, DIM))
        assert len(latent) == 0

    def test_rejects_2d_tensor(self):
        # A 2D tensor lacks the batch axis required by (batch, seq, dim).
        with pytest.raises(ValueError, match=r"\(batch, seq, dim\)"):
            BatchedLatentSequence(torch.zeros(SEQ, DIM))

    def test_rejects_4d_tensor(self):
        # A 4D tensor carries an extra axis the value object forbids.
        with pytest.raises(ValueError, match=r"\(batch, seq, dim\)"):
            BatchedLatentSequence(torch.zeros(BATCH, SEQ, DIM, 4))

    def test_error_reports_actual_ndim_and_shape(self):
        # The message must surface the offending ndim and shape (substring only).
        with pytest.raises(ValueError) as exc:
            BatchedLatentSequence(torch.zeros(3, 16, 8, 4))

        assert "batch, seq, dim" in str(exc.value)
        assert "4" in str(exc.value)  # ndim
        assert "3" in str(exc.value) and "16" in str(exc.value)  # shape
        assert "8" in str(exc.value) and "4" in str(exc.value)  # shape


class TestBatchedLatentSequenceEquality:
    def test_identity_for_same_instance(self):
        # eq=False contract: an instance compares equal only to itself.
        latent = BatchedLatentSequence(torch.zeros(BATCH, SEQ, DIM))
        assert latent == latent

    def test_distinct_instances_are_not_equal(self):
        # Equal-valued but distinct instances must NOT be equal, and the
        # comparison must not raise "boolean value of Tensor is ambiguous".
        assert BatchedLatentSequence(
            torch.zeros(BATCH, SEQ, DIM)
        ) != BatchedLatentSequence(torch.zeros(BATCH, SEQ, DIM))


class TestBatchedLatentSequenceLen:
    def test_len_is_batch_axis(self):
        assert len(BatchedLatentSequence(torch.zeros(BATCH, SEQ, DIM))) == BATCH

    def test_len_of_empty_is_zero(self):
        assert len(BatchedLatentSequence(torch.zeros(0, SEQ, DIM))) == 0


class TestBatchedLatentSequenceIndexing:
    def test_int_index_returns_latent_sequence(self):
        # __getitem__ addresses the batch axis, restoring a LatentSequence.
        batch = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM)
        )

        seq = batch[1]

        assert type(seq) is LatentSequence
        assert torch.equal(seq.tensor, batch.tensor[1])

    def test_negative_index_returns_last_sequence(self):
        batch = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM)
        )

        seq = batch[-1]

        assert type(seq) is LatentSequence
        assert torch.equal(seq.tensor, batch.tensor[BATCH - 1])

    def test_slice_returns_batched_latent_sequence(self):
        batch = BatchedLatentSequence(torch.arange(4 * SEQ * DIM).reshape(4, SEQ, DIM))

        sub = batch[1:3]

        assert type(sub) is BatchedLatentSequence
        assert len(sub) == 2
        assert torch.equal(sub.tensor[0], batch.tensor[1])
        assert torch.equal(sub.tensor[1], batch.tensor[2])

    def test_out_of_range_int_raises_index_error(self):
        batch = BatchedLatentSequence(torch.zeros(BATCH, SEQ, DIM))
        with pytest.raises(IndexError):
            _ = batch[BATCH]


class TestBatchedLatentSequenceDeviceTransfer:
    # integration-real: real CPU torch device transfer.

    def test_device_reflects_tensor(self):
        assert BatchedLatentSequence(
            torch.zeros(BATCH, SEQ, DIM)
        ).device == torch.device("cpu")

    def test_device_available_for_empty_batch(self):
        assert BatchedLatentSequence(torch.zeros(0, SEQ, DIM)).device == torch.device(
            "cpu"
        )

    def test_to_returns_new_batch(self):
        # `to` is not in-place: it returns a fresh BatchedLatentSequence.
        batch = BatchedLatentSequence(torch.zeros(BATCH, SEQ, DIM))

        result = batch.to("cpu")

        assert type(result) is BatchedLatentSequence
        assert result is not batch
        assert torch.equal(result.tensor, batch.tensor)

    def test_to_accepts_torch_device(self):
        batch = BatchedLatentSequence(torch.zeros(BATCH, SEQ, DIM))

        result = batch.to(torch.device("cpu"))

        assert type(result) is BatchedLatentSequence
        assert torch.equal(result.tensor, batch.tensor)


class TestBatchedLatentSequenceIterBatch:
    def test_iter_batch_yields_latent_sequences(self):
        # iter_batch walks the batch axis (dim=0), restoring one
        # LatentSequence per entry, in order. It is the alias of __iter__.
        batch = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM)
        )

        items = list(batch.iter_batch())

        assert len(items) == BATCH
        assert all(type(item) is LatentSequence for item in items)
        for i, item in enumerate(items):
            assert torch.equal(item.tensor, batch.tensor[i])

    def test_iter_batch_matches_default_iteration(self):
        # iter_batch and __iter__ must yield the same entries.
        batch = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM)
        )

        via_iter_batch = list(batch.iter_batch())
        via_iter = list(batch)

        assert len(via_iter_batch) == len(via_iter)
        for a, b in zip(via_iter_batch, via_iter, strict=True):
            assert torch.equal(a.tensor, b.tensor)

    def test_iter_batch_empty_yields_nothing(self):
        assert list(BatchedLatentSequence(torch.zeros(0, SEQ, DIM)).iter_batch()) == []


class TestBatchedLatentSequenceIterSequence:
    def test_iter_sequence_yields_batched_latents(self):
        # iter_sequence walks the seq axis (dim=1), restoring one
        # BatchedLatent per step of shape (batch, dim), in order.
        batch = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM)
        )

        items = list(batch.iter_sequence())

        assert len(items) == SEQ
        assert all(type(item) is BatchedLatent for item in items)
        for step, item in enumerate(items):
            assert item.tensor.shape == (BATCH, DIM)
            assert torch.equal(item.tensor, batch.tensor[:, step])

    def test_iter_sequence_empty_seq_yields_nothing(self):
        # A (batch, 0, dim) tensor has no seq steps to walk.
        batch = BatchedLatentSequence(torch.zeros(BATCH, 0, DIM))
        assert list(batch.iter_sequence()) == []


class TestBatchedLatentSequenceFromSequences:
    def test_stacks_sequences_on_batch_axis(self):
        # from_sequences stacks LatentSequences on a NEW leading batch axis
        # (dim=0); per-entry torch.equal pins element identity and order.
        sequences = [_distinct_latent_sequence(0), _distinct_latent_sequence(1)]

        result = BatchedLatentSequence.from_sequences(sequences)

        assert type(result) is BatchedLatentSequence
        assert len(result) == 2
        assert result.tensor.shape == (2, SEQ, DIM)
        assert torch.equal(result.tensor[0], sequences[0].tensor)
        assert torch.equal(result.tensor[1], sequences[1].tensor)

    def test_single_sequence(self):
        sequence = _distinct_latent_sequence(3)

        result = BatchedLatentSequence.from_sequences([sequence])

        assert type(result) is BatchedLatentSequence
        assert len(result) == 1
        assert result.tensor.shape == (1, SEQ, DIM)
        assert torch.equal(result.tensor[0], sequence.tensor)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one sequence" contract.
        with pytest.raises(ValueError, match="at least one sequence"):
            BatchedLatentSequence.from_sequences([])

    def test_accepts_generator(self):
        result = BatchedLatentSequence.from_sequences(
            _distinct_latent_sequence(i) for i in range(2)
        )

        assert type(result) is BatchedLatentSequence
        assert len(result) == 2
        assert result.tensor.shape == (2, SEQ, DIM)


class TestBatchedLatentSequenceFromBatches:
    def test_stacks_batches_on_seq_axis(self):
        # from_batches stacks BatchedLatents on the seq axis (dim=1), so each
        # input BatchedLatent (batch, dim) becomes one seq step; the tensor[:, i]
        # checks pin that stacking lands on axis 1, not axis 0.
        batches = [_distinct_batched_latent(0), _distinct_batched_latent(1)]

        result = BatchedLatentSequence.from_batches(batches)

        assert type(result) is BatchedLatentSequence
        assert result.tensor.shape == (BATCH, 2, DIM)
        assert torch.equal(result.tensor[:, 0], batches[0].tensor)
        assert torch.equal(result.tensor[:, 1], batches[1].tensor)

    def test_single_batch(self):
        batch = _distinct_batched_latent(5)

        result = BatchedLatentSequence.from_batches([batch])

        assert type(result) is BatchedLatentSequence
        assert result.tensor.shape == (BATCH, 1, DIM)
        assert torch.equal(result.tensor[:, 0], batch.tensor)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one batch" contract.
        with pytest.raises(ValueError, match="at least one batch"):
            BatchedLatentSequence.from_batches([])

    def test_accepts_generator(self):
        result = BatchedLatentSequence.from_batches(
            _distinct_batched_latent(i) for i in range(2)
        )

        assert type(result) is BatchedLatentSequence
        assert result.tensor.shape == (BATCH, 2, DIM)

    def test_from_batches_round_trips_iter_sequence(self):
        # iter_sequence and from_batches are inverses along the seq axis:
        # splitting into per-step BatchedLatents then re-stacking reproduces
        # the original tensor.
        original = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM)
        )

        rebuilt = BatchedLatentSequence.from_batches(original.iter_sequence())

        assert torch.equal(rebuilt.tensor, original.tensor)


class TestBatchedLatentSequenceShift:
    def test_shifts_sequence_back_by_one_with_zero_prefix(self):
        # [e_1, …, e_T] -> [0, e_1, …, e_{T-1}]: each seq step moves to the
        # next index, index 0 becomes zeros, and the last input step e_T is
        # dropped. Checked per batch row so shifting is confirmed independent
        # across the batch axis.
        batch = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM)
        )

        result = batch.shift()

        assert result.tensor.shape == (BATCH, SEQ, DIM)
        for b in range(BATCH):
            assert torch.equal(result.tensor[b, 0], torch.zeros(DIM, dtype=torch.long))
            for step in range(1, SEQ):
                assert torch.equal(result.tensor[b, step], batch.tensor[b, step - 1])
            # The original last step is pushed out, not present at any index.
            assert not any(
                torch.equal(result.tensor[b, step], batch.tensor[b, SEQ - 1])
                for step in range(SEQ)
            )

    def test_does_not_mutate_original(self):
        # shift is non-destructive: it returns a fresh instance and leaves the
        # source tensor untouched (mirrors the `to` contract).
        original = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM)
        )
        before = original.tensor.clone()

        result = original.shift()

        assert type(result) is BatchedLatentSequence
        assert result is not original
        assert torch.equal(original.tensor, before)

    def test_preserves_dtype(self):
        batch = BatchedLatentSequence(
            torch.arange(BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM).float()
        )

        result = batch.shift()

        assert result.tensor.dtype == batch.tensor.dtype

    def test_seq_one_becomes_all_zeros(self):
        # With a single step the sole element is pushed out, leaving only the
        # zero prefix; shape is preserved.
        batch = BatchedLatentSequence(
            torch.arange(BATCH * 1 * DIM).reshape(BATCH, 1, DIM)
        )

        result = batch.shift()

        assert result.tensor.shape == (BATCH, 1, DIM)
        assert torch.equal(result.tensor, torch.zeros(BATCH, 1, DIM, dtype=torch.long))

    def test_empty_seq_returns_empty(self):
        # A (batch, 0, dim) sequence has nothing to shift; it must pass through
        # without raising and keep its shape.
        batch = BatchedLatentSequence(torch.zeros(BATCH, 0, DIM))

        result = batch.shift()

        assert result.tensor.shape == (BATCH, 0, DIM)

    def test_empty_batch_returns_empty(self):
        # A (0, seq, dim) batch has no rows; it must pass through without
        # raising and keep its shape.
        batch = BatchedLatentSequence(torch.zeros(0, SEQ, DIM))

        result = batch.shift()

        assert result.tensor.shape == (0, SEQ, DIM)
