"""Behaviour spec for ``exp.types.focus``.

These tests translate the approved spec for the immutable ``Focus``
value object (the action ``a = (p, z)``: gaze point + zoom) into
executable form. They are written against the *spec*, not any
implementation: an implementation that diverges should make these red.
Scenarios are grouped into one class per behaviour area.

The crop behaviour mirrors the former ``Image.focus`` exactly (the
move is a relocation, not a change), so ``TestCall`` reuses the same
four scenarios. Tensors are constructed deterministically on real CPU
torch (no mocking of torch) per the project testing strategy.
"""

import pytest
import torch

from exp.types.focus import BatchedFocusSequence, Focus, FocusSequence
from exp.types.image import Image, ImageSequence


class TestConstruction:
    def test_rejects_out_of_range_point(self):
        # Each point component must lie in [-1, 1]; 1.5 is outside.
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            Focus((1.5, 0.0), 1.0)

    def test_rejects_out_of_range_zoom(self):
        # zoom must lie in the closed interval [0, 1]; -0.1 is below it.
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Focus((0.0, 0.0), -0.1)

    def test_accepts_zero_zoom(self):
        # zoom is a CLOSED interval [0, 1]: 0.0 is a valid lower bound.
        focus = Focus((0.0, 0.0), 0.0)

        assert focus.zoom == 0.0


class TestTensor:
    def test_packs_point_and_zoom_as_float32(self):
        # tensor() yields [x, y, zoom] as a CPU float32 vector of shape (3,).
        out = Focus((0.3, -0.5), 0.25).tensor()

        assert out.shape == (3,)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, torch.tensor([0.3, -0.5, 0.25]))


class TestCall:
    def test_full_view_returns_whole_image(self):
        # zoom=1, point=center on a square image returns the image unchanged.
        img = Image(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4))

        out = Focus((0.0, 0.0), 1.0)(img)

        assert torch.equal(out.tensor, img.tensor)

    def test_output_side_is_round_zoom_times_side(self):
        out = Focus((0.0, 0.0), 0.5)(Image(torch.zeros(3, 8, 8)))

        assert out.is_squared
        assert out.size == (4, 4)

    def test_squares_non_square_input_first(self):
        # (4, 8) squares to side 8, so a full-zoom focus yields 8x8.
        out = Focus((0.0, 0.0), 1.0)(Image(torch.zeros(3, 4, 8)))

        assert out.size == (8, 8)

    def test_corner_pads_out_of_bounds(self):
        # Bottom-right gaze with partial zoom spills past the edge -> zero pad.
        out = Focus((1.0, 1.0), 0.5)(Image(torch.ones(1, 4, 4)))

        assert out.size == (2, 2)
        assert (out.tensor == 0).any()


# A focus row is [x, y, zoom]; these helpers build value-distinct rows so the
# per-row torch.equal checks below pin both element identity and ordering.
def _focus_row(index: int) -> Focus:
    # Stay inside [-1, 1] / [0, 1] so the built Focus is in-range; small unique
    # offsets per index keep every row distinguishable.
    return Focus((0.1 * index, -0.1 * index), 0.1 * index)


def _distinct_focus_sequence(index: int) -> FocusSequence:
    # A unique (seq, 3) block per index: arange offset by a per-index stride.
    stride = 4 * 3
    block = torch.arange(
        index * stride, (index + 1) * stride, dtype=torch.float32
    ).reshape(4, 3)
    return FocusSequence(block)


class TestFocusSequenceConstruction:
    def test_accepts_2d_tensor(self):
        seq = FocusSequence(torch.zeros(5, 3))
        assert seq.tensor.shape == (5, 3)

    def test_accepts_empty_sequence(self):
        # An empty (0, 3) sequence is a valid value, not an error.
        seq = FocusSequence(torch.zeros(0, 3))
        assert seq.tensor.shape == (0, 3)

    def test_rejects_1d_tensor(self):
        # A 1D tensor lacks the leading seq axis required by (seq, 3).
        with pytest.raises(ValueError, match=r"\(seq, 3\)"):
            FocusSequence(torch.zeros(3))

    def test_rejects_3d_tensor(self):
        # A 3D tensor carries an extra (batch) axis the sequence forbids.
        with pytest.raises(ValueError, match=r"\(seq, 3\)"):
            FocusSequence(torch.zeros(2, 5, 3))

    def test_rejects_wrong_last_dim(self):
        # The last axis must be exactly 3 ([x, y, zoom]); 2 is rejected.
        with pytest.raises(ValueError, match=r"\(seq, 3\)"):
            FocusSequence(torch.zeros(5, 2))

    def test_error_reports_actual_ndim_and_shape(self):
        # The message must surface the offending ndim and shape (substring only).
        with pytest.raises(ValueError) as exc:
            FocusSequence(torch.zeros(2, 5, 4))

        assert "(seq, 3)" in str(exc.value)
        assert "3" in str(exc.value)  # ndim
        assert "5" in str(exc.value) and "4" in str(exc.value)  # shape


class TestFocusSequenceValidate:
    def test_in_range_is_valid(self):
        seq = FocusSequence(torch.tensor([[0.0, 0.0, 0.5], [-0.5, 0.5, 0.2]]))

        assert seq.is_valid()

    def test_boundary_values_are_valid(self):
        # point=±1 and zoom=0 / zoom=1 are all on the CLOSED-interval boundary.
        seq = FocusSequence(torch.tensor([[1.0, -1.0, 0.0], [-1.0, 1.0, 1.0]]))

        assert seq.is_valid()

    def test_out_of_range_point_is_invalid(self):
        # point=1.5 exceeds the [-1, 1] bound.
        seq = FocusSequence(torch.tensor([[1.5, 0.0, 0.5]]))

        assert not seq.is_valid()

    def test_negative_zoom_is_invalid(self):
        # zoom=-0.1 is below the [0, 1] bound.
        seq = FocusSequence(torch.tensor([[0.0, 0.0, -0.1]]))

        assert not seq.is_valid()

    def test_zoom_above_one_is_invalid(self):
        # zoom=1.2 exceeds the [0, 1] bound.
        seq = FocusSequence(torch.tensor([[0.0, 0.0, 1.2]]))

        assert not seq.is_valid()

    def test_validate_passes_silently_when_in_range(self):
        seq = FocusSequence(torch.tensor([[0.0, 0.0, 0.5]]))

        assert seq.validate() is None

    def test_validate_raises_when_out_of_range(self):
        # validate() surfaces the out-of-range condition (substring match only).
        seq = FocusSequence(torch.tensor([[0.0, 0.0, -0.1]]))

        with pytest.raises(ValueError, match="out of range"):
            seq.validate()


class TestFocusSequenceFromFocuses:
    def test_stacks_multiple_focuses(self):
        # Three value-distinct Focuses stack into a (seq, 3) sequence; the
        # per-row torch.equal checks pin both element identity and order.
        focuses = [_focus_row(0), _focus_row(1), _focus_row(2)]

        result = FocusSequence.from_focuses(focuses)

        assert type(result) is FocusSequence
        assert result.tensor.shape == (3, 3)
        assert torch.equal(result.tensor[0], focuses[0].tensor())
        assert torch.equal(result.tensor[1], focuses[1].tensor())
        assert torch.equal(result.tensor[2], focuses[2].tensor())

    def test_single_focus(self):
        # The minimum non-empty input is one Focus, yielding leading axis 1.
        focus = _focus_row(2)

        result = FocusSequence.from_focuses([focus])

        assert type(result) is FocusSequence
        assert result.tensor.shape == (1, 3)
        assert torch.equal(result.tensor[0], focus.tensor())

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one" contract (substring match only).
        with pytest.raises(ValueError, match="at least one"):
            FocusSequence.from_focuses([])

    def test_accepts_generator(self):
        # A one-shot generator (N>=1) must work: the empty check and the stack
        # both succeed, proving the input is materialized before being scanned.
        result = FocusSequence.from_focuses(_focus_row(i) for i in range(2))

        assert type(result) is FocusSequence
        assert result.tensor.shape == (2, 3)

    def test_is_float32(self):
        # tensor() yields float32 and torch.stack carries that dtype through.
        result = FocusSequence.from_focuses([_focus_row(1)])

        assert result.tensor.dtype == torch.float32


class TestFocusSequenceDeviceTransfer:
    def test_device_reflects_tensor(self):
        assert FocusSequence(torch.zeros(5, 3)).device == torch.device("cpu")

    def test_device_available_for_empty_sequence(self):
        assert FocusSequence(torch.zeros(0, 3)).device == torch.device("cpu")

    def test_to_returns_new_sequence(self):
        # `to` is not in-place: it returns a fresh FocusSequence with equal values.
        seq = FocusSequence(torch.zeros(5, 3))

        result = seq.to("cpu")

        assert type(result) is FocusSequence
        assert result is not seq
        assert torch.equal(result.tensor, seq.tensor)

    def test_to_accepts_torch_device(self):
        seq = FocusSequence(torch.zeros(5, 3))

        result = seq.to(torch.device("cpu"))

        assert type(result) is FocusSequence
        assert torch.equal(result.tensor, seq.tensor)


class TestFocusSequenceIter:
    def test_iterates_rows_as_focuses(self):
        # Iteration yields one Focus per row, in order; each rebuilt Focus's
        # tensor() round-trips to the stored row (float32 conversion is exact).
        focuses = [_focus_row(0), _focus_row(1), _focus_row(2)]
        seq = FocusSequence.from_focuses(focuses)

        items = list(seq)

        assert len(items) == 3
        assert all(type(f) is Focus for f in items)
        for i, focus in enumerate(items):
            assert torch.equal(focus.tensor(), seq.tensor[i])

    def test_iterates_empty_as_no_focuses(self):
        assert list(FocusSequence(torch.zeros(0, 3))) == []


class TestFocusSequenceApply:
    def test_returns_image_sequence_with_one_frame_per_focus(self):
        # apply maps each of the 3 focuses to one observation frame.
        seq = FocusSequence.from_focuses(
            [Focus((0.0, 0.0), 1.0), Focus((0.0, 0.0), 0.5), Focus((0.5, -0.5), 0.3)]
        )
        image = Image(torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8))

        result = seq.apply(image, 4)

        assert type(result) is ImageSequence
        assert len(result) == 3

    def test_mixed_zoom_frames_share_uniform_shape_with_int_size(self):
        # The whole point: full-zoom and partial-zoom focuses crop to DIFFERENT
        # sizes, yet apply resizes each to `size` so they stack into one
        # (seq, C, size, size) ImageSequence. int size -> square.
        seq = FocusSequence.from_focuses(
            [Focus((0.0, 0.0), 1.0), Focus((0.0, 0.0), 0.25), Focus((1.0, 1.0), 0.5)]
        )
        image = Image(torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8))

        result = seq.apply(image, 6)

        assert result.tensor.shape == (3, 3, 6, 6)

    def test_mixed_zoom_frames_share_uniform_shape_with_tuple_size(self):
        # Same uniformity guarantee with a (h, w) tuple size.
        seq = FocusSequence.from_focuses(
            [Focus((0.0, 0.0), 1.0), Focus((-0.5, 0.5), 0.4)]
        )
        image = Image(torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8))

        result = seq.apply(image, (5, 7))

        assert result.tensor.shape == (2, 3, 5, 7)

    def test_full_zoom_center_frame_has_expected_shape(self):
        # A full-zoom center Focus is resized to `size`; resampling forbids a
        # pixel-exact check, so we pin len and the per-frame shape only.
        seq = FocusSequence.from_focuses([Focus((0.0, 0.0), 1.0)])
        image = Image(torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8))

        result = seq.apply(image, 4)

        assert len(result) == 1
        assert result[0].size == (4, 4)
        assert result[0].channels == 3


class TestBatchedFocusSequenceConstruction:
    def test_accepts_3d_tensor(self):
        batch = BatchedFocusSequence(torch.zeros(2, 5, 3))
        assert batch.tensor.shape == (2, 5, 3)

    def test_accepts_empty_batch(self):
        # An empty (0, seq, 3) batch is a valid value, not an error.
        batch = BatchedFocusSequence(torch.zeros(0, 5, 3))
        assert batch.tensor.shape == (0, 5, 3)

    def test_rejects_2d_tensor(self):
        # A 2D tensor lacks the leading batch axis required by (batch, seq, 3).
        with pytest.raises(ValueError, match=r"\(batch, seq, 3\)"):
            BatchedFocusSequence(torch.zeros(5, 3))

    def test_rejects_4d_tensor(self):
        # A 4D tensor carries an extra axis the batched sequence forbids.
        with pytest.raises(ValueError, match=r"\(batch, seq, 3\)"):
            BatchedFocusSequence(torch.zeros(2, 2, 5, 3))

    def test_rejects_wrong_last_dim(self):
        # The last axis must be exactly 3 ([x, y, zoom]); 4 is rejected.
        with pytest.raises(ValueError, match=r"\(batch, seq, 3\)"):
            BatchedFocusSequence(torch.zeros(2, 5, 4))

    def test_error_reports_actual_ndim_and_shape(self):
        # The message must surface the offending ndim and shape (substring only).
        with pytest.raises(ValueError) as exc:
            BatchedFocusSequence(torch.zeros(5, 4))

        assert "(batch, seq, 3)" in str(exc.value)
        assert "2" in str(exc.value)  # ndim
        assert "5" in str(exc.value) and "4" in str(exc.value)  # shape


class TestBatchedFocusSequenceValidate:
    def test_in_range_is_valid(self):
        batch = BatchedFocusSequence(torch.zeros(2, 5, 3))

        assert batch.is_valid()

    def test_boundary_values_are_valid(self):
        # point=±1 and zoom=0 / zoom=1 are all on the CLOSED-interval boundary.
        batch = BatchedFocusSequence(
            torch.tensor([[[1.0, -1.0, 0.0], [-1.0, 1.0, 1.0]]])
        )

        assert batch.is_valid()

    def test_out_of_range_point_is_invalid(self):
        batch = BatchedFocusSequence(torch.tensor([[[1.5, 0.0, 0.5]]]))

        assert not batch.is_valid()

    def test_negative_zoom_is_invalid(self):
        batch = BatchedFocusSequence(torch.tensor([[[0.0, 0.0, -0.1]]]))

        assert not batch.is_valid()

    def test_zoom_above_one_is_invalid(self):
        batch = BatchedFocusSequence(torch.tensor([[[0.0, 0.0, 1.2]]]))

        assert not batch.is_valid()

    def test_validate_passes_silently_when_in_range(self):
        batch = BatchedFocusSequence(torch.tensor([[[0.0, 0.0, 0.5]]]))

        assert batch.validate() is None

    def test_validate_raises_when_out_of_range(self):
        # validate() surfaces the out-of-range condition (substring match only).
        batch = BatchedFocusSequence(torch.tensor([[[0.0, 0.0, 1.2]]]))

        with pytest.raises(ValueError, match="out of range"):
            batch.validate()


class TestBatchedFocusSequenceFromSequences:
    def test_stacks_multiple_sequences(self):
        # Two value-distinct sequences stack into (batch, seq, 3); the per-entry
        # torch.equal checks pin both element identity and order.
        sequences = [_distinct_focus_sequence(0), _distinct_focus_sequence(1)]

        result = BatchedFocusSequence.from_sequences(sequences)

        assert type(result) is BatchedFocusSequence
        assert result.tensor.shape == (2, 4, 3)
        assert torch.equal(result.tensor[0], sequences[0].tensor)
        assert torch.equal(result.tensor[1], sequences[1].tensor)

    def test_single_sequence(self):
        # The minimum non-empty input is one FocusSequence, yielding batch 1.
        sequence = _distinct_focus_sequence(3)

        result = BatchedFocusSequence.from_sequences([sequence])

        assert type(result) is BatchedFocusSequence
        assert result.tensor.shape == (1, 4, 3)
        assert torch.equal(result.tensor[0], sequence.tensor)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one" contract (substring match only).
        with pytest.raises(ValueError, match="at least one"):
            BatchedFocusSequence.from_sequences([])

    def test_accepts_generator(self):
        # A one-shot generator (N>=1) must work: materialization lets the empty
        # check and the stack both scan the same elements.
        result = BatchedFocusSequence.from_sequences(
            _distinct_focus_sequence(i) for i in range(2)
        )

        assert type(result) is BatchedFocusSequence
        assert result.tensor.shape == (2, 4, 3)

    def test_is_float32(self):
        # The stacked sequences are float32; torch.stack carries that through.
        result = BatchedFocusSequence.from_sequences([_distinct_focus_sequence(0)])

        assert result.tensor.dtype == torch.float32


class TestBatchedFocusSequenceDeviceTransfer:
    def test_device_reflects_tensor(self):
        assert BatchedFocusSequence(torch.zeros(2, 5, 3)).device == torch.device("cpu")

    def test_device_available_for_empty_batch(self):
        assert BatchedFocusSequence(torch.zeros(0, 5, 3)).device == torch.device("cpu")

    def test_to_returns_new_batch(self):
        # `to` is not in-place: it returns a fresh BatchedFocusSequence.
        batch = BatchedFocusSequence(torch.zeros(2, 5, 3))

        result = batch.to("cpu")

        assert type(result) is BatchedFocusSequence
        assert result is not batch
        assert torch.equal(result.tensor, batch.tensor)

    def test_to_accepts_torch_device(self):
        batch = BatchedFocusSequence(torch.zeros(2, 5, 3))

        result = batch.to(torch.device("cpu"))

        assert type(result) is BatchedFocusSequence
        assert torch.equal(result.tensor, batch.tensor)


class TestBatchedFocusSequenceIter:
    def test_iterates_entries_as_focus_sequences(self):
        # Iteration yields one FocusSequence per batch entry, in order.
        sequences = [_distinct_focus_sequence(0), _distinct_focus_sequence(1)]
        batch = BatchedFocusSequence.from_sequences(sequences)

        items = list(batch)

        assert len(items) == 2
        assert all(type(e) is FocusSequence for e in items)
        for i, entry in enumerate(items):
            assert torch.equal(entry.tensor, batch.tensor[i])

    def test_iterates_empty_as_no_entries(self):
        assert list(BatchedFocusSequence(torch.zeros(0, 5, 3))) == []
