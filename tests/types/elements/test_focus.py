"""Behaviour spec for ``exp.types.elements.focus``.

These tests translate the approved spec for the immutable focus value
objects (the action ``a = (point, zoom)``: gaze point + zoom, plus its
sequence / batch / batched-sequence collection types) into executable
form. They are written against the *spec*, not any implementation: an
implementation that diverges should make these red. Scenarios are
grouped into one class per behaviour area.

The crop behaviour of ``Focus.__call__`` mirrors the former
``Image.focus`` (square-pad first, ``round(zoom * side)`` crop side,
out-of-bounds regions zero-padded), so ``TestCall`` pins those four
scenarios. Tensors are constructed deterministically on real CPU torch
(no mocking of torch) per the project testing strategy.
"""

import pytest
import torch

from exp.types.elements.focus import (
    FOCUS_DIM,
    BatchedFocus,
    BatchedFocusSequence,
    Focus,
    FocusSequence,
)
from exp.types.elements.image import Image, ImageSequence


class TestFocusDim:
    """``FOCUS_DIM`` names the [x, y, zoom] width shared by the family."""

    def test_focus_dim_is_three(self):
        # The action row is [x, y, zoom]: exactly three components. Pinned as a
        # semantic invariant (not a bare literal duplicate): downstream models
        # concatenate FOCUS_DIM + latent_dim, so this width is part of the
        # contract, and it must equal what a real Focus tensor actually carries.
        assert FOCUS_DIM == 3
        assert Focus.init((0.0, 0.0), 0.5).tensor.shape == (FOCUS_DIM,)


class TestFocusInit:
    """``Focus.init(point, zoom)`` range-checks its arguments."""

    def test_rejects_out_of_range_point(self):
        # Each point component must lie in [-1, 1]; 1.5 is outside.
        with pytest.raises(ValueError, match=r"Focus point must be within \[-1, 1\]"):
            Focus.init((1.5, 0.0), 1.0)

    def test_rejects_out_of_range_zoom(self):
        # zoom must lie in the closed interval [0, 1]; -0.1 is below it.
        with pytest.raises(ValueError, match=r"Focus zoom must be within \[0, 1\]"):
            Focus.init((0.0, 0.0), -0.1)

    def test_accepts_boundary_point_and_zoom(self):
        # point=±1 and zoom=0 / zoom=1 are all on the CLOSED-interval boundary.
        low = Focus.init((-1.0, -1.0), 0.0)
        high = Focus.init((1.0, 1.0), 1.0)

        assert low.zoom == 0.0
        assert high.zoom == 1.0

    def test_point_and_zoom_round_trip_through_properties(self):
        # init packs [x, y, zoom]; .point / .zoom read them back (float32 exact
        # for these representable values on the point, approx on zoom is not
        # needed here as all three are exactly representable in float32).
        focus = Focus.init((0.5, -0.5), 0.25)

        assert focus.point == (0.5, -0.5)
        assert focus.zoom == 0.25


class TestFocusTensor:
    """``Focus.tensor`` is an attribute (not a method) holding [x, y, zoom]."""

    def test_tensor_is_float32_vector_of_x_y_zoom(self):
        # .tensor yields [x, y, zoom] as a CPU float32 vector of shape (3,).
        out = Focus.init((0.5, -0.25), 0.75).tensor

        assert out.shape == (3,)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, torch.tensor([0.5, -0.25, 0.75]))

    def test_direct_construction_checks_shape_only(self):
        # Focus(tensor) is the raw constructor: it validates shape but NOT the
        # value range, so an out-of-range tensor still builds successfully.
        focus = Focus(torch.tensor([5.0, -9.0, 3.0]))

        assert focus.tensor.shape == (3,)

    def test_direct_construction_rejects_wrong_ndim(self):
        # A 2D tensor lacks the single (3,) axis the leaf Focus requires.
        with pytest.raises(ValueError, match=r"\(3,\)"):
            Focus(torch.zeros(2, 3))

    def test_direct_construction_rejects_wrong_last_dim(self):
        # The single axis must be exactly length 3 ([x, y, zoom]); 2 is rejected.
        with pytest.raises(ValueError, match=r"\(3,\)"):
            Focus(torch.zeros(2))


class TestFocusValidate:
    """``Focus`` inherits the family range check: is_valid / validate."""

    def test_in_range_is_valid(self):
        assert Focus(torch.tensor([0.5, -0.5, 0.25])).is_valid()

    def test_boundary_values_are_valid(self):
        # point=±1 and zoom=0 / zoom=1 are all on the CLOSED-interval boundary.
        assert Focus(torch.tensor([1.0, -1.0, 0.0])).is_valid()
        assert Focus(torch.tensor([-1.0, 1.0, 1.0])).is_valid()

    def test_out_of_range_point_is_invalid(self):
        # point=1.5 exceeds the [-1, 1] bound.
        assert not Focus(torch.tensor([1.5, 0.0, 0.5])).is_valid()

    def test_out_of_range_zoom_is_invalid(self):
        # zoom=1.2 exceeds the [0, 1] bound.
        assert not Focus(torch.tensor([0.0, 0.0, 1.2])).is_valid()

    def test_validate_passes_silently_when_in_range(self):
        assert Focus(torch.tensor([0.0, 0.0, 0.5])).validate() is None

    def test_validate_raises_when_out_of_range(self):
        # validate() surfaces the out-of-range condition (substring match only).
        with pytest.raises(ValueError, match="out of range"):
            Focus(torch.tensor([0.0, 0.0, 1.2])).validate()


class TestCall:
    """``Focus.__call__(image)`` square-crops per the (point, zoom) action."""

    def test_full_view_returns_whole_image(self):
        # zoom=1, point=center on a square image returns the image unchanged.
        img = Image(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4))

        out = Focus.init((0.0, 0.0), 1.0)(img)

        assert torch.equal(out.tensor, img.tensor)

    def test_output_side_is_round_zoom_times_side(self):
        # zoom=0.5 of an 8-wide side rounds to a 4x4 square crop.
        out = Focus.init((0.0, 0.0), 0.5)(Image(torch.zeros(3, 8, 8)))

        assert out.is_squared
        assert out.size == (4, 4)

    def test_squares_non_square_input_first(self):
        # (4, 8) squares to side 8, so a full-zoom focus yields 8x8.
        out = Focus.init((0.0, 0.0), 1.0)(Image(torch.zeros(3, 4, 8)))

        assert out.size == (8, 8)

    def test_corner_pads_out_of_bounds(self):
        # Bottom-right gaze with partial zoom spills past the edge -> zero pad.
        out = Focus.init((1.0, 1.0), 0.5)(Image(torch.ones(1, 4, 4)))

        assert out.size == (2, 2)
        assert (out.tensor == 0).any()


# A focus row is [x, y, zoom]; these helpers build value-distinct, in-range rows
# so the per-row torch.equal checks below pin both element identity and ordering.
def _focus_row(index: int) -> Focus:
    # Stay inside [-1, 1] / [0, 1] so the built Focus is in-range; small unique
    # offsets per index keep every row distinguishable.
    return Focus.init((0.1 * index, -0.1 * index), 0.1 * index)


def _distinct_focus_sequence(index: int) -> FocusSequence:
    # A unique (seq, 3) block per index: arange offset by a per-index stride.
    stride = 4 * 3
    block = torch.arange(
        index * stride, (index + 1) * stride, dtype=torch.float32
    ).reshape(4, 3)
    return FocusSequence(block)


def _distinct_batched_focus(index: int) -> BatchedFocus:
    # A unique (batch, 3) block per index for from_batches / iter_sequence tests.
    stride = 2 * 3
    block = torch.arange(
        index * stride, (index + 1) * stride, dtype=torch.float32
    ).reshape(2, 3)
    return BatchedFocus(block)


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
        assert "ndim=3" in str(exc.value)
        assert "(2, 5, 4)" in str(exc.value)


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


class TestFocusSequenceFromElements:
    def test_stacks_multiple_focuses(self):
        # Three value-distinct Focuses stack into a (seq, 3) sequence; the
        # per-row torch.equal checks pin both element identity and order.
        focuses = [_focus_row(0), _focus_row(1), _focus_row(2)]

        result = FocusSequence.from_elements(focuses)

        assert type(result) is FocusSequence
        assert result.tensor.shape == (3, 3)
        assert torch.equal(result.tensor[0], focuses[0].tensor)
        assert torch.equal(result.tensor[1], focuses[1].tensor)
        assert torch.equal(result.tensor[2], focuses[2].tensor)

    def test_single_focus(self):
        # The minimum non-empty input is one Focus, yielding leading axis 1.
        focus = _focus_row(2)

        result = FocusSequence.from_elements([focus])

        assert type(result) is FocusSequence
        assert result.tensor.shape == (1, 3)
        assert torch.equal(result.tensor[0], focus.tensor)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one" contract (substring match only).
        with pytest.raises(ValueError, match="at least one element"):
            FocusSequence.from_elements([])

    def test_accepts_generator(self):
        # A one-shot generator (N>=1) must work: the empty check and the stack
        # both succeed, proving the input is materialized before being scanned.
        result = FocusSequence.from_elements(_focus_row(i) for i in range(2))

        assert type(result) is FocusSequence
        assert result.tensor.shape == (2, 3)

    def test_is_float32(self):
        # Focus tensors are float32 and torch.stack carries that dtype through.
        result = FocusSequence.from_elements([_focus_row(1)])

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


class TestFocusSequenceIndexing:
    def test_int_index_returns_focus(self):
        # A single index yields the element type (Focus), not a sequence.
        seq = FocusSequence.from_elements([_focus_row(0), _focus_row(1)])

        item = seq[1]

        assert type(item) is Focus
        assert torch.equal(item.tensor, seq.tensor[1])

    def test_slice_index_returns_focus_sequence(self):
        # A slice preserves the collection type and selects the sub-range.
        seq = FocusSequence.from_elements([_focus_row(i) for i in range(4)])

        sub = seq[1:3]

        assert type(sub) is FocusSequence
        assert sub.tensor.shape == (2, 3)
        assert torch.equal(sub.tensor, seq.tensor[1:3])

    def test_len_counts_rows(self):
        seq = FocusSequence.from_elements([_focus_row(i) for i in range(3)])

        assert len(seq) == 3


class TestFocusSequenceIter:
    def test_iterates_rows_as_focuses(self):
        # Iteration yields one Focus per row, in order; each rebuilt Focus's
        # tensor round-trips to the stored row (float32 conversion is exact).
        focuses = [_focus_row(0), _focus_row(1), _focus_row(2)]
        seq = FocusSequence.from_elements(focuses)

        items = list(seq)

        assert len(items) == 3
        assert all(type(f) is Focus for f in items)
        for i, focus in enumerate(items):
            assert torch.equal(focus.tensor, seq.tensor[i])

    def test_iterates_empty_as_no_focuses(self):
        assert list(FocusSequence(torch.zeros(0, 3))) == []


class TestFocusSequenceApply:
    def test_returns_image_sequence_with_one_frame_per_focus(self):
        # apply maps each of the 3 focuses to one observation frame.
        seq = FocusSequence.from_elements(
            [
                Focus.init((0.0, 0.0), 1.0),
                Focus.init((0.0, 0.0), 0.5),
                Focus.init((0.5, -0.5), 0.3),
            ]
        )
        image = Image(torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8))

        result = seq.apply(image, 4)

        assert type(result) is ImageSequence
        assert len(result) == 3

    def test_mixed_zoom_frames_share_uniform_shape_with_int_size(self):
        # The whole point: full-zoom and partial-zoom focuses crop to DIFFERENT
        # sizes, yet apply resizes each to `size` so they stack into one
        # (seq, C, size, size) ImageSequence. int size -> square.
        seq = FocusSequence.from_elements(
            [
                Focus.init((0.0, 0.0), 1.0),
                Focus.init((0.0, 0.0), 0.25),
                Focus.init((1.0, 1.0), 0.5),
            ]
        )
        image = Image(torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8))

        result = seq.apply(image, 6)

        assert result.tensor.shape == (3, 3, 6, 6)

    def test_mixed_zoom_frames_share_uniform_shape_with_tuple_size(self):
        # Same uniformity guarantee with a (h, w) tuple size.
        seq = FocusSequence.from_elements(
            [Focus.init((0.0, 0.0), 1.0), Focus.init((-0.5, 0.5), 0.4)]
        )
        image = Image(torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8))

        result = seq.apply(image, (5, 7))

        assert result.tensor.shape == (2, 3, 5, 7)

    def test_full_zoom_center_frame_has_expected_shape(self):
        # A full-zoom center Focus is resized to `size`; resampling forbids a
        # pixel-exact check, so we pin len and the per-frame shape only.
        seq = FocusSequence.from_elements([Focus.init((0.0, 0.0), 1.0)])
        image = Image(torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8))

        result = seq.apply(image, 4)

        assert len(result) == 1
        assert result[0].size == (4, 4)
        assert result[0].channels == 3


class TestBatchedFocusConstruction:
    def test_accepts_2d_tensor(self):
        batch = BatchedFocus(torch.zeros(4, 3))
        assert batch.tensor.shape == (4, 3)

    def test_accepts_empty_batch(self):
        # An empty (0, 3) batch is a valid value, not an error.
        batch = BatchedFocus(torch.zeros(0, 3))
        assert batch.tensor.shape == (0, 3)

    def test_rejects_1d_tensor(self):
        # A 1D tensor lacks the leading batch axis required by (batch, 3).
        with pytest.raises(ValueError, match=r"\(batch, 3\)"):
            BatchedFocus(torch.zeros(3))

    def test_rejects_3d_tensor(self):
        # A 3D tensor carries an extra (seq) axis the batch forbids.
        with pytest.raises(ValueError, match=r"\(batch, 3\)"):
            BatchedFocus(torch.zeros(2, 4, 3))

    def test_rejects_wrong_last_dim(self):
        # The last axis must be exactly 3 ([x, y, zoom]); 2 is rejected.
        with pytest.raises(ValueError, match=r"\(batch, 3\)"):
            BatchedFocus(torch.zeros(4, 2))

    def test_error_reports_actual_ndim_and_shape(self):
        # The message must surface the offending ndim and shape (substring only).
        with pytest.raises(ValueError) as exc:
            BatchedFocus(torch.zeros(2, 4, 5))

        assert "(batch, 3)" in str(exc.value)
        assert "ndim=3" in str(exc.value)
        assert "(2, 4, 5)" in str(exc.value)


class TestBatchedFocusValidate:
    def test_in_range_is_valid(self):
        batch = BatchedFocus(torch.tensor([[0.0, 0.0, 0.5], [-0.5, 0.5, 0.2]]))

        assert batch.is_valid()

    def test_boundary_values_are_valid(self):
        batch = BatchedFocus(torch.tensor([[1.0, -1.0, 0.0], [-1.0, 1.0, 1.0]]))

        assert batch.is_valid()

    def test_out_of_range_point_is_invalid(self):
        batch = BatchedFocus(torch.tensor([[1.5, 0.0, 0.5]]))

        assert not batch.is_valid()

    def test_zoom_above_one_is_invalid(self):
        batch = BatchedFocus(torch.tensor([[0.0, 0.0, 1.2]]))

        assert not batch.is_valid()

    def test_validate_passes_silently_when_in_range(self):
        batch = BatchedFocus(torch.tensor([[0.0, 0.0, 0.5]]))

        assert batch.validate() is None

    def test_validate_raises_when_out_of_range(self):
        # validate() surfaces the out-of-range condition (substring match only).
        batch = BatchedFocus(torch.tensor([[0.0, 0.0, 1.2]]))

        with pytest.raises(ValueError, match="out of range"):
            batch.validate()


class TestBatchedFocusFromElements:
    def test_stacks_multiple_focuses(self):
        # Value-distinct Focuses stack along the batch axis; per-row torch.equal
        # pins element identity and order.
        focuses = [_focus_row(0), _focus_row(1), _focus_row(2)]

        result = BatchedFocus.from_elements(focuses)

        assert type(result) is BatchedFocus
        assert result.tensor.shape == (3, 3)
        assert torch.equal(result.tensor[0], focuses[0].tensor)
        assert torch.equal(result.tensor[2], focuses[2].tensor)

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one element"):
            BatchedFocus.from_elements([])

    def test_accepts_generator(self):
        result = BatchedFocus.from_elements(_focus_row(i) for i in range(2))

        assert type(result) is BatchedFocus
        assert result.tensor.shape == (2, 3)


class TestBatchedFocusIndexing:
    def test_int_index_returns_focus(self):
        # A single index yields the element type (Focus), not a batch.
        batch = BatchedFocus.from_elements([_focus_row(0), _focus_row(1)])

        item = batch[0]

        assert type(item) is Focus
        assert torch.equal(item.tensor, batch.tensor[0])

    def test_slice_index_returns_batched_focus(self):
        batch = BatchedFocus.from_elements([_focus_row(i) for i in range(4)])

        sub = batch[1:3]

        assert type(sub) is BatchedFocus
        assert sub.tensor.shape == (2, 3)

    def test_len_counts_rows(self):
        batch = BatchedFocus.from_elements([_focus_row(i) for i in range(3)])

        assert len(batch) == 3


class TestBatchedFocusIter:
    def test_iterates_rows_as_focuses(self):
        # Iteration yields one Focus per batch row, in order.
        focuses = [_focus_row(0), _focus_row(1)]
        batch = BatchedFocus.from_elements(focuses)

        items = list(batch)

        assert len(items) == 2
        assert all(type(f) is Focus for f in items)
        for i, focus in enumerate(items):
            assert torch.equal(focus.tensor, batch.tensor[i])

    def test_iterates_empty_as_no_focuses(self):
        assert list(BatchedFocus(torch.zeros(0, 3))) == []


class TestBatchedFocusDeviceTransfer:
    def test_device_reflects_tensor(self):
        assert BatchedFocus(torch.zeros(4, 3)).device == torch.device("cpu")

    def test_to_returns_new_batch(self):
        # `to` is not in-place: it returns a fresh BatchedFocus with equal values.
        batch = BatchedFocus(torch.zeros(4, 3))

        result = batch.to("cpu")

        assert type(result) is BatchedFocus
        assert result is not batch
        assert torch.equal(result.tensor, batch.tensor)


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
        assert "ndim=2" in str(exc.value)
        assert "(5, 4)" in str(exc.value)


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
    def test_stacks_multiple_sequences_on_batch_axis(self):
        # Two value-distinct sequences stack into (batch, seq, 3) along dim=0;
        # the per-entry torch.equal checks pin element identity and order.
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
        # Empty input violates the "at least one sequence" contract (substring).
        with pytest.raises(ValueError, match="at least one sequence"):
            BatchedFocusSequence.from_sequences([])

    def test_accepts_generator(self):
        # A one-shot generator (N>=1) must work: materialization lets the empty
        # check and the stack both scan the same elements.
        result = BatchedFocusSequence.from_sequences(
            _distinct_focus_sequence(i) for i in range(2)
        )

        assert type(result) is BatchedFocusSequence
        assert result.tensor.shape == (2, 4, 3)


class TestBatchedFocusSequenceFromBatches:
    def test_stacks_multiple_batches_on_seq_axis(self):
        # from_batches stacks along dim=1 (the seq axis): N BatchedFocus of
        # shape (batch, 3) become one (batch, N, 3) batched sequence. The
        # per-column torch.equal checks pin identity and seq ordering.
        batches = [_distinct_batched_focus(0), _distinct_batched_focus(1)]

        result = BatchedFocusSequence.from_batches(batches)

        assert type(result) is BatchedFocusSequence
        assert result.tensor.shape == (2, 2, 3)
        assert torch.equal(result.tensor[:, 0], batches[0].tensor)
        assert torch.equal(result.tensor[:, 1], batches[1].tensor)

    def test_single_batch(self):
        # One BatchedFocus (batch, 3) yields seq length 1: (batch, 1, 3).
        batch = _distinct_batched_focus(4)

        result = BatchedFocusSequence.from_batches([batch])

        assert type(result) is BatchedFocusSequence
        assert result.tensor.shape == (2, 1, 3)
        assert torch.equal(result.tensor[:, 0], batch.tensor)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one batch" contract (substring).
        with pytest.raises(ValueError, match="at least one batch"):
            BatchedFocusSequence.from_batches([])

    def test_accepts_generator(self):
        result = BatchedFocusSequence.from_batches(
            _distinct_batched_focus(i) for i in range(3)
        )

        assert type(result) is BatchedFocusSequence
        assert result.tensor.shape == (2, 3, 3)


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
    def test_iter_yields_focus_sequences_over_batch_axis(self):
        # Default iteration walks the batch axis (dim=0), yielding one
        # FocusSequence per batch entry, in order.
        sequences = [_distinct_focus_sequence(0), _distinct_focus_sequence(1)]
        batch = BatchedFocusSequence.from_sequences(sequences)

        items = list(batch)

        assert len(items) == 2
        assert all(type(e) is FocusSequence for e in items)
        for i, entry in enumerate(items):
            assert torch.equal(entry.tensor, batch.tensor[i])

    def test_iter_batch_matches_default_iteration(self):
        # iter_batch() is documented as the same walk as __iter__ (batch axis).
        batch = BatchedFocusSequence(
            torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        )

        via_iter = [e.tensor for e in batch]
        via_iter_batch = [e.tensor for e in batch.iter_batch()]

        assert len(via_iter) == len(via_iter_batch)
        assert all(
            torch.equal(a, b) for a, b in zip(via_iter, via_iter_batch, strict=True)
        )

    def test_iter_sequence_yields_batched_focus_over_seq_axis(self):
        # iter_sequence() walks the seq axis (dim=1): each yielded BatchedFocus
        # is the (batch, 3) column at a given time step, in seq order.
        batch = BatchedFocusSequence(
            torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        )

        steps = list(batch.iter_sequence())

        assert len(steps) == 4
        assert all(type(s) is BatchedFocus for s in steps)
        for t, step in enumerate(steps):
            assert torch.equal(step.tensor, batch.tensor[:, t])

    def test_iterates_empty_as_no_entries(self):
        assert list(BatchedFocusSequence(torch.zeros(0, 5, 3))) == []
