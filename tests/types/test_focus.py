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

from exp.types.focus import Focus
from exp.types.image import Image


class TestConstruction:
    def test_rejects_out_of_range_point(self):
        # Each point component must lie in [-1, 1]; 1.5 is outside.
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            Focus((1.5, 0.0), 1.0)

    def test_rejects_out_of_range_zoom(self):
        # zoom must lie in (0, 1]; 0.0 is outside (open lower bound).
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            Focus((0.0, 0.0), 0.0)


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
