"""Behaviour spec for ``exp.types.image``.

These tests translate the approved spec for the immutable ``Image``
value object into executable form. They are written against the *spec*,
not any implementation: an implementation that diverges should make
these red. Scenarios are grouped into one class per behaviour area.

Tensors are constructed deterministically on real CPU torch (no mocking
of torch) per the project testing strategy.
"""

import pytest
import torch

from exp.types.image import ChannelFormat, Image


class TestConstruction:
    def test_rejects_2d_tensor(self):
        # A 2D tensor lacks the channel axis required by (C, H, W).
        with pytest.raises(ValueError, match="C, H, W"):
            Image(torch.zeros(8, 16))

    def test_rejects_4d_tensor(self):
        # A 4D tensor carries a batch axis the value object forbids.
        with pytest.raises(ValueError, match="C, H, W"):
            Image(torch.zeros(1, 3, 8, 16))

    def test_error_reports_actual_shape(self):
        # The message must surface the offending shape (substring check only).
        with pytest.raises(ValueError) as exc:
            Image(torch.zeros(8, 16))

        assert "C, H, W" in str(exc.value)
        assert "8" in str(exc.value) and "16" in str(exc.value)


class TestShapeProperties:
    def test_chw_axis_mapping(self):
        # Non-symmetric shape catches a swapped H/W axis assignment.
        img = Image(torch.zeros(3, 8, 16))

        assert img.channels == 3
        assert img.height == 8
        assert img.width == 16

    def test_size_is_height_width(self):
        # size pins (height, width) order, matching Size2d / resize.
        assert Image(torch.zeros(3, 8, 16)).size == (8, 16)

    def test_is_squared(self):
        assert Image(torch.zeros(3, 8, 8)).is_squared
        assert not Image(torch.zeros(3, 8, 16)).is_squared


class TestEquality:
    def test_identity_for_same_instance(self):
        # eq=False contract: an instance compares equal only to itself.
        img = Image(torch.zeros(3, 8, 16))
        assert img == img

    def test_distinct_instances_are_not_equal(self):
        # Equal-valued but distinct instances must NOT be equal, and the
        # comparison must not raise "boolean value of Tensor is ambiguous".
        assert Image(torch.zeros(3, 8, 16)) != Image(torch.zeros(3, 8, 16))


class TestChannelFormat:
    def test_inferred_from_channels(self):
        assert Image(torch.zeros(1, 4, 4)).channel_format is ChannelFormat.GRAY
        assert Image(torch.zeros(3, 4, 4)).channel_format is ChannelFormat.RGB
        assert Image(torch.zeros(4, 4, 4)).channel_format is ChannelFormat.RGBA

    def test_rejects_unsupported_channel_count(self):
        # A 2-channel tensor maps to no ChannelFormat member.
        with pytest.raises(ValueError):
            _ = Image(torch.zeros(2, 4, 4)).channel_format

    def test_enum_values_are_channel_counts(self):
        # value == channel count is the invariant channel_format relies on.
        assert ChannelFormat(1) is ChannelFormat.GRAY
        assert ChannelFormat(3) is ChannelFormat.RGB
        assert ChannelFormat(4) is ChannelFormat.RGBA

    def test_enum_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ChannelFormat(2)


class TestDtypeCasts:
    def test_float(self):
        out = Image(torch.zeros(3, 4, 4, dtype=torch.uint8)).float()
        assert out.tensor.dtype == torch.float32

    def test_uint8(self):
        out = Image(torch.ones(3, 4, 4)).uint8()
        assert out.tensor.dtype == torch.uint8


class TestStandardize:
    def test_sets_mean_and_std(self):
        torch.manual_seed(0)
        img = Image(torch.randn(3, 16, 16) * 5 + 2)

        out = img.standardize(mean=1.0, std=2.0)

        torch.testing.assert_close(
            out.tensor.mean(), torch.tensor(1.0), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            out.tensor.std(), torch.tensor(2.0), atol=1e-4, rtol=0
        )

    def test_constant_image_has_no_nan(self):
        # Zero-variance input must not divide by zero; all pixels become `mean`.
        out = Image(torch.full((3, 4, 4), 7.0)).standardize(mean=3.0, std=2.0)

        assert torch.isfinite(out.tensor).all()
        torch.testing.assert_close(out.tensor, torch.full((3, 4, 4), 3.0))


class TestNormalize:
    def test_sets_min_and_max(self):
        out = Image(torch.tensor([[[0.0, 2.0], [4.0, 8.0]]])).normalize(
            min=0.0, max=1.0
        )

        torch.testing.assert_close(out.tensor.min(), torch.tensor(0.0))
        torch.testing.assert_close(out.tensor.max(), torch.tensor(1.0))

    def test_constant_image_has_no_div_by_zero(self):
        out = Image(torch.full((1, 4, 4), 5.0)).normalize(min=0.0, max=255.0)

        assert torch.isfinite(out.tensor).all()
        torch.testing.assert_close(out.tensor, torch.zeros(1, 4, 4))


class TestAsChannelFormat:
    def test_same_is_identity(self):
        img = Image(torch.zeros(3, 4, 4))
        assert img.as_channel_format(ChannelFormat.RGB) is img

    def test_gray_to_rgb_replicates(self):
        img = Image(torch.arange(16, dtype=torch.uint8).reshape(1, 4, 4))

        rgb = img.as_channel_format(ChannelFormat.RGB)

        assert rgb.channels == 3
        for c in range(3):
            assert torch.equal(rgb.tensor[c], img.tensor[0])

    def test_rgb_to_rgba_adds_opaque_alpha(self):
        rgba = Image(torch.zeros(3, 4, 4, dtype=torch.uint8)).as_channel_format(
            ChannelFormat.RGBA
        )

        assert rgba.channels == 4
        assert torch.equal(rgba.tensor[3], torch.full((4, 4), 255, dtype=torch.uint8))

    def test_rgba_to_rgb_drops_alpha(self):
        rgb = Image(torch.zeros(4, 4, 4, dtype=torch.uint8)).as_channel_format(
            ChannelFormat.RGB
        )
        assert rgb.channels == 3

    def test_rgb_to_gray(self):
        gray = Image(torch.zeros(3, 4, 4, dtype=torch.uint8)).as_channel_format(
            ChannelFormat.GRAY
        )
        assert gray.channels == 1


class TestSquarePad:
    def test_makes_squared(self):
        out = Image(torch.ones(3, 4, 8)).square_pad()

        assert out.is_squared
        assert out.size == (8, 8)

    def test_centers_data_and_fills_border(self):
        # H=2, W=4: pad 1px top and bottom (diff=2, symmetric).
        out = Image(torch.ones(1, 2, 4)).square_pad(fill_value=0)

        assert out.size == (4, 4)
        assert torch.equal(out.tensor[0, 1:3, :], torch.ones(2, 4))
        assert torch.equal(out.tensor[0, 0, :], torch.zeros(4))
        assert torch.equal(out.tensor[0, 3, :], torch.zeros(4))

    def test_noop_when_squared(self):
        img = Image(torch.ones(3, 5, 5))
        assert img.square_pad() is img


class TestFocus:
    def test_full_view_returns_whole_image(self):
        # zoom=1, point=center on a square image returns the image unchanged.
        img = Image(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4))

        out = img.focus((0.0, 0.0), 1.0)

        assert torch.equal(out.tensor, img.tensor)

    def test_output_side_is_round_zoom_times_side(self):
        out = Image(torch.zeros(3, 8, 8)).focus((0.0, 0.0), 0.5)

        assert out.is_squared
        assert out.size == (4, 4)

    def test_squares_non_square_input_first(self):
        # (4, 8) squares to side 8, so a full-zoom focus yields 8x8.
        out = Image(torch.zeros(3, 4, 8)).focus((0.0, 0.0), 1.0)
        assert out.size == (8, 8)

    def test_corner_pads_out_of_bounds(self):
        # Bottom-right gaze with partial zoom spills past the edge -> zero pad.
        out = Image(torch.ones(1, 4, 4)).focus((1.0, 1.0), 0.5)

        assert out.size == (2, 2)
        assert (out.tensor == 0).any()

    def test_rejects_out_of_range_point(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            Image(torch.zeros(3, 4, 4)).focus((1.5, 0.0), 1.0)

    def test_rejects_out_of_range_zoom(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            Image(torch.zeros(3, 4, 4)).focus((0.0, 0.0), 0.0)


class TestResize:
    def test_int_to_square(self):
        assert Image(torch.zeros(3, 4, 8)).resize(16).size == (16, 16)

    def test_tuple_is_hw(self):
        assert Image(torch.zeros(3, 4, 8)).resize((6, 10)).size == (6, 10)

    def test_preserves_dtype(self):
        out = Image(torch.zeros(3, 4, 4, dtype=torch.uint8)).resize(8)
        assert out.tensor.dtype == torch.uint8


class TestDeviceTransfer:
    def test_to_returns_new_image(self):
        # `to` is not in-place: it returns a fresh Image with equal values.
        img = Image(torch.zeros(3, 8, 16))

        result = img.to("cpu")

        assert isinstance(result, Image)
        assert result is not img
        assert torch.equal(result.tensor, img.tensor)

    def test_to_accepts_torch_device(self):
        # DeviceLike is `torch.device | str`; the torch.device form must work.
        img = Image(torch.zeros(3, 8, 16))

        result = img.to(torch.device("cpu"))

        assert isinstance(result, Image)
        assert torch.equal(result.tensor, img.tensor)

    def test_device_reflects_tensor(self):
        assert Image(torch.zeros(3, 8, 16)).device == torch.device("cpu")


class TestLoadSave:
    # integration-real: real file I/O under tmp_path.

    def test_save_load_png_round_trip(self, tmp_path):
        # save normalizes to [0,255] then writes uint8; PNG decode is
        # lossless, so load returns exactly the saved (normalized) tensor.
        img = Image((torch.rand(3, 8, 8) * 255).to(torch.uint8))
        path = tmp_path / "img.png"

        img.save(path)
        loaded = Image.load(path)

        expected = img.normalize(0, 255).uint8().tensor
        assert loaded.tensor.dtype == torch.uint8
        assert loaded.tensor.shape == (3, 8, 8)
        assert torch.equal(loaded.tensor, expected)

    def test_load_preserves_original_channels(self, tmp_path):
        path = tmp_path / "g.png"
        Image((torch.rand(1, 8, 8) * 255).to(torch.uint8)).save(path)

        assert Image.load(path).channels == 1

    def test_save_load_rgba_round_trip(self, tmp_path):
        # RGBA (4ch) も PNG で保存・読み戻しできる (load の UNCHANGED と対称)。
        img = Image((torch.rand(4, 8, 8) * 255).to(torch.uint8))
        path = tmp_path / "a.png"

        img.save(path)
        loaded = Image.load(path)

        assert loaded.channels == 4
        assert torch.equal(loaded.tensor, img.normalize(0, 255).uint8().tensor)

    def test_save_rgba_as_jpeg_raises(self, tmp_path):
        # JPEG は alpha 非対応。暗黙に PNG へ切り替えず明示的にエラーにする。
        with pytest.raises(ValueError, match="alpha"):
            Image(torch.zeros(4, 8, 8, dtype=torch.uint8)).save(tmp_path / "a.jpg")
