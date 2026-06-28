"""Behaviour spec for ``exp.types.image.Image``.

These tests translate the approved spec for the immutable ``Image``
value object into executable form. They are written against the *spec*,
not any implementation: an implementation that diverges should make
these red.

Tensors are constructed deterministically on real CPU torch (no seed, no
mocking of torch) per the project testing strategy.
"""

import pytest
import torch

from exp.types.image import ChannelFormat, Image

# --- unit -----------------------------------------------------------------


def test_post_init_rejects_2d_tensor():
    # Arrange: a 2D tensor lacks the channel axis required by (C, H, W).
    not_chw = torch.zeros(8, 16)

    # Act / Assert: __attrs_post_init__ must reject non-3D input. The message
    # is checked by substring only (the exact wording is not part of the spec).
    with pytest.raises(ValueError, match="C, H, W"):
        Image(not_chw)


def test_post_init_rejects_4d_tensor():
    # Arrange: a 4D tensor carries a batch axis the value object forbids.
    batched = torch.zeros(1, 3, 8, 16)

    with pytest.raises(ValueError, match="C, H, W"):
        Image(batched)


def test_post_init_error_reports_actual_shape():
    # The error message must surface the offending shape so callers can
    # diagnose it. Substring check only -- not a full-message assertion.
    not_chw = torch.zeros(8, 16)

    with pytest.raises(ValueError) as exc:
        Image(not_chw)

    assert "C, H, W" in str(exc.value)
    # The actual shape (here a 2D tensor) is part of the contract.
    assert "8" in str(exc.value) and "16" in str(exc.value)


def test_dimension_properties_match_chw():
    # Non-symmetric shape so a wrong axis assignment (e.g. swapping H/W)
    # would be caught -- this pins the CHW axis mapping, not a trivial
    # getter round-trip.
    img = Image(torch.zeros(3, 8, 16))

    assert img.channels == 3
    assert img.height == 8
    assert img.width == 16


def test_eq_is_identity_for_same_instance():
    # eq=False contract: an instance compares equal only to itself.
    img = Image(torch.zeros(3, 8, 16))

    assert img == img


def test_eq_distinguishes_equal_valued_instances():
    # Two distinct instances holding equal values must NOT be equal
    # (identity semantics), and the comparison must not raise the
    # "boolean value of Tensor is ambiguous" error.
    first = Image(torch.zeros(3, 8, 16))
    second = Image(torch.zeros(3, 8, 16))

    assert first != second


# --- integration-real -----------------------------------------------------


def test_to_returns_new_image():
    # `to` is not in-place: it returns a fresh Image with equal values.
    img = Image(torch.zeros(3, 8, 16))

    result = img.to("cpu")

    assert isinstance(result, Image)
    assert result is not img
    assert torch.equal(result.tensor, img.tensor)


def test_to_accepts_torch_device():
    # DeviceLike is `torch.device | str`; the torch.device form must work too.
    img = Image(torch.zeros(3, 8, 16))

    result = img.to(torch.device("cpu"))

    assert isinstance(result, Image)
    assert torch.equal(result.tensor, img.tensor)


def test_device_property_reflects_tensor():
    # The device property reports the underlying tensor's device.
    img = Image(torch.zeros(3, 8, 16))

    assert img.device == torch.device("cpu")


# --- added properties ------------------------------------------------------


def test_size_returns_height_width():
    # size pins (height, width) order, matching Size2d / resize.
    assert Image(torch.zeros(3, 8, 16)).size == (8, 16)


def test_channel_format_inferred_from_channels():
    assert Image(torch.zeros(1, 4, 4)).channel_format is ChannelFormat.GRAY
    assert Image(torch.zeros(3, 4, 4)).channel_format is ChannelFormat.RGB
    assert Image(torch.zeros(4, 4, 4)).channel_format is ChannelFormat.RGBA


def test_channel_format_rejects_unsupported_channel_count():
    # A 2-channel tensor maps to no ChannelFormat member.
    with pytest.raises(ValueError):
        _ = Image(torch.zeros(2, 4, 4)).channel_format


def test_is_squared():
    assert Image(torch.zeros(3, 8, 8)).is_squared
    assert not Image(torch.zeros(3, 8, 16)).is_squared


# --- ChannelFormat enum ----------------------------------------------------


def test_channel_format_values_are_channel_counts():
    assert ChannelFormat(1) is ChannelFormat.GRAY
    assert ChannelFormat(3) is ChannelFormat.RGB
    assert ChannelFormat(4) is ChannelFormat.RGBA


def test_channel_format_invalid_value_raises():
    with pytest.raises(ValueError):
        ChannelFormat(2)


# --- dtype casts -----------------------------------------------------------


def test_float_cast():
    out = Image(torch.zeros(3, 4, 4, dtype=torch.uint8)).float()
    assert out.tensor.dtype == torch.float32


def test_uint8_cast():
    out = Image(torch.ones(3, 4, 4)).uint8()
    assert out.tensor.dtype == torch.uint8


# --- standardize / normalize ----------------------------------------------


def test_standardize_sets_mean_and_std():
    torch.manual_seed(0)
    img = Image(torch.randn(3, 16, 16) * 5 + 2)

    out = img.standardize(mean=1.0, std=2.0)

    torch.testing.assert_close(out.tensor.mean(), torch.tensor(1.0), atol=1e-4, rtol=0)
    torch.testing.assert_close(out.tensor.std(), torch.tensor(2.0), atol=1e-4, rtol=0)


def test_standardize_constant_image_has_no_nan():
    # Zero-variance input must not divide by zero; all pixels become `mean`.
    out = Image(torch.full((3, 4, 4), 7.0)).standardize(mean=3.0, std=2.0)

    assert torch.isfinite(out.tensor).all()
    torch.testing.assert_close(out.tensor, torch.full((3, 4, 4), 3.0))


def test_normalize_sets_min_and_max():
    img = Image(torch.tensor([[[0.0, 2.0], [4.0, 8.0]]]))

    out = img.normalize(min=0.0, max=1.0)

    torch.testing.assert_close(out.tensor.min(), torch.tensor(0.0))
    torch.testing.assert_close(out.tensor.max(), torch.tensor(1.0))


def test_normalize_constant_image_has_no_div_by_zero():
    out = Image(torch.full((1, 4, 4), 5.0)).normalize(min=0.0, max=255.0)

    assert torch.isfinite(out.tensor).all()
    torch.testing.assert_close(out.tensor, torch.zeros(1, 4, 4))


# --- as_channel_format -----------------------------------------------------


def test_as_channel_format_same_is_identity():
    img = Image(torch.zeros(3, 4, 4))
    assert img.as_channel_format(ChannelFormat.RGB) is img


def test_as_channel_format_gray_to_rgb_replicates():
    img = Image(torch.arange(16, dtype=torch.uint8).reshape(1, 4, 4))

    rgb = img.as_channel_format(ChannelFormat.RGB)

    assert rgb.channels == 3
    for c in range(3):
        assert torch.equal(rgb.tensor[c], img.tensor[0])


def test_as_channel_format_rgb_to_rgba_adds_opaque_alpha():
    rgba = Image(torch.zeros(3, 4, 4, dtype=torch.uint8)).as_channel_format(
        ChannelFormat.RGBA
    )

    assert rgba.channels == 4
    assert torch.equal(rgba.tensor[3], torch.full((4, 4), 255, dtype=torch.uint8))


def test_as_channel_format_rgba_to_rgb_drops_alpha():
    rgb = Image(torch.zeros(4, 4, 4, dtype=torch.uint8)).as_channel_format(
        ChannelFormat.RGB
    )
    assert rgb.channels == 3


def test_as_channel_format_rgb_to_gray():
    gray = Image(torch.zeros(3, 4, 4, dtype=torch.uint8)).as_channel_format(
        ChannelFormat.GRAY
    )
    assert gray.channels == 1


# --- square_pad ------------------------------------------------------------


def test_square_pad_makes_squared():
    out = Image(torch.ones(3, 4, 8)).square_pad()

    assert out.is_squared
    assert out.size == (8, 8)


def test_square_pad_centers_data_and_fills_border():
    # H=2, W=4: pad 1px top and bottom (diff=2, symmetric).
    out = Image(torch.ones(1, 2, 4)).square_pad(fill_value=0)

    assert out.size == (4, 4)
    assert torch.equal(out.tensor[0, 1:3, :], torch.ones(2, 4))
    assert torch.equal(out.tensor[0, 0, :], torch.zeros(4))
    assert torch.equal(out.tensor[0, 3, :], torch.zeros(4))


def test_square_pad_noop_when_squared():
    img = Image(torch.ones(3, 5, 5))
    assert img.square_pad() is img


# --- focus -----------------------------------------------------------------


def test_focus_full_view_returns_whole_image():
    # zoom=1, point=center on a square image returns the image unchanged.
    img = Image(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4))

    out = img.focus((0.0, 0.0), 1.0)

    assert torch.equal(out.tensor, img.tensor)


def test_focus_output_side_is_round_zoom_times_side():
    out = Image(torch.zeros(3, 8, 8)).focus((0.0, 0.0), 0.5)

    assert out.is_squared
    assert out.size == (4, 4)


def test_focus_squares_non_square_input_first():
    # (4, 8) squares to side 8, so a full-zoom focus yields 8x8.
    out = Image(torch.zeros(3, 4, 8)).focus((0.0, 0.0), 1.0)
    assert out.size == (8, 8)


def test_focus_corner_pads_out_of_bounds():
    # Bottom-right gaze with partial zoom spills past the edge -> zero pad.
    out = Image(torch.ones(1, 4, 4)).focus((1.0, 1.0), 0.5)

    assert out.size == (2, 2)
    assert (out.tensor == 0).any()


def test_focus_rejects_out_of_range_point():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        Image(torch.zeros(3, 4, 4)).focus((1.5, 0.0), 1.0)


def test_focus_rejects_out_of_range_zoom():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        Image(torch.zeros(3, 4, 4)).focus((0.0, 0.0), 0.0)


# --- resize ----------------------------------------------------------------


def test_resize_int_to_square():
    assert Image(torch.zeros(3, 4, 8)).resize(16).size == (16, 16)


def test_resize_tuple_is_hw():
    assert Image(torch.zeros(3, 4, 8)).resize((6, 10)).size == (6, 10)


def test_resize_preserves_dtype():
    out = Image(torch.zeros(3, 4, 4, dtype=torch.uint8)).resize(8)
    assert out.tensor.dtype == torch.uint8


# --- integration-real: load / save -----------------------------------------


def test_save_load_png_round_trip(tmp_path):
    # save normalizes to [0,255] then writes uint8; PNG decode is lossless,
    # so load must return exactly the saved (normalized) tensor.
    img = Image((torch.rand(3, 8, 8) * 255).to(torch.uint8))
    path = tmp_path / "img.png"

    img.save(path)
    loaded = Image.load(path)

    expected = img.normalize(0, 255).uint8().tensor
    assert loaded.tensor.dtype == torch.uint8
    assert loaded.tensor.shape == (3, 8, 8)
    assert torch.equal(loaded.tensor, expected)


def test_load_preserves_original_channels(tmp_path):
    path = tmp_path / "g.png"
    Image((torch.rand(1, 8, 8) * 255).to(torch.uint8)).save(path)

    assert Image.load(path).channels == 1


def test_save_rejects_rgba(tmp_path):
    with pytest.raises(ValueError, match="channel"):
        Image(torch.zeros(4, 4, 4, dtype=torch.uint8)).save(tmp_path / "a.png")
