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

from exp.types.image import Image

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
