"""Public API contract pins for the ``exp`` package.

IMPORTANT: every test in this file is a *contract pin*, NOT a behaviour
test. Public names, base classes, and type aliases re-exported from
``exp`` are depended on by external callers, so we fix them deliberately
to catch accidental renames / removals (Hyrum's law mitigation). These
would otherwise count as tautological tests and live nowhere else.
"""

import pytest

import exp


@pytest.mark.api_contract
def test_exp_exports():
    # Contract pin (not a behaviour test): the public export surface of `exp`
    # is exactly these symbols, and each is actually present.
    expected = {
        "BatchedFocus",
        "BatchedFocusSequence",
        "BatchedImage",
        "BatchedImageSequence",
        "BatchedLatent",
        "BatchedLatentSequence",
        "ChannelFormat",
        "DeviceLike",
        "DeviceTransferMixin",
        "Focus",
        "FocusSequence",
        "Image",
        "ImageSequence",
        "Latent",
        "LatentSequence",
        "ScalarTensor",
        "Size2d",
        "SupportsDeviceTransfer",
        "size_2d_to_tuple",
    }

    assert set(exp.__all__) == expected
    for name in expected:
        assert hasattr(exp, name)

    # Contract pin: internal base 系 (Element / ElementArray / エイリアス /
    # BatchedElementSequence) は公開しない。exp からは import できないこと。
    for base_name in (
        "Element",
        "ElementArray",
        "ElementSequence",
        "BatchedElement",
        "BatchedElementSequence",
    ):
        assert not hasattr(exp, base_name)


@pytest.mark.api_contract
def test_image_is_device_transfer_mixin():
    # Contract pin (not a behaviour test): Image's public base-class
    # relationship is part of the API surface external code may rely on.
    assert issubclass(exp.Image, exp.DeviceTransferMixin)


@pytest.mark.api_contract
def test_sequence_types_are_device_transfer_mixins():
    # Contract pin (not a behaviour test): all element value objects share
    # Image's DeviceTransferMixin base, which external code may rely on.
    for cls in (
        exp.Focus,
        exp.Latent,
        exp.ImageSequence,
        exp.FocusSequence,
        exp.LatentSequence,
        exp.BatchedImage,
        exp.BatchedFocus,
        exp.BatchedLatent,
        exp.BatchedImageSequence,
        exp.BatchedFocusSequence,
        exp.BatchedLatentSequence,
    ):
        assert issubclass(cls, exp.DeviceTransferMixin)
