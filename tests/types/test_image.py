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

from exp.types.image import BatchedImageSequence, ChannelFormat, Image, ImageSequence


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


class TestImageSequenceConstruction:
    def test_rejects_3d_tensor(self):
        # A 3D tensor lacks the leading length axis required by (len, C, H, W).
        with pytest.raises(ValueError, match=r"len, C, H, W"):
            ImageSequence(torch.zeros(3, 8, 16))

    def test_rejects_5d_tensor(self):
        # A 5D tensor carries an extra batch axis the sequence forbids.
        with pytest.raises(ValueError, match=r"len, C, H, W"):
            ImageSequence(torch.zeros(2, 5, 3, 8, 16))

    def test_error_reports_actual_ndim_and_shape(self):
        # The message must surface the offending ndim and shape (substring only).
        with pytest.raises(ValueError) as exc:
            ImageSequence(torch.zeros(3, 8, 16))

        assert "len, C, H, W" in str(exc.value)
        assert "3" in str(exc.value)  # ndim
        assert "8" in str(exc.value) and "16" in str(exc.value)  # shape

    def test_accepts_4d_tensor(self):
        seq = ImageSequence(torch.zeros(5, 3, 8, 16))
        assert len(seq) == 5

    def test_accepts_empty_sequence(self):
        # An empty (0, C, H, W) sequence is a valid value, not an error.
        seq = ImageSequence(torch.zeros(0, 3, 8, 16))
        assert len(seq) == 0


class TestImageSequenceIndexing:
    def test_int_index_returns_image(self):
        seq = ImageSequence(torch.arange(5 * 3 * 8 * 16).reshape(5, 3, 8, 16))

        frame = seq[2]

        assert type(frame) is Image
        assert torch.equal(frame.tensor, seq.tensor[2])

    def test_negative_index_returns_last_image(self):
        seq = ImageSequence(torch.arange(5 * 3 * 8 * 16).reshape(5, 3, 8, 16))

        frame = seq[-1]

        assert type(frame) is Image
        assert torch.equal(frame.tensor, seq.tensor[4])

    def test_slice_returns_image_sequence(self):
        seq = ImageSequence(torch.arange(5 * 3 * 8 * 16).reshape(5, 3, 8, 16))

        sub = seq[1:3]

        assert type(sub) is ImageSequence
        assert len(sub) == 2
        assert torch.equal(sub.tensor[0], seq.tensor[1])
        assert torch.equal(sub.tensor[1], seq.tensor[2])

    def test_out_of_range_int_raises_index_error(self):
        seq = ImageSequence(torch.zeros(5, 3, 8, 16))
        with pytest.raises(IndexError):
            _ = seq[5]


class TestImageSequenceLen:
    def test_len_is_leading_axis(self):
        assert len(ImageSequence(torch.zeros(5, 3, 8, 16))) == 5

    def test_len_of_empty_is_zero(self):
        assert len(ImageSequence(torch.zeros(0, 3, 8, 16))) == 0


class TestImageSequenceIter:
    def test_iterates_frames_as_images(self):
        seq = ImageSequence(torch.arange(5 * 3 * 8 * 16).reshape(5, 3, 8, 16))

        frames = list(seq)

        assert len(frames) == 5
        assert all(type(f) is Image for f in frames)
        for i, frame in enumerate(frames):
            assert torch.equal(frame.tensor, seq.tensor[i])

    def test_iterates_empty_as_no_frames(self):
        assert list(ImageSequence(torch.zeros(0, 3, 8, 16))) == []


class TestImageSequenceDeviceTransfer:
    def test_device_reflects_tensor(self):
        assert ImageSequence(torch.zeros(5, 3, 8, 16)).device == torch.device("cpu")

    def test_device_available_for_empty_sequence(self):
        assert ImageSequence(torch.zeros(0, 3, 8, 16)).device == torch.device("cpu")

    def test_to_returns_new_sequence(self):
        # `to` is not in-place: it returns a fresh ImageSequence with equal values.
        seq = ImageSequence(torch.zeros(5, 3, 8, 16))

        result = seq.to("cpu")

        assert type(result) is ImageSequence
        assert result is not seq
        assert torch.equal(result.tensor, seq.tensor)

    def test_to_accepts_torch_device(self):
        seq = ImageSequence(torch.zeros(5, 3, 8, 16))

        result = seq.to(torch.device("cpu"))

        assert type(result) is ImageSequence
        assert torch.equal(result.tensor, seq.tensor)


class TestBatchedImageSequenceConstruction:
    def test_rejects_4d_tensor(self):
        # A 4D tensor lacks the leading batch axis required by (batch, len, C, H, W).
        with pytest.raises(ValueError, match=r"batch, len, C, H, W"):
            BatchedImageSequence(torch.zeros(5, 3, 8, 16))

    def test_rejects_6d_tensor(self):
        # A 6D tensor carries an extra axis the batched sequence forbids.
        with pytest.raises(ValueError, match=r"batch, len, C, H, W"):
            BatchedImageSequence(torch.zeros(2, 2, 5, 3, 8, 16))

    def test_error_reports_actual_ndim_and_shape(self):
        # The message must surface the offending ndim and shape (substring only).
        with pytest.raises(ValueError) as exc:
            BatchedImageSequence(torch.zeros(5, 3, 8, 16))

        assert "batch, len, C, H, W" in str(exc.value)
        assert "4" in str(exc.value)  # ndim
        assert "8" in str(exc.value) and "16" in str(exc.value)  # shape

    def test_accepts_5d_tensor(self):
        batch = BatchedImageSequence(torch.zeros(2, 5, 3, 8, 16))
        assert len(batch) == 2

    def test_accepts_empty_batch(self):
        # An empty (0, len, C, H, W) batch is a valid value, not an error.
        batch = BatchedImageSequence(torch.zeros(0, 5, 3, 8, 16))
        assert len(batch) == 0


class TestBatchedImageSequenceIndexing:
    def test_int_index_returns_image_sequence(self):
        batch = BatchedImageSequence(
            torch.arange(2 * 5 * 3 * 8 * 16).reshape(2, 5, 3, 8, 16)
        )

        seq = batch[1]

        assert type(seq) is ImageSequence
        assert torch.equal(seq.tensor, batch.tensor[1])

    def test_negative_index_returns_last_sequence(self):
        batch = BatchedImageSequence(
            torch.arange(2 * 5 * 3 * 8 * 16).reshape(2, 5, 3, 8, 16)
        )

        seq = batch[-1]

        assert type(seq) is ImageSequence
        assert torch.equal(seq.tensor, batch.tensor[1])

    def test_slice_returns_batched_image_sequence(self):
        batch = BatchedImageSequence(
            torch.arange(4 * 5 * 3 * 8 * 16).reshape(4, 5, 3, 8, 16)
        )

        sub = batch[1:3]

        assert type(sub) is BatchedImageSequence
        assert len(sub) == 2
        assert torch.equal(sub.tensor[0], batch.tensor[1])
        assert torch.equal(sub.tensor[1], batch.tensor[2])

    def test_out_of_range_int_raises_index_error(self):
        batch = BatchedImageSequence(torch.zeros(2, 5, 3, 8, 16))
        with pytest.raises(IndexError):
            _ = batch[2]


class TestBatchedImageSequenceLen:
    def test_len_is_leading_axis(self):
        assert len(BatchedImageSequence(torch.zeros(2, 5, 3, 8, 16))) == 2

    def test_len_of_empty_is_zero(self):
        assert len(BatchedImageSequence(torch.zeros(0, 5, 3, 8, 16))) == 0


class TestBatchedImageSequenceIter:
    def test_iterates_entries_as_image_sequences(self):
        batch = BatchedImageSequence(
            torch.arange(2 * 5 * 3 * 8 * 16).reshape(2, 5, 3, 8, 16)
        )

        entries = list(batch)

        assert len(entries) == 2
        assert all(type(e) is ImageSequence for e in entries)
        for i, entry in enumerate(entries):
            assert torch.equal(entry.tensor, batch.tensor[i])

    def test_iterates_empty_as_no_entries(self):
        assert list(BatchedImageSequence(torch.zeros(0, 5, 3, 8, 16))) == []


class TestBatchedImageSequenceDeviceTransfer:
    def test_device_reflects_tensor(self):
        assert BatchedImageSequence(torch.zeros(2, 5, 3, 8, 16)).device == torch.device(
            "cpu"
        )

    def test_device_available_for_empty_batch(self):
        assert BatchedImageSequence(torch.zeros(0, 5, 3, 8, 16)).device == torch.device(
            "cpu"
        )

    def test_to_returns_new_batch(self):
        # `to` is not in-place: it returns a fresh BatchedImageSequence.
        batch = BatchedImageSequence(torch.zeros(2, 5, 3, 8, 16))

        result = batch.to("cpu")

        assert type(result) is BatchedImageSequence
        assert result is not batch
        assert torch.equal(result.tensor, batch.tensor)

    def test_to_accepts_torch_device(self):
        batch = BatchedImageSequence(torch.zeros(2, 5, 3, 8, 16))

        result = batch.to(torch.device("cpu"))

        assert type(result) is BatchedImageSequence
        assert torch.equal(result.tensor, batch.tensor)


def _distinct_image(index: int) -> Image:
    # Build an Image whose pixels are unique to `index` so order-preservation
    # is observable: each (3, 8, 16) tensor is a contiguous arange offset by a
    # per-index stride, leaving no two frames equal.
    stride = 3 * 8 * 16
    return Image(torch.arange(index * stride, (index + 1) * stride).reshape(3, 8, 16))


def _distinct_sequence(index: int) -> ImageSequence:
    # Same idea at the sequence level: a unique (5, 3, 8, 16) block per index.
    stride = 5 * 3 * 8 * 16
    return ImageSequence(
        torch.arange(index * stride, (index + 1) * stride).reshape(5, 3, 8, 16)
    )


class TestImageSequenceFromImages:
    def test_stacks_multiple_images(self):
        # Three value-distinct Images stack into a (len, C, H, W) sequence; the
        # per-frame torch.equal checks pin both element identity and order.
        images = [_distinct_image(0), _distinct_image(1), _distinct_image(2)]

        result = ImageSequence.from_images(images)

        assert type(result) is ImageSequence
        assert len(result) == 3
        assert result.tensor.shape == (3, 3, 8, 16)
        assert torch.equal(result.tensor[0], images[0].tensor)
        assert torch.equal(result.tensor[1], images[1].tensor)
        assert torch.equal(result.tensor[2], images[2].tensor)

    def test_single_image(self):
        # The minimum non-empty input is one Image, yielding leading axis 1.
        image = _distinct_image(7)

        result = ImageSequence.from_images([image])

        assert type(result) is ImageSequence
        assert len(result) == 1
        assert result.tensor.shape == (1, 3, 8, 16)
        assert torch.equal(result.tensor[0], image.tensor)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one" contract (substring match only).
        with pytest.raises(ValueError, match="at least one"):
            ImageSequence.from_images([])

    def test_accepts_generator(self):
        # A one-shot generator (N>=1) must work: the empty check and the stack
        # both succeed, proving the input is materialized before being scanned.
        result = ImageSequence.from_images(_distinct_image(i) for i in range(2))

        assert type(result) is ImageSequence
        assert len(result) == 2
        assert result.tensor.shape == (2, 3, 8, 16)

    def test_preserves_dtype(self):
        # torch.stack carries the element dtype through; uint8 must survive.
        images = [Image(torch.zeros(3, 8, 16, dtype=torch.uint8)) for _ in range(2)]

        result = ImageSequence.from_images(images)

        assert result.tensor.dtype == torch.uint8


class TestBatchedImageSequenceFromSequences:
    def test_stacks_multiple_sequences(self):
        # Two value-distinct sequences stack into (batch, len, C, H, W); the
        # per-entry torch.equal checks pin both element identity and order.
        sequences = [_distinct_sequence(0), _distinct_sequence(1)]

        result = BatchedImageSequence.from_sequences(sequences)

        assert type(result) is BatchedImageSequence
        assert len(result) == 2
        assert result.tensor.shape == (2, 5, 3, 8, 16)
        assert torch.equal(result.tensor[0], sequences[0].tensor)
        assert torch.equal(result.tensor[1], sequences[1].tensor)

    def test_single_sequence(self):
        # The minimum non-empty input is one ImageSequence, yielding batch 1.
        sequence = _distinct_sequence(3)

        result = BatchedImageSequence.from_sequences([sequence])

        assert type(result) is BatchedImageSequence
        assert len(result) == 1
        assert result.tensor.shape == (1, 5, 3, 8, 16)
        assert torch.equal(result.tensor[0], sequence.tensor)

    def test_empty_raises_value_error(self):
        # Empty input violates the "at least one" contract (substring match only).
        with pytest.raises(ValueError, match="at least one"):
            BatchedImageSequence.from_sequences([])

    def test_accepts_generator(self):
        # A one-shot generator (N>=1) must work: materialization lets the empty
        # check and the stack both scan the same elements.
        result = BatchedImageSequence.from_sequences(
            _distinct_sequence(i) for i in range(2)
        )

        assert type(result) is BatchedImageSequence
        assert len(result) == 2
        assert result.tensor.shape == (2, 5, 3, 8, 16)
