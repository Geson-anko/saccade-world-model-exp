"""Behaviour spec for ``exp.data.dataset``.

These tests translate the approved spec for the glimpse data pipeline
(``random_focus_sequence`` / ``GlimpseDataset``) into executable form.
They are written against the *spec*, not any implementation: an
implementation that diverges should make these red. Scenarios are
grouped into one class per public function / behaviour area.

Only the public surface (``from exp.data import ...``) is exercised;
path collection is verified through ``len`` and determinism, never by
peeking at private attributes. Datasets read real image files written
under ``tmp_path`` and run through real torchvision I/O and real CPU
torch (no mocking) per the project testing strategy (integration-real).
Randomness is made deterministic via seeded ``torch.Generator``.
"""

import pytest
import torch

from exp.data import GlimpseDataset, random_focus_sequence
from exp.types import (
    FocusSequence,
    ImageSequence,
)
from exp.types.elements.image import Image


def _write_image(
    path, *, channels: int = 3, height: int = 12, width: int = 12, seed: int = 0
) -> None:
    # Write a real image file at `path` (extension decides the format). A
    # seeded generator keeps the pixel content deterministic so determinism
    # tests compare like with like.
    path.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    tensor = (torch.rand(channels, height, width, generator=gen) * 255).to(torch.uint8)
    Image(tensor).save(path)


def _populated_dir(root):
    # A small on-disk image corpus: two files at the top level, two in a
    # subdirectory (recursive collection + mixed / uppercase suffixes), and
    # one non-image file that must be ignored. Returns the image count.
    _write_image(root / "a.png", seed=1)
    _write_image(root / "b.jpg", seed=2)
    _write_image(root / "sub" / "c.jpeg", seed=3)
    _write_image(root / "sub" / "d.PNG", seed=4)
    (root / "notes.txt").write_text("not an image")
    return 4


class TestRandomFocusSequence:
    def test_shape_and_dtype(self):
        # A length-5 request yields a (5, 3) float32 [x, y, zoom] sequence.
        seq = random_focus_sequence(5)

        assert type(seq) is FocusSequence
        assert seq.tensor.shape == (5, 3)
        assert seq.tensor.dtype == torch.float32

    def test_large_sample_stays_in_valid_range(self):
        # x, y in [-1, 1] and zoom in [0, 1]: a large sample must satisfy the
        # public range contract (is_valid), catching a bad scale/offset.
        seq = random_focus_sequence(1000, torch.Generator().manual_seed(0))

        assert seq.is_valid()

    def test_same_seed_generators_produce_identical_sequences(self):
        # Reproducibility contract: two generators seeded alike draw the same
        # action sequence.
        a = random_focus_sequence(8, torch.Generator().manual_seed(42))
        b = random_focus_sequence(8, torch.Generator().manual_seed(42))

        assert torch.equal(a.tensor, b.tensor)

    def test_consecutive_draws_from_one_generator_differ(self):
        # A single generator advances between draws, so back-to-back calls
        # yield different action sequences (not frozen output).
        gen = torch.Generator().manual_seed(42)

        first = random_focus_sequence(8, gen)
        second = random_focus_sequence(8, gen)

        assert not torch.equal(first.tensor, second.tensor)

    def test_point_components_take_negative_values(self):
        # x and y are sampled over [-1, 1], not [0, 1]: a large sample must
        # contain negatives in BOTH point columns. This kills a [0, 1] scale
        # mistakenly applied to the point.
        seq = random_focus_sequence(2000, torch.Generator().manual_seed(0))

        assert (seq.tensor[:, 0] < 0).any()
        assert (seq.tensor[:, 1] < 0).any()

    def test_zoom_never_negative(self):
        # zoom is sampled over [0, 1]: a large sample must have no negatives in
        # the zoom column, pinning the asymmetric offset (point vs zoom).
        seq = random_focus_sequence(2000, torch.Generator().manual_seed(0))

        assert (seq.tensor[:, 2] >= 0).all()


class TestGlimpseDatasetInit:
    # integration-real: real image files written under tmp_path.

    def test_collects_images_recursively_ignoring_non_images(self, tmp_path):
        # png / jpg / jpeg (incl. uppercase .PNG) under nested dirs are
        # collected; notes.txt is ignored. Verified via len (no private peek).
        expected = _populated_dir(tmp_path)

        ds = GlimpseDataset(tmp_path, image_size=8, seq_len=3)

        assert len(ds) == expected

    def test_empty_dir_raises_value_error(self, tmp_path):
        # A directory with no image files violates the "at least one image"
        # precondition (substring match only).
        with pytest.raises(ValueError, match="no image files"):
            GlimpseDataset(tmp_path, image_size=8, seq_len=3)

    def test_seq_len_zero_raises_value_error(self, tmp_path):
        # seq_len must be >= 1 (t=1 carries no information); 0 is rejected.
        _write_image(tmp_path / "a.png", seed=1)

        with pytest.raises(ValueError, match="seq_len"):
            GlimpseDataset(tmp_path, image_size=8, seq_len=0)


class TestGetItem:
    # integration-real: real image files written under tmp_path.

    def test_returns_focus_and_image_sequence_with_int_size(self, tmp_path):
        # ds[i] -> (FocusSequence (S, 3), ImageSequence (S, 3, s, s)); int
        # image_size means square h == w == s, both float32.
        _write_image(tmp_path / "a.png", channels=3, height=20, width=30, seed=1)
        ds = GlimpseDataset(tmp_path, image_size=8, seq_len=4)

        focuses, observations = ds[0]

        assert type(focuses) is FocusSequence
        assert type(observations) is ImageSequence
        assert focuses.tensor.shape == (4, 3)
        assert focuses.tensor.dtype == torch.float32
        assert observations.tensor.shape == (4, 3, 8, 8)
        assert observations.tensor.dtype == torch.float32

    def test_tuple_size_yields_non_square_observations(self, tmp_path):
        # A (h, w) image_size crops each observation to exactly (h, w).
        _write_image(tmp_path / "a.png", channels=3, height=20, width=30, seed=1)
        ds = GlimpseDataset(tmp_path, image_size=(16, 24), seq_len=3)

        _, observations = ds[0]

        assert observations.tensor.shape == (3, 3, 16, 24)

    def test_grayscale_non_square_source_yields_three_channel_observations(
        self, tmp_path
    ):
        # A 1-channel non-square source is promoted to RGB before cropping, so
        # observations always carry 3 channels regardless of source format.
        _write_image(tmp_path / "gray.png", channels=1, height=18, width=25, seed=1)
        ds = GlimpseDataset(tmp_path, image_size=8, seq_len=3)

        _, observations = ds[0]

        assert observations.tensor.shape == (3, 3, 8, 8)

    def test_minimal_single_image_single_step(self, tmp_path):
        # The smallest valid episode: one image, seq_len=1.
        _write_image(tmp_path / "a.png", seed=1)
        ds = GlimpseDataset(tmp_path, image_size=8, seq_len=1)

        focuses, observations = ds[0]

        assert len(ds) == 1
        assert focuses.tensor.shape == (1, 3)
        assert observations.tensor.shape == (1, 3, 8, 8)

    def test_same_seed_datasets_produce_identical_items(self, tmp_path):
        # Reproducibility contract at the dataset level: same directory + same
        # seeded generator => ds[0] returns identical actions AND observations.
        # This also verifies deterministic, order-stable path collection.
        _populated_dir(tmp_path)
        ds_a = GlimpseDataset(
            tmp_path,
            image_size=8,
            seq_len=4,
            generator=torch.Generator().manual_seed(7),
        )
        ds_b = GlimpseDataset(
            tmp_path,
            image_size=8,
            seq_len=4,
            generator=torch.Generator().manual_seed(7),
        )

        focuses_a, obs_a = ds_a[0]
        focuses_b, obs_b = ds_b[0]

        assert torch.equal(focuses_a.tensor, focuses_b.tensor)
        assert torch.equal(obs_a.tensor, obs_b.tensor)
