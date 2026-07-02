"""Behaviour spec for ``exp.data.dataloader``.

``collate_glimpses`` / ``GlimpseDataLoader`` turn per-episode samples
into batched value objects. Tests exercise only the public surface
(``from exp.data import ...``) with real image files under ``tmp_path``
and real CPU torch / DataLoader machinery (integration-real, no
mocking).
"""

import pytest
import torch

from exp.data import (
    GlimpseDataLoader,
    GlimpseDataset,
    collate_glimpses,
)
from exp.types import (
    BatchedFocusSequence,
    BatchedImageSequence,
)
from exp.types.elements.image import Image


def _write_image(path, *, seed: int = 0) -> None:
    # Write a real 3x12x12 PNG at `path` with seeded deterministic pixels.
    path.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    tensor = (torch.rand(3, 12, 12, generator=gen) * 255).to(torch.uint8)
    Image(tensor).save(path)


def _dataset(root, count: int = 4) -> GlimpseDataset:
    for i in range(count):
        _write_image(root / f"{i}.png", seed=i)
    return GlimpseDataset(root, image_size=8, seq_len=4)


class TestCollateGlimpses:
    # integration-real: real image files written under tmp_path.

    def test_stacks_hand_built_batch_on_batch_axis(self, tmp_path):
        # Two (FocusSequence, ImageSequence) items stack into a batch axis:
        # (B, S, 3) actions and (B, S, 3, s, s) observations.
        ds = _dataset(tmp_path, count=2)

        batch = [ds[0], ds[1]]
        batched_focuses, batched_observations = collate_glimpses(batch)

        assert type(batched_focuses) is BatchedFocusSequence
        assert type(batched_observations) is BatchedImageSequence
        assert batched_focuses.tensor.shape == (2, 4, 3)
        assert batched_observations.tensor.shape == (2, 4, 3, 8, 8)

    def test_empty_batch_raises_value_error(self):
        # An empty batch has no sequences to stack (substring match only).
        with pytest.raises(ValueError, match="at least one"):
            collate_glimpses([])


class TestGlimpseDataLoader:
    # integration-real: real image files + real DataLoader iteration.

    def test_yields_batched_value_objects_without_explicit_collate(self, tmp_path):
        # The whole point of GlimpseDataLoader: no collate_fn= at the call
        # site, yet batches come out as the batched value objects.
        ds = _dataset(tmp_path)
        loader = GlimpseDataLoader(ds, batch_size=2)

        batched_focuses, batched_observations = next(iter(loader))

        assert type(batched_focuses) is BatchedFocusSequence
        assert type(batched_observations) is BatchedImageSequence
        assert batched_focuses.tensor.shape == (2, 4, 3)
        assert batched_observations.tensor.shape == (2, 4, 3, 8, 8)

    def test_full_iteration_covers_dataset(self, tmp_path):
        # 4 episodes at batch_size=2 -> exactly 2 batches, 4 episodes total.
        ds = _dataset(tmp_path, count=4)
        loader = GlimpseDataLoader(ds, batch_size=2)

        batch_sizes = [focuses.tensor.shape[0] for focuses, _ in loader]

        assert batch_sizes == [2, 2]

    def test_passing_collate_fn_is_rejected(self, tmp_path):
        # collate_fn is pinned to collate_glimpses; callers cannot replace it.
        ds = _dataset(tmp_path, count=2)

        with pytest.raises(TypeError, match="collate_fn"):
            GlimpseDataLoader(ds, batch_size=2, collate_fn=collate_glimpses)
