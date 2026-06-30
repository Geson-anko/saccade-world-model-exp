"""Behaviour spec for ``exp.models.encoder``.

Translates the approved spec for ``ImageEncoder`` into executable form.
``ImageEncoder`` encodes an observation glimpse sequence
``BatchedImageSequence (B, S, C, H, W)`` into one latent vector per
glimpse ``BatchedLatentSequence (B, S, latent_dim)`` via the
LeWorldModel path ViT [CLS] -> Linear -> BatchNorm.

These are integration-real tests: the real ViT / Linear / BatchNorm
stack runs on real CPU torch tensors with fixed seeds (no mocking of
torch internals, per the project testing strategy). Only public surface
is exercised — construction, the call contract, the ``latent_dim``
attribute, gradient flow, eval determinism, and validation delegated to
the underlying ViT.

The ViT architecture (embed_dim / depth / num_heads / patch size) is
fixed inside ``ImageEncoder``, so tests vary only the public knobs:
``image_size`` (a small multiple of the patch size, 16), ``in_channels``
and ``latent_dim``.
"""

import pytest
import torch

from exp.models.encoder import ImageEncoder
from exp.types.image import BatchedImageSequence
from exp.types.latent import BatchedLatentSequence

# Small but spec-valid config. image 32x32 with the fixed patch size 16
# gives a 2x2 grid; latent_dim 8 keeps tensors tiny.
_IMAGE_SIZE = 32
_IN_CHANNELS = 3
_LATENT_DIM = 8


def _make_encoder():
    return ImageEncoder(
        image_size=_IMAGE_SIZE,
        in_channels=_IN_CHANNELS,
        latent_dim=_LATENT_DIM,
    )


def _make_input(batch: int, seq: int) -> BatchedImageSequence:
    # B*S >= 2 keeps train-mode BatchNorm variance well defined.
    torch.manual_seed(0)
    return BatchedImageSequence(
        torch.randn(batch, seq, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE)
    )


class TestForwardShape:
    def test_returns_batched_latent_sequence(self):
        # The call contract's return type is the latent value object.
        encoder = _make_encoder()

        out = encoder(_make_input(batch=2, seq=3))

        assert type(out) is BatchedLatentSequence

    def test_collapses_glimpse_to_latent_vector(self):
        # (B, S, C, H, W) -> (B, S, latent_dim): one latent per glimpse.
        encoder = _make_encoder()

        out = encoder(_make_input(batch=2, seq=3))

        assert out.tensor.shape == (2, 3, _LATENT_DIM)


class TestPublicAttributes:
    def test_exposes_latent_dim(self):
        # latent_dim is public so downstream knows its input width.
        assert _make_encoder().latent_dim == _LATENT_DIM


class TestGradientFlow:
    def test_backward_reaches_vit(self):
        # The ViT trunk must receive gradient: at least one of its
        # parameters has a non-None grad after backward through the CLS path.
        encoder = _make_encoder()

        encoder(_make_input(batch=2, seq=3)).tensor.sum().backward()

        vit_grads = [p.grad for p in encoder.vit.parameters() if p.grad is not None]
        assert vit_grads, "no ViT parameter received a gradient"

    def test_backward_reaches_projection(self):
        # The Linear projection is on the path; its weight must get grad.
        encoder = _make_encoder()

        encoder(_make_input(batch=2, seq=3)).tensor.sum().backward()

        assert encoder.proj.weight.grad is not None

    def test_backward_reaches_batchnorm(self):
        # The final BatchNorm is on the path (regression guard for its
        # placement / reshape); its affine weight must get grad.
        encoder = _make_encoder()

        encoder(_make_input(batch=2, seq=3)).tensor.sum().backward()

        assert encoder.bn.weight.grad is not None


class TestEvalDeterminism:
    def test_repeated_eval_forward_matches(self):
        # In eval mode BatchNorm uses fixed running stats and dropout is off,
        # so the same input must yield the same latents across two forwards.
        encoder = _make_encoder()
        encoder.eval()
        x = _make_input(batch=2, seq=3)

        with torch.no_grad():
            first = encoder(x)
            second = encoder(x)

        torch.testing.assert_close(first.tensor, second.tensor)


class TestValidation:
    def test_image_size_not_divisible_by_patch_raises(self):
        # image_size 20 is not divisible by the fixed patch size 16; the
        # underlying ViT rejects it, and the error must propagate at
        # construction time (validation is delegated, not duplicated).
        with pytest.raises(ValueError):
            ImageEncoder(
                image_size=20, in_channels=_IN_CHANNELS, latent_dim=_LATENT_DIM
            )
