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
from exp.types import (
    BatchedImage,
    BatchedImageSequence,
    BatchedLatent,
    BatchedLatentSequence,
    Image,
    ImageSequence,
    Latent,
    LatentSequence,
)
from tests.helpers import parametrize_device

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


def _make_image() -> Image:
    # A single glimpse (C, H, W); reshape collapses it to N=1 rows.
    torch.manual_seed(0)
    return Image(torch.randn(_IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE))


def _make_batched_image(batch: int) -> BatchedImage:
    # A batch of glimpses (batch, C, H, W); reshape collapses it to N=batch.
    torch.manual_seed(0)
    return BatchedImage(torch.randn(batch, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE))


def _make_image_sequence(seq: int) -> ImageSequence:
    # A single glimpse sequence (len, C, H, W); reshape collapses it to N=len.
    torch.manual_seed(0)
    return ImageSequence(torch.randn(seq, _IN_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE))


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


class TestImageEncoderDevice:
    # Smoke test: the ViT -> Linear -> BatchNorm path must run on each device
    # and return a latent whose tensor lands there.

    @parametrize_device
    def test_forward_output_on_device(self, device: str):
        encoder = _make_encoder().to(device)
        x = _make_input(batch=2, seq=3).to(device)

        out = encoder(x)

        assert out.tensor.device == torch.device(device)


class TestRankPreservingReturnType:
    # The call contract maps each input rank to the matching latent value
    # object: the encoder must not up- or down-rank its output. Each glimpse
    # of every rank still collapses to one latent vector. Run in eval() so the
    # Image (N=1) case does not hit the train-mode BatchNorm constraint, which
    # has its own dedicated test below.

    def test_image_returns_latent(self):
        encoder = _make_encoder().eval()

        out = encoder(_make_image())

        assert type(out) is Latent

    def test_batched_image_returns_batched_latent(self):
        encoder = _make_encoder().eval()

        out = encoder(_make_batched_image(batch=2))

        assert type(out) is BatchedLatent

    def test_image_sequence_returns_latent_sequence(self):
        encoder = _make_encoder().eval()

        out = encoder(_make_image_sequence(seq=3))

        assert type(out) is LatentSequence


class TestRankPreservingShape:
    # Each rank keeps its leading axes and replaces (C, H, W) with
    # (latent_dim,): one latent per glimpse. (batch, seq) is already covered
    # by TestForwardShape. Run in eval() for the same N=1 reason as above.

    def test_image_shape(self):
        # (C, H, W) -> (latent_dim,)
        encoder = _make_encoder().eval()

        out = encoder(_make_image())

        assert out.tensor.shape == (_LATENT_DIM,)

    def test_batched_image_shape(self):
        # (batch, C, H, W) -> (batch, latent_dim)
        encoder = _make_encoder().eval()

        out = encoder(_make_batched_image(batch=2))

        assert out.tensor.shape == (2, _LATENT_DIM)

    def test_image_sequence_shape(self):
        # (len, C, H, W) -> (seq, latent_dim)
        encoder = _make_encoder().eval()

        out = encoder(_make_image_sequence(seq=3))

        assert out.tensor.shape == (3, _LATENT_DIM)


class TestNonBatchedGradientFlow:
    # The reshape-based generalization must not sever the gradient path for
    # the newly supported ranks. One representative rank (ImageSequence, which
    # gives N=seq>=2 rows so train-mode BatchNorm is well defined) exercises
    # the whole ViT -> Linear -> BatchNorm chain; the per-layer coverage for
    # the (batch, seq) path already lives in TestGradientFlow.

    def test_backward_reaches_vit(self):
        encoder = _make_encoder()

        encoder(_make_image_sequence(seq=3)).tensor.sum().backward()

        vit_grads = [p.grad for p in encoder.vit.parameters() if p.grad is not None]
        assert vit_grads, "no ViT parameter received a gradient"

    def test_backward_reaches_projection(self):
        encoder = _make_encoder()

        encoder(_make_image_sequence(seq=3)).tensor.sum().backward()

        assert encoder.proj.weight.grad is not None

    def test_backward_reaches_batchnorm(self):
        encoder = _make_encoder()

        encoder(_make_image_sequence(seq=3)).tensor.sum().backward()

        assert encoder.bn.weight.grad is not None


class TestSingleImageBatchNormConstraint:
    # Documented, intentionally-unchanged constraint: a lone Image collapses to
    # N=1 rows, and BatchNorm1d cannot compute batch variance over a single
    # sample in train mode. This guards that the reshape generalization does
    # NOT silently paper over the N=1 case.

    def test_single_image_in_train_mode_raises(self):
        encoder = _make_encoder()  # train mode is the default

        with pytest.raises(ValueError):
            encoder(_make_image())

    def test_image_sequence_in_train_mode_is_allowed(self):
        # Contrast: a length>=2 sequence gives N>=2 rows, so train-mode
        # BatchNorm variance is well defined and the same path succeeds.
        encoder = _make_encoder()  # train mode is the default

        out = encoder(_make_image_sequence(seq=2))

        assert out.tensor.shape == (2, _LATENT_DIM)

    def test_single_image_in_eval_mode_is_allowed(self):
        # Contrast: eval mode uses fixed running stats instead of per-batch
        # variance, so the same lone Image passes through fine.
        encoder = _make_encoder().eval()

        with torch.no_grad():
            out = encoder(_make_image())

        assert out.tensor.shape == (_LATENT_DIM,)
