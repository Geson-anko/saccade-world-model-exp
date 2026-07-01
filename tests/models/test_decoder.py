"""Behaviour spec for ``exp.models.decoder``.

Translates the approved public contract for ``ImageDecoder`` into
executable form. ``ImageDecoder`` is the mirror image of ``ImageEncoder``:
it decodes a latent (sequence) ``Latent`` family value object back into an
image (sequence) ``Image`` family value object via a CNN decoder trunk
(``ConvDecoder``: nearest-upsample + conv). It is the visualization /
evaluation read-out described in CLAUDE.md decision #6.

These are integration-real tests: the real upsample+conv decoder stack runs
on real CPU torch tensors with fixed seeds (no mocking of torch internals,
per the project testing strategy). Only the public surface is exercised —
construction / validation, the call contract (rank-preserving type and
shape), the ``latent_dim`` attribute, gradient flow into the decoder's own
parameters, the detach contract (a public pin), eval determinism, and
device placement.

The internal decoder architecture (``ConvDecoder`` and its ``_`` layers)
is never imported or asserted on directly; the contract is verified only
through ``ImageDecoder``.

Config notes: the output side length must be ``init_spatial * 2**N`` for a
non-negative integer ``N`` (each upsample stage doubles the spatial size).
``_OUT_SIZE = 32 = 4 * 2**3`` with ``init_spatial=4`` satisfies this.
``out_channels`` must be a ChannelFormat value (1, 3 or 4).
"""

import pytest
import torch

from exp.models.decoder import ImageDecoder
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

# Small but spec-valid config. latent_dim 8 keeps tensors tiny; out_channels
# 3 is a valid ChannelFormat (RGB); output side 32 == init_spatial(4) * 2**3
# so the 2-power upsample constraint is satisfied. base_channels 16 keeps the
# conv trunk cheap.
_LATENT_DIM = 8
_OUT_CHANNELS = 3
_OUT_SIZE = 32


def _make_decoder():
    return ImageDecoder(
        latent_dim=_LATENT_DIM,
        out_channels=_OUT_CHANNELS,
        image_size=_OUT_SIZE,
        base_channels=16,
        init_spatial=4,
    )


def _latent() -> Latent:
    # A single latent (dim,); decodes to one image (C, s', s').
    torch.manual_seed(0)
    return Latent(torch.randn(_LATENT_DIM))


def _batched_latent(b: int) -> BatchedLatent:
    # A batch of latents (b, dim); decodes to (b, C, s', s').
    torch.manual_seed(0)
    return BatchedLatent(torch.randn(b, _LATENT_DIM))


def _latent_sequence(s: int) -> LatentSequence:
    # A single latent sequence (s, dim); decodes to (s, C, s', s').
    torch.manual_seed(0)
    return LatentSequence(torch.randn(s, _LATENT_DIM))


def _batched_latent_sequence(b: int, s: int) -> BatchedLatentSequence:
    # (b, s, dim); decodes to (b, s, C, s', s').
    torch.manual_seed(0)
    return BatchedLatentSequence(torch.randn(b, s, _LATENT_DIM))


class TestForwardShape:
    def test_returns_batched_image_sequence(self):
        # The call contract's return type for the full-rank input is the
        # image sequence value object.
        decoder = _make_decoder()

        out = decoder(_batched_latent_sequence(b=2, s=3))

        assert type(out) is BatchedImageSequence

    def test_expands_latent_to_image(self):
        # (B, S, dim) -> (B, S, C, s', s'): one image per latent.
        decoder = _make_decoder()

        out = decoder(_batched_latent_sequence(b=2, s=3))

        assert out.tensor.shape == (2, 3, _OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)


class TestRankPreservingReturnType:
    # The call contract maps each input rank to the matching image value
    # object: the decoder must not up- or down-rank its output. Each latent of
    # every rank expands to one image.

    def test_latent_returns_image_in_train_mode(self):
        # Regression guard: unlike ImageEncoder's BatchNorm, ImageDecoder's
        # internal normalization is batch-independent (GroupNorm), so a lone
        # Latent (N=1) must pass in the default train mode without raising.
        decoder = _make_decoder()  # train mode is the default

        out = decoder(_latent())

        assert type(out) is Image

    def test_batched_latent_returns_batched_image(self):
        decoder = _make_decoder()

        out = decoder(_batched_latent(b=2))

        assert type(out) is BatchedImage

    def test_latent_sequence_returns_image_sequence(self):
        decoder = _make_decoder()

        out = decoder(_latent_sequence(s=3))

        assert type(out) is ImageSequence


class TestRankPreservingShape:
    # Each rank keeps its leading axes and replaces (dim,) with
    # (C, s', s'): one image per latent. The (batch, seq) rank is already
    # covered by TestForwardShape.

    def test_latent_shape(self):
        # (dim,) -> (C, s', s')
        decoder = _make_decoder()

        out = decoder(_latent())

        assert out.tensor.shape == (_OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)

    def test_batched_latent_shape(self):
        # (b, dim) -> (b, C, s', s')
        decoder = _make_decoder()

        out = decoder(_batched_latent(b=2))

        assert out.tensor.shape == (2, _OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)

    def test_latent_sequence_shape(self):
        # (s, dim) -> (s, C, s', s')
        decoder = _make_decoder()

        out = decoder(_latent_sequence(s=3))

        assert out.tensor.shape == (3, _OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)


class TestPublicAttributes:
    def test_exposes_latent_dim(self):
        # latent_dim is public so upstream knows the width it must feed.
        assert _make_decoder().latent_dim == _LATENT_DIM


class TestGradientFlow:
    def test_backward_reaches_decoder_parameters(self):
        # The decoder trunk (Linear / conv-transpose) must receive gradient:
        # at least one of the decoder's own learnable parameters has a
        # non-None grad after backward through the full-rank path.
        decoder = _make_decoder()

        decoder(_batched_latent_sequence(b=2, s=3)).tensor.sum().backward()

        grads = [p.grad for p in decoder.parameters() if p.grad is not None]
        assert grads, "no decoder parameter received a gradient"


class TestDetachContract:
    # PUBLIC CONTRACT PIN — this is not a behaviour test. It fixes the
    # decision that ImageDecoder is a pure function that does NOT detach its
    # input: severing the upstream gradient path is the caller's
    # responsibility (CLAUDE.md #6: the decoder is a detached read-out, but
    # the detach is applied by the caller, not baked into the decoder). If a
    # future refactor makes the decoder detach internally, this pin fails and
    # forces a conscious re-decision.

    def test_input_gradient_is_not_detached(self):
        decoder = _make_decoder()
        torch.manual_seed(0)
        t = torch.randn(2, 3, _LATENT_DIM).requires_grad_(True)

        decoder(BatchedLatentSequence(t)).tensor.sum().backward()

        assert t.grad is not None


class TestEvalDeterminism:
    def test_repeated_eval_forward_matches(self):
        # In eval mode normalization uses fixed stats and any dropout is off,
        # so the same input must yield the same image across two forwards.
        decoder = _make_decoder()
        decoder.eval()
        x = _batched_latent_sequence(b=2, s=3)

        with torch.no_grad():
            first = decoder(x)
            second = decoder(x)

        torch.testing.assert_close(first.tensor, second.tensor)


class TestValidation:
    def test_image_size_not_power_of_two_multiple_raises(self):
        # 20 cannot be written as init_spatial(4) * 2**N for integer N, so the
        # upsample stage count is undefined; construction must reject it.
        with pytest.raises(ValueError):
            ImageDecoder(
                latent_dim=_LATENT_DIM,
                out_channels=_OUT_CHANNELS,
                image_size=20,
                base_channels=16,
                init_spatial=4,
            )

    def test_out_channels_not_channel_format_raises(self):
        # out_channels must be a ChannelFormat value (1, 3 or 4); 2 is not.
        with pytest.raises(ValueError):
            ImageDecoder(
                latent_dim=_LATENT_DIM,
                out_channels=2,
                image_size=_OUT_SIZE,
                base_channels=16,
                init_spatial=4,
            )

    def test_non_positive_latent_dim_raises(self):
        # A width of 0 has no meaning as an input dimension.
        with pytest.raises(ValueError):
            ImageDecoder(
                latent_dim=0,
                out_channels=_OUT_CHANNELS,
                image_size=_OUT_SIZE,
                base_channels=16,
                init_spatial=4,
            )


class TestImageDecoderDevice:
    # Smoke test: the decoder trunk must run on each device and return an
    # image whose tensor lands there.

    @parametrize_device
    def test_forward_output_on_device(self, device: str):
        decoder = _make_decoder().to(device)
        x = _batched_latent_sequence(b=2, s=3).to(device)

        out = decoder(x)

        assert out.tensor.device == torch.device(device)
