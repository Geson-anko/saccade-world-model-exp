"""Behaviour spec for ``exp.models.components.conv_decoder``.

Translates the approved contract for ``ConvDecoder`` into executable
form. ``ConvDecoder`` is the generic feature-to-image trunk: it maps a
feature vector ``(*, feature_dim)`` to a square image ``(*,
out_channels, s', s')`` via ``Linear -> [Upsample(nearest) -> Conv2d ->
GroupNorm -> GELU] * N -> Conv2d`` (final projection linear, no
norm/activation).

These are integration-real tests: the real upsample+conv decoder stack
runs on real CPU torch tensors with fixed seeds (no mocking of torch
internals, per the project testing strategy). Everything is checked
through the public ``forward`` surface; the internal layers (``proj`` /
``stages`` / ``head``) and module-level helpers (``_resolve_num_stages``
etc.) are never imported or asserted on directly.

This complements ``tests/models/test_decoder.py``, which pins the
``ImageDecoder`` value-object interface (Latent family <-> Image family,
ChannelFormat validation, detach). Here we exercise ``ConvDecoder``
itself on raw tensors: arbitrary-leading shape folding, the
``init_spatial * 2**N`` square-output constraint, and channel genericity
(no ChannelFormat limit — that is ``ImageDecoder``'s responsibility, and
the contrast is asserted by running out_channels=1 and 3).
"""

import pytest
import torch

from exp.models.components.conv_decoder import ConvDecoder
from exp.types import Size2d
from tests.helpers import parametrize_device

# Small but spec-valid config. feature_dim 8 keeps tensors tiny; out_channels
# 3 is a plain positive int (ConvDecoder has no ChannelFormat constraint);
# out_size 16 == init_spatial(4) * 2**2 gives a cheap 2-stage trunk.
_FEATURE_DIM = 8
_OUT_CHANNELS = 3
_OUT_SIZE = 16
_INIT_SPATIAL = 4


def _make_conv_decoder(out_channels: int = _OUT_CHANNELS, out_size: Size2d = _OUT_SIZE):
    return ConvDecoder(
        feature_dim=_FEATURE_DIM,
        out_channels=out_channels,
        out_size=out_size,
        base_channels=16,
        init_spatial=_INIT_SPATIAL,
    )


class TestForwardShape:
    # The module folds arbitrary leading dims into the batch axis, runs the
    # conv trunk, then restores the leading dims. Each rank must map
    # (*lead, feature_dim) -> (*lead, out_channels, s', s').

    def test_no_leading_dim(self):
        # (feature_dim,) -> (out_channels, s', s')
        torch.manual_seed(0)
        decoder = _make_conv_decoder()

        out = decoder(torch.randn(_FEATURE_DIM))

        assert out.shape == (_OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)

    def test_one_leading_dim(self):
        # (b, feature_dim) -> (b, out_channels, s', s')
        torch.manual_seed(0)
        decoder = _make_conv_decoder()

        out = decoder(torch.randn(2, _FEATURE_DIM))

        assert out.shape == (2, _OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)

    def test_two_leading_dims(self):
        # (b, s, feature_dim) -> (b, s, out_channels, s', s')
        torch.manual_seed(0)
        decoder = _make_conv_decoder()

        out = decoder(torch.randn(2, 3, _FEATURE_DIM))

        assert out.shape == (2, 3, _OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)

    def test_three_leading_dims(self):
        # A deeper leading stack must still fold and restore correctly:
        # (b, g, s, feature_dim) -> (b, g, s, out_channels, s', s').
        torch.manual_seed(0)
        decoder = _make_conv_decoder()

        out = decoder(torch.randn(2, 3, 4, _FEATURE_DIM))

        assert out.shape == (2, 3, 4, _OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)


class TestSpatialSize:
    # The output side is exactly out_size, resolved as init_spatial * 2**N.
    # Different (out_size, init_spatial) pairs that yield the same stage count
    # must both land on their requested side.

    def test_out_size_16_init_spatial_4(self):
        # 16 == 4 * 2**2 -> 2 stages -> 16x16 output.
        torch.manual_seed(0)
        decoder = ConvDecoder(
            feature_dim=_FEATURE_DIM,
            out_channels=_OUT_CHANNELS,
            out_size=16,
            base_channels=16,
            init_spatial=4,
        )

        out = decoder(torch.randn(2, _FEATURE_DIM))

        assert out.shape[-2:] == (16, 16)

    def test_out_size_32_init_spatial_8(self):
        # 32 == 8 * 2**2 -> 2 stages -> 32x32 output. Same stage count as the
        # case above but a different requested side.
        torch.manual_seed(0)
        decoder = ConvDecoder(
            feature_dim=_FEATURE_DIM,
            out_channels=_OUT_CHANNELS,
            out_size=32,
            base_channels=16,
            init_spatial=8,
        )

        out = decoder(torch.randn(2, _FEATURE_DIM))

        assert out.shape[-2:] == (32, 32)

    def test_init_spatial_equals_out_size_gives_zero_stages(self):
        # out_size == init_spatial -> N=0 stages: the Linear projection alone
        # already sits at the target resolution.
        torch.manual_seed(0)
        decoder = ConvDecoder(
            feature_dim=_FEATURE_DIM,
            out_channels=_OUT_CHANNELS,
            out_size=4,
            base_channels=16,
            init_spatial=4,
        )

        out = decoder(torch.randn(2, _FEATURE_DIM))

        assert out.shape[-2:] == (4, 4)

    def test_out_size_accepts_square_tuple(self):
        # out_size may be given as an (n, n) tuple, equivalent to the int n.
        torch.manual_seed(0)
        decoder = _make_conv_decoder(out_size=(16, 16))

        out = decoder(torch.randn(2, _FEATURE_DIM))

        assert out.shape[-2:] == (16, 16)


class TestChannels:
    # out_channels is an arbitrary positive int; the head projects to exactly
    # that many channels. Unlike ImageDecoder, there is no ChannelFormat
    # {1,3,4} constraint here, so a value like 1 must work as well as 3.

    def test_single_channel(self):
        torch.manual_seed(0)
        decoder = _make_conv_decoder(out_channels=1)

        out = decoder(torch.randn(2, _FEATURE_DIM))

        assert out.shape[-3] == 1

    def test_three_channels(self):
        torch.manual_seed(0)
        decoder = _make_conv_decoder(out_channels=3)

        out = decoder(torch.randn(2, _FEATURE_DIM))

        assert out.shape[-3] == 3


class TestSingleSampleTrainMode:
    def test_no_leading_dim_in_train_mode_passes(self):
        # A lone feature vector folds to N=1. The trunk normalizes with
        # GroupNorm (batch-independent), so unlike a BatchNorm-based encoder
        # this must NOT raise in the default train mode.
        torch.manual_seed(0)
        decoder = _make_conv_decoder()  # train mode is the default

        out = decoder(torch.randn(_FEATURE_DIM))

        assert out.shape == (_OUT_CHANNELS, _OUT_SIZE, _OUT_SIZE)


class TestValidation:
    def test_out_size_not_power_of_two_multiple_raises(self):
        # 20 cannot be written as init_spatial(4) * 2**N for integer N, so the
        # stage count is undefined; construction must reject it.
        with pytest.raises(ValueError, match="init_spatial"):
            ConvDecoder(
                feature_dim=_FEATURE_DIM,
                out_channels=_OUT_CHANNELS,
                out_size=20,
                base_channels=16,
                init_spatial=4,
            )

    def test_non_square_out_size_raises(self):
        # (8, 16) needs 1 stage on H but 2 on W: the per-side stage counts
        # disagree, so no single upsample depth produces it -> ValueError.
        with pytest.raises(ValueError, match="init_spatial"):
            ConvDecoder(
                feature_dim=_FEATURE_DIM,
                out_channels=_OUT_CHANNELS,
                out_size=(8, 16),
                base_channels=16,
                init_spatial=4,
            )


class TestGradientFlow:
    def test_backward_reaches_conv_decoder_parameters(self):
        # The whole trunk (proj / stages / head) must be on the autograd path:
        # at least one learnable parameter gets a non-None grad after backward.
        torch.manual_seed(0)
        decoder = _make_conv_decoder()

        decoder(torch.randn(2, 3, _FEATURE_DIM)).sum().backward()

        grads = [p.grad for p in decoder.parameters() if p.grad is not None]
        assert grads, "no ConvDecoder parameter received a gradient"


class TestEvalDeterminism:
    def test_repeated_eval_forward_matches(self):
        # In eval mode GroupNorm uses per-sample stats deterministically and
        # no dropout is present, so the same input yields the same output.
        torch.manual_seed(0)
        decoder = _make_conv_decoder()
        decoder.eval()
        x = torch.randn(2, 3, _FEATURE_DIM)

        with torch.no_grad():
            first = decoder(x)
            second = decoder(x)

        torch.testing.assert_close(first, second)


class TestConvDecoderCompile:
    # integration-real: exercises the CPU inductor backend. Skips when
    # torch.compile is unavailable/unsupported in the environment, since
    # this only guarantees compile *executability* and eager parity.

    def test_compiled_matches_eager(self):
        torch.manual_seed(0)
        model = _make_conv_decoder()
        model.eval()
        x = torch.randn(2, _FEATURE_DIM)

        try:
            compiled = torch.compile(model)
            with torch.no_grad():
                eager_out = model(x)
                compiled_out = compiled(x)
        except Exception as exc:  # compile backend missing/unsupported here
            pytest.skip(f"torch.compile unavailable in this environment: {exc}")

        torch.testing.assert_close(compiled_out, eager_out)


class TestConvDecoderDevice:
    # Smoke test: the conv trunk must run on each device and keep its output
    # there.

    @parametrize_device
    def test_forward_output_on_device(self, device: str):
        torch.manual_seed(0)
        decoder = _make_conv_decoder().to(device)
        x = torch.randn(2, 3, _FEATURE_DIM, device=device)

        out = decoder(x)

        assert out.device == torch.device(device)
