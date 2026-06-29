"""Behaviour spec for ``exp.models.norm.RMSNorm``.

Translates the spec for root-mean-square layer normalisation into
executable form. ``RMSNorm(dim, *, eps=1e-6)`` applies ``x *
rsqrt(mean(x**2, -1, keepdim=True) + eps) * weight`` with a learnable
``weight`` initialised to ones, over the last dimension and arbitrary
leading batch dims. Tests run on real CPU torch tensors (no mocking of
torch internals, per the project testing strategy).
"""

import torch

from exp.models.norm import RMSNorm

_DIM = 8


class TestRMSNormShape:
    def test_preserves_shape(self):
        # Normalisation is elementwise over the last dim: shape is unchanged.
        torch.manual_seed(0)
        norm = RMSNorm(_DIM)

        out = norm(torch.randn(_DIM))

        assert out.shape == (_DIM,)

    def test_preserves_arbitrary_leading_batch(self):
        # (B, T, dim) -> (B, T, dim): leading batch dims are untouched.
        torch.manual_seed(0)
        norm = RMSNorm(_DIM)

        out = norm(torch.randn(2, 5, _DIM))

        assert out.shape == (2, 5, _DIM)


class TestRMSNormContract:
    def test_output_rms_is_approximately_one_with_default_weight(self):
        # With weight=ones and a tiny eps, RMSNorm rescales each row to unit
        # root-mean-square: sqrt(mean(out**2, -1)) ~= 1.
        torch.manual_seed(0)
        norm = RMSNorm(_DIM)
        # Scale the input large so eps is negligible relative to mean(x**2).
        x = torch.randn(4, _DIM) * 10.0

        out = norm(x)

        rms = out.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones(4), rtol=1e-3, atol=1e-3)

    def test_weight_scales_the_normalised_output(self):
        # The affine weight multiplies the normalised result per-channel: a
        # uniform weight of c scales the output RMS to c.
        torch.manual_seed(0)
        norm = RMSNorm(_DIM)
        scale = 3.0
        with torch.no_grad():
            norm.weight.fill_(scale)
        x = torch.randn(4, _DIM) * 10.0

        out = norm(x)

        rms = out.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.full((4,), scale), rtol=1e-3, atol=1e-3)
