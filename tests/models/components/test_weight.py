"""Behaviour spec for ``exp.models.components.weight``.

Translates the approved spec for weight initialisation helpers into
executable form. Tests are written against the *spec*, not any
implementation, on real CPU torch modules (no mocking of torch).

- ``init_weights`` is applied via ``nn.Module.apply`` style usage and is
  expected to set biases to zero, LayerNorm weights to one, and leave
  unrelated modules untouched.
"""

import torch
from torch import nn

from exp.models.components.weight import init_weights


class TestInitWeights:
    def test_linear_bias_zeroed_and_weight_finite(self):
        # Linear weight -> trunc_normal (finite), bias -> 0.
        layer = nn.Linear(8, 4)
        nn.init.constant_(layer.bias, 5.0)

        init_weights(layer)

        assert torch.equal(layer.bias, torch.zeros(4))
        assert torch.isfinite(layer.weight).all()

    def test_conv2d_bias_zeroed_and_weight_finite(self):
        # Conv2d is treated like Linear: weight trunc_normal, bias -> 0.
        conv = nn.Conv2d(3, 6, kernel_size=3)
        assert conv.bias is not None
        nn.init.constant_(conv.bias, 5.0)

        init_weights(conv)

        assert torch.equal(conv.bias, torch.zeros(6))
        assert torch.isfinite(conv.weight).all()

    def test_layernorm_weight_one_bias_zero(self):
        # LayerNorm -> weight all 1, bias all 0 (affine reset to identity).
        norm = nn.LayerNorm(8)
        nn.init.constant_(norm.weight, 3.0)
        nn.init.constant_(norm.bias, 3.0)

        init_weights(norm)

        assert torch.equal(norm.weight, torch.ones(8))
        assert torch.equal(norm.bias, torch.zeros(8))

    def test_unhandled_module_is_left_untouched(self):
        # A module not matched by any case (GELU has no params) must not raise.
        init_weights(nn.GELU())

    def test_apply_recurses_over_submodules(self):
        # `apply` is the intended usage: every Linear bias should be zeroed.
        fc1 = nn.Linear(8, 8)
        norm = nn.LayerNorm(8)
        fc2 = nn.Linear(8, 4)
        for layer in (fc1, norm, fc2):
            nn.init.constant_(layer.bias, 9.0)

        nn.Sequential(fc1, norm, fc2).apply(init_weights)

        assert torch.equal(fc1.bias, torch.zeros(8))
        assert torch.equal(norm.weight, torch.ones(8))
        assert torch.equal(fc2.bias, torch.zeros(4))
