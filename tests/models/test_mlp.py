"""Behaviour spec for ``exp.models.mlp.Mlp``.

``Mlp`` is a thin Linear-GELU-dropout-Linear-dropout block relocated out
of ``vit.py``. The only contract worth pinning is its shape mapping
``(*, in_features) -> (*, out_features)`` over arbitrary leading batch
dims. Tests run on real CPU torch tensors with a fixed seed (no mocking
of torch internals, per the project testing strategy).
"""

import torch

from exp.models.mlp import Mlp
from tests.helpers import parametrize_device

_IN_FEATURES = 6
_HIDDEN_FEATURES = 24
_OUT_FEATURES = 10


class TestMlpShape:
    def test_maps_in_features_to_out_features(self):
        # (in_features,) -> (out_features,): empty leading batch.
        torch.manual_seed(0)
        mlp = Mlp(_IN_FEATURES, _HIDDEN_FEATURES, _OUT_FEATURES)

        out = mlp(torch.randn(_IN_FEATURES))

        assert out.shape == (_OUT_FEATURES,)

    def test_preserves_arbitrary_leading_batch(self):
        # (B, T, in_features) -> (B, T, out_features): leading dims untouched.
        torch.manual_seed(0)
        mlp = Mlp(_IN_FEATURES, _HIDDEN_FEATURES, _OUT_FEATURES)

        out = mlp(torch.randn(2, 5, _IN_FEATURES))

        assert out.shape == (2, 5, _OUT_FEATURES)


class TestMlpDevice:
    # Smoke test: forward must compute on each device and land its output there.

    @parametrize_device
    def test_forward_output_on_device(self, device: str):
        torch.manual_seed(0)
        mlp = Mlp(_IN_FEATURES, _HIDDEN_FEATURES, _OUT_FEATURES).to(device)
        x = torch.randn(2, 5, _IN_FEATURES, device=device)

        out = mlp(x)

        assert out.device == torch.device(device)
