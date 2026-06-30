"""Behaviour spec for ``exp.models.base.SequenceModel``.

Translates the approved spec for the sequence-model abstract base into
executable form. Tests are written against the *spec* of the public
contract, not any concrete implementation.

``SequenceModel`` is a project-owned ABC, so per the testing strategy we
exercise it through a minimal concrete fake (integration-with-fakes) on
real CPU torch tensors with a fixed seed. The fake implements only the
abstract hooks ``_forward`` / ``_step``; the public ``forward`` / ``step``
are the base's ``@final`` validation wrappers and are never redefined in
the fake.

Pinned surfaces:
- the abstract contract (base and hook-missing subclasses are not
  instantiable);
- input/output shape validation in the ``@final`` wrappers;
- ``__call__`` dispatch through ``nn.Module`` hooks, hidden threading,
  arbitrary leading batch dims, and gradient flow;
- ``step`` single-timestep behaviour and ``SequenceModel[None]`` support.
"""

import pytest
import torch
import torch.nn as nn

from exp.models.base import SequenceModel
from tests.helpers import parametrize_device

# Small feature dim reused across the fakes and inputs.
_DIM = 4


class _EchoSequenceModel(SequenceModel[torch.Tensor]):
    """Spec-conformant fake: hidden is observable and a param carries grad.

    ``_forward`` adds ``hidden`` (broadcast over the seq axis) so
    callers can observe that hidden reaches the hook, and returns the
    last sequence token as the new hidden. ``_step`` delegates to the
    public ``forward`` over a singleton seq axis.
    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(_DIM, _DIM)

    def _forward(self, x, hidden=None):
        out = self.lin(x)
        if hidden is not None:
            out = out + hidden.unsqueeze(-2)
        return out, x[..., -1, :]

    def _step(self, x, hidden=None):
        out, h = self(x.unsqueeze(-2), hidden)
        return out.squeeze(-2), h


class _BadForwardShapeModel(SequenceModel[torch.Tensor]):
    """Fake whose ``_forward`` returns the wrong shape (drops last dim).

    Only ``out`` (the first element) carries the bad shape; the base's
    shape check inspects ``out`` alone. The hidden is a valid ``Tensor``
    (the sequence-tail token) so the override matches
    ``THidden=Tensor``.
    """

    def _forward(self, x, hidden=None):
        return x[..., :-1], x[..., -1, :]

    def _step(self, x, hidden=None):
        return x, x


class _BadStepShapeModel(SequenceModel[torch.Tensor]):
    """Fake whose ``_step`` returns the wrong shape (drops last dim).

    Only ``out`` (the first element) carries the bad shape; the hidden
    is a valid ``Tensor`` so the override matches ``THidden=Tensor``.
    """

    def _forward(self, x, hidden=None):
        return x, x[..., -1, :]

    def _step(self, x, hidden=None):
        return x[..., :-1], x


class _NoneHiddenModel(SequenceModel[None]):
    """``SequenceModel[None]`` fake: hooks return ``None`` as the hidden."""

    def _forward(self, x, hidden=None):
        return x, None

    def _step(self, x, hidden=None):
        return x, None


class TestSequenceModelContract:
    def test_abstract_base_cannot_be_instantiated(self):
        # Abstract hooks make the base non-instantiable. This pins the design
        # decision that the base is a contract, not a usable model.
        with pytest.raises(TypeError):
            SequenceModel()  # type: ignore[abstract]

    def test_subclass_missing_a_hook_cannot_be_instantiated(self):
        # Implementing only _forward leaves _step abstract, so the subclass is
        # still abstract and instantiation must fail.
        class _OnlyForward(SequenceModel[torch.Tensor]):
            def _forward(self, x, hidden=None):
                return x, x[..., -1, :]

        with pytest.raises(TypeError):
            _OnlyForward()  # type: ignore[abstract]


class TestSequenceModelValidation:
    def test_forward_rejects_1d_input_with_seq_dim_hint(self):
        # forward requires (*, seq, dim) i.e. ndim >= 2; a 1D tensor is invalid.
        model = _EchoSequenceModel()

        with pytest.raises(ValueError, match=r"\(\*, seq, dim\)"):
            model.forward(torch.randn(_DIM))

    def test_forward_rejects_scalar_input(self):
        # A scalar (ndim 0) is below the seq+dim requirement for forward.
        model = _EchoSequenceModel()

        with pytest.raises(ValueError, match=r"\(\*, seq, dim\)"):
            model.forward(torch.randn(()))

    def test_step_rejects_scalar_input_with_dim_hint(self):
        # step requires (*, dim) i.e. ndim >= 1; a scalar is invalid.
        model = _EchoSequenceModel()

        with pytest.raises(ValueError, match=r"\(\*, dim\)"):
            model.step(torch.randn(()))

    def test_forward_rejects_hook_output_with_wrong_shape(self):
        # When _forward returns out whose shape != input, the base must reject.
        model = _BadForwardShapeModel()

        with pytest.raises(ValueError, match="must return"):
            model(torch.randn(2, 3, _DIM))

    def test_step_rejects_hook_output_with_wrong_shape(self):
        # When _step returns out whose shape != input, the base must reject.
        model = _BadStepShapeModel()

        with pytest.raises(ValueError, match="must return"):
            model.step(torch.randn(2, _DIM))

    def test_forward_accepts_well_formed_input_and_preserves_shape(self):
        # Normal path: no raise, and output keeps the input (*, seq, dim) shape.
        torch.manual_seed(0)
        model = _EchoSequenceModel()
        x = torch.randn(2, 3, _DIM)

        out, _ = model(x)

        assert out.shape == x.shape

    def test_step_accepts_well_formed_input_and_preserves_shape(self):
        # Normal path: no raise, and output keeps the input (*, dim) shape.
        torch.manual_seed(0)
        model = _EchoSequenceModel()
        x = torch.randn(2, _DIM)

        out, _ = model.step(x)

        assert out.shape == x.shape


class TestSequenceModelCall:
    def test_call_returns_output_and_hidden_shapes(self):
        # model(x) with x:(B, T, D) -> out (B, T, D) and hidden (B, D).
        torch.manual_seed(0)
        model = _EchoSequenceModel()
        x = torch.randn(2, 3, _DIM)

        out, hidden = model(x)

        assert out.shape == (2, 3, _DIM)
        assert hidden.shape == (2, _DIM)

    def test_passing_hidden_changes_the_output(self):
        # Threading a non-None hidden must reach _forward and alter the output,
        # observed via the echo fake (which adds hidden over the seq axis).
        torch.manual_seed(0)
        model = _EchoSequenceModel()
        x = torch.randn(2, 3, _DIM)

        out_no_hidden, _ = model(x)
        hidden = torch.ones(2, _DIM)
        out_with_hidden, _ = model(x, hidden)

        assert not torch.allclose(out_no_hidden, out_with_hidden)

    def test_call_supports_arbitrary_leading_batch_dims(self):
        # Extra leading batch dims are preserved: (2, 3, T, D) -> (2, 3, T, D).
        torch.manual_seed(0)
        model = _EchoSequenceModel()
        x = torch.randn(2, 3, 5, _DIM)

        out, _ = model(x)

        assert out.shape == (2, 3, 5, _DIM)

    def test_call_dispatches_through_module_hooks_so_gradients_flow(self):
        # model(x) must go through nn.Module.__call__ -> forward -> _forward so
        # autograd hooks stay live and a parameter receives a gradient.
        torch.manual_seed(0)
        model = _EchoSequenceModel()
        x = torch.randn(2, 3, _DIM)

        model(x)[0].sum().backward()

        assert model.lin.weight.grad is not None


class TestSequenceModelStep:
    def test_step_returns_output_and_hidden_shapes(self):
        # step(x_t, hidden) with x_t:(B, D) -> out (B, D) and hidden (B, D).
        torch.manual_seed(0)
        model = _EchoSequenceModel()
        x_t = torch.randn(2, _DIM)
        hidden = torch.zeros(2, _DIM)

        out, new_hidden = model.step(x_t, hidden)

        assert out.shape == (2, _DIM)
        assert new_hidden.shape == (2, _DIM)

    def test_none_hidden_model_forward_returns_none_hidden(self):
        # SequenceModel[None]: a model with no recurrent state may return None
        # as the hidden from forward.
        torch.manual_seed(0)
        model = _NoneHiddenModel()
        x = torch.randn(2, 3, _DIM)

        out, hidden = model(x)

        assert out.shape == x.shape
        assert hidden is None

    def test_none_hidden_model_step_returns_none_hidden(self):
        # SequenceModel[None]: step may likewise return None as the hidden.
        torch.manual_seed(0)
        model = _NoneHiddenModel()
        x_t = torch.randn(2, _DIM)

        out, hidden = model.step(x_t)

        assert out.shape == x_t.shape
        assert hidden is None


class TestSequenceModelDevice:
    # Smoke test: the @final validation wrappers must pass device through, so
    # forward over a concrete fake keeps output and hidden on each device.

    @parametrize_device
    def test_forward_output_on_device(self, device: str):
        torch.manual_seed(0)
        model = _EchoSequenceModel().to(device)
        x = torch.randn(2, 3, _DIM, device=device)

        out, hidden = model(x)

        assert out.device == torch.device(device)
        assert hidden.device == torch.device(device)
