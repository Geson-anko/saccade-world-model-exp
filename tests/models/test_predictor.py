"""Behaviour spec for ``exp.models.predictor``.

Translates the approved spec for ``Predictor`` into executable form.
``Predictor`` is the sequence-model stage of the world model: it fuses the
next action ``Focus`` with the current observation embedding ``Latent`` at
the input of an injected ``SequenceModel`` and projects the model output
back to a latent, realising ``(a_{<=t}, e_{<=t}), a_{t+1} -> e_hat_{t+1}``.

Per CLAUDE.md decision #5 the encoder never sees the action; the action is
injected here, so ``Predictor`` concatenates ``focus`` and ``latent`` along
the feature axis (``FOCUS_DIM + latent_dim``) and feeds them to the
sequence model.

These are integration-real tests: rather than a fake sequence model we
inject a small real ``MinGRU`` (the only concrete ``SequenceModel``) and
run the real Linear / MinGRU stack on real CPU torch tensors with fixed
seeds (no mocking of torch internals, per the project testing strategy).
This is faithful to "real resource first" -- MinGRU is lightweight, and a
fake would only re-assert our own assumptions about the injected contract.

The input-type -> output-type / dispatch contract is the core surface:
- ``Focus`` / ``BatchedFocus`` (no seq axis)      -> ``sequence_model.step``
- ``FocusSequence`` / ``BatchedFocusSequence``     -> ``sequence_model.forward``
and the returned value-object type is decided by the ``focus`` type.
"""

import pytest
import torch

from exp.models.components.mingru import MinGRU
from exp.models.predictor import Predictor
from exp.types import (
    BatchedFocus,
    BatchedFocusSequence,
    BatchedLatent,
    BatchedLatentSequence,
    Focus,
    FocusSequence,
    Latent,
    LatentSequence,
)
from tests.helpers import parametrize_device

# latent_dim (5) is deliberately distinct from FOCUS_DIM (3) and the sequence
# model's feature width (8): if the concat / projection widths were wired
# wrongly, the mismatched dims would surface as a shape error rather than
# silently passing.
_LATENT_DIM = 5
_SEQ_DIM = 8
_DEPTH = 1

_BATCH = 2
_SEQ_LEN = 4


def _make_predictor() -> Predictor[torch.Tensor]:
    torch.manual_seed(0)
    return Predictor(_LATENT_DIM, MinGRU(_SEQ_DIM, _DEPTH))


def _focus() -> Focus:
    return Focus(torch.randn(3))


def _latent() -> Latent:
    return Latent(torch.randn(_LATENT_DIM))


def _batched_focus() -> BatchedFocus:
    return BatchedFocus(torch.randn(_BATCH, 3))


def _batched_latent() -> BatchedLatent:
    return BatchedLatent(torch.randn(_BATCH, _LATENT_DIM))


def _focus_sequence() -> FocusSequence:
    return FocusSequence(torch.randn(_SEQ_LEN, 3))


def _latent_sequence() -> LatentSequence:
    return LatentSequence(torch.randn(_SEQ_LEN, _LATENT_DIM))


def _batched_focus_sequence() -> BatchedFocusSequence:
    return BatchedFocusSequence(torch.randn(_BATCH, _SEQ_LEN, 3))


def _batched_latent_sequence() -> BatchedLatentSequence:
    return BatchedLatentSequence(torch.randn(_BATCH, _SEQ_LEN, _LATENT_DIM))


class TestForwardShape:
    """Each input family maps to its matching latent family at latent_dim.

    The hidden is the injected MinGRU's belief ``(*, depth, dim)``:
    leading batch dims of the input are preserved on it, and it is
    always at the sequence model's own width, never latent_dim.
    """

    def test_focus_returns_latent(self):
        # A single (Focus, Latent) with no batch/seq axis dispatches to step
        # and yields one Latent of shape (latent_dim,); hidden is (depth, dim).
        out, hidden = _make_predictor()(_focus(), _latent())

        assert type(out) is Latent
        assert out.tensor.shape == (_LATENT_DIM,)
        assert hidden.shape == (_DEPTH, _SEQ_DIM)

    def test_batched_focus_returns_batched_latent(self):
        # (BatchedFocus, BatchedLatent): batch axis preserved, no seq axis ->
        # step. Out (batch, latent_dim); hidden (batch, depth, dim).
        out, hidden = _make_predictor()(_batched_focus(), _batched_latent())

        assert type(out) is BatchedLatent
        assert out.tensor.shape == (_BATCH, _LATENT_DIM)
        assert hidden.shape == (_BATCH, _DEPTH, _SEQ_DIM)

    def test_focus_sequence_returns_latent_sequence(self):
        # (FocusSequence, LatentSequence) has a seq axis -> forward. Out
        # (seq, latent_dim); hidden (depth, dim) (belief at the sequence tail).
        out, hidden = _make_predictor()(_focus_sequence(), _latent_sequence())

        assert type(out) is LatentSequence
        assert out.tensor.shape == (_SEQ_LEN, _LATENT_DIM)
        assert hidden.shape == (_DEPTH, _SEQ_DIM)

    def test_batched_focus_sequence_returns_batched_latent_sequence(self):
        # (BatchedFocusSequence, BatchedLatentSequence): batch + seq axes ->
        # forward. Out (batch, seq, latent_dim); hidden (batch, depth, dim).
        out, hidden = _make_predictor()(
            _batched_focus_sequence(), _batched_latent_sequence()
        )

        assert type(out) is BatchedLatentSequence
        assert out.tensor.shape == (_BATCH, _SEQ_LEN, _LATENT_DIM)
        assert hidden.shape == (_BATCH, _DEPTH, _SEQ_DIM)


class TestPublicAttributes:
    def test_exposes_latent_dim(self):
        # latent_dim is public so downstream knows the predictor's in/out width.
        assert _make_predictor().latent_dim == _LATENT_DIM


class TestHiddenContinuity:
    def test_split_sequence_with_threaded_hidden_matches_full_forward(self):
        # The step/forward wiring is correct only if a belief returned from one
        # forward can be threaded into the next: processing the sequence in two
        # chunks and passing the first chunk's hidden must equal a single
        # forward over the whole sequence. Mirrors the MinGRU continuity test,
        # but through the Predictor (input_proj -> forward -> output_proj), so
        # it pins that the predictor forwards hidden untouched. double() for
        # numerical headroom.
        predictor = _make_predictor().double()
        focus = FocusSequence(torch.randn(_SEQ_LEN, 3, dtype=torch.double))
        latent = LatentSequence(torch.randn(_SEQ_LEN, _LATENT_DIM, dtype=torch.double))
        split = 2

        full_out, full_h = predictor(focus, latent)
        first_out, first_h = predictor(focus[:split], latent[:split])
        second_out, second_h = predictor(focus[split:], latent[split:], first_h)

        torch.testing.assert_close(
            torch.cat([first_out.tensor, second_out.tensor], dim=0), full_out.tensor
        )
        torch.testing.assert_close(second_h, full_h)


class TestGradientFlow:
    def test_backward_reaches_input_projection(self):
        # The input Linear (focus||latent -> sequence width) is on the path.
        predictor = _make_predictor()

        out, _ = predictor(_focus_sequence(), _latent_sequence())
        out.tensor.sum().backward()

        assert predictor.input_proj.weight.grad is not None

    def test_backward_reaches_output_projection(self):
        # The output Linear (sequence width -> latent_dim) is on the path.
        predictor = _make_predictor()

        out, _ = predictor(_focus_sequence(), _latent_sequence())
        out.tensor.sum().backward()

        assert predictor.output_proj.weight.grad is not None

    def test_backward_reaches_injected_sequence_model(self):
        # Gradient must flow through the DI-injected sequence model too, so the
        # belief pathway is trained end-to-end: at least one MinGRU parameter
        # receives a non-None grad.
        predictor = _make_predictor()

        out, _ = predictor(_focus_sequence(), _latent_sequence())
        out.tensor.sum().backward()

        grads = [
            p.grad for p in predictor.sequence_model.parameters() if p.grad is not None
        ]
        assert grads, "no sequence-model parameter received a gradient"


class TestValidation:
    def test_non_positive_latent_dim_raises(self):
        # latent_dim must be a positive feature width (substring match only).
        with pytest.raises(ValueError, match="latent_dim"):
            Predictor(0, MinGRU(_SEQ_DIM, _DEPTH))

    def test_negative_latent_dim_raises(self):
        with pytest.raises(ValueError, match="latent_dim"):
            Predictor(-1, MinGRU(_SEQ_DIM, _DEPTH))


class TestPredictorDevice:
    # Smoke test: the input_proj -> sequence_model -> output_proj path must run
    # on each device and keep both the output latent and the belief there.

    @parametrize_device
    def test_forward_output_on_device(self, device: str):
        predictor = _make_predictor().to(device)
        focus = _focus_sequence().to(device)
        latent = _latent_sequence().to(device)

        out, hidden = predictor(focus, latent)

        assert out.tensor.device == torch.device(device)
        assert hidden.device == torch.device(device)
