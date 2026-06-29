"""Behaviour spec for ``exp.models.mingru``.

Translates the approved minGRU spec (Feng et al. 2024, arXiv:2410.01201)
into executable form. Tests are written against the *spec*, not any
implementation, on real CPU torch tensors with fixed seeds (no mocking of
torch internals, per the project testing strategy).

minGRU's gate/candidate depend only on the input ``x_t`` (not on the
previous hidden), so the recurrence ``h_t = (1-z_t)*h_{t-1} + z_t*g(h~_t)``
is linear and computed by a log-space parallel scan. The private numerical
helpers (``_g`` / ``_log_g`` / ``_parallel_scan_log``) are NOT imported
directly; everything is checked through the public surfaces
``MinGRULayer`` / ``MinGRUBlock`` / ``MinGRU``.

The independent reference for the parallel forward is the public ``step``
recurrence looped over time: ``MinGRULayer.step`` uses the plain recurrence
``h = (1-z)*h_prev + z*g(h~)`` (no log-space), so forward==step-loop pins
the parallel scan against a different code path (tautology avoided).

Gate saturation is forced through the public ``fc_z`` parameters: rewriting
``fc_z.weight``/``fc_z.bias`` under ``no_grad`` drives ``z = sigmoid(fc_z(x))``
to ~1 (large positive bias) or ~0 (large negative bias), independent of the
input.

Base-class contract (``forward``/``step`` shape validation and the ``@final``
wrappers) is already covered by ``tests/models/test_base.py`` and is not
re-tested here.
"""

import pytest
import torch

from exp.models.mingru import MinGRU, MinGRULayer

_INPUT_DIM = 4
_HIDDEN_DIM = 6
_SEQ_LEN = 7
_DIM = 8
_DEPTH = 3


def _step_loop(layer, x, hidden=None):
    """Reference: run ``layer.step`` across the seq axis (dim=-2).

    Returns the stacked per-step hidden sequence and the final hidden,
    mirroring ``MinGRULayer.forward``'s ``(out, h_last)`` contract.
    """
    seq_len = x.shape[-2]
    h = hidden
    outs = []
    for t in range(seq_len):
        h = layer.step(x[..., t, :], h)
        outs.append(h)
    return torch.stack(outs, dim=-2), h


def _saturate_gate(layer, *, bias_value):
    """Force ``z = sigmoid(fc_z(x))`` to a saturated constant.

    Zeroing ``fc_z.weight`` makes the gate input-independent; a large
    positive ``bias_value`` drives ``z -> 1``, a large negative one ``z -> 0``.
    """
    with torch.no_grad():
        layer.fc_z.weight.zero_()
        layer.fc_z.bias.fill_(bias_value)


class TestMinGRULayerMath:
    def test_hidden_is_strictly_positive(self):
        # The candidate g(h~) is a continuous positive activation and the
        # gate convex-combines positive quantities, so every hidden > 0.
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM)
        x = torch.randn(3, _SEQ_LEN, _INPUT_DIM)

        out, h_last = layer(x)

        assert (out > 0).all()
        assert (h_last > 0).all()

    def test_parallel_forward_matches_step_loop_from_zeros(self):
        # The log-space parallel scan must equal the plain step recurrence
        # looped over time, starting from the default zeros initial hidden.
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM).double()
        x = torch.randn(3, _SEQ_LEN, _INPUT_DIM, dtype=torch.double)

        out, h_last = layer(x)
        ref_out, ref_h_last = _step_loop(layer, x)

        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(h_last, ref_h_last)

    def test_parallel_forward_matches_step_loop_from_random_hidden(self):
        # Same equivalence, but threading a non-zero initial hidden (the
        # h_0 side-term in the scan) so that branch is exercised too.
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM).double()
        x = torch.randn(3, _SEQ_LEN, _INPUT_DIM, dtype=torch.double)
        h0 = torch.randn(3, _HIDDEN_DIM, dtype=torch.double)

        out, h_last = layer(x, h0)
        ref_out, ref_h_last = _step_loop(layer, x, h0)

        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(h_last, ref_h_last)

    def test_gate_one_makes_output_equal_candidate_independent_of_hidden(self):
        # With z=1 the recurrence collapses to h_t = g(h~_t), so the output
        # is fully determined by the input and ignores the initial hidden.
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM)
        _saturate_gate(layer, bias_value=30.0)
        x = torch.randn(3, _SEQ_LEN, _INPUT_DIM)

        out_from_zeros, _ = layer(x)
        out_from_random, _ = layer(x, torch.randn(3, _HIDDEN_DIM))

        torch.testing.assert_close(out_from_zeros, out_from_random)

    def test_gate_zero_holds_the_initial_hidden(self):
        # With z=0 the recurrence is h_t = h_{t-1}, so every step sticks to
        # the initial hidden; from zeros the whole output sequence is zero.
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM)
        _saturate_gate(layer, bias_value=-30.0)
        x = torch.randn(3, _SEQ_LEN, _INPUT_DIM)

        out, h_last = layer(x)

        torch.testing.assert_close(out, torch.zeros_like(out))
        torch.testing.assert_close(h_last, torch.zeros_like(h_last))

    def test_backward_through_negative_candidate_branch_is_finite(self):
        # _log_g must clamp with relu so the x < -0.5 branch never feeds
        # log(negative)=NaN into backward. Drive fc_h hard negative and
        # check every gradient is finite.
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM)
        with torch.no_grad():
            layer.fc_h.bias.fill_(-50.0)
        x = torch.randn(3, _SEQ_LEN, _INPUT_DIM, requires_grad=True)

        out, _ = layer(x)
        out.sum().backward()

        assert torch.isfinite(out).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for p in layer.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


class TestMinGRULayerShape:
    def test_maps_input_dim_to_hidden_dim_over_sequence(self):
        # (len, input_dim) -> out (len, hidden_dim), h_last (hidden_dim).
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM)

        out, h_last = layer(torch.randn(_SEQ_LEN, _INPUT_DIM))

        assert out.shape == (_SEQ_LEN, _HIDDEN_DIM)
        assert h_last.shape == (_HIDDEN_DIM,)

    def test_preserves_arbitrary_leading_batch(self):
        # (B, G, len, input_dim) -> out (B, G, len, hidden_dim),
        # h_last (B, G, hidden_dim).
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM)

        out, h_last = layer(torch.randn(2, 3, _SEQ_LEN, _INPUT_DIM))

        assert out.shape == (2, 3, _SEQ_LEN, _HIDDEN_DIM)
        assert h_last.shape == (2, 3, _HIDDEN_DIM)

    def test_step_maps_input_dim_to_hidden_dim(self):
        # step(x_t) with x_t:(B, input_dim) -> h_t (B, hidden_dim).
        torch.manual_seed(0)
        layer = MinGRULayer(_INPUT_DIM, _HIDDEN_DIM)

        h_t = layer.step(torch.randn(2, _INPUT_DIM))

        assert h_t.shape == (2, _HIDDEN_DIM)


class TestMinGRUShape:
    def test_maps_sequence_in_to_same_dim_out(self):
        # MinGRU is a SequenceModel: in=out=dim. (seq, dim) -> out (seq, dim),
        # h_last (depth, dim).
        torch.manual_seed(0)
        model = MinGRU(_DIM, _DEPTH)

        out, h_last = model(torch.randn(_SEQ_LEN, _DIM))

        assert out.shape == (_SEQ_LEN, _DIM)
        assert h_last.shape == (_DEPTH, _DIM)

    def test_preserves_arbitrary_leading_batch(self):
        # (B, G, seq, dim) -> out (B, G, seq, dim), h_last (B, G, depth, dim).
        torch.manual_seed(0)
        model = MinGRU(_DIM, _DEPTH)

        out, h_last = model(torch.randn(2, 3, _SEQ_LEN, _DIM))

        assert out.shape == (2, 3, _SEQ_LEN, _DIM)
        assert h_last.shape == (2, 3, _DEPTH, _DIM)

    def test_step_maps_token_in_to_same_dim_out(self):
        # step(x_t) with x_t:(B, dim) -> out (B, dim), h_last (B, depth, dim).
        torch.manual_seed(0)
        model = MinGRU(_DIM, _DEPTH)

        out, h_last = model.step(torch.randn(2, _DIM))

        assert out.shape == (2, _DIM)
        assert h_last.shape == (2, _DEPTH, _DIM)

    def test_exposes_dim_and_depth(self):
        # dim/depth are part of the public contract (used downstream to size
        # the belief state).
        model = MinGRU(_DIM, _DEPTH)

        assert model.dim == _DIM
        assert model.depth == _DEPTH


class TestMinGRUValidation:
    def test_non_positive_dim_raises(self):
        with pytest.raises(ValueError, match="dim"):
            MinGRU(0, _DEPTH)

    def test_non_positive_depth_raises(self):
        with pytest.raises(ValueError, match="depth"):
            MinGRU(_DIM, 0)


class TestMinGRUBehaviour:
    def test_parallel_forward_matches_step_loop(self):
        # End-to-end (all depth layers + norms): the parallel forward must
        # equal the public step recurrence looped over time from zeros.
        torch.manual_seed(0)
        model = MinGRU(_DIM, _DEPTH).double()
        x = torch.randn(2, _SEQ_LEN, _DIM, dtype=torch.double)

        out, h_last = model(x)

        hidden = None
        step_outs = []
        for t in range(_SEQ_LEN):
            out_t, hidden = model.step(x[..., t, :], hidden)
            step_outs.append(out_t)
        ref_out = torch.stack(step_outs, dim=-2)

        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(h_last, hidden)

    def test_split_sequence_with_threaded_hidden_matches_full_forward(self):
        # Hidden roundtrip continuity: processing the sequence in two chunks
        # and threading the returned hidden must equal a single forward.
        torch.manual_seed(0)
        model = MinGRU(_DIM, _DEPTH).double()
        x = torch.randn(2, _SEQ_LEN, _DIM, dtype=torch.double)
        split = 3

        full_out, full_h = model(x)
        first_out, first_h = model(x[..., :split, :])
        second_out, second_h = model(x[..., split:, :], first_h)

        torch.testing.assert_close(torch.cat([first_out, second_out], dim=-2), full_out)
        torch.testing.assert_close(second_h, full_h)

    def test_backward_populates_every_block_gradient(self):
        # A scalar loss must reach parameters in every block (all depth layers
        # participate, not just the last).
        torch.manual_seed(0)
        model = MinGRU(_DIM, _DEPTH)
        x = torch.randn(2, _SEQ_LEN, _DIM)

        model(x)[0].sum().backward()

        for block in model.blocks:
            block_grads = [p.grad for p in block.parameters() if p.grad is not None]
            assert block_grads, "a block received no gradient"
            assert any(g.abs().sum() > 0 for g in block_grads)

    def test_eval_is_deterministic(self):
        # In eval mode (dropout disabled) repeated forwards must match.
        torch.manual_seed(0)
        model = MinGRU(_DIM, _DEPTH, dropout=0.5)
        model.eval()
        x = torch.randn(2, _SEQ_LEN, _DIM)

        with torch.no_grad():
            first, _ = model(x)
            second, _ = model(x)

        torch.testing.assert_close(first, second)


class TestMinGRUCompile:
    # integration-real: exercises the CPU inductor backend. The log-space
    # scan uses logcumsumexp, which the inductor backend does not always
    # support, so we skip (surfacing the reason) rather than fail when
    # compilation is unavailable. This only guarantees compile executability
    # and eager parity.

    def test_compiled_matches_eager(self):
        torch.manual_seed(0)
        model = MinGRU(_DIM, _DEPTH)
        model.eval()
        x = torch.randn(2, _SEQ_LEN, _DIM)

        try:
            compiled = torch.compile(model)
            with torch.no_grad():
                eager_out, _ = model(x)
                compiled_out, _ = compiled(x)
        except Exception as exc:  # inductor may lack logcumsumexp support
            pytest.skip(f"torch.compile unavailable in this environment: {exc}")

        torch.testing.assert_close(compiled_out, eager_out)
