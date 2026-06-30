"""Behaviour spec for ``exp.loss``.

Translates the approved spec for the two loss functors into executable
form:

- ``MSELoss`` — mean/sum squared error over latent sequences. Returns a
  ``ScalarTensor`` value plus a detached per-position ``MSELossInfo``.
- ``SIGReg`` — step-wise Epps-Pulley regularisation of the embedding
  distribution. Returns ``gamma * pure`` as a ``ScalarTensor`` plus a
  float-valued ``SIGRegInfo``.

Per the spec, ``TotalLoss`` does not exist and is intentionally not
tested here.

These are written against the *spec*, not any implementation: a
divergent implementation should make them red. Losses run on real CPU
torch tensors with fixed seeds (no mocking of torch) per the project
testing strategy. SIGReg is stochastic (random projections), so seeds
are pinned immediately before each compared compute to keep the
distributional comparisons fair and deterministic.
"""

import pytest
import torch

from exp.loss import MSELoss, SIGReg
from exp.types.latent import BatchedLatentSequence


def _latent(tensor: torch.Tensor) -> BatchedLatentSequence:
    return BatchedLatentSequence(tensor)


class TestMSELossValue:
    def test_returns_scalar_tensor_value(self):
        # First return is the ScalarTensor value object, holding a 0-dim tensor.
        mse = MSELoss()
        pred = _latent(torch.zeros(2, 3, 4))
        target = _latent(torch.ones(2, 3, 4))

        value, _ = mse(pred, target)

        assert value.tensor.ndim == 0

    def test_mean_of_unit_difference_is_one(self):
        # pred=0, target=1 -> every (pred-target)**2 == 1 -> mean == 1.0.
        mse = MSELoss(reduction="mean")
        pred = _latent(torch.zeros(2, 3, 4))
        target = _latent(torch.ones(2, 3, 4))

        value, _ = mse(pred, target)

        torch.testing.assert_close(value.tensor, torch.tensor(1.0))

    def test_identical_inputs_give_zero(self):
        # pred == target -> squared difference is 0 everywhere.
        mse = MSELoss()
        same = torch.randn(2, 3, 4)

        value, _ = mse(_latent(same), _latent(same.clone()))

        torch.testing.assert_close(value.tensor, torch.tensor(0.0))

    def test_mean_scales_with_squared_gap(self):
        # pred=0, target=2 -> each squared diff == 4 -> mean == 4.0.
        mse = MSELoss()
        pred = _latent(torch.zeros(2, 3, 4))
        target = _latent(2.0 * torch.ones(2, 3, 4))

        value, _ = mse(pred, target)

        torch.testing.assert_close(value.tensor, torch.tensor(4.0))

    def test_sum_reduction_totals_over_all_elements(self):
        # reduction="sum": pred=0, target=1 over (2,3,4) -> 24 elements * 1.
        mse = MSELoss(reduction="sum")
        pred = _latent(torch.zeros(2, 3, 4))
        target = _latent(torch.ones(2, 3, 4))

        value, _ = mse(pred, target)

        torch.testing.assert_close(value.tensor, torch.tensor(24.0))


class TestMSELossReductionValidation:
    def test_rejects_none_reduction_at_construction(self):
        # Only "mean"/"sum" are allowed; "none" is rejected when built.
        with pytest.raises(ValueError, match="reduction must be"):
            MSELoss(reduction="none")

    def test_rejects_unknown_reduction_at_construction(self):
        # An arbitrary string is rejected at construction time.
        with pytest.raises(ValueError, match="reduction must be"):
            MSELoss(reduction="foo")


class TestMSELossShapeValidation:
    def test_rejects_mismatched_feature_dim(self):
        # Differing last (dim) axis: (2,3,4) vs (2,3,8) -> ValueError on call.
        mse = MSELoss()
        pred = _latent(torch.zeros(2, 3, 4))
        target = _latent(torch.zeros(2, 3, 8))

        with pytest.raises(ValueError, match="same shape"):
            mse(pred, target)

    def test_rejects_mismatched_seq_dim(self):
        # Differing seq axis: (2,3,4) vs (2,4,4) -> ValueError on call.
        mse = MSELoss()
        pred = _latent(torch.zeros(2, 3, 4))
        target = _latent(torch.zeros(2, 4, 4))

        with pytest.raises(ValueError, match="same shape"):
            mse(pred, target)


class TestMSELossInfo:
    def test_elementwise_has_batch_seq_shape(self):
        # info["elementwise"] is the per-position MSE -> shape (B, S).
        mse = MSELoss()
        pred = _latent(torch.zeros(2, 3, 4))
        target = _latent(torch.ones(2, 3, 4))

        _, info = mse(pred, target)

        assert info["elementwise"].shape == (2, 3)

    def test_elementwise_is_detached(self):
        # The recorded tensor is for logging/debug only -> detached.
        mse = MSELoss()
        pred = _latent(torch.zeros(2, 3, 4, requires_grad=True))
        target = _latent(torch.ones(2, 3, 4))

        _, info = mse(pred, target)

        assert info["elementwise"].requires_grad is False


class TestMSELossGradient:
    def test_gradient_flows_to_both_inputs(self):
        # No stop-grad: backward through the value reaches both prediction
        # and target tensors.
        mse = MSELoss()
        pred_tensor = torch.zeros(2, 3, 4, requires_grad=True)
        target_tensor = torch.ones(2, 3, 4, requires_grad=True)

        value, _ = mse(_latent(pred_tensor), _latent(target_tensor))
        value.tensor.backward()

        assert pred_tensor.grad is not None
        assert target_tensor.grad is not None


class TestMSELossReuse:
    def test_same_instance_handles_multiple_calls(self):
        # A functor must be reusable: a second call with new inputs works
        # and reflects the new data.
        mse = MSELoss()

        first, _ = mse(_latent(torch.zeros(2, 3, 4)), _latent(torch.ones(2, 3, 4)))
        second, _ = mse(
            _latent(torch.zeros(2, 3, 4)), _latent(2.0 * torch.ones(2, 3, 4))
        )

        torch.testing.assert_close(first.tensor, torch.tensor(1.0))
        torch.testing.assert_close(second.tensor, torch.tensor(4.0))


# --- SIGReg ---------------------------------------------------------------

# Large-N config for the distributional comparisons: enough batch (N axis)
# and projections that Gaussian vs non-Gaussian separates cleanly, not at a
# threshold edge.
_SIG_BATCH = 64
_SIG_SEQ = 8
_SIG_DIM = 16


class TestSIGRegValue:
    def test_returns_scalar_tensor_value(self):
        # First return is a ScalarTensor holding a 0-dim tensor.
        torch.manual_seed(0)
        sig = SIGReg()
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        value, _ = sig(emb)

        assert value.tensor.ndim == 0

    def test_pure_is_non_negative(self):
        # pure is a sum of squared characteristic-function residuals -> >= 0.
        torch.manual_seed(0)
        sig = SIGReg()
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        _, info = sig(emb)

        assert info["pure"] >= 0

    def test_info_fields_are_python_floats(self):
        # info carries floats (not tensors) for logging.
        torch.manual_seed(0)
        sig = SIGReg()
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        _, info = sig(emb)

        assert isinstance(info["output"], float)
        assert isinstance(info["pure"], float)

    def test_value_matches_reported_output(self):
        # The ScalarTensor's float value equals info["output"] (= gamma*pure).
        torch.manual_seed(0)
        sig = SIGReg(gamma=0.1)
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        value, info = sig(emb)

        torch.testing.assert_close(float(value), info["output"])


class TestSIGRegGammaScaling:
    def test_output_scales_linearly_with_gamma(self):
        # gamma multiplies pure. Pin the seed right before each compute so
        # both runs draw the same projections; output(gamma=2) == 2*output(1).
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        torch.manual_seed(0)
        _, info_one = SIGReg(gamma=1.0)(emb)
        torch.manual_seed(0)
        _, info_two = SIGReg(gamma=2.0)(emb)

        torch.testing.assert_close(
            info_two["output"], 2.0 * info_one["output"], rtol=1e-4, atol=1e-6
        )

    def test_pure_is_independent_of_gamma(self):
        # pure is the unscaled statistic; with matched projections it is
        # identical regardless of gamma.
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        torch.manual_seed(0)
        _, info_one = SIGReg(gamma=1.0)(emb)
        torch.manual_seed(0)
        _, info_two = SIGReg(gamma=2.0)(emb)

        torch.testing.assert_close(
            info_one["pure"], info_two["pure"], rtol=1e-4, atol=1e-6
        )

    def test_zero_gamma_gives_zero_output(self):
        # gamma=0 disables the contribution entirely.
        torch.manual_seed(0)
        sig = SIGReg(gamma=0.0)
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        value, info = sig(emb)

        assert info["output"] == 0.0
        torch.testing.assert_close(value.tensor, torch.tensor(0.0))


class TestSIGRegGradient:
    def test_gradient_flows_to_embeddings(self):
        # The regulariser must push gradient back to the embeddings
        # (projection is sampled under no_grad, but the projected product is
        # not), and that gradient must be non-trivial.
        torch.manual_seed(0)
        sig = SIGReg()
        emb_tensor = torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM, requires_grad=True)

        value, _ = sig(_latent(emb_tensor))
        value.tensor.backward()

        assert emb_tensor.grad is not None
        assert torch.any(emb_tensor.grad != 0)


class TestSIGRegDistributionalContract:
    def test_gaussian_input_has_lower_pure_than_non_gaussian(self):
        # The statistic measures deviation from a standard normal: a standard
        # Gaussian batch must score lower pure than a scaled/shifted one.
        # Match projections across both runs for a fair comparison.
        torch.manual_seed(123)
        base = torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM)
        gaussian = base
        non_gaussian = base * 5.0 + 3.0

        sig = SIGReg()
        torch.manual_seed(7)
        _, info_gaussian = sig(_latent(gaussian))
        torch.manual_seed(7)
        _, info_non_gaussian = sig(_latent(non_gaussian))

        assert info_gaussian["pure"] < info_non_gaussian["pure"]

    def test_shifted_input_is_penalised_versus_centred(self):
        # An off-mean batch (randn + 10) must score higher pure than a
        # centred standard-normal one, evidencing a live mean-penalising
        # gradient. Same base sample, matched projections.
        torch.manual_seed(321)
        base = torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM)
        centred = base
        shifted = base + 10.0

        sig = SIGReg()
        torch.manual_seed(11)
        _, info_centred = sig(_latent(centred))
        torch.manual_seed(11)
        _, info_shifted = sig(_latent(shifted))

        assert info_shifted["pure"] > info_centred["pure"]


class TestSIGRegStochasticProjection:
    def test_unseeded_repeats_differ(self):
        # Projections are resampled every call; without a pinned seed two
        # calls on the same input must not coincide.
        sig = SIGReg()
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        torch.manual_seed(0)
        first, _ = sig(emb)
        # Advance the RNG so the next draw differs.
        _ = torch.randn(1000)
        second, _ = sig(emb)

        assert float(first) != float(second)

    def test_matched_seed_repeats_coincide(self):
        # Pinning the same seed before each call reproduces the projection
        # and therefore the output exactly.
        sig = SIGReg()
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        torch.manual_seed(42)
        first, _ = sig(emb)
        torch.manual_seed(42)
        second, _ = sig(emb)

        torch.testing.assert_close(first.tensor, second.tensor)


class TestSIGRegHyperparameterValidation:
    def test_rejects_non_positive_num_projections(self):
        # At least one projection direction is required.
        with pytest.raises(ValueError, match="num_projections"):
            SIGReg(num_projections=0)

    def test_rejects_even_num_points(self):
        # num_points must be an odd integer.
        with pytest.raises(ValueError, match="num_points"):
            SIGReg(num_points=16)

    def test_rejects_too_few_num_points(self):
        # num_points must be >= 3.
        with pytest.raises(ValueError, match="num_points"):
            SIGReg(num_points=1)


class TestSIGRegHyperparameterVariation:
    def test_small_hyperparameters_still_return_scalar(self):
        # Reduced num_projections / num_points must still yield a 0-dim value.
        torch.manual_seed(0)
        sig = SIGReg(num_projections=8, num_points=5)
        emb = _latent(torch.randn(_SIG_BATCH, _SIG_SEQ, _SIG_DIM))

        value, _ = sig(emb)

        assert value.tensor.ndim == 0
