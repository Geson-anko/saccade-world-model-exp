"""Behaviour spec for ``exp.types.tensor``.

Translates the approved spec for the immutable ``ScalarTensor`` value
object into executable form. ``ScalarTensor`` wraps a single-element
tensor (the loss-value return type), normalising it to shape ``()``
while preserving the autograd path, and exposes ``float`` / ``int``
casts.

These are written against the *spec*, not any implementation: a
divergent implementation should make them red. Tensors are constructed
deterministically on real CPU torch (no mocking of torch) per the
project testing strategy. Scenarios are grouped one class per behaviour
area.
"""

import pytest
import torch

from exp.types.tensor import ScalarTensor


class TestConstruction:
    def test_accepts_zero_dim_tensor(self):
        # shape () already has numel()==1 and is the canonical form.
        st = ScalarTensor(torch.tensor(3.5))

        assert st.tensor.shape == ()

    def test_accepts_singleton_1d_tensor_and_normalises_to_scalar(self):
        # (1,) has numel()==1; it is accepted and reshaped to ().
        st = ScalarTensor(torch.zeros(1))

        assert st.tensor.shape == ()

    def test_accepts_singleton_2d_tensor_and_normalises_to_scalar(self):
        # (1, 1) also has numel()==1; accepted and reshaped to ().
        st = ScalarTensor(torch.zeros(1, 1))

        assert st.tensor.shape == ()

    def test_rejects_1d_tensor_with_multiple_elements(self):
        # (2,) has numel()==2; rejected. Message names "single-element"
        # and surfaces the offending shape (substring checks only).
        with pytest.raises(ValueError, match="single-element") as exc:
            ScalarTensor(torch.zeros(2))

        assert "2" in str(exc.value)

    def test_rejects_2d_tensor_with_multiple_elements(self):
        # (2, 3) has numel()==6; rejected with the same contract.
        with pytest.raises(ValueError, match="single-element") as exc:
            ScalarTensor(torch.zeros(2, 3))

        assert "2" in str(exc.value) and "3" in str(exc.value)


class TestScalarCasts:
    def test_float_returns_python_float_value(self):
        # float(st) yields the contained scalar as a Python float.
        assert float(ScalarTensor(torch.tensor(3.5))) == 3.5

    def test_int_truncates_toward_zero(self):
        # int(st) follows Python/torch int() semantics: truncation.
        assert int(ScalarTensor(torch.tensor(3.9))) == 3


class TestGradientFlow:
    def test_backward_through_tensor_reaches_input(self):
        # Normalisation uses reshape (a view), so the autograd path is
        # preserved: backward through st.tensor populates the input grad.
        x = torch.tensor(2.0, requires_grad=True)
        st = ScalarTensor(x)

        st.tensor.backward()

        assert x.grad is not None


class TestEquality:
    def test_identity_for_same_instance(self):
        # eq=False contract: an instance compares equal only to itself.
        st = ScalarTensor(torch.tensor(1.0))

        assert st == st

    def test_distinct_instances_with_equal_value_are_not_equal(self):
        # Equal-valued but distinct instances must NOT be equal (eq=False),
        # and the comparison must not raise on tensor ambiguity.
        assert ScalarTensor(torch.tensor(1.0)) != ScalarTensor(torch.tensor(1.0))
