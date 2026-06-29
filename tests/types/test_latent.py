"""Behaviour spec for ``exp.types.latent``.

These tests translate the approved spec for the immutable
``BatchedLatentSequence`` value object into executable form. They are
written against the *spec*, not any implementation: an implementation
that diverges should make these red. Scenarios are grouped into one
class per behaviour area.

Tensors are constructed deterministically on real CPU torch (no mocking
of torch) per the project testing strategy.

Unlike the image sequence types, ``BatchedLatentSequence`` intentionally
provides no indexing / iteration / device-transfer surface (the spec
defers those until a single-entry latent type exists), so none is
exercised here.
"""

import pytest
import torch
from exp.types.latent import BatchedLatentSequence


class TestConstruction:
    def test_accepts_3d_tensor(self):
        # (batch, seq, dim) is the one valid rank; the tensor passes through.
        latent = BatchedLatentSequence(torch.zeros(2, 3, 16))

        assert latent.tensor.shape == (2, 3, 16)

    def test_rejects_2d_tensor(self):
        # A 2D tensor lacks the batch axis required by (batch, seq, dim).
        with pytest.raises(ValueError, match="batch, seq, dim"):
            BatchedLatentSequence(torch.zeros(3, 16))

    def test_rejects_4d_tensor(self):
        # A 4D tensor carries an extra axis the value object forbids.
        with pytest.raises(ValueError, match="batch, seq, dim"):
            BatchedLatentSequence(torch.zeros(2, 3, 16, 4))

    def test_error_reports_actual_shape(self):
        # The message must surface the offending shape (substring check only).
        with pytest.raises(ValueError) as exc:
            BatchedLatentSequence(torch.zeros(3, 16, 8, 4))

        assert "batch, seq, dim" in str(exc.value)
        assert "3" in str(exc.value) and "16" in str(exc.value)
        assert "8" in str(exc.value) and "4" in str(exc.value)


class TestEquality:
    def test_identity_for_same_instance(self):
        # eq=False contract: an instance compares equal only to itself.
        latent = BatchedLatentSequence(torch.zeros(2, 3, 16))
        assert latent == latent

    def test_distinct_instances_are_not_equal(self):
        # Equal-valued but distinct instances must NOT be equal, and the
        # comparison must not raise "boolean value of Tensor is ambiguous".
        assert BatchedLatentSequence(torch.zeros(2, 3, 16)) != BatchedLatentSequence(
            torch.zeros(2, 3, 16)
        )
