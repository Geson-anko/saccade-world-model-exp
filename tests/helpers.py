"""Device parametrization helper for model tests.

``parametrize_device`` parametrizes a test's ``device`` argument over
CPU and every available CUDA device, so a single method verifies a model
computes on each. The first param is always ``"cpu"`` (runs everywhere;
keeps the param set non-empty without a GPU). Each ``"cuda:i"`` carries
the ``gpu`` marker for ``-m gpu`` / ``-m "not gpu"`` selection.
"""

import pytest
import torch

__all__ = ["parametrize_device"]

# CPU first (no marker; runs everywhere and keeps the param set non-empty
# without a GPU), then every available CUDA device with the ``gpu`` marker.
_DEVICE_PARAMS = [
    pytest.param("cpu"),
    *(
        pytest.param(f"cuda:{i}", marks=pytest.mark.gpu)
        for i in range(torch.cuda.device_count())
    ),
]

parametrize_device = pytest.mark.parametrize("device", _DEVICE_PARAMS)
