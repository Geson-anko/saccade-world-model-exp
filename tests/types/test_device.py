"""Behaviour spec for ``exp.types.device.DeviceTransferMixin``.

The mixin is an ABC whose ``to`` / ``device`` members are abstract. The
single behavioural pin here encodes the design decision that the mixin
exists to *force* subclasses to implement device transfer -- so it must
not be directly instantiable.
"""

import pytest

from exp.types.device import DeviceTransferMixin


def test_mixin_cannot_be_instantiated():
    # Abstract members make direct instantiation a TypeError. This pins the
    # "subclasses must implement device transfer" decision, not an
    # implementation detail of how the ABC is declared.
    with pytest.raises(TypeError):
        DeviceTransferMixin()  # type: ignore[abstract]
