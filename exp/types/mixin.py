import abc
from typing import Self

import torch

__all__ = ["DeviceLike", "DeviceTransferMixin"]

type DeviceLike = torch.device | str  # PEP 695 (Python 3.13)


class DeviceTransferMixin(abc.ABC):
    """サブクラスに device 転送 (to / device property) を強制する抽象基底 (状態を持たない純粋 IF)。"""

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    @abc.abstractmethod
    def to(self, device: DeviceLike) -> Self: ...
