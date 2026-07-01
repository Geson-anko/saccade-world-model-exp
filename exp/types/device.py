import abc
from typing import Protocol, Self

import torch

__all__ = ["DeviceLike", "DeviceTransferMixin", "SupportsDeviceTransfer"]

type DeviceLike = torch.device | str  # PEP 695 (Python 3.13)


class DeviceTransferMixin(abc.ABC):
    """サブクラスに device 転送 (to / device property) を強制する抽象基底 (状態を持たない純粋 IF)。"""

    # 非 slotted 基底が MRO に居ると子の slots=True が無効化される (__dict__ が生える)。
    # 空 slots を宣言して attrs 値オブジェクト側の slots を全チェーンで実効化する。
    __slots__ = ()

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    @abc.abstractmethod
    def to(self, device: DeviceLike) -> Self: ...


class SupportsDeviceTransfer(Protocol):
    """Device 転送 (.to(device)) を持つものを構造的に表す annotation 用 Protocol。

    DeviceTransferMixin のサブクラスに加え、torch.Tensor / torch.nn.Module も `to`
    を持つため構造適合する (継承を強制せず注釈として広く受けるための型)。
    """

    def to(self, device: torch.device) -> Self: ...
