from __future__ import annotations

from typing import TypedDict, final

import attrs
import torch

from .types import BatchedImageSequence, BatchedLatentSequence, ScalarTensor

__all__ = ["MSELoss", "SIGReg", "MSELossInfo", "SIGRegInfo"]


class MSELossInfo(TypedDict):
    """MSELoss の記録/デバッグ専用 info。tensor は detach、scalar は float 化済み。"""

    output: float  # MSE 値 (第 1 戻り値の float)
    # (B, S) detached: 位置ごとの MSE。先頭 2 軸 (batch, seq) を除く全特徴軸で
    # mean する (latent は D、image は C,H,W)。
    elementwise: torch.Tensor


class SIGRegInfo(TypedDict):
    """SIGReg の記録/デバッグ専用 info。scalar は float 化済み。"""

    output: float  # γ·pure (第 1 戻り値の float)
    pure: float  # 未スケールの SIGReg 統計量


@final
@attrs.define(frozen=True, slots=True)
class MSELoss:
    """潜在系列の平均二乗誤差を計算する functor。

    config (reduction) をコンストラクタで受け取り、data (prediction, target) を
    ``__call__`` で受け取る。
    """

    reduction: str = "mean"

    def __attrs_post_init__(self) -> None:
        if self.reduction not in ("mean", "sum"):
            raise ValueError(
                f'MSELoss reduction must be "mean" or "sum", got {self.reduction!r}'
            )

    def __call__[T: (BatchedLatentSequence, BatchedImageSequence)](
        self,
        prediction: T,
        target: T,
    ) -> tuple[ScalarTensor, MSELossInfo]:
        if prediction.tensor.shape != target.tensor.shape:
            raise ValueError(
                f"MSELoss expects prediction and target of the same shape, got "
                f"prediction={tuple(prediction.tensor.shape)} "
                f"target={tuple(target.tensor.shape)}"
            )
        diff2 = (
            prediction.tensor - target.tensor
        ) ** 2  # latent (B,S,D) / image (B,S,C,H,W)
        value = diff2.mean() if self.reduction == "mean" else diff2.sum()
        # 先頭 2 軸 (batch, seq) を除く全特徴軸で mean → (B, S)
        elementwise = diff2.mean(dim=tuple(range(2, diff2.ndim))).detach()
        info: MSELossInfo = {
            "output": float(value),
            "elementwise": elementwise,
        }
        return ScalarTensor(value), info


@final
@attrs.define(frozen=True, slots=True)
class SIGReg:
    """step-wise Epps-Pulley による埋め込み分布の正則化を計算する functor。

    各 step の埋め込み (N=batch 軸) をランダム射影し、参照特性関数 (標準正規) との
    Epps-Pulley 統計量を求める。射影方向 ``A`` は毎 ``__call__`` 再サンプリングし、
    その生成・正規化のみ ``torch.no_grad()`` 内で行う。射影 ``H = Z @ A`` は no_grad の
    外なので埋め込みへ勾配が流れる (stop-grad なし)。
    """

    num_projections: int = 1024  # M
    num_points: int = 17  # K (奇数)
    gamma: float = 0.1  # ≈ λ。固定 (学習中不変)

    def __attrs_post_init__(self) -> None:
        if self.num_projections < 1:
            raise ValueError(
                f"SIGReg num_projections must be >= 1, got {self.num_projections}"
            )
        if self.num_points < 3 or self.num_points % 2 == 0:
            raise ValueError(
                f"SIGReg num_points must be an odd int >= 3, got {self.num_points}"
            )

    def __call__(
        self, embeddings: BatchedLatentSequence
    ) -> tuple[ScalarTensor, SIGRegInfo]:
        pure = self._pure(embeddings)
        scaled = self.gamma * pure
        info: SIGRegInfo = {"output": float(scaled), "pure": float(pure)}
        return ScalarTensor(scaled), info

    def _pure(self, embeddings: BatchedLatentSequence) -> torch.Tensor:
        z = embeddings.tensor  # (B, S, D)
        Z = z.transpose(0, 1)  # (S, B, D)  N 軸 = B
        device, dtype = Z.device, Z.dtype
        B, D = z.shape[0], z.shape[-1]
        M, K = self.num_projections, self.num_points
        # 方向は学習対象でない。Z @ A は no_grad の外なので Z へ grad は流れる
        with torch.no_grad():
            A = torch.randn(D, M, device=device, dtype=dtype)
            A = A / A.norm(p=2, dim=0, keepdim=True)  # 列ごと単位ベクトル
        H = Z @ A  # (S, B, M)
        t = torch.linspace(0.0, 3.0, K, device=device, dtype=dtype)  # (K,)
        phi = torch.exp(-0.5 * t * t)  # (K,) 参照 CF かつ重み窓
        dt = 3.0 / (K - 1)
        trap = torch.full((K,), 2 * dt, device=device, dtype=dtype)
        trap[0] = dt
        trap[-1] = dt
        weights = trap * phi  # (K,)
        phase = H.unsqueeze(-1) * t  # (S, B, M, K)
        cos_mean = phase.cos().mean(dim=1)  # (S, M, K)  Re φ_N (N=B 軸で平均)
        sin_mean = phase.sin().mean(dim=1)  # (S, M, K)  Im φ_N
        err = (cos_mean - phi) ** 2 + sin_mean**2  # (S, M, K)
        return ((err @ weights) * B).mean()  # () : ×N(=B) → S・M 平均
