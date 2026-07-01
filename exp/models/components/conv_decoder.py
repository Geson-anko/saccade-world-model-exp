"""特徴ベクトルを画像へ展開する汎用 CNN デコーダ ConvDecoder。

Linear で初期空間解像度へ射影したのち、``Upsample(nearest) → Conv2d → GroupNorm → GELU``
を段積みして解像度を倍々に拡大する。最終段は norm/活性を 挟まない linear 出力 (入力画像が standardize
済みで値域不定なため sigmoid/tanh を付けない)。
"""

import math
from functools import partial
from typing import override

import torch
import torch.nn as nn

from exp.types.size import Size2d, size_2d_to_tuple

from .weight import init_weights

__all__ = ["ConvDecoder"]

# 段ごとにチャンネルを半減させる際の下限。これ以上は減らさずクランプする。
_CHANNEL_FLOOR = 16


def _num_groups(channels: int) -> int:
    """GroupNorm の num_groups (min(32, channels))。channels は 2 冪前提。"""
    return min(32, channels)


def _resolve_num_stages(out_hw: tuple[int, int], init_spatial: int) -> int:
    """各辺が ``init_spatial * 2**N`` (N 非負整数) で表せる共通の N を返す。

    各辺の N (``side == init_spatial * 2**N`` となる非負整数、なければ None) を求め、
    両辺で一致すればそれを、いずれかが None か食い違えば ValueError を送出する。
    """

    def side_stages(side: int) -> int | None:
        if side < init_spatial or side % init_spatial != 0:
            return None
        ratio = side // init_spatial
        if ratio & (ratio - 1) != 0:  # ratio が 2 冪でない
            return None
        return ratio.bit_length() - 1

    stages_h = side_stages(out_hw[0])
    stages_w = side_stages(out_hw[1])
    if stages_h is None or stages_w is None or stages_h != stages_w:
        raise ValueError(
            f"out_size {out_hw} の各辺は init_spatial ({init_spatial}) * 2**N "
            f"(N は非負整数) で表せる必要がある"
        )
    return stages_h


class ConvDecoder(nn.Module):
    """特徴ベクトル ``(*, feature_dim)`` を画像 ``(*, C, s', s')`` へ展開する CNN デコーダ。

    ``Linear`` で ``(N, base_channels, init_spatial, init_spatial)`` へ射影し、
    ``N = log2(out_size / init_spatial)`` 段の ``Upsample(scale=2, nearest) →
    Conv2d(3x3) → GroupNorm → GELU`` で空間解像度を倍々に拡大する。段ごとに
    チャンネルを半減し ``_CHANNEL_FLOOR`` で下限クランプ、最終 ``Conv2d(3x3)`` で
    ``out_channels`` へ射影する (最終段は norm/活性なし)。

    leading 次元は本モジュールが畳み込む/復元する。``forward`` は ``(*, feature_dim)``
    を受け、``math.prod(lead) or 1`` で 4D へ潰して Conv2d 群を通し、``(*lead, C,
    s', s')`` へ戻す (``math.prod(()) == 1`` なので lead が空でも N=1 で通る。
    GroupNorm はバッチ非依存なので N=1 でも問題ない)。入力は detach しない。

    前提条件: ``out_size`` の各辺は ``init_spatial * 2**N`` (N 非負整数) で表せること。
    表せない場合は構築時に ValueError を送出する。
    """

    def __init__(
        self,
        feature_dim: int,
        out_channels: int,
        out_size: Size2d,
        *,
        base_channels: int = 128,
        init_spatial: int = 4,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        out_hw = size_2d_to_tuple(out_size)
        num_stages = _resolve_num_stages(out_hw, init_spatial)

        self._base_channels = base_channels
        self._init_spatial = init_spatial

        self.proj = nn.Linear(feature_dim, base_channels * init_spatial**2)

        stages: list[nn.Module] = []
        c_in = base_channels
        for k in range(num_stages):
            c_out = max(base_channels // 2 ** (k + 1), _CHANNEL_FLOOR)
            stages.append(nn.Upsample(scale_factor=2, mode="nearest"))
            stages.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            stages.append(nn.GroupNorm(_num_groups(c_out), c_out))
            stages.append(nn.GELU())
            c_in = c_out
        self.stages = nn.Sequential(*stages)
        # 最終射影は norm/活性なしの linear 出力 (値域を固定しない)。
        self.head = nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1)

        # 子に自己初期化コンポーネントを持たないので素直に apply してよい。
        # GroupNorm は init_weights の対象外なので既定 (weight=1, bias=0) のまま。
        self.apply(partial(init_weights, init_std=init_std))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, _ = x.shape
        z = self.proj(x)  # (*lead, base_channels * init_spatial**2)
        # Conv2d は 4D のみを受けるので leading 次元を潰す。math.prod(()) == 1。
        z = z.reshape(
            math.prod(lead) or 1,
            self._base_channels,
            self._init_spatial,
            self._init_spatial,
        )
        z = self.stages(z)
        img = self.head(z)  # (N, out_channels, s', s')
        _, c, h, w = img.shape
        return img.reshape(*lead, c, h, w)
