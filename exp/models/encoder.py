"""観測画像列を 1 グリンプス 1 潜在ベクトルへ符号化する ImageEncoder。

LeWorldModel に倣い、ViT の CLS トークンを取り出し Linear 射影 → BatchNorm で
観測埋め込みを作る。BatchNorm をバッチ方向正規化として最終操作に置くことで、崩壊防止 SIGReg
(埋め込み分布をバッチ方向で等方ガウスに寄せる) と整合させる。
"""

from collections.abc import Callable
from typing import override

import torch.nn as nn

from exp.types.image import BatchedImageSequence
from exp.types.latent import BatchedLatentSequence
from exp.types.size import Size2d

from .components.vit import VisionTransformer
from .components.weight import init_weights

__all__ = ["ImageEncoder"]


class ImageEncoder(nn.Module):
    """観測画像列を潜在系列へ符号化する Encoder。

    入力 ``BatchedImageSequence`` の ``(batch, seq, C, H, W)``
    を受け取り、各グリンプスを 1 つの潜在ベクトルへ落とした ``BatchedLatentSequence`` の ``(batch,
    seq, latent_dim)`` を返す。経路は ViT CLS トークン → Linear → BatchNorm
    (活性は挟まない)。

    BatchNorm は ``(batch * seq, latent_dim)`` を 1 母集団として適用する
    (全グリンプス埋め込みを 1 分布として統計を取り SIGReg と整合させるため)。train
    時はバッチ内グリンプスに依存し、eval 時は running stats で決定的になる (LeWorldModel
    と同じ意図的挙動)。

    前提条件:     ``image_size`` は内部 ViT の ``patch_size`` (既定 16)
    で割り切れること。割り切れない場合は ViT     構築時に ValueError が送出される。学習は実質 ``batch *
    seq >> 1`` を想定する (train 時     BatchNorm の分散が定義されるため)。
    """

    # nn.Module.__call__ は Any を返すため、forward の型を呼び出し側へ伝える
    # (型注釈のみ。runtime は nn.Module.__call__ が hooks 経由で forward に dispatch)。
    __call__: Callable[[BatchedImageSequence], BatchedLatentSequence]

    def __init__(
        self,
        image_size: Size2d,
        in_channels: int,
        latent_dim: int,
        *,
        patch_size: Size2d = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.latent_dim: int = latent_dim  # 公開属性 (下流が入力次元を知る)
        # ViT のアーキは LeWorldModel 準拠の既定値 (上書き可)。num_heads は
        # head_dim (= embed_dim / num_heads) が 4 の倍数になるよう選ぶこと
        # (例 192/3=64。AxialRoPE 制約)。崩せば VisionTransformer 構築時に ValueError。
        self.vit = VisionTransformer(
            image_size,
            patch_size,
            in_channels,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio=mlp_ratio,
            init_std=init_std,
        )  # 構築時に自己初期化 (_fix_init_weight 含む)
        self.proj = nn.Linear(self.vit.embed_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)
        # self.apply(init_weights) は使わない。子の vit を再初期化し ViT の
        # _fix_init_weight rescale を壊すため、新規層 (proj) だけ直接初期化する。
        # bn は PyTorch 既定 (weight=1, bias=0) のままで十分。
        init_weights(self.proj, init_std=init_std)

    @override
    def forward(self, x: BatchedImageSequence) -> BatchedLatentSequence:
        t = self.vit(x.tensor)  # (B, S, n_patches + 1, embed_dim)
        cls = t[..., 0, :]  # (B, S, embed_dim) CLS トークン
        z = self.proj(cls)  # (B, S, latent_dim)
        b, s = z.shape[0], z.shape[1]
        # BatchNorm1d は (B, S, dim) を (N, C, L) と解釈するため、明示 reshape で
        # (B * S, latent_dim) を 1 母集団として正規化する。
        z = self.bn(z.reshape(b * s, self.latent_dim)).reshape(b, s, self.latent_dim)
        return BatchedLatentSequence(z)
