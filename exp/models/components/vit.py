"""画像エンコーダ VisionTransformer と内部 building block。

位置符号は 2D axial RoPE、Attention は SDPA を用いる。
"""

import math
from collections.abc import Callable
from functools import partial
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.types.size import Size2d, size_2d_to_tuple

from .mlp import Mlp
from .weight import init_weights

__all__ = [
    "AxialRoPE",
    "Attention",
    "Block",
    "Mlp",
    "PatchEmbed",
    "VisionTransformer",
]


class PatchEmbed(nn.Module):
    """画像をパッチ分割し埋め込む (Conv2d で kernel=stride=patch)。

    出力は行優先 (Hp 外・Wp 内) で flatten した (B, n_patches, embed_dim)。
    """

    def __init__(
        self,
        image_size: Size2d,
        patch_size: Size2d,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        image_hw = size_2d_to_tuple(image_size)
        patch_hw = size_2d_to_tuple(patch_size)
        if image_hw[0] % patch_hw[0] != 0 or image_hw[1] % patch_hw[1] != 0:
            raise ValueError(
                f"image_size {image_hw} は patch_size {patch_hw} で割り切れない"
            )
        self.grid_hw: tuple[int, int] = (
            image_hw[0] // patch_hw[0],
            image_hw[1] // patch_hw[1],
        )
        self.num_patches: int = self.grid_hw[0] * self.grid_hw[1]
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_hw, stride=patch_hw
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class AxialRoPE(nn.Module):
    """2D axial RoPE。head_dim を y 軸 (前半)・x 軸 (後半) に 2 分割して回す。

    パッチ位置は行優先 flatten ((y, x) で x が内側) で並べる。cos/sin は half-split 方式で
    precompute し、persistent でない buffer に保持する。
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        grid_hw: tuple[int, int],
        theta: float = 10000.0,
        num_prefix_tokens: int = 0,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
        axis_dim = head_dim // 2
        n_freqs = head_dim // 4
        hp, wp = grid_hw

        i = torch.arange(n_freqs, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (2.0 * i / axis_dim))  # (n_freqs,)

        # 行優先 flatten: pos_y は外側 (行)、pos_x は内側 (列)。
        ys = torch.arange(hp, dtype=torch.float32)
        xs = torch.arange(wp, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos_y = grid_y.reshape(-1)  # (n_patches,)
        pos_x = grid_x.reshape(-1)  # (n_patches,)

        angle_y = pos_y[:, None] * inv_freq  # (n_patches, n_freqs)
        angle_x = pos_x[:, None] * inv_freq  # (n_patches, n_freqs)

        # 各軸内で half-split (lo/hi に同一角度) → 軸ごとに (n_patches, axis_dim)、
        # y 軸ぶん前半・x 軸ぶん後半に連結して (n_patches, head_dim)。
        axis_angle_y = torch.cat([angle_y, angle_y], dim=-1)
        axis_angle_x = torch.cat([angle_x, angle_x], dim=-1)
        angle = torch.cat([axis_angle_y, axis_angle_x], dim=-1)  # (n_patches, head_dim)

        # prefix トークン (CLS など) は空間位置を持たないので回転しない。角度 0 の行
        # を先頭に連結 (cos=1, sin=0 → 恒等回転) する。
        if num_prefix_tokens > 0:
            prefix = torch.zeros(num_prefix_tokens, head_dim, dtype=torch.float32)
            angle = torch.cat(
                [prefix, angle], dim=0
            )  # (num_prefix + n_patches, head_dim)

        cos = angle.cos()[None, None]  # (1, 1, num_prefix + n_patches, head_dim)
        sin = angle.sin()[None, None]
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # y 軸ブロック・x 軸ブロックに 2 分割し、各ブロック内で half-split。
        def half_split(block: torch.Tensor) -> torch.Tensor:
            lo, hi = block.chunk(2, dim=-1)
            return torch.cat([-hi, lo], dim=-1)

        y_block, x_block = x.chunk(2, dim=-1)
        return torch.cat([half_split(y_block), half_split(x_block)], dim=-1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, heads, n_patches, head_dim)
        return x * self.cos + self._rotate_half(x) * self.sin


class Attention(nn.Module):
    """Qkv 射影 → q/k に RoPE → SDPA → 出力射影。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        grid_hw: tuple[int, int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_prefix_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = AxialRoPE(
            self.head_dim, grid_hw, num_prefix_tokens=num_prefix_tokens
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)
        q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    """Pre-norm Transformer ブロック。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        grid_hw: tuple[int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        num_prefix_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(
            embed_dim,
            num_heads,
            grid_hw,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=dropout,
            num_prefix_tokens=num_prefix_tokens,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _fix_init_weight(attn_proj: nn.Linear, mlp_out: nn.Linear, layer_id: int) -> None:
    """深さに応じて Transformer 層の出力射影を 1/sqrt(2*layer_id) で縮める。

    深い積層での出力分散の増大を抑え、学習を安定させる (iJEPA 系の rescale)。
    `layer_id` は 1 始まり。

    Args:
        attn_proj: attention の出力射影 Linear。
        mlp_out: MLP の出力 Linear。
        layer_id: 層の深さ (1 始まり)。

    Raises:
        ValueError: layer_id が 0 のとき。
    """
    if layer_id == 0:
        raise ValueError(f"layer_id must be non-zero (1-based), got {layer_id}")
    scale = math.sqrt(2.0 * layer_id)
    attn_proj.weight.data.div_(scale)
    mlp_out.weight.data.div_(scale)


class VisionTransformer(nn.Module):
    """画像を符号化する Vision Transformer。

    任意の leading batch 次元 (*, C, H, W) を受け取り (*, n_patches + 1,
    embed_dim) を返す。先頭 (index 0) は集約用の CLS トークンで、以降がパッチ。位置符号は 2D axial
    RoPE (CLS は恒等回転)。
    """

    # nn.Module.__call__ は Any を返すため、forward の型を呼び出し側へ伝える
    # (型注釈のみ。runtime は nn.Module.__call__ が hooks 経由で forward に dispatch)。
    __call__: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        image_size: Size2d,
        patch_size: Size2d,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.grid_hw: tuple[int, int] = self.patch_embed.grid_hw
        self.n_patches: int = self.patch_embed.num_patches
        self.embed_dim: int = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    self.grid_hw,
                    mlp_ratio,
                    qkv_bias,
                    dropout,
                    attn_drop,
                    num_prefix_tokens=1,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.apply(partial(init_weights, init_std=init_std))
        for i, blk in enumerate(self.blocks):
            assert isinstance(blk, Block)
            _fix_init_weight(blk.attn.proj, blk.mlp.fc2, i + 1)
        # cls_token は Parameter なので self.apply(init_weights) が触らない。明示初期化。
        nn.init.trunc_normal_(self.cls_token, std=init_std)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, c, h, w = x.shape
        xf = x.reshape(math.prod(lead), c, h, w)  # math.prod(()) == 1
        t = self.patch_embed(xf)  # (N, n_patches, embed_dim)
        cls = self.cls_token.expand(t.shape[0], -1, -1)  # (N, 1, embed_dim)
        t = torch.cat([cls, t], dim=1)  # (N, n_patches + 1, embed_dim) index0=CLS
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t.reshape(*lead, self.n_patches + 1, self.embed_dim)
