"""潜在 (列) を画像 (列) へ復元する可視化用 ImageDecoder。

ImageEncoder の鏡写し。``Latent`` 族の値オブジェクトを受け、内部の CNN デコーダ ``ConvDecoder`` で
``Image`` 族へ復元する。CLAUDE.md 決定 #6 の可視化・評価専用の read-out に相当する。
"""

from typing import overload, override

import torch.nn as nn

from exp.types import (
    BatchedImage,
    BatchedImageSequence,
    BatchedLatent,
    BatchedLatentSequence,
    ChannelFormat,
    Image,
    ImageSequence,
    Latent,
    LatentSequence,
    Size2d,
)

from .components.conv_decoder import ConvDecoder

__all__ = ["ImageDecoder"]


class ImageDecoder(nn.Module):
    """潜在 (列) を画像 (列) へ復元する Decoder (ImageEncoder の鏡写し)。

    入力の leading 次元に応じて 4 パターンを 1 経路で処理する: ``Latent (dim,) → Image (C,
    s', s')`` / ``BatchedLatent (batch, dim) → BatchedImage (batch, C,
    s', s')`` / ``LatentSequence (seq, dim) → ImageSequence (seq, C, s',
    s')`` / ``BatchedLatentSequence (batch, seq, dim) →
    BatchedImageSequence (batch, seq, C, s', s')``。leading 次元の畳み込み/復元は内部
    ``ConvDecoder`` が担う。

    本モジュールは入力を detach しない。CLAUDE.md #6 の detached read-out は、 呼び出し側が
    ``e.detach()`` してから渡す責務であり Decoder には焼き込まない。 ピクセル損失の逆伝播は Decoder
    自身のパラメータにのみ流れる (呼び出し側が detach していれば upstream には流れない)。

    前提条件: ``latent_dim`` は正、``out_channels`` は ChannelFormat 値 (1/3/4)、
    ``image_size`` の各辺は ``init_spatial * 2**N`` で表せること (最後の制約は
    ``ConvDecoder`` 構築時に検証される)。
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        image_size: Size2d,
        *,
        base_channels: int = 128,
        init_spatial: int = 4,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if out_channels not in tuple(ChannelFormat):
            raise ValueError(
                f"out_channels must be a ChannelFormat value "
                f"{[int(c) for c in ChannelFormat]}, got {out_channels}"
            )
        self.latent_dim: int = latent_dim  # 公開属性 (上流が入力次元を知る)
        # ConvDecoder が自己初期化するので ImageDecoder 側で init_weights を重ねない。
        self.decoder = ConvDecoder(
            latent_dim,
            out_channels,
            image_size,
            base_channels=base_channels,
            init_spatial=init_spatial,
            init_std=init_std,
        )

    # nn.Module.__call__ は Any を返すため、forward の入出力対応を呼び出し側へ
    # overload で伝える実メソッドとして __call__ を定義する (runtime は
    # super().__call__ 経由で通常どおり hooks → forward に dispatch)。
    @overload
    def __call__(self, x: Latent) -> Image: ...
    @overload
    def __call__(self, x: BatchedLatent) -> BatchedImage: ...
    @overload
    def __call__(self, x: LatentSequence) -> ImageSequence: ...
    @overload
    def __call__(self, x: BatchedLatentSequence) -> BatchedImageSequence: ...
    def __call__(
        self, x: Latent | BatchedLatent | LatentSequence | BatchedLatentSequence
    ) -> Image | BatchedImage | ImageSequence | BatchedImageSequence:
        return super().__call__(x)

    @override
    def forward(
        self, x: Latent | BatchedLatent | LatentSequence | BatchedLatentSequence
    ) -> Image | BatchedImage | ImageSequence | BatchedImageSequence:
        # detach しない (#6 の detached read-out は呼び出し側責務)。
        img = self.decoder(x.tensor)  # (*lead, C, s', s')
        match x:
            case BatchedLatentSequence():
                return BatchedImageSequence(img)
            case BatchedLatent():
                return BatchedImage(img)
            case LatentSequence():
                return ImageSequence(img)
            case Latent():
                return Image(img)
            case _:
                raise TypeError(f"unsupported input type: {type(x).__name__}")
