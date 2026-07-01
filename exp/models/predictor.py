"""行動と観測埋め込みの履歴から次グリンプスの潜在を予測する Predictor。

CLAUDE.md 決定事項 #5「Encoder は観測 ``o`` のみを見る。``a`` は系列モデル側で 注入する」に従い、行動
``Focus`` と観測埋め込み ``Latent`` を系列モデルの入口で 結合する。``sequence_model``
は差し替え可能な部品として外から注入する (DI)。系列 軸のない ``Focus`` / ``BatchedFocus`` は
``sequence_model.step`` へ、系列軸を持つ ``FocusSequence`` /
``BatchedFocusSequence`` は ``sequence_model.forward`` へ振る。
"""

from typing import overload

import torch
import torch.nn as nn

from exp.types import (
    FOCUS_DIM,
    BatchedFocus,
    BatchedFocusSequence,
    BatchedLatent,
    BatchedLatentSequence,
    Focus,
    FocusSequence,
    Latent,
    LatentSequence,
)

from .components import SequenceModel, init_weights

__all__ = ["Predictor"]


class Predictor[THidden](nn.Module):
    """``(a_{≤t}, e_{≤t}), a_{t+1} → ê_{t+1}`` を具体化した上位モデル。

    行動 ``Focus`` (``FOCUS_DIM``) と観測埋め込み ``Latent`` (``latent_dim``) を
    最終軸で結合し、``Linear`` で ``sequence_model.dim`` へ射影してから注入された
    ``sequence_model`` に通し、出力を ``Linear`` で ``latent_dim`` へ戻して潜在を
    予測する。出力の値オブジェクト型は入力 ``focus`` の型から決まる。

    Attributes:
        latent_dim: 観測埋め込み・予測潜在の特徴次元。
    """

    def __init__(
        self,
        latent_dim: int,
        sequence_model: SequenceModel[THidden],
        *,
        init_std: float = 0.02,
    ) -> None:
        """Predictor を構築する。

        Raises:
            ValueError: ``latent_dim`` が正でないとき (潜在次元として成立しないため)。
        """
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim: int = latent_dim
        self.sequence_model = sequence_model
        self.input_proj = nn.Linear(FOCUS_DIM + latent_dim, sequence_model.dim)
        self.output_proj = nn.Linear(sequence_model.dim, latent_dim)
        # 注入された sequence_model の初期化には触れず、新規 Linear のみ初期化する。
        init_weights(self.input_proj, init_std=init_std)
        init_weights(self.output_proj, init_std=init_std)

    @overload
    def __call__(
        self, focus: Focus, latent: Latent, hidden: THidden | None = None
    ) -> tuple[Latent, THidden]: ...
    @overload
    def __call__(
        self,
        focus: BatchedFocus,
        latent: BatchedLatent,
        hidden: THidden | None = None,
    ) -> tuple[BatchedLatent, THidden]: ...
    @overload
    def __call__(
        self,
        focus: FocusSequence,
        latent: LatentSequence,
        hidden: THidden | None = None,
    ) -> tuple[LatentSequence, THidden]: ...
    @overload
    def __call__(
        self,
        focus: BatchedFocusSequence,
        latent: BatchedLatentSequence,
        hidden: THidden | None = None,
    ) -> tuple[BatchedLatentSequence, THidden]: ...
    def __call__(
        self,
        focus: Focus | BatchedFocus | FocusSequence | BatchedFocusSequence,
        latent: Latent | BatchedLatent | LatentSequence | BatchedLatentSequence,
        hidden: THidden | None = None,
    ) -> tuple[
        Latent | BatchedLatent | LatentSequence | BatchedLatentSequence, THidden
    ]:
        # nn.Module.__call__ は Any を返すため、forward の戻り型を呼び出し側へ伝える
        # 薄い委譲。runtime は super().__call__ が hooks 経由で forward に dispatch。
        return super().__call__(focus, latent, hidden)

    def forward(
        self,
        focus: Focus | BatchedFocus | FocusSequence | BatchedFocusSequence,
        latent: Latent | BatchedLatent | LatentSequence | BatchedLatentSequence,
        hidden: THidden | None = None,
    ) -> tuple[
        Latent | BatchedLatent | LatentSequence | BatchedLatentSequence, THidden
    ]:
        x = self.input_proj(torch.cat([focus.tensor, latent.tensor], dim=-1))
        match focus:
            case Focus():
                raw, hidden = self.sequence_model.step(x, hidden)
                return Latent(self.output_proj(raw)), hidden
            case BatchedFocus():
                raw, hidden = self.sequence_model.step(x, hidden)
                return BatchedLatent(self.output_proj(raw)), hidden
            case FocusSequence():
                raw, hidden = self.sequence_model.forward(x, hidden)
                return LatentSequence(self.output_proj(raw)), hidden
            case BatchedFocusSequence():
                raw, hidden = self.sequence_model.forward(x, hidden)
                return BatchedLatentSequence(self.output_proj(raw)), hidden
            case _:
                raise TypeError(f"Unsupported focus type: {type(focus).__name__}")
