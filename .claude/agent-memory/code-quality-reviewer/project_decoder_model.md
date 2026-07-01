---
name: decoder-model
description: ImageDecoder (可視化 read-out) / ConvDecoder (CNN トランク) の public 面と構造イディオム、実施済み refactor の記録
metadata:
  type: project
---

exp/models/decoder.py の `ImageDecoder` と exp/models/components/conv_decoder.py の
`ConvDecoder`。CLAUDE.md 決定 #6 の可視化・評価専用 detached read-out。ImageEncoder の鏡写し。

**Why:** 系列モデルの belief から画像を復元できるか (層 2 評価) を測る道具。世界モデルの
目的関数ではない。

**How to apply (public 面 = 変更禁止):**
- `ImageDecoder.__init__(latent_dim, out_channels, image_size, *, base_channels=128, init_spatial=4, init_std=0.02)`、公開属性 `latent_dim`、`@overload __call__` の 4 パターン (Latent 族 → Image 族)、`forward` の型契約。`__all__ = ["ImageDecoder"]`。
- `ConvDecoder.__init__(feature_dim, out_channels, out_size, *, base_channels=128, init_spatial=4, init_std=0.02)`、`forward(x: Tensor) -> Tensor`。`__all__ = ["ConvDecoder"]`。
- detach 契約は public pin (test_input_gradient_is_not_detached)。Decoder は入力を detach しない。detach は呼び出し側責務。

**構造イディオム (Encoder / ViT と一貫):**
- `forward` の leading 次元畳み込み/復元は `math.prod(lead) or 1` (VisionTransformer.forward と同型)。GroupNorm 採用でバッチ非依存 → 単一 Latent (N=1) が train モードでも通る (Encoder の BatchNorm と異なる点。テストが回帰ガード)。
- `forward`/`match` の順序は full-rank (BatchedLatentSequence) → 単一 (Latent) の降順、末尾に `case _: raise TypeError`。Encoder と完全一致なので趣味的書き換え不要。
- private helper: `_num_groups` (GroupNorm groups), `_resolve_num_stages` (out_size 検証)。両方 `__init__` で使うので `_CHANNEL_FLOOR` 定数とともにクラス直前へ集約 (vit.py の `_fix_init_weight` がクラス直前に置く流儀に合わせる)。

**実施済み refactor (2026-07-01, feat/2026-07-01/conv-decoder):**
- `_side_stages` (1 用途ヘルパ) を `_resolve_num_stages` 内の nested fn `side_stages` へ畳み込み。単一用途の module-level ヘルパは過剰分割。PatchEmbed が h/w 検証をインラインで書く流儀に合わせた。
- `_resolve_num_stages` をクラス後→クラス前 (`_num_groups` の隣) へ移動し構築ヘルパを一箇所に集約。
- components/__init__.py の `__all__` を「系列モデル → 画像系 (VisionTransformer, ConvDecoder) → 共通ブロック → weight」の意味順に整理 (ConvDecoder を VisionTransformer の隣へ)。`__all__` の順序変更は集合不変なので public 影響なし。docstring に "CNN デコーダ" を追記。
- 変更しなかった: forward/match (Encoder と一貫し既にクリーン)、`_num_groups`/`_CHANNEL_FLOOR` (適切に private/定数化済み)。

関連: [[vit-component]] (画像系モデルの対), [[predictor-model]]。
