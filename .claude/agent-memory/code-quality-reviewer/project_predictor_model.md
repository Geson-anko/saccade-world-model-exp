---
name: project-predictor-model
description: exp/models/predictor.py の Predictor 上位モデルと SequenceModel.dim 契約・FOCUS_DIM 定数の public 面と refactor 境界メモ
metadata:
  type: project
---

`exp/models/predictor.py` の `Predictor[THidden](nn.Module)` は世界モデルの系列モデル段。
`Focus` (FOCUS_DIM=3) と観測埋め込み `Latent` (latent_dim) を最終軸で cat → `input_proj`
(Linear) → 注入された `SequenceModel` → `output_proj` (Linear) で latent へ戻す。DI で
`sequence_model` を外注 (CLAUDE.md 決定 #5)。

**Why:** `(a_{<=t}, e_{<=t}), a_{t+1} -> e_hat_{t+1}` の自己回帰予測を具体化する層。

**How to apply (public 面・refactor 境界):**
- public: `Predictor`(`__all__`)、属性 `latent_dim` / `input_proj` / `output_proj`
  / `sequence_model` はテストが直接触る → 公開扱い、改名不可。
- `SequenceModel.dim` は基底の抽象 `@property` (契約)。下流が射影幅を決めるための面で、
  `MinGRU` は `self._dim` + `@property @override dim` で実装 (読み取り面は不変)。
  `test_base.py` が「dim 未実装サブクラスは instantiate 不可」を pin。
- `FOCUS_DIM` は `elements/focus.py` 定義、`exp.types` / `exp.types.elements` まで
  re-export され `__all__` 登録済み → public。マジックナンバー 3 の置換元。
- `forward` の 4-case `match focus` (Focus/BatchedFocus→step, FocusSequence/
  BatchedFocusSequence→forward、各 case で対応 Latent 型に wrap) は**あえて明示のまま**が最適。
  出力型は overload の戻り型として型付けされるため、辞書 mapping 化やヘルパ抽出は型推論を
  壊し投機的抽象化になる。2026-07-01 レビューで「これ以上触らない」と判断済み。
- 型区別の注意: `ElementSequence` と `BatchedElement` は同じ `ElementArray` エイリアス
  (`elements/base.py`) なので isinstance で区別不可。match は Focus / BatchedFocus /
  FocusSequence / BatchedFocusSequence の具象型で分岐している。
