---
name: project-loss-module
description: exp/loss.py の public 面 (MSELoss / SIGReg / MSELossInfo / SIGRegInfo) と固定の設計判断・数値不変条件・refactor 境界メモ
metadata:
  type: project
---

`exp/loss.py` は損失 functor 2 種。`exp.loss` 直下に公開 ([[project-types-package]] の ScalarTensor を戻り値に使う)。

公開シンボル (`__all__`): `MSELoss` / `SIGReg` / `MSELossInfo` / `SIGRegInfo`。テストは `tests/test_loss.py` (spec ピン)。

**固定の設計判断 (再議論しない・refactor で崩すな):**
- functor 形式: config をコンストラクタ・data を `__call__` で受ける。基底クラス/継承なし。`@final @attrs.define(frozen=True, slots=True)`。
- 戻り値は `(ScalarTensor, info)` の 2 値。info (TypedDict) は detach/float 化の記録専用。
- `MSELossInfo.elementwise`: (B, S) detached。`MSELoss` の reduction は "mean"/"sum" のみ (`__attrs_post_init__` で検証)。
- `SIGReg`: step-wise Epps-Pulley。第1戻り値は γ·pure。`SIGRegInfo` は `output`(=γ·pure float)/`pure`(float)。
- SIGReg 数値不変条件: `Z` を標準化しない・`A` 生成のみ no_grad・`H = Z@A` は no_grad の外 (stop-grad なし)・グリッド [0,3]・台形重み・×N(=B)。これらは変えない。
- 例外メッセージの substring (`"reduction must be"` / `"same shape"` / `"single-element"` / `"num_projections"` / `"num_points"`) はテストがピンしている。変えるな。

**実施済み refactor (2026-06-30, branch feat/20260630/loss):**
- TypedDict フィールドの長い trailing inline コメントが annotation を括弧で折り返していた → コメントをフィールド上行へ。
- `with (torch.no_grad()):` の inline コメント reflow → コメントを `with` の上行へ出し `with torch.no_grad():` に。
- `_pure` 末尾の一度きり中間変数 `pure` を直接 return。
- `device,dtype` / `B,D` の unpack と `M,K` エイリアスは複数回使用 or 数式記号 (M/K/N) との対応で残置 (冗長でない)。
- 結果: 262 tests green / pyright 0 errors。public API 不変・tests 不変。
