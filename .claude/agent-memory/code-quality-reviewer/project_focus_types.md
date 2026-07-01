---
name: project-focus-types
description: exp/types/elements/focus.py の public 面と設計不変条件。Focus 行動値オブジェクト + FocusSequence / BatchedFocus / BatchedFocusSequence の refactor 境界メモ
metadata:
  type: project
---

`exp/types/elements/focus.py` はサッカード行動 `a=(point, zoom)` の値オブジェクト群。[[project-element-base]] の基底を継承し、image.py が手本。（旧パス `exp/types/focus.py` は 2026-06 の elements/ サブパッケージ移設で廃止）

public 面（契約テストと spec テストでピン、絶対不変）:
- `Focus`（`@final`+`@attrs.define(slots=True,frozen=True,eq=False)`, Element 継承, `_SHAPE=[3]`）: `point`/`zoom` property、`init(point,zoom)`(値域検証つき classmethod)、`tensor` 属性→`(3,)` float32、`__call__(image)→Image`（`square_pad()` 後に crop）。Focus(tensor) 直接構築は shape のみ検証し値域は検証しない。
- `FocusSequence`/`BatchedFocus`/`BatchedFocusSequence`（`_FocusValidation` mixin + base 収集型）: `_SHAPE=[...,3]`、`is_valid`/`validate`、`apply`(FocusSequence のみ)。空系列許容。
- エラーメッセージ substring pin: `Focus point must be within [-1, 1]`/`Focus zoom must be within [0, 1]`（init）/`out of range`（validate）/base 由来 `(seq, 3)`/`(batch, 3)`/`(batch, seq, 3)`/`at least one ...`。

設計不変条件:
- **zoom は閉区間 [0, 1]**（point は [-1, 1]）。2026-06-30 に旧 `(0, 1]` から `[0, 1]` へ変更。spec テストが pin。
- `_FocusValidation` mixin が 3 収集クラスの唯一の検証実体（rule-of-three を満たす正当な mixin）。`is_valid` は point/zoom を 0-dim bool tensor の `&` で判定。`apply` は `focus(image).resize(size)` を `ImageSequence.from_elements(...)` に流す（crop ロジック再利用のため正しい）。

**Why:** 系列モデルへ流す行動系列の基盤型。Hyrum's law 緩和で public 表面が契約・spec テストで固定。

**How to apply:** refactor 時は上記 public シンボル名・シグネチャ・エラー substring・`DeviceTransferMixin`(base 経由) 継承・相対 import を変えない。2026-07-01 レビュー: (1) 単一呼び出し元しかない module-level `_focus_tensor_is_valid` を `is_valid` 本体へインライン化（過剰抽象化除去、private・非直テスト・呼び出し元1で安全）。(2) `__all__` を package 支配的な多行+trailing comma へ揃えた。docformatter が `_FocusValidation` docstring の `point∈`→`Point∈` 自動 capitalize・Focus docstring の zoom 行を折り返し（触れると自動発生・許容）。
