---
name: project-focus-types
description: exp/types/focus.py の public 面と設計不変条件。Focus 行動値オブジェクト + FocusSequence / BatchedFocusSequence の refactor 境界メモ
metadata:
  type: project
---

`exp/types/focus.py` はサッカード行動 `a=(point, zoom)` の値オブジェクト群。[[project-types-package]] と同じ基盤層に属し、`image.py`（`ImageSequence`/`BatchedImageSequence`）が手本。

public 面（契約テストと spec テストでピン、絶対不変）:
- `Focus`（`@attrs.define(frozen=True)`、eq=True で hashable）: `point: tuple[float,float]`・`zoom: float`、`tensor()`→`(3,)` float32 CPU、`__call__(image)→Image`（crop は `square_pad()` 後に実施）。
- `FocusSequence`/`BatchedFocusSequence`（`@final` + `@attrs.define(slots=True, frozen=True, eq=False)` + `DeviceTransferMixin`）: `tensor` を内包、`(seq,3)`/`(batch,seq,3)`、`device`/`to`/`is_valid`/`validate`/`from_focuses`/`from_sequences`/`apply(image,size)`。空系列（要素0）は許容。
- エラーメッセージ substring がテストで pin: `(seq, 3)` / `(batch, seq, 3)`（+ndim・shape）/ `out of range`（validate）/ `at least one`（from_* 空入力）/ Focus zoom の `[0, 1]`・point の `[-1, 1]`。

設計不変条件:
- **zoom は閉区間 [0, 1]**（point は [-1, 1]）。2026-06-30 に旧 `(0, 1]` から `[0, 1]` へ変更（zoom=0 を valid 下限として許容）。spec テスト `test_accepts_zero_zoom` が pin。
- `_focus_tensor_is_valid(t)` が 2 クラスの唯一の検証 helper（point/zoom 範囲を 0-dim bool tensor の `&` で判定）。`validate` のエラーメッセージは 2 クラスで完全重複だが、手本 image.py の `from_images`/`from_sequences` と同流儀でクラスごとに独立保持（「2 回まで OK」・薄い helper 化は過剰）。

**Why:** 系列モデルへ流す行動系列の基盤型。Hyrum's law 緩和で public 表面が契約・spec テストで固定。

**How to apply:** refactor 時は上記 public シンボル名・シグネチャ・エラー substring・`DeviceTransferMixin` 継承・相対 import を変えない。命名/コメント密度/post_init メッセージ形式/`from_*` の `list()`→空チェック→`torch.stack` 構造は image.py に馴染ませる。2026-06-30 レビュー: `_focus_tensor_is_valid` を `point_ok`/`zoom_ok` の 2 行に分離（明示性）、`from_focuses`/`from_sequences` の内包表記変数を `f`/`s`→`focus`/`sequence` に揃えて image.py に一致（一貫性）。`apply` の `Focus(...)(image).resize(size)` 再構築は crop ロジック再利用のため正しく、保留。
