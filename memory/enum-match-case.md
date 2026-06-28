---
name: enum-match-case
description: enum などのパターンマッチ分岐は if/elif でなく match-case をできる限り使う (網羅性チェックが効く)
metadata:
  type: feedback
---

enum (や値による場合分け) の分岐は `if/elif` ではなく `match-case` をできる限り使う。

**Why:** match-case は enum メンバーを全列挙したとき pyright が網羅性 (exhaustiveness) を静的に検証でき、メンバー追加時に未処理の case を型エラーで検出できる。`case _` を置かず全メンバーを明示すれば、分岐内での変数代入漏れや関数の暗黙 None 返却も pyright が拾う。

**How to apply:** `match value:` ＋ 各 `case EnumMember:` を全メンバー分書く (網羅時は `case _` 不要)。実例は `exp/types/image.py` の `as_channel_format` / `_as_rgb` (`ChannelFormat` 分岐) と `save` (拡張子分岐)。関連: [[shadow-builtin-method]]。
