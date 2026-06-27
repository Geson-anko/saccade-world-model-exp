---
name: attrs-tensor-eq
description: torch.Tensor を field に持つ attrs クラスは eq=False が必須 (自動 __eq__ が壊れる)
metadata:
  type: feedback
---

`torch.Tensor` を field に持つ `@attrs.define` クラスは `eq=False` を付ける。

**Why:** attrs 自動生成の `__eq__` は `self.t == other.t` を bool 評価するが、tensor の `==` は要素ごとの bool tensor を返すため `Boolean value of Tensor ... is ambiguous` を投げる。しかもクラス定義時ではなく `a == b` を評価した瞬間に初めて落ちる (静かな破損)。shape 違いだと別の謎エラーになる。

**How to apply:** 値オブジェクトには `@attrs.define(slots=True, frozen=True, eq=False)` (identity 比較)。値等価が要るなら field 単位で `attrs.field(eq=attrs.cmp_using(eq=torch.equal))` を使う (ただし torch.equal は dtype 盲目で float32==float64 が True になる)。テストでの値一致は `==` でなく `torch.equal` / `torch.testing.assert_close` を直接使う。実例は `exp/types/image.py` の `Image`。
