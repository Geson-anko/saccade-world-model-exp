---
name: nn-module-call-overload
description: nn.Module.__call__ を shape 別 overload で型付けする定石 (private Protocol + type:ignore) と pyright standard の落とし穴
metadata:
  type: feedback
---

`nn.Module` サブクラスの `__call__` を入力型ごとに overload で型付けするときの定石。

**やり方:**
- private な `Protocol` サブクラス (`_XxxCall`, `__all__` に入れない) を定義し、その中で `__call__` を `@overload` 複数本で書く。
- クラス本体では `__call__: _XxxCall` の annotation のみ (代入なし)。runtime は `nn.Module.__call__` が hooks 経由で forward に dispatch するため実挙動は変わらない。
- `forward` 自体は overload せず Union 型 1 本 (引数も戻り値も Union) で実装。出力ラップは `match x: case ConcreteClass(): ...` で分岐 (frozen attrs 具象クラスは互いに素なので判別可能)。

**Why / 落とし穴:** `Callable[[X], Y]` 形の annotation では pyright standard で `reportIncompatibleMethodOverride` が出ないが、`Protocol` (overload 付き) を `__call__` に当てると `nn.Module.__call__` (= `Callable[..., Any]`) との override 互換判定に引っかかりエラーになる。回避は当該行のみ `# type: ignore[reportIncompatibleMethodOverride]` (意図的な型絞り込み)。これは Protocol overload に本質的に伴うもので、プロジェクト全体で pyright standard・`typeCheckingMode = "standard"`。

**How to apply:** encoder/decoder/sequence-model など `exp/models/` で複数 shape を受ける Module を型付けするとき同じ形にする。overload 解決の確認は `reveal_type` を書いた一時スクリプトに `uv run pyright` を掛けると各戻り型が information で出る。実例: `exp/models/encoder.py` の `_ImageEncoderCall`。参考にした既存 overload は `exp/types/elements/base.py` の `ElementArray.__getitem__`。
