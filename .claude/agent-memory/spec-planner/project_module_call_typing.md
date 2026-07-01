---
name: project-module-call-typing
description: nn.Module.__call__ の型付け方針 (Callable annotation trick と Protocol overload) — Encoder 系仕様で踏襲する
metadata:
  type: project
---

`nn.Module.__call__` は `Any` を返すため、forward の型を呼び出し側へ伝える型付け trick をプロジェクト標準とする。実装ファイルは `exp/models/`。

- **単一シグネチャで足りる場合**: クラス本体に `__call__: Callable[[In], Out]` という **annotation のみ (代入なし)** を書く。runtime 無影響、pyright 向けのみ。`exp/models/encoder.py` L40 が原型。
- **複数入出力対応 (overload) が要る場合**: `Callable` は複数シグネチャを表現できないため、private Protocol に `__call__` を `@overload` 定義し、そのインスタンス型を annotation に使う (typeshed 標準パターン)。Protocol 名は private 規約で `_`prefix・`__all__` 非掲載 (例 `_ImageEncoderCall`)。`forward` 自体は overload せず Union 型 1 本で足りる (直接 `.forward()` 呼びが無く hooks 経由 dispatch のため)。

**Why:** nn.Module の型を呼び出し側に伝えつつ、private 規約・シンプルさ優先 (forward 二重 overload を避ける) を両立するため。2026-07-01 の ImageEncoder.__call__ overload 化仕様で確定。
**How to apply:** Encoder/Decoder 等 nn.Module 派生の入出力型を型レベルで公開する仕様では、単一なら Callable annotation、複数対応なら private Protocol overload を採る。

値オブジェクトのラップ分岐は `if/elif isinstance` ではなく `match x: case ClassName():` を使う (4 具象 Element クラスは互いに素・全て `@final` なので case 順序は可読性優先で自由、`case _` で TypeError フォールバック)。関連: [[project-image-value-objects]] [[project-testing-conventions]]
