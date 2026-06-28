---
name: fail-loud-no-implicit-fallback
description: 暗黙的なエラー回避・自動フォールバック (フォーマット/型の勝手な切替) は禁忌。不整合は握りつぶさず明示的に raise する
metadata:
  type: feedback
---

不整合や非対応の入力に対し、**暗黙的にエラーを回避したり別の挙動へ自動フォールバックするのは禁忌**。明示的に例外を投げ、呼び出し側に判断させる。

**Why:** オーナーの強い方針 (「暗黙的にエラーを回避する行為は禁忌」)。暗黙の回避 (例: JPEG に alpha 付き画像を渡されたとき勝手に PNG へ切り替えて保存する、サイレントに型/チャンネルを変換する等) はバグを隠し、呼び出し側の意図と異なる結果を静かに生む。CLAUDE.md の「混乱を隠さない」の具体化。

**How to apply:** 拡張子・dtype・チャンネル等の不整合は処理前に検出して `ValueError` 等を投げる。ライブラリ任せの曖昧なエラーに頼らず、プロジェクトのコードで明示的に弾く。実例は `exp/types/image.py` の `save` (JPEG×RGBA を明示的に ValueError)。関連: [[enum-match-case]]。
