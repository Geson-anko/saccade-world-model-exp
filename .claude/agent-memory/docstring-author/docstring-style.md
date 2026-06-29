---
name: docstring-style
description: saccade-world-model-exp の docstring 様式 (日本語ベース・reST ref・記号原形・shape 契約)
metadata:
  type: project
---

`exp/` 配下の docstring は **日本語ベース**で書く (CLAUDE.md は英語識別子だが、docstring 本文は日本語が既存慣習)。

**How to apply:**

- **言語:** 本文は日本語。識別子・記号は原形のまま (`p`, `z`, `a`, `o`, `e`, `b_t`, `h̃`, `z_t`, `dim`, `depth`)。数式は ``h_t = (1−z_t)·h_{t−1} + z_t·h̃_t`` のように reST double-backtick で囲む。
- **クロスリファレンス:** reST ロールを使う。`:class:`SequenceModel``、`:class:`~torch.nn.Module``。Google style の `Args:`/`Returns:`/`Raises:`/`Attributes:` セクションと併用する (Sphinx napoleon 流の混在が既存スタイル)。
- **shape 契約を明示:** 系列モデル系は引数・戻り値の tensor shape を `(*, len, input_dim)` のように必ず書く。任意 leading batch は `(*, ...)` で表す。これは読み手が最も必要とする情報。
- **密度:** 1 行 summary で足りる building block (`Mlp`, `RMSNorm`) は 1 行に留める。非自明な数値安定性・lifecycle・belief 形状などは body / `Attributes:` / `Raises:` を足す。
- **非自明コメント:** torch.where の両ブランチ評価で NaN が backward に逆流するから relu でクランプ、等「なぜ」を 1 行で。`# ...` は why のみ、what は書かない。
- **行長 88、double quote。** docstring 追記で伸びても `uv run ruff format` に従う (PEP257 summary + 空行 + body)。
- **private helper にも 1 行 docstring を付けてよい** (`_g` / `_log_g` / `_parallel_scan_log` は付いている)。意図 (連続正値活性・log-space scan 等) を 1 行で。

検証: `uv run ruff check && uv run ruff format --check` / `uv run pyright` / `uv run pytest tests/models -q` がすべて green であること。doctest は実行されない (`--doctest-modules` 無効)。
