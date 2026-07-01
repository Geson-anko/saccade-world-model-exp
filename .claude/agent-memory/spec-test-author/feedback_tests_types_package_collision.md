---
name: tests-types-package-collision
description: tests/types/** が stdlib `types` と衝突して collection error になる harness 問題と回避
metadata:
  type: feedback
---

`tests/types/` 配下に `__init__.py` があり `tests/__init__.py` が無いと、pytest
(default `prepend` import mode) は test module を top-level package `types` 下
(`types.elements.test_image` など) として import しようとし、stdlib の `types`
と衝突して `ModuleNotFoundError: ... 'types' is not a package` で **全 collection
が落ちる**。`tests/types/**` の全ファイルが対象になり、個別ファイルだけを
`uv run pytest <file>` しても再現する。

**Why:** elements 移行 (`exp/types/*.py` → `exp/types/elements/*.py`) の途中で
`tests/types/__init__.py` と `tests/types/elements/__init__.py` が追加された一方
`tests/__init__.py` が無い working-tree 状態で踏んだ (2026-07-01)。テスト内容の
バグではなく harness (package tree) の問題。

**How to apply:** 自分のテストファイル単体が collection error になっても、まず
`git status` で `tests/types*/__init__.py` の有無と `tests/__init__.py` の欠落を
疑う。切り分けは `touch tests/__init__.py` を一時的に置いて
`uv run pytest <file>` が緑になるか確認する (緑ならテスト内容は正しく、harness
問題)。恒久修正 (`tests/__init__.py` 追加 or `tests/types*/__init__.py` 削除 or
`--import-mode=importlib`) は担当範囲を超えるので orchestrator に報告して判断を
仰ぐ。probe で置いた `tests/__init__.py` は必ず消して working-tree を汚さない。
[[testing-conventions]]
