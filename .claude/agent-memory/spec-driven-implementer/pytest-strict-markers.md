---
name: pytest-strict-markers
description: pytest --strict-markers が有効なので新規 marker は pyproject.toml に登録が必須
metadata:
  type: project
---

`[tool.pytest.ini_options].addopts` に `--strict-markers` が入っている。

**Why:** 未登録 marker の typo を防ぐためのプロジェクト方針。
**How to apply:** 別エージェントが新しい `@pytest.mark.<name>` を使うテストを書く場合、
実装側は `[tool.pytest.ini_options]` の `markers` リストにその marker を登録しておく
必要がある (未登録だと pytest がエラーで止まる)。

既存登録: `api_contract: 公開 API 契約ピン (振る舞いテストではない)`。
