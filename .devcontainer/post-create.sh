#!/usr/bin/env bash
# .devcontainer/post-create.sh
#
# container 内 dev ユーザーで実行する冪等な初期化レシピ。
#   1. uv sync         — dev 依存 (ruff / pyright / pytest / pre-commit) を解決
#   2. pre-commit hook — git commit 時のフックを登録 (best-effort)
#   3. Claude 設定     — コンテナ用 settings.container.json を settings.local.json に展開
set -euo pipefail

cd /workspace

echo "info: uv sync ..."
uv sync

echo "info: pre-commit install ..."
uv run pre-commit install --install-hooks || echo "warning: pre-commit install に失敗。hook 登録を skip して続行します。"

# コンテナ内では bypassPermissions 等を効かせる。settings.local.json は gitignore 済み
# なので host 側の settings.json (より厳格な allowlist) を上書きせず共存できる。
if [[ -f .claude/settings.container.json ]]; then
  cp .claude/settings.container.json .claude/settings.local.json
  echo "info: applied .claude/settings.container.json -> .claude/settings.local.json"
fi

echo "info: done. 'just --list' で利用可能なレシピを確認してください。"
