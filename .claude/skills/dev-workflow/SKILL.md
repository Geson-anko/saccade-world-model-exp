---
name: dev-workflow
description: saccade-world-model-exp の開発フロー (uv / just / devcontainer / pre-commit / memory)。依存追加・整形・型チェック・テスト・GPU 確認・コミット前ゲートのコマンドと前提をまとめる。作業着手前やコマンドの叩き方に迷ったら参照する。
---

# 開発フロー (dev-workflow)

`saccade-world-model-exp` の環境・ツール・コマンドの早見表。配布物ではない**研究用コードベース**であり、品質ゲートは GitHub CI ではなく **ローカルの `just run`** で担保する。

## 環境

- **devcontainer (docker-compose, GPU)** 上で開発する。VS Code の "Reopen in Container" で起動。
  - `.devcontainer/initialize.sh` (host) が uid/gid を `.devcontainer/.env` に書き出す。
  - `.devcontainer/post-create.sh` (container) が `uv sync` / `pre-commit install` / `settings.container.json` の展開を行う。
- **uv** がパッケージ管理と Python (`.python-version` = 3.13) を担う。
- GPU は compose の device 予約で全機 (RTX 4090 + RTX 4060) を公開する。

## パッケージ管理 (uv)

- 依存導入: `uv sync`（dev グループのみ。ランタイム依存は現状空）。
- 依存追加: `uv add <pkg>`。ランタイム依存は torch / torchvision / numpy / attrs を採用済み。
- 任意コマンド実行: `uv run <cmd>`（プロジェクトの venv 内で実行）。
- `uv.lock` はコミットする（再現性）。

## just レシピ（主インターフェース）

| レシピ | 内容 |
|---|---|
| `just setup` | `uv sync` + `pre-commit install` |
| `just format` | 全 pre-commit hook を実行（ruff check+format、ファイル衛生、shell など） |
| `just lint` | `ruff check`（autofix なし） |
| `just type` | `pyright`（standard mode） |
| `just test` | `pytest -v` |
| `just run` | `format` → `test` → `type`。**コミット前にこれを通す** |
| `just clean` | キャッシュ削除 |

## ソース構成

- モジュール系ソースは **フラットな `exp/`**（src-layout にしない）。
- 学習・評価のエントリーポイントは `scripts/`（または repo ルート）。実行は `uv run python -m scripts.<name>`（repo ルートから）か、ルート直下のスクリプトを `uv run python <file>`。
- 配布パッケージ化しない（`tool.uv.package = false`、`build-system` なし）。`import exp` はテストの `pythonpath = "."` で解決。

## GPU

- devcontainer 内で `nvidia-smi` を叩き、RTX 4090 / 4060 が見えることを確認する。
- GPU 選択は `CUDA_VISIBLE_DEVICES`（学習は通常 4090 = `CUDA_VISIBLE_DEVICES=0`）。compose は全 GPU を予約しているので可視性はランタイムで絞る。
- 特定 GPU に固定したい場合は `.devcontainer/compose.yml` の `count: all` を `device_ids: ["0"]` に変更。

## memory（プロジェクト知識）

- プロジェクト固有の知見は `memory/` に 1 ファイル 1 事項で蓄積し、`memory/MEMORY.md` に 1 行索引を張る。
- エージェント固有メモリは `memory/agents/<agent-name>/`。
- コードから読み取れること（構造・規約）は memory に書かない。

## Claude 設定

- host: `.claude/settings.json`（粒度の細かい allow/deny）。
- devcontainer: `post-create.sh` が `.claude/settings.container.json` を `.claude/settings.local.json` に展開し、コンテナ内では bypassPermissions + xhigh effort を効かせる（`settings.local.json` は gitignore 済み）。
