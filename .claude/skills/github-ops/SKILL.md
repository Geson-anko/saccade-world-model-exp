---
name: github-ops
description: "Use when performing GitHub operations from the devcontainer — creating/reviewing PRs (`gh pr create`, `gh pr view`, `gh pr checks`), managing issues, pushing branches, or authenticating gh CLI. Triggers: 'gh pr create', 'gh pr', 'gh issue', 'gh auth', 'PR を作成', 'PR を送る', 'pull request を作る', 'git push', 'PR レビュー', 'issue を立てる'."
version: 0.1.0
---

# GitHub Operations (gh CLI)

`saccade-world-model-exp` での GitHub 連携 (PR / issue / push) のための skill。**`gh` CLI と `git` は devcontainer 内で実行する前提**で、Dockerfile に `gh` を同梱済み (`.claude/settings.container.json` で `Bash(git push:*)` / `Bash(gh:*)` を allow 済み)。GitHub CI は持たないので、PR の検証は **ローカルの `just run` 緑** が前提になる。

> **既定の完走点 = PR 作成。** この skill が読み込まれた作業では、**基本的に `gh pr create` で PR を作るところまで完走する** (ブランチ push で止めない)。流れは「作業ブランチ → 実装・コミット → `just run` 緑 → push → `gh pr create`」。PR 作成はマージではないので [安全規約](#6-安全規約) (main へ直接 push しない / `gh pr merge` はユーザー判断) と矛盾しない。明示的に「push まで」「draft で止めて」等の指示がある場合のみ完走しない。

______________________________________________________________________

## 1. 認証 (gh auth)

`gh` の認証情報はコンテナ image には焼かれない。

### 経路 A: `GH_TOKEN` 環境変数

- shell / 環境に `GH_TOKEN=ghp_xxx` (または fine-grained PAT) を入れる。`gh` は `GH_TOKEN` を最優先で参照するため `gh auth login` 不要
- PAT に必要な scope: `repo` (private を扱うなら) / `read:org`
- token を commit / PR body に貼らない

### 経路 B: `gh auth login`

- `gh auth login --hostname github.com --git-protocol https --web` を実行 (host 側 browser で device flow を承認)
- 認証情報は `~/.config/gh/hosts.yml` に書かれる。本プロジェクトの devcontainer は `~/.config/gh` を **named volume** にしているため、再ビルドをまたいで永続化される

### 確認

```bash
gh auth status                # 認証状態と scope を表示
gh api user --jq .login       # 自分の username を返せれば OK
```

______________________________________________________________________

## 2. ブランチを push する

- **`main` に直接 commit しない**。作業ブランチは `main` から `<種別>/<日付>/<内容>` で分岐 (種別 = `feature` / `fix` / `refactor` / `docs` / `chore`)
- 例: `feature/20260627/saccade-crop` / `fix/20260627/padding-edge`

```bash
git switch -c feature/$(date +%Y%m%d)/<short-slug> main
# ... 作業・コミット ...
git push -u origin HEAD        # 初回 push。-u で upstream を貼る
git push                       # 2 回目以降
```

`git push --force` は使わない。rebase 直後など必要時のみ `--force-with-lease`。

______________________________________________________________________

## 3. PR を作成する (`gh pr create`)

```bash
gh pr create \
  --base main \
  --title "<種別>(<スコープ>): <内容>" \
  --body "$(cat <<'EOF'
## Summary
- <変更点 1 つ目>
- <変更点 2 つ目>

## Test plan
- [ ] `just run` が green (format → test → type)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- タイトルは **コミットメッセージと同じ形式** (`<種別>(<スコープ>): <内容>`、種別 = `feat` / `fix` / `docs` / `style` / `refactor` / `test` / `chore`、スコープ = `exp` / `scripts` / `devcontainer` / `docs` など)
- `--draft` で WIP。`--base main` は明示する習慣を推奨
- **PR body の末尾に `🤖 Generated with [Claude Code](https://claude.com/claude-code)` を入れる** (ハーネス規約)
- コミットメッセージの末尾には trailer 行 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` を入れる (ハーネス規約)
- **`main` へのマージはユーザーが判断・実行する**。Claude から `gh pr merge` は基本叩かない
- `--body "$(cat <<'EOF' ... EOF)"` の **シングルクォート** が重要 (無いと `$` / backtick が展開される)

______________________________________________________________________

## 4. PR をレビュー・確認する

```bash
gh pr list                          # 開いている PR 一覧
gh pr view <番号>                   # メタ情報 + body
gh pr view <番号> --comments        # コメント込み
gh pr diff <番号>                   # diff
gh pr checks <番号>                 # status (本プロジェクトは CI 無しのため空のことが多い)
gh pr review <番号> --comment --body "..."
gh pr review <番号> --approve
```

URL からの参照も可: `gh pr view https://github.com/<owner>/<repo>/pull/123`。

______________________________________________________________________

## 5. Issue 操作

```bash
gh issue create --title "..." --body "..." --label bug
gh issue list --state open
gh issue view <番号> --comments
gh issue close <番号> --comment "..."
```

bug tracker は GitHub Issues に集約 (Linear / Jira 等の外部 tracker は使わない)。

______________________________________________________________________

## 6. 安全規約

- **`main` への直接 push / force-push を絶対にしない**。ユーザー指示があっても `feature/...` ブランチを切ってから PR にする
- `git push --force-with-lease` は **同名 branch を自分が rebase した直後** のみ。`main` や他人が触るブランチには使わない
- `gh pr merge` / `gh pr close` / `gh release create` / `gh repo edit` は **ユーザー判断**
- PR description / commit body に secret や token を貼らない

______________________________________________________________________

## 7. ローカル PR ドラフト

```bash
git log main..HEAD --oneline       # PR に含まれる commits
git diff main...HEAD               # PR 全体の diff (... に注意; .. ではない)
```

これを材料に PR title / body を組み立て、最後に `gh pr create` で投げる。

______________________________________________________________________

## 8. トラブルシュート

- `gh: command not found` — image が古い。devcontainer を再ビルド (VS Code「Dev Containers: Rebuild Container」)
- `HTTP 401: Bad credentials` — `GH_TOKEN` が空 / 期限切れ / scope 不足。`gh auth status` で確認
- `Updates were rejected ...` — `main` が進んでいる。作業ブランチで取り込む ([merge-main skill](../merge-main/SKILL.md))
- `pre-commit hook が push を遮る` — `--no-verify` で回避せず、`just run` で落ちている根本原因を直す

______________________________________________________________________

## 9. 関連参照

- [`.devcontainer/Dockerfile`](../../../.devcontainer/Dockerfile) — `gh` のインストール
- [`.claude/settings.container.json`](../../settings.container.json) — `Bash(git push:*)` / `Bash(gh:*)` の allow
- [merge-main skill](../merge-main/SKILL.md) — PR 前に `main` を取り込む手順
