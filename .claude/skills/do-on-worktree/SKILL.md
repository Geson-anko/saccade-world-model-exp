---
name: do-on-worktree
description: "現在のディレクトリで別タスクが進行中のまま、新しいタスクを隔離した git worktree 内でバックグラウンド実行する。作業ツリーの未コミット変更に一切触れずに 2 本目の作業を並行で回したいときに使う。Triggers: '/do-on-worktree', 'worktree で裏で', 'worktree で並行', '今の作業を止めずに別タスク', 'バックグラウンドで別の作業', '別タスクを並行で回す', 'run in a worktree', 'background worktree task'."
version: 0.1.0
---

# Do on worktree (隔離 worktree でバックグラウンド実行)

current directory で **タスク A が進行中**のまま、**タスク B** を別の git worktree 内で **バックグラウンド実行**するための手順。worktree は同じ `.git` を共有する別 checkout なので、タスク B はタスク A の作業ツリー（未コミット変更を含む）に一切触れずに進む。

中核は `Agent` tool の **`isolation: "worktree"` ＋ `run_in_background: true`**。ハーネスが `.claude/worktrees/<id>/`（`.gitignore` 済み）に worktree を切り、その中でサブエージェントを非同期に走らせる。フォアグラウンドのセッションはタスク A を続行できる。

`/do-on-worktree <タスク内容>` の **引数がタスク B の内容**。引数が無ければ「何を裏で回すか」を確認してから起動する。

______________________________________________________________________

## いつ使う / 使わないか

**使う:**

- タスク A の未コミット変更を抱えたまま、独立したタスク B を並行で進めたい
- タスク B が作業ツリーを派手に書き換える（大量生成・一括置換・`just run` を何度も回す）ので、A の作業ツリーを汚したくない
- 結果を後でブランチ単位で取り込む（or 捨てる）前提で、思い切った変更を隔離して試したい

**使わない（別手段が素直）:**

- タスク B が **タスク A の未コミット変更に依存**する → worktree はその変更を見られない。A を先に commit するか、in-place で続ける
- A と B が **同じ file を別意図で編集**し、後で手で突き合わせが要る → 並行より逐次が安い
- current dir が既に clean で、単に作業を分けたいだけ → 普通に `git switch -c` でブランチを切る方が軽い（worktree 隔離の旨味は「未コミット状態の保護」と「作業ツリーを汚さないこと」）
- ごく短いタスク → 後述の通り worktree ごとに `uv sync` が要る。セットアップ費がタスクを上回るなら隔離しない

______________________________________________________________________

## 起動手順

`Agent` tool を次の形で 1 回呼ぶ。

```
Agent({
  description: "<3〜5 語のタスク名>",
  subagent_type: "general-purpose",   // タスクに応じて選ぶ。実装一式なら spec 系より general-purpose が無難
  isolation: "worktree",              // 隔離 worktree を切る（必須）
  run_in_background: true,            // 裏で回す（必須）。これでフォアグラウンドはタスク A を続行
  prompt: "<下記テンプレで組んだ自己完結のブリーフ>"
})
```

- `isolation` と `run_in_background` の **両方**を付ける。片方でも欠けると「隔離 worktree でバックグラウンド」にならない
- サブエージェントは**こちらの会話を共有しない**。prompt は単体で完結させる（背景・タスク・完了条件を全部書く）

### バックグラウンドエージェントへの prompt テンプレ

```
あなたは saccade-world-model-exp の隔離 git worktree の中にいる。
main の checkout では無関係の作業が進行中で、その未コミット変更はこの worktree には無い。
あなたの worktree は **コミット済み状態からの clean な checkout** から始まる前提で動くこと。
まず CLAUDE.md と .claude/skills/dev-workflow を読んで開発フローに従う。

## タスク
<タスク B の具体的な内容と完了条件>

## 進め方（この順で）
1. ブランチを切る: `git switch -c <種別>/$(date +%Y%m%d)/<slug>`（種別 = feat/fix/refactor/docs/chore）
2. この worktree 専用の venv を作る: `uv sync`（worktree には .venv が無い。torch があるので最初に済ませる）
3. 実装する。仮定を勝手に置かない／頼まれていない改変をしない（CLAUDE.md の開発原則に従う）
4. `just run`（format → test → type）を **green** にする。落ちたままにしない
5. **ブランチに commit する**（重要: コミットしないと worktree 撤去時に作業が消える。commit さえすれば共有 .git のブランチに残る）。
   commit message は `<種別>(<スコープ>): <内容>`、末尾に `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
6. push / PR は **指示が無ければやらない**（取り込みは呼び出し側が判断する）

## 最終メッセージで必ず返すこと（これが呼び出し側に渡る結果になる）
- 作成したブランチ名
- 変更点の要約（触った file）
- `just run` の結果（green か、残課題があるか）
- 取り込み時の注意（あれば）
```

> なぜ commit が必須か: `isolation: "worktree"` の worktree は「変更が無ければ自動撤去」される。**commit した内容は共有 `.git` のブランチに残る**ので、worktree が消えても `/workspace` 側から `git switch <branch>` / `git log <branch>` で回収できる。未コミットのまま終えると失われる。

______________________________________________________________________

## 進捗の監視と結果の回収

- バックグラウンドエージェントが**完了すると自動で通知が来て**こちらが再起動される。その**最終メッセージが結果**として渡る（ユーザーには直接表示されないので、要点はこちらが伝える）
- 走行中に覗く: `TaskOutput` / `Monitor` で進捗を見る。ブランチの中身は `/workspace` から `git log <branch> --oneline` / `git show <branch>` で直接見える（同じ `.git` を共有しているため）
- 追加指示を送る: `SendMessage`（エージェントの id / name 宛）。文脈を保ったまま継続させられる
- 中止する: `TaskStop`
- 注意: 走行中の worktree が checkout しているブランチは、`/workspace` 側で同時に `git switch` できない（git は同一ブランチの二重 checkout を禁ずる）。走行中はブランチを覗くだけにし、`git switch` は完了・撤去後に行う

______________________________________________________________________

## 完了後（取り込み or 破棄）

タスク A が一段落してから判断する。

- **取り込む**: 作業ブランチに `git switch <branch>` して続ける、または PR にする。PR 化は [merge-main skill](../merge-main/SKILL.md) で main を取り込んでから [github-ops skill](../github-ops/SKILL.md) の `gh pr create`
- **破棄する**: ブランチを消す（`git branch -D <branch>`）。worktree が残っていれば `git worktree remove <path>` で撤去（変更が無ければ自動撤去済みのことが多い）
- 取り残し確認: `git worktree list` と `git worktree prune`（撤去済みディレクトリの残メタを掃除）

______________________________________________________________________

## やってはいけないこと

- `run_in_background` を付けずに起動して**フォアグラウンドを塞ぐ**（タスク A が止まる。並行の意味が消える）
- バックグラウンドエージェントに **「main の未コミット変更がある前提」**で指示する（worktree には無い。clean checkout から始まる）
- worktree 内で **commit せずに終える**（作業が消える）
- 隔離 worktree から **勝手に push / PR / `gh pr merge`** する（取り込みは呼び出し側＝ユーザー判断）
- タスク B が A の変更に依存しているのに無理に隔離する（見えないものを参照して破綻する）

______________________________________________________________________

## 手動 worktree（永続させて自分で覗きたいとき）

ハーネス任せでなく、消えない worktree を既知パスに置いて後から `cd` して触りたい場合のみ：

```bash
git worktree add .claude/worktrees/<slug> -b <種別>/$(date +%Y%m%d)/<slug>   # .claude/worktrees/ は .gitignore 済み
# そのパスを cwd にしたエージェント or Bash でタスク B を回す（uv sync → 実装 → just run → commit）
git worktree remove .claude/worktrees/<slug>                                  # 用が済んだら撤去
```

通常は上の `Agent` 隔離（自動撤去つき）で十分。手動は「成果物 file を後でこの目で確認したい」等の特別な事情があるときだけ。

______________________________________________________________________

## 関連参照

- [dev-workflow skill](../dev-workflow/SKILL.md) — `just run` / uv / コミット前ゲート（バックグラウンドエージェントにも同じ規約を守らせる）
- [merge-main skill](../merge-main/SKILL.md) / [github-ops skill](../github-ops/SKILL.md) — 完了後にブランチを取り込む・PR にする手順
