# プロジェクトメモリ

`saccade-world-model-exp` 固有のメモリインデックス。詳細は各ファイルへ。

各セッション開始時、または規約が関係するタスク着手前にここを確認する。新しい規約・知見が見つかったらファイルを足し、ここから 1 行リンクを張る。コードから読み取れること（構造・規約・git 履歴）は memory に書かない。

エージェント固有メモリは frontmatter `memory: project` により `.claude/agent-memory/<agent-name>/` に自動保存される（spec-planner / spec-driven-implementer / spec-test-author / code-quality-reviewer / docstring-author）。

## user（ユーザー像）

<!-- 例: - [User role](user_role.md) — Japanese-speaking developer; 返信は日本語、コード/識別子は英語 -->

## feedback（規約・ガイドライン）

- [attrs + torch eq](attrs-tensor-eq.md) — Tensor を持つ attrs クラスは eq=False 必須 (自動 __eq__ が壊れる)
- [docformatter 日本語](docformatter-japanese.md) — 日本語 docstring は日本語始まり・短文で書く (capitalize / 途中改行を回避)
- [組み込みシャドウ](shadow-builtin-method.md) — メソッド名が組み込み型と同名だと同クラスの型注釈が壊れる (_float 別名で回避)
- [enum は match-case](enum-match-case.md) — enum 等の分岐は if/elif でなく match-case を使う (網羅性チェックが効く)
- [明示的に fail](fail-loud-no-implicit-fallback.md) — 暗黙のエラー回避・自動フォールバックは禁忌、不整合は明示的に raise

## project（実装上の固有事情）

<!-- 進行中の作業・決定とその理由をここに 1 行で。 -->

## reference（外部ツール・パス）

<!-- 外部リソースへのポインタをここに 1 行で。 -->
