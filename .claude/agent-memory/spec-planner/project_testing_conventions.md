---
name: project-testing-conventions
description: saccade-world-model-exp の仕様に織り込むテスト規約 (real tensor / substring 検証 / 1対1ミラー / private 非テスト)
metadata:
  type: project
---

仕様書を書くとき、受け入れ基準・テストシナリオは以下のプロジェクトテスト規約に整合させる:

- 実 CPU tensor を実ライブラリ（torch 等）に通す。3rd-party の internals はモックしない。自前 ABC の fake のみ許可。
- private（`_prefix`）は直接テストせず公開面越しに検証する。
- tests は `exp/` を 1 対 1 ミラー（`exp/types/image.py` ↔ `tests/types/test_image.py`）。
- 異常系は例外メッセージの**完全一致ではなく substring** で検証（`pytest.raises(..., match="...")`）。仕様でエラー契約を書くときは「substring X を含むこと」の形にする。
- 3rd-party の例外型に依存したテストは脆いので避ける。自分が定義した契約（自前の ValueError 等）だけを検証対象にし、ライブラリ任せの例外挙動はテストで固定しない。
- 公開 API 契約は `tests/test_api_contract.py` でピン。`__all__` や `exp` 公開シンボルを増やさない変更ならこのファイルは変更不要、と仕様に明記する。

**Why:** real-resource 優先・mock 最小の testing-strategy skill に沿うため。substring 検証はメッセージ文言の自由度を保ちつつ契約を固定する手段。
**How to apply:** 受け入れ基準と spec-test-author 向けシナリオを書く全ての仕様で適用する。

詳細根拠は `.claude/skills/testing-strategy` を参照。関連: [[project-image-value-objects]]
