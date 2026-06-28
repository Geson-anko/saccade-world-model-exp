---
name: shadow-builtin-method
description: クラスのメソッド名が組み込み型 (float 等) と同名だと、同クラスの型注釈がそのメソッドを指し pyright が壊れる
metadata:
  type: feedback
---

クラスで組み込み型と同名のメソッド (`float`, `int`, `bytes` 等) を定義すると、**同じクラス内のシグネチャの型注釈**でその名前がメソッドを指してしまい、pyright が `クラスが必要ですが ... を受け取りました (reportGeneralTypeIssues)` を出す。

**Why:** メソッド `def float(self) ...` はクラス名前空間に `float` を束縛する。型注釈 `mean: float` はクラススコープで名前解決されるため、組み込み `float` ではなくそのメソッド (`(self) -> Self`) を参照する。関数ボディ内の `float(x)` 呼び出しはビルトインを見るので**実行時は壊れず型チェックだけが落ちる** (静かな破損)。

**How to apply:** メソッド名を変えられない (公開 API 要件) なら、モジュールレベルで別名 `_float = float` を保持し型注釈側を `_float` にする。`from __future__ import annotations` やメソッド定義順の入れ替えでは直らない (pyright はクラス全体のシンボルを先に収集するため)。実例は `exp/types/image.py` の `Image.float` と `_float`。関連: [[attrs-tensor-eq]]。
