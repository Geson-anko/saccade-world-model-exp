---
name: docformatter-japanese
description: docformatter は日本語 docstring を壊す (先頭 capitalize + 全角途中改行)。日本語始まり・短文で安定させる
metadata:
  type: feedback
---

pre-commit の docformatter は英語前提で、日本語 docstring を壊す。

**Why:** (1) summary 先頭の ASCII 単語を capitalize する (文頭の `device` → `Device`)。(2) description を文字数 79 で機械再ラップし、日本語の禁則を無視して `。 状態` のように全角文の途中で改行する。

**How to apply:** 日本語 docstring は (1) 日本語始まりにして capitalize を回避、(2) 1 行サマリ or 短い 1 パラグラフに収めて再ラップを抑える。docformatter は冪等なので、一度通った形なら以降は安定。

補足: `pre-commit run -a` (`just format` / `just run` 内) は **tracked/staged ファイルのみ** 対象。新規ファイルは `git add` 前だと docformatter を素通りし、commit 時 (staged 後) に初めて整形される。新規追加時の「`just run` は green なのに commit で整形が走る」はこれが原因。
