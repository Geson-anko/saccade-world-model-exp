---
name: mingru-numerical-core
description: minGRU (exp/models/mingru.py) の log-space scan で踏みやすい NaN/不一致の罠と、正しさの検証法
metadata:
  type: feedback
---

minGRU の log-space 並列 scan を実装/変更するときの load-bearing な注意点。

**Why:** `torch.where` は両ブランチを必ず評価するため、`_log_g` で
`(x+0.5).log()` を直書きすると `x<-0.5` で `log(負)=NaN` が backward に逆流する。
zeros 初期 hidden を `log(h_0)` 経由で扱うと `log(0)=-inf` を踏む。これらは
forward の値は正しく見えても勾配だけ壊れるため気付きにくい。

**How to apply:**
- `_log_g(x)` は `torch.where(x>=0, log(relu(x)+0.5), -softplus(-x))`。`relu` クランプ必須。
- ゲートは `log z = -softplus(-k)`, `log(1-z) = -softplus(k)` (`k = fc_z(x)`)。
  `log(sigmoid)` 直書きより数値安定。
- h_0 は log を経由せず別項で足し戻す: `decay = exp(cumsum(log(1-z)))`,
  `h = scanned + decay * h_0.unsqueeze(-2)`。scan 本体 (`_parallel_scan_log`)
  は h_0 を含まない純粋系列にする。
- `step` の直接漸化式 (`(1-z)*h_prev + z*g(h̃)`) は scan を呼ばず独立実装にする。
  両者が機械イプシロン (float64 で ~1e-16) で一致するのが正しさの証拠。一致しなければ
  どちらかにバグがある。float32 だと許容差が緩むので検証は `.double()` 推奨。
- `_g` と `_log_g` は exp(`_log_g`) == `_g` を満たすよう対で保つ
  (x>=0: x+0.5 / log(x+0.5)、x<0: sigmoid(x) / log sigmoid = -softplus(-x))。

検証スクリプトを scratchpad で回すときは cwd がリセットされるので
`PYTHONPATH=/workspace uv run --project /workspace python ...` で実行する。
