---
name: mingru-component
description: exp/models/mingru.py の public 面と refactor 境界 (MinGRULayer/Block/MinGRU、log-space scan の数値不変条件)
metadata:
  type: project
---

`exp/models/mingru.py` = minGRU 系列モデル (Feng et al. 2024, arXiv:2410.01201)。`MinGRU` のみ `SequenceModel[torch.Tensor]` を継承、`MinGRULayer`/`MinGRUBlock` は素の `nn.Module` 内部部品。

public 面 (変更不可):
- `__all__`: MinGRU, MinGRUBlock, MinGRULayer。helper は private `_g`/`_log_g`/`_parallel_scan_log`。
- `MinGRULayer(input_dim, hidden_dim, *, init_std=0.02)`、属性 `fc_z`/`fc_h`、`forward(x, hidden=None)->(out, h_last)`、`step(x_t, h_prev=None)->h_t` (戻り Tensor 単体、tuple でない)。テストが `fc_z.weight/bias` を no_grad 書換、`step` を T 回ループして並列 forward の独立参照にする。
- `MinGRUBlock(dim, *, mlp_ratio=4.0, dropout=0.0, init_std=0.02)`、`MinGRU(dim, depth, *, mlp_ratio, dropout, init_std)`、公開属性 `dim`/`depth`/`blocks`。
- `MinGRU` は `_forward`/`_step` フックのみ実装し public `forward`/`step` は基底 (base.py) の `@final` ラッパに任せる。

数値不変条件 (振る舞いを変えない):
- `_log_g` は `torch.where` の両ブランチ評価対策で `relu` クランプ必須 (`(x+0.5).log()` 直書き禁止、x<-0.5 で NaN が backward 逆流)。テスト `test_backward_through_negative_candidate_branch_is_finite` がピン。
- log-space scan: `log z = -softplus(-k)`、`log(1-z) = -softplus(k)`、h_0 別項分離 (`decay = exp(cumsum(log(1-z)))`、`h = scanned + decay*h_0`)。zeros 初期で `log(0)=-inf` を踏まない設計。
- `init_hidden` は無い (hidden=None で zeros 初期)。

**Why:** spec-driven-implementer 初版直後の refactor 対象 (ブランチ feature/20260629/mingru)。
**How to apply (実施済み refactor の記録):**
- `MinGRULayer.__init__` の `self._init_std = init_std` は dead だったので削除 (param は uniform signature のため signature 上は残す。実初期化は `MinGRU.apply(init_weights)` が唯一の源)。
- `MinGRULayer.forward` の `decay` は `hidden is not None` の時だけ計算 (hidden=None の common path で無駄計算を避ける)。
- `MinGRU._forward`/`_step` は per-layer hidden 分配・stack・最終 norm が完全重複していたので private `_run_blocks(x, hidden, run_block)` に統合。`run_block` lambda が並列(`blk(...)`)/単一(`blk.step(...)`)の違いだけを担う。`torch.compile` parity テスト (TestMinGRUCompile) は lambda 経由でも pass する。
- `mlp.py`/`norm.py` は元から最小 clean、refactor 不要。

関連: [[vit-component]] (Mlp は vit から mlp.py へ移設), [[project-types-package]]
