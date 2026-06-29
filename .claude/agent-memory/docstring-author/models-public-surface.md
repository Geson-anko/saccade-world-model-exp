---
name: models-public-surface
description: exp/models の公開 API 構造と座標・記号規約 (SequenceModel / minGRU / vit)
metadata:
  type: project
---

`exp/models/` の公開面と記号規約。docstring 執筆時の shape 契約・用語の参照元。

**How to apply:**

- **`exp/models/__init__.py` は空** (re-export しない)。各モジュールが `__all__` で公開面を明示。`exp` 公開面は `tests/test_api_contract.py` がピン。新規 building block は `from exp.models.<mod> import ...` で直接 import される前提。
- **記号規約 (CLAUDE.md 由来):** `p=(x,y), x,y∈[-1,1]` 注視点 / `z∈(0,1]` 切り取り辺長 / `a=(p,z)` 行動 / `o` 観測 / `e=E(o)` 埋め込み / `b_t` belief (系列モデルの内部状態)。docstring でこれらに触れるときは原形を保つ。
- **`SequenceModel[THidden]` (base.py):** 系列モデル抽象基底。public `forward(x,hidden=None)->(out,hidden)` / `step` は `@final` 検証ラッパ。サブクラスは `_forward`/`_step` フックのみ実装。**out.shape==x.shape を基底が強制 (in=out=dim)**。hidden は None で初期状態、戻りを次チャンクへ。
- **minGRU (mingru.py):** `MinGRU` のみ `SequenceModel[torch.Tensor]` を継承。belief hidden は `(*, depth, dim)` (全ブロック末尾 hidden を depth 軸 stack)。`MinGRULayer`/`MinGRUBlock` は素の nn.Module 内部部品。`MinGRULayer.forward` は log-space 並列 scan、`.step` は直接漸化式でその独立参照 (tautology 回避)。helper `_g`/`_log_g`/`_parallel_scan_log` は private。論文 arXiv:2410.01201 準拠。
- **数値安定性の決まり文句:** `_log_g` の relu クランプ (torch.where 両ブランチ評価 → x<-0.5 で log(負)=NaN が backward 逆流を防ぐ)、h_0 別項分離 (zeros で log(0)=-inf を踏まない)、log z=-softplus(-k) / log(1-z)=-softplus(k)。
- **vit.py:** `Mlp` は `mlp.py` から import し `__all__` で再 export 維持。Transformer 系は別軸 (RoPE/SDPA)。
