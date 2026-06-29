# minGRU 実装計画の下書き（`exp/models/components/mingru.py`）

> **ステータス: 下書き（実装未着手）。** これは将来の `SequenceModel` ベースクラス（系列モデルの抽象基底・実装契約）の **足がかりとなる先行する具体例**。`SequenceModel` の策定は別の会話で行うため本文書では扱わないが、minGRU がその契約に素直に乗るよう forward の shape 契約だけ先に合わせておく。コード（`mingru.py` / テスト）は `SequenceModel` 契約確定後に着手する。

## Context

サッカード世界モデルの系列モデル主軸として minGRU（Feng et al. 2024, "Were RNNs All We Needed?", arXiv:2410.01201）を採用する。minGRU はゲート・候補が入力 `x_t` のみ依存（`h_{t-1}` 非依存）で、線形漸化式 `h_t = (1−z_t)·h_{t−1} + z_t·h̃_t` を **並列スキャンで一括計算**できる。固定長の再帰状態が belief になる。

参考にした自プロジェクト pamiq-curiosity-exp の `qgru.py`/`qlstm.py` は Layer→Block→Stack の3層構成・`forward(x, hidden=None)→(out, hidden)`・初期 hidden zeros・`depth` 多層化。ただし pamiq の `scan` は非 log-space かつ forget gate に `linspace` decay 改変が入り論文と異なるため、構造のみ参考にし数値コアは論文の log-space 版に従う。

## shape 契約（将来の SequenceModel に合わせる）

- **入力 `x: (*, len, dim)` → 出力 `out: (*, len, dim)`**、加えて hidden を授受する: `forward(x, hidden=None) -> (out, hidden)`。
- **hidden の shape は契約で固定しない**（モデル依存。Transformer を KV キャッシュで RNN 的に回すと毎回形が変わるため）。minGRU では hidden は末尾隠れ状態（単層 `(*, dim)` / 多層 `(*, depth, dim)`）。
- 将来 `SequenceModel`（別会話で策定）を継承する想定。hidden 型は将来 `Generic[THidden]` 等で表現される見込みだが、本下書きでは minGRU 固有の `torch.Tensor` として書き、`SequenceModel` 確定後に差し替える。

## 数値コア（論文 Appendix 準拠・実装者が自前で再構成）

- `z_t = σ(fc_z(x_t))`、`h̃_t = g(fc_h(x_t))`、`h_t = (1−z_t)⊙h_{t−1} + z_t⊙h̃_t`。
- 連続正値活性 `g(x) = x+0.5 (x≥0) else σ(x)`、その log `log_g(x) = log(relu(x)+0.5) (x≥0) else −softplus(−x)`。
  - **`(x+0.5).log()` を直書きしない**。`torch.where` は両ブランチを評価するため `x<−0.5` で `log(負)=NaN` が backward に逆流する。必ず `relu` でクランプ。
- log 係数: `k = fc_z(x)` として `log z = −softplus(−k)`、`log(1−z) = −softplus(k)`。
- **h_0 は別項分離**（pamiq `QGRULayer` 方式）: inner log-space scan `_parallel_scan_log(log_coeffs, log_values)`（`cumsum`+`logcumsumexp`、`log_coeffs=log(1−z)`、`log_values=log z + log_g(h̃)`）で h_0 なしの純粋系列を計算し、`exp(cumsum(log(1−z)))_t ⊙ h_0` を別に加算。これで h_0 を実 hidden（正値制約なし）として扱え、`zeros` で `log(0)=−inf` を踏まず、チャンク跨ぎ（末尾 hidden 再投入）も連続。`h̃` の `g` で hidden 正値性は保つ。
- **初期隠れ状態は zeros 固定**（ユーザー決定）。`init_hidden` は zeros を返す。learnable は実装しない。

## 周辺アーキテクチャ（norm / FFN / conv）

論文は「minimal」を標榜し**多層化の周辺機構（norm/FFN/conv）を規定していない**（規定するのは漸化式の簡約: ゲートの状態依存除去・候補の range 制限除去・cell state 除去）。実構成は参照実装 lucidrains/minGRU-pytorch の言語モデル `minLM` に従う:

- **pre-norm RMSNorm**（LayerNorm ではない）。block は `norm(x) → minGRU → +residual`、`ff_norm(x) → FFN → +residual`、最終に RMSNorm。
- **FFN は Linear-GELU-Linear（expansion ×4）**（SwiGLU ではない）。
- **causal depthwise Conv1d**（kernel=3, groups=dim, 因果 pad）は optional。

採用方針（下書き、実装時に確定）:
- **Norm = RMSNorm**（参照実装準拠）。saccade に未実装のため `exp/models/components/norm.py` に新規実装（`vit.py` の `init_weights` 様式に整合）。
- **FFN = Linear-GELU-Linear ×4**。**`vit.py` の `Mlp` を `exp/models/components/mlp.py` に切り出し**、`vit.py` と `mingru.py` の双方が参照（ユーザー指示）。`vit.py` は `from .mlp import Mlp` に変更（外科的変更）。
- **conv は初版では省略**（minimal を尊重）。必要なら後続で `enable_conv` 的に追加。

## 公開 API（下書き）

`exp/models/components/mingru.py`（新規）— `__all__ = ["MinGRULayer", "MinGRUBlock", "MinGRU"]`、helper は private `_g`/`_log_g`/`_parallel_scan_log`、`__init__.py` は re-export しない、`exp` 公開面に出さない（`test_api_contract.py` 不変）。

- `MinGRULayer(input_dim, hidden_dim, *, init_std=0.02)`: `fc_z`/`fc_h = nn.Linear(input_dim, hidden_dim)` のみ（出力射影なし）。`forward(x, hidden=None) -> (out, h_last)`（`x:(*,len,input_dim)`、`out:(*,len,hidden_dim)`、`h_last:(*,hidden_dim)`、並列 log-space + h_0 別項、任意 leading batch を reshape）。`step(x_t, h_prev) -> h_t`（直接漸化式。forward(len=1) で代替可だが薄い step も提供）。`init_hidden(batch_shape=(), *, device, dtype) -> (*batch_shape, hidden_dim)` zeros。
- `MinGRUBlock(dim, *, mlp_ratio=4.0, dropout=0.0, init_std=0.02)`: `RMSNorm → MinGRULayer(dim,dim) → +residual`、`RMSNorm → Mlp → +residual`。`forward(x, hidden=None) -> (out, h_last)`、`step`、`init_hidden` を委譲。
- `MinGRU(dim, depth, *, mlp_ratio=4.0, dropout=0.0, init_std=0.02)`: `nn.ModuleList([MinGRUBlock(dim,...) for _ in range(depth)])`。public 属性 `dim`/`depth`。`forward(x, hidden=None) -> (out, h_last)`（`hidden:(*,depth,dim)` を層に分配、各層末尾を stack）。`step`、`init_hidden(...) -> (*batch_shape, depth, dim)`。`dim<=0`/`depth<=0` で `ValueError`（substring 検証可能なメッセージ）。

依存切り出し:
- `exp/models/components/mlp.py`（新規）: `vit.py` の `Mlp` を移設、`__all__ = ["Mlp"]`。
- `exp/models/components/norm.py`（新規）: `RMSNorm`、`__all__ = ["RMSNorm"]`。

## テスト計画（下書き）`tests/models/components/test_mingru.py`

testing-strategy 準拠（real CPU tensor + seed 固定、private 直接 import 禁止、torch internals モック禁止）。`test_vit.py` のクラス分けに倣う。逐次参照は public `step` を T 回ループ（`_seq_reference`）。z 固定は public `fc_z.weight/bias` を `no_grad` で書換。

- `TestMinGRUMath`: hidden 正値性 / 並列==逐次一致（zeros 初期・ランダム初期、float32 で落ちたら `.double()` 版）/ z=1 で出力=候補（h_0 非依存）/ z=0 で初期 hidden 張り付き / where 分岐の勾配 finite。
- `TestMinGRUShape`: 単層・多層・step の shape `(*, len, dim)`、公開属性。
- `TestMinGRUValidation`: `dim<=0`/`depth<=0` の ValueError substring。
- `TestMinGRUBehaviour`: 多層 並列==逐次、hidden roundtrip 連続性、全層に勾配流通、eval 決定性。
- `TestMinGRUInitHidden`: 戻り shape、device/dtype 整合、zeros 値。
- `TestMinGRUCompile`（integration-real）: `torch.compile` parity（`logcumsumexp` 未対応に備え try/except skip、理由を握り潰さない）。
- `mlp.py` 切り出しは既存 `test_vit.py` を緑のまま保つ（import 経路変更のみ）。`RMSNorm` には最小の shape/正規化テストを追加。

書かないテスト: import 可能性 / 継承 / 定数 literal / getter-setter / private 直接 import / 例外メッセージ完全一致。

## 後続フェーズの実装フロー（参考・今回は実行しない）

`SequenceModel` 契約確定後に、ブランチ `feature/<日付>/mingru` で TDD マルチエージェント（`spec-test-author` × `spec-driven-implementer` 並列 → `code-quality-reviewer` → `docstring-author`）。`just run`（format→test→type）緑で push → `gh pr create --base main`。`main` 直 push/merge はしない。

## 主要ファイル（後続フェーズ）

- `exp/models/components/mingru.py`（新規・実装先）
- `exp/models/components/mlp.py`（新規・`vit.py` から `Mlp` 切り出し）
- `exp/models/components/norm.py`（新規・`RMSNorm`）
- `exp/models/components/vit.py`（`Mlp` を `mlp.py` から import に変更）
- `exp/models/components/weight.py`（`init_weights`）
- `tests/models/components/test_mingru.py`（新規）
