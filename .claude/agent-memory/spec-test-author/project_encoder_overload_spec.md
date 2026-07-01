---
name: encoder-overload-spec
description: ImageEncoder.__call__ rank-preserving overload spec (Image/Batched/Sequence/BatchedSequence) → test patterns
metadata:
  type: project
---

`exp/models/encoder.py` の `ImageEncoder.__call__` を 4 rank 対応へ拡張する仕様のテストを
`tests/models/test_encoder.py` に追加した。

**仕様 (rank-preserving encode)**: 各 glimpse を 1 latent vector へ潰す。rank は保存する。
- `Image (C,H,W)` → `Latent (dim,)`
- `BatchedImage (batch,C,H,W)` → `BatchedLatent (batch,dim)`
- `ImageSequence (len,C,H,W)` → `LatentSequence (seq,dim)`
- `BatchedImageSequence (batch,len,C,H,W)` → `BatchedLatentSequence (batch,seq,dim)` (既存維持)

**Why**: SequenceModel / Decoder / データローダが glimpse を色々な rank で encoder に渡すため、
`__call__` を private Protocol の `@overload` 4 本で型付け、`forward` は Union 1 本で実装。
出力ラップは `match x` の順序分岐 (BatchedImageSequence → BatchedImage → ImageSequence →
Image → `case _: raise TypeError`)。tensor は `reshape(-1, latent_dim)` で `(N, dim)` に潰して
BatchNorm1d → 元 shape に復元、という汎化。

**How to apply (テスト設計の勘所)**:
- BatchNorm N=1 制約は「変更しない」既知制約。`Image` 単体 (N=1) を **train モード**で通すと
  `BatchNorm1d` は **`ValueError`** (`"Expected more than 1 value per channel when training"`)
  を投げる。regression guard として `pytest.raises(ValueError)` で pin (メッセージ完全一致は
  しない)。対比で `ImageSequence(len>=2)` は train でも通る、`Image` も eval() なら通る、を書いて
  制約の境界を明示した。
- rank 保存は「型」と「shape」を別テストクラスに分けて検証 (`TestRankPreservingReturnType` /
  `TestRankPreservingShape`)。`type(out) is Latent` で厳密判定。
- N=1 に触れない正常系 (return type / shape) は `eval()` で回すと BatchNorm N=1 制約に引っかから
  ない。train モード固有の検証は `TestSingleImageBatchNormConstraint` に隔離する。
- gradient flow は全 rank に増やさず 1 代表 rank (`ImageSequence`, N=seq>=2 で train BatchNorm
  が定義される) のみ。per-layer 網羅は既存 `TestGradientFlow` (batch,seq) が担保済み。

**実装者への申し送り**: `match` 分岐は必ず具体度の高い順 (BatchedImageSequence → BatchedImage →
ImageSequence → Image)。ImageSequence と BatchedImage はどちらも ndim=4 だが**別クラス**なので
`isinstance`/`match case ClassName()` で正しく分岐する (ndim では区別不能)。テストは 4 型を厳密に
`type(out) is ...` で判定するため、`BatchedImage` を `ImageSequence` として扱う等の取り違えは即 red。
