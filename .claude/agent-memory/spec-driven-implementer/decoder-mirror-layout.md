---
name: decoder-mirror-layout
description: ImageDecoder / ConvDecoder の構成 — Encoder の鏡写し、leading 次元の担当分担、GroupNorm と init_weights の関係
metadata:
  type: project
---

`exp/models/decoder.py` の `ImageDecoder` は `exp/models/encoder.py` の `ImageEncoder` の鏡写しで、内部 CNN トランクは `exp/models/components/conv_decoder.py` の `ConvDecoder`。ViT が Encoder のトランクなのと対称。

**Why:** 「潜在 → 画像」の可視化 read-out (CLAUDE.md #6)。Encoder と同じ overload / match / 公開 latent_dim / 新規層のみ init する配慮を踏襲する方針が確立済み。

**How to apply:**
- leading 次元 (`*lead`) の畳み込み/復元は **トランク側 (ConvDecoder)** が担う (`ImageDecoder` ではない)。Encoder 側は BatchNorm の reshape 都合で Encoder 本体が担っていたが、Decoder は Conv2d が 4D 必須なので `ConvDecoder.forward` で `reshape(math.prod(lead) or 1, ...)` → conv → `(*lead, C, s', s')`。`math.prod(()) == 1` で `Latent` (lead=()) が N=1 で通る。
- Decoder の内部 norm は **GroupNorm** (バッチ非依存)。Encoder の BatchNorm と違い N=1 (単一 Latent) を train モードで通しても落ちない。テストがこの差 (`test_latent_returns_image_in_train_mode`) を明示的にピンしている。
- GroupNorm は `init_weights` (weight.py) の match 対象外 (Linear/Conv2d/ConvTranspose2d/LayerNorm のみ)。`self.apply(init_weights)` しても既定 (weight=1, bias=0) のまま残るので、子に ViT のような自己初期化トランクを持たない ConvDecoder は素直に apply してよい。ImageDecoder 側は ConvDecoder が自己初期化するので init_weights を重ねない。
- **detach しない** のが公開契約 (`TestDetachContract` がピン)。#6 の detached read-out は呼び出し側が `e.detach()` してから渡す責務。Decoder 内で detach するとこのテストが落ちる。
- ChannelFormat 検証は `out_channels not in tuple(ChannelFormat)` (IntEnum なので 1/3/4 と int 比較が成立)。
- `exp/models/__init__.py` は空で encoder も decoder も re-export しない。import は `from exp.models.decoder import ImageDecoder`。
