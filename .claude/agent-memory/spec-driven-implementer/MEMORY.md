# spec-driven-implementer memory

- [exp/types layout](exp-types-layout.md) — attrs frozen 値オブジェクト + DeviceTransferMixin の構成と load-bearing な注意点
- [pytest strict-markers](pytest-strict-markers.md) — --strict-markers 有効、新規 marker は pyproject.toml に登録必須
- [minGRU 数値コア](mingru-numerical-core.md) — log-space scan の NaN/-inf 罠 (relu クランプ・h_0 別項) と scan==step 一致による検証
- [nn.Module.__call__ overload](nn-module-call-overload.md) — shape 別 overload の定石 (private Protocol + type:ignore) と pyright standard の override 互換落とし穴
- [Decoder 鏡写し layout](decoder-mirror-layout.md) — ImageDecoder/ConvDecoder の構成、leading 次元はトランク担当、GroupNorm×init_weights、detach しない契約
