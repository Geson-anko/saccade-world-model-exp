---
name: exp-types-layout
description: exp/types/ パッケージの値オブジェクト実装パターン (attrs frozen + DeviceTransferMixin)
metadata:
  type: project
---

`exp/types/` は torch tensor を内包する不変な値オブジェクト群を置くパッケージ。

- `mixin.py`: device 転送語彙。`type DeviceLike = torch.device | str` (PEP 695) と
  `DeviceTransferMixin(abc.ABC)`。後者は state を持たない純粋 IF で、`device`
  property と `to(device) -> Self` を abstract で強制する。
- `image.py`: `Image` — `@attrs.define(slots=True, frozen=True, eq=False)`。
  `tensor: torch.Tensor` (C, H, W) 1 field。`channels`/`height`/`width` は
  `tensor.shape[0..2]`、`device` は `tensor.device`。

**Why:** 仕様 (2026-06-27, image-types) の確定事項。
**How to apply:** 同種の値オブジェクトを追加するときはこの構成に揃える。

load-bearing な注意点 (実機検証済み: attrs 26.1.0 / torch 2.12.1 / Python 3.13 / pyright standard):
- `eq=False` は必須。既定 `eq=True` だと attrs が `tensor == tensor` を生成し
  boolean ambiguity の `RuntimeError` で死ぬ。
- `to()` は `type(self)(self.tensor.to(device))` を返す (サブクラスで正しい型を返す)。
  `Image(...)` ハードコードは禁止。
- デコレータ順序: ABC の abstract property は `@property`→`@abc.abstractmethod`。
  具象 property は `@property`(外)→`@override`(内)。逆順は pyright エラー。
- frozen でも内部 tensor の in-place 変更 (add_ 等) は防げない点を docstring に明記する。
- 検証は `ndim == 3` のみ。dtype/値域/channels 値は検証しない (投機実装を避ける)。
