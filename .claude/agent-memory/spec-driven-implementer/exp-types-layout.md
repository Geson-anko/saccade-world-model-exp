---
name: exp-types-layout
description: exp/types/ パッケージの値オブジェクト実装パターン (attrs frozen + DeviceTransferMixin)
metadata:
  type: project
---

`exp/types/` は torch tensor を内包する不変な値オブジェクト群を置くパッケージ。

- `device.py` (旧 `mixin.py`、2026-06-29 にリネーム済み): device 転送語彙。
  `type DeviceLike = torch.device | str` (PEP 695) と `DeviceTransferMixin(abc.ABC)`。
  後者は state を持たない純粋 IF で、`device` property と `to(device) -> Self` を
  abstract で強制する。
- `image.py`: `Image` — `@attrs.define(slots=True, frozen=True, eq=False)`。
  `tensor: torch.Tensor` (C, H, W) 1 field。`channels`/`height`/`width` は
  `tensor.shape[0..2]`、`device` は `tensor.device`。
- `focus.py`: `Focus` — 行動 a=(point, zoom) の値オブジェクト。
  `@attrs.define(frozen=True)` (slots=True 既定、eq=True 既定で hashable)。
  tensor を内包しないスカラー値オブジェクトなので `eq=False` は不要・むしろ
  `eq=True` で値等価が自然。`__call__(image: Image) -> Image` で正方切り取り、
  `tensor() -> (3,) float32` で行動ベクトル [x, y, zoom] を返す。crop ロジックは
  旧 `Image.focus` から移設 (Image.focus は削除済み)。

**Why:** 仕様 (2026-06-27, image-types / 2026-06-29, focus 値オブジェクト化) の確定事項。
**How to apply:** 同種の値オブジェクトを追加するときはこの構成に揃える。
tensor を持たないスカラー値オブジェクトは `eq` 既定 (True) のまま hashable にできる。

load-bearing な注意点 (実機検証済み: attrs 26.1.0 / torch 2.12.1 / Python 3.13 / pyright standard):
- tensor 内包型は `eq=False` 必須。既定 `eq=True` だと attrs が `tensor == tensor` を
  生成し boolean ambiguity の `RuntimeError` で死ぬ (Focus はスカラーのみなので該当しない)。
- tensor 内包の値オブジェクト (`Image` / `ImageSequence` / `BatchedImageSequence`)
  は `@final` (typing.final を `@attrs.define` の直上=最外側に付与)。継承されない
  前提なので自己同型メソッド (`to` / `float` / `__getitem__` slice 等) の戻り値は
  `Self` ではなく具象クラス名で注釈し、本体も `type(self)(...)` ではなく具象名
  コンストラクタ (`Image(...)` 等) を直接呼ぶ (2026-06-29 リファクタで確定)。
  `from __future__ import annotations` 併用で自クラス前方参照が効く。
  classmethod (`load` / `from_images` / `from_sequences`) の本体は `cls(...)` を維持。
  基底 `DeviceTransferMixin.to(-> Self)` の override を具象戻り値にしても pyright
  standard は LSP 互換と判定しエラーにならない (`@final` で Self=具象型に解決)。
  注: ChannelFormat (member 持ち IntEnum) は言語仕様で既に継承禁止なので @final は付けない。
- デコレータ順序: ABC の abstract property は `@property`→`@abc.abstractmethod`。
  具象 property は `@property`(外)→`@override`(内)。逆順は pyright エラー。
- frozen でも内部 tensor の in-place 変更 (add_ 等) は防げない点を docstring に明記する。
- 検証は `ndim == 3` のみ。dtype/値域/channels 値は検証しない (投機実装を避ける)。
