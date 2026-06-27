---
name: project-types-package
description: exp/types/ はプロジェクト最基盤の型パッケージ。Image 値オブジェクトと DeviceTransferMixin を提供し、public API が契約テストで固定されている
metadata:
  type: project
---

`exp/types/` はプロジェクト最基盤の型パッケージ。`exp.{DeviceLike, DeviceTransferMixin, Image}` を再 export する。

- `mixin.py`: `DeviceLike = torch.device | str`（PEP 695 type alias）+ `DeviceTransferMixin`(ABC、状態なし、`device` property と `to` を abstract で強制)
- `image.py`: `Image`（attrs `define(slots=True, frozen=True, eq=False)` の不変値オブジェクト、`(C,H,W)` tensor を内包、`channels/height/width/device` property と `to`）

**Why:** 下流の系列モデル・エンコーダ・データローダ実装すべてが依存する基盤型。Hyrum's law 緩和のため public 表面が契約で固定されている。

**How to apply:** この層を refactor する際は以下を不変条件として絶対に変えない（設計判断として確定済み）:
公開シンボル名、`Image(tensor)` シグネチャ、property 名 `channels/height/width/device`、`to(device)` の `type(self)(...)` 新インスタンス返却、`attrs.define(slots=True, frozen=True, eq=False)`、`ndim==3` 検証、`@property`→`@override` のデコレータ順序、相対 import。`tests/test_api_contract.py`（`@pytest.mark.api_contract`）が export 集合と base class 関係をピンしている。

`channels/height/width` の 3 property は `self.tensor.shape[i]` の同形だが、抽象化すると CHW 軸の意味づけ（明示性）が失われるため統合しないのが正解。2026-06-27 時点で image.py 48行 / mixin.py 24行、リファクタ余地なしと判定済み。
