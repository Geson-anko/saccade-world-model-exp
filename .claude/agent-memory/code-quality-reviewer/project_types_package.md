---
name: project-types-package
description: exp/types/ はプロジェクト最基盤の型パッケージ。tensor 値オブジェクト群 (elements/) と DeviceTransferMixin を提供し、public API が契約テストで固定されている
metadata:
  type: project
---

`exp/types/` はプロジェクト最基盤の型パッケージ。`exp` / `exp.types` が値オブジェクト群と device 抽象を再 export する。

構成（2026-07 時点）:
- `device.py`: `DeviceLike = torch.device | str`（PEP 695）+ `DeviceTransferMixin`(ABC、`__slots__=()` で子の slots を全チェーン実効化、`device` property と `to` を abstract 強制) + `SupportsDeviceTransfer`(Protocol、torch.Tensor/Module も構造適合)
- `size.py`: `Size2d = int | tuple[int,int]`（PEP 695）+ `size_2d_to_tuple`
- `tensor.py`: `ScalarTensor`（スカラー損失値の値オブジェクト。[[project-loss-module]] が使用）
- `elements/`: tensor 値オブジェクト群のサブパッケージ。[[project-element-base]]（内部基底 base.py）+ 具象 image.py / focus.py / latent.py。`elements/__init__.py` が 13 シンボルを re-export。詳細は各 memory 参照。

**この層の確立済みパターン（refactor 時に尊重する、自分の好みで崩さない）:**
- すべての具象値オブジェクトは `@final` + `@attrs.define(slots=True, frozen=True, eq=False)` + `Element`/`ElementArray`/`BatchedElementSequence` 継承。`_NDIM`/`_SHAPE_DESC`/(任意で)`_SHAPE` の ClassVar を宣言順この順で定義（3 モジュールで一貫、崩さない）。
- `_item_type`/`_batch_type` は `@classmethod`+`@override`、`type[...]` 返却。3 モジュールで完全一致。
- 具象モジュールの `__all__` は多行 + magic trailing comma が package 内の支配的スタイル（base/image/latent/両 __init__）。1 行に収まっても多行に揃える。
- docstring: docformatter 対策で日本語始まり・短い description。CJK 長文は 132 桁近辺で docformatter が途中改行し stray space が入ることがある（image.py BatchedImageSequence 等に既存の軽微アーティファクト）。手で直しても再 mangle される恐れがあり過剰反応しない。
- 具象間の docstring 粒度: 収集型クラスは「要素 0（空〜）も許容する」の不変条件注記を全モジュールで揃える（2026-07 に latent へ追記して統一）。leaf の semantic 説明は型ごとに正当に異なる（Image=変換系/Focus=point,zoom/Latent=最小）ので無理に揃えない。

**Why:** 下流の系列モデル・エンコーダ・データローダ実装すべてが依存する基盤型。Hyrum's law 緩和のため public 表面が契約で固定されている。

**How to apply:** `tests/test_api_contract.py`（`@pytest.mark.api_contract`）が `exp.__all__`(19)/`exp.types.__all__`(19) の集合、base 系(Element/ElementArray/エイリアス/BatchedElementSequence)が exp から import できないこと、全値オブジェクトが DeviceTransferMixin サブクラスであることをピン。refactor では公開シンボル名・シグネチャ・エラー substring・継承・相対 import・デコレータ付け方を変えない。
