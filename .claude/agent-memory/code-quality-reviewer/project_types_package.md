---
name: project-types-package
description: exp/types/ はプロジェクト最基盤の型パッケージ。Image 値オブジェクトと DeviceTransferMixin を提供し、public API が契約テストで固定されている
metadata:
  type: project
---

`exp/types/` はプロジェクト最基盤の型パッケージ。`exp.{DeviceLike, DeviceTransferMixin, Image}` を再 export する。

- `mixin.py`: `DeviceLike = torch.device | str`（PEP 695 type alias）+ `DeviceTransferMixin`(ABC、状態なし、`device` property と `to` を abstract で強制)
- `image.py`: `Image`（attrs `define(slots=True, frozen=True, eq=False)` の不変値オブジェクト、`(C,H,W)` tensor を内包）。property `channels/height/width/size/channel_format/is_squared/device`、変換系メソッド `to/float/uint8/standardize/normalize/as_channel_format/square_pad/focus/resize`、I/O `load/save`、`ChannelFormat`(IntEnum GRAY=1/RGB=3/RGBA=4) を提供。変換系は in-place 不可で必ず新 Image を返す（`type(self)(...)`）。
- `size.py`: `Size2d = int | tuple[int,int]`（PEP 695）+ `size_2d_to_tuple`（int→正方 (n,n)、tuple は (h,w) passthrough）

**この層の確立済みパターン（refactor 時に尊重する、自分の好みで崩さない）:**
- `float` メソッドが組み込み `float` をシャドウするため型注釈には module-level `_float = float` 別名を使う。消すと pyright が壊れる（意図的な回避策）。
- docstring は docformatter 対策で**日本語始まり・短い1行 description**。英語始まりは先頭 capitalize、長い description は日本語途中で折り返される実害あり。
- zero-guard は `standardize`/`normalize` で `float(stat) == 0.0` 系。2 メソッドの構造は似るが分岐の戻り値が異なり、統合より独立可読性を優先（「2回まで OK」）。
- private helper `_as_rgb`/`_alpha_like` は適切な粒度。`channels/height/width` 3 property は同形だが CHW 軸の明示性のため統合しない。

**Why:** 下流の系列モデル・エンコーダ・データローダ実装すべてが依存する基盤型。Hyrum's law 緩和のため public 表面が契約で固定されている。

**How to apply:** この層を refactor する際は以下を不変条件として絶対に変えない（設計判断として確定済み）:
公開シンボル名、`Image(tensor)` シグネチャ、property 名 `channels/height/width/device`、`to(device)` の `type(self)(...)` 新インスタンス返却、`attrs.define(slots=True, frozen=True, eq=False)`、`ndim==3` 検証、`@property`→`@override` のデコレータ順序、相対 import。`tests/test_api_contract.py`（`@pytest.mark.api_contract`）が export 集合と base class 関係をピンしている。

2026-06-28 の image-transforms 拡張をレビュー: 実装はほぼクリーン。唯一の客観的 DRY 修正として `focus` 内の `squared = self if self.is_squared else self.square_pad()` を `self.square_pad()` に簡素化（square_pad が既に is_squared noop を内包し self を返すので二重チェック）。`focus` の x/y 対称座標計算 (cx/cy/left/top) はちょうど 2 回・1 メソッド内なので closure 抽出は過剰と判断し保留（「外科的変更・過度な抽象化を避ける」）。enum 値テスト等の冗長気味テストは spec-test-author 専任のため削除せず観測のみ報告した。
