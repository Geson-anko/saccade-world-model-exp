---
name: project-image-value-objects
description: exp/types/image.py の値オブジェクト群の設計判断 (final 化・Self→具象型・from_* コンストラクタ契約)
metadata:
  type: project
---

`exp/types/image.py` は不変な tensor ラッパ値オブジェクトを定義する: `Image` `(C,H,W)` / `ImageSequence` `(len,C,H,W)` / `BatchedImageSequence` `(batch,len,C,H,W)`、すべて `@attrs.define(slots=True, frozen=True, eq=False)`。`ChannelFormat(IntEnum)` は GRAY=1/RGB=3/RGBA=4（値=チャンネル数）。`DeviceTransferMixin`（`exp/types/device.py`、ABC、`to -> Self` と `device` を強制）を継承。

確定した型注釈方針（2026-06-29 のリファクタで決定）:
- この 3 値オブジェクトは `typing.final`（継承禁止）。`ChannelFormat` には付けない（member 付き enum は言語仕様で既にサブクラス不可、付与は冗長）。
- 自己同型を返すメソッドは `typing.Self` を使わず**具象クラス名**で注釈する。`from __future__ import annotations` でクラス本体内の自クラス前方参照を可能にしている。
- インスタンスメソッド内は `type(self)(...)` ではなく具象名コンストラクタ。ただし classmethod 本体は `cls(...)` 慣用を維持。
- `final` クラスでは基底の `to -> Self`（= 当該具象型に解決）と override の具象型注釈が pyright で両立するので `DeviceTransferMixin.to` は変更不要。

**Why:** 継承されない設計を型レベルで固定し、戻り値を具象型に確定させて可読性・型解決精度を上げるため。
**How to apply:** この系統の値オブジェクトを新規/拡張する仕様では同方針（final + 具象型注釈 + future annotations）を踏襲する。

`from_*` コンストラクタ契約: `ImageSequence.from_images(Iterable[Image])` / `BatchedImageSequence.from_sequences(Iterable[ImageSequence])` は torch.stack で dim=0 に積む。最小 1 要素、空は `ValueError`（メッセージ substring `"at least one"` が test 契約）。one-shot iterable 対応のため**先に list 化**してから空判定。shape/len 不一致の独自検証はせず torch.stack の例外に委ねる（投機的実装禁止＝開発原則のシンプルさ優先）。

関連方針: [[project-testing-conventions]]
