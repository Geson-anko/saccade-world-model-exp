---
name: project-element-base
description: exp/types/elements/base.py の内部共通基底 (Element / ElementArray / BatchedElementSequence)。Image/Focus/Latent の shape 検証・device 転送・collection・stack を集約
metadata:
  type: project
---

`exp/types/elements/base.py` は tensor 値オブジェクトの内部共通基底。**非公開**（`exp`/`exp.types` の `__all__` に載せない）だが実装ロジックを持つため直接テストする（`tests/types/elements/test_base.py` が purpose-built fake subclass 経由で検証）。[[project-types-package]] の一部。

軸設計（3 クラス）:
- `Element`（`DeviceTransferMixin`+abc.ABC、attrs frozen slots eq=False）: 末端 tensor 1 個を保持。`_NDIM`/`_SHAPE_DESC`/`_SHAPE`(任意, 各軸期待サイズ, None=wildcard, 不変条件 len==_NDIM) の ClassVar でサブクラスが shape 規約を宣言。`__attrs_post_init__` が ndim→per-axis の順で検証しエラーに shape_desc/ndim/shape を出す。`device`/`to` を `@override` で実装（`to` は `type(self)(...)` 新インスタンス）。
- `ElementArray[T: Element]`: 先頭 1 軸で積んだ収集型。`_item_type()`(abstract) で要素型復元。`__len__`/`__getitem__`(int→T, slice→Self overload)/`__iter__`/`from_elements`(空は ValueError "at least one element") を提供。`ElementSequence`/`BatchedElement` は**単純代入エイリアス**（PEP695 type 文は基底 subscript 不可なので `= ElementArray`）。意味は命名で区別。
- `BatchedElementSequence[TBatch, TSeq]`(ElementArray[TSeq]): (batch, seq, ...) 2 軸。`iter_batch`(=__iter__, dim0→TSeq)/`iter_sequence`(dim1 unbind→TBatch)/`from_sequences`(dim0 stack)/`from_batches`(dim1 stack)。`_batch_type()` abstract。__getitem__/__len__ は batch 軸。

エラー substring がテストで pin: `at least one element`/`at least one sequence`/`at least one batch`。

**Why:** 2026-06 に image.py/focus.py/latent.py が個別に持っていた shape 検証・collection protocol・stack ファクトリ・device 転送を DRY 統合した成果。旧 `from_images`/`from_focuses` は `from_elements`/`from_sequences`/`from_batches` に一本化済み（rule-of-three を満たす正当な抽象）。

**How to apply:** refactor では ElementArray の dunder には `@override` を付けない／`device`・`to`・`_item_type`・`_batch_type` には付ける方針を崩さない。エイリアスの単純代入は type 文に変えない（subscript 不可で壊れる）。base は非公開のまま。2 軸 from_sequences/from_batches・iter_batch/iter_sequence は意図的設計で削除・改名しない。
