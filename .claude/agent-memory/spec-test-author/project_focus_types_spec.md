---
name: focus-types-spec
description: Spec→test conventions for exp/types/focus.py (Focus + FocusSequence/BatchedFocusSequence value objects, zoom range [0,1])
metadata:
  type: project
---

`exp/types/focus.py` holds the action value objects. `FocusSequence` /
`BatchedFocusSequence` behaviour tests added on branch
`feat/20260630/focus-sequence` (spec-first; impl by parallel agent, already in tree).

**Why:** these mirror the Image-sequence patterns ([[image-types-spec]]) and recur
for future (seq, *) value objects.

**How to apply:**
- `Focus` zoom range is the CLOSED interval `[0, 1]` (changed from `(0, 1]`).
  `zoom=0.0` and `zoom=1.0` are BOTH valid; rejection sample must be `-0.1`
  (or `1.5`). post_init message substring is `"[0, 1]"` (regex `\[0, 1\]`),
  point is `"[-1, 1]"`.
- `FocusSequence.tensor` is `(seq, 3)`, `BatchedFocusSequence.tensor` is
  `(batch, seq, 3)`; each row is `[x, y, zoom]`. Both subclass
  `DeviceTransferMixin` (identity-style, `eq=False`, immutable).
- post_init validates RANK + last-dim ONLY (`ndim`/`shape[-1]!=3`), NOT value
  range. Empty `(0,3)` / `(0,seq,3)` are VALID. Error substrings:
  `"(seq, 3)"` / `"(batch, seq, 3)"` plus actual ndim+shape (substring only).
- Range checking is split out into `is_valid() -> bool` (point∈[-1,1] AND
  zoom∈[0,1], closed) and `validate() -> None` (raises ValueError substring
  `"out of range"` when invalid, else returns None silently). Test `validate()`
  passing path with `assert seq.validate() is None`.
- `from_focuses` / `from_sequences`: dim0 stack, order-preserving (pin each row
  with `torch.equal` against `.tensor()` / `.tensor`), single→leading axis 1,
  empty→ValueError substring `"at least one"`, generator accepted, float32.
- `FocusSequence.apply(image, size) -> ImageSequence`: each Focus crops then
  resizes to `size`, so MIXED zooms (full + partial) still stack into a uniform
  `(seq, C, size, size)`. This uniformity is the headline behaviour — test with
  full-zoom + partial-zoom + corner focus mixed, both int and (h,w) size. Resize
  resamples, so pin shape/len only, NOT pixel-exact values.
- API pins (`tests/test_api_contract.py`): both names added to
  `test_exp_exports` expected set; two `issubclass(..., DeviceTransferMixin)`
  lines appended to the combined sequence-types pin. Both already exported at
  top-level `exp.__all__`.
- Helpers `_focus_row(i)` (in-range distinct Focus) and
  `_distinct_focus_sequence(i)` ((4,3) arange block) live at module level, mirror
  the `_distinct_image` / `_distinct_sequence` pattern in test_image.py.
