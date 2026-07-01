---
name: latent-types-spec
description: Spec→test conventions for exp/types/elements/latent.py (Latent + LatentSequence/BatchedLatent/BatchedLatentSequence, ndim-only, ElementArray base)
metadata:
  type: project
---

`exp/types/elements/latent.py` holds the latent value objects. Full family
behaviour tests added on branch `refactor/2026-07-01/element-array-2axis`
(spec-first; impl by parallel agent, already in tree).

**Why:** the value-object family was refactored onto a shared `ElementArray`
base ([[element-array-base]] if written); latent mirrors image/focus patterns
([[image-types-spec]], [[focus-types-spec]]) but with NO extra shape constraint.

**How to apply:**
- IMPORT PATH CHANGED: modules moved from `exp/types/<name>.py` to
  `exp/types/elements/<name>.py`. Import `from exp.types.elements.latent import
  BatchedLatent, BatchedLatentSequence, Latent, LatentSequence`. The old
  `exp.types.latent` path no longer resolves (the pre-refactor test_latent.py
  imported it and would fail collection). Other tests still on the old path
  (tests/models/test_encoder.py, tests/test_loss.py) are red until updated by
  their owners — not my concern.
- Four classes, all `@final attrs.define(slots frozen eq=False)` identity value
  objects on `DeviceTransferMixin`: Latent `(dim,)` ndim1 / LatentSequence
  `(seq, dim)` ndim2 / BatchedLatent `(batch, dim)` ndim2 / BatchedLatentSequence
  `(batch, seq, dim)` ndim3. Validation is ndim-ONLY (dim sizes free, no `_SHAPE`),
  error substrings `(dim,)` / `(seq, dim)` / `(batch, dim)` / `(batch, seq, dim)`
  plus actual ndim+shape (substring only; escape parens in `match=` regex).
- ElementArray base gives all collection types: `len`/`__getitem__`
  (int→element, slice→same container, use `type(x) is Cls`)/`__iter__`/
  `from_elements` (dim0 stack, empty→ValueError "at least one element",
  generator OK). Empty leading axis `(0,...)` is VALID.
- BatchedLatentSequence (two-axis) adds: `iter_batch()` == `__iter__` walks
  batch axis dim0 → yields LatentSequence; `iter_sequence()` walks seq axis dim1
  (torch.unbind) → yields BatchedLatent of shape `(batch, dim)` (pin with
  `tensor[:, step]`); `from_sequences` dim0 stack of LatentSequence (empty→
  "at least one sequence"); `from_batches` dim1 stack of BatchedLatent (empty→
  "at least one batch"). NOTE the distinct empty-messages: element/sequence/batch.
  `from_batches(bls.iter_sequence())` round-trips to the original tensor — good
  inverse test. focus/image tests do NOT cover iter_batch/iter_sequence/
  from_batches (those are exercised first here).
- Module-level helpers `_distinct_latent(i)` / `_distinct_latent_sequence(i)` /
  `_distinct_batched_latent(i)` (arange blocks) mirror `_distinct_image` etc.
  Axis sizes deliberately distinct (BATCH=2, SEQ=3, DIM=4) so a swapped axis in
  indexing/iter/stack is observable.
- Suite collection is broken by an untracked `tests/types/__init__.py` +
  `tests/types/elements/__init__.py` shadowing stdlib `types`; self-verify by
  moving them aside then restoring — see [[types-package-shadows-stdlib]].
