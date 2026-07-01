---
name: predictor-spec
description: Predictor[THidden] spec→test — first DI-composed model; focus||latent fusion, focus-type-decides-output-type dispatch (step vs forward), MinGRU injected as real resource
metadata:
  type: project
---

`exp/models/predictor.py` — `Predictor[THidden](nn.Module)` is the sequence-model
stage: fuses next action `Focus` + current embedding `Latent` at the input of an
injected `SequenceModel`, projects back to a latent. First model built by DI
(receives a `SequenceModel` rather than constructing one, unlike `ImageEncoder`).

**Why (test shape):** the load-bearing contract is *input-type decides everything*:
- `Focus`/`BatchedFocus` (no seq axis) → `sequence_model.step`
- `FocusSequence`/`BatchedFocusSequence` (seq axis) → `sequence_model.forward`
- returned value-object type mirrors the `focus` type (Latent family).

**How to apply:**
- Inject a real small `MinGRU(dim=8, depth=1)` as `sequence_model`, not a fake —
  MinGRU is lightweight so "real resource first" beats a self-owned-ABC fake here
  (a `SequenceModel` fake would only re-assert our own assumptions). See
  [[torch-module-tests]] and [[sequence-model-spec]].
- Pick `latent_dim` distinct from FOCUS_DIM(3) and seq width(8) (used 5) so a
  mis-wired concat/projection surfaces as a shape error instead of passing.
- Hidden is MinGRU's belief `(*, depth, dim)` at the *sequence* width, never
  latent_dim; leading batch dims of the input are preserved on it.
- Continuity test (split seq, thread first hidden into second, compare to full
  forward) is the real proof the step/forward dispatch + hidden pass-through are
  wired right — mirrors MinGRU's own continuity test but through Predictor's
  input_proj→forward→output_proj. Use `.double()` for numerical headroom.
- The `case _: raise TypeError` branch in Predictor.forward is unreachable from
  valid inputs (the 4 focus types are @final, and `.tensor` access precedes the
  match), so don't test it — would need a contrived value-object-mimicking fake.
- Grad-flow tests assert non-None grad on `input_proj.weight`, `output_proj.weight`,
  and at least one `sequence_model` param (end-to-end belief training).

Sidecar changes tested alongside: `SequenceModel.dim` became an abstract property
(added `dim` to the 4 fakes in test_base.py + a "missing dim → not instantiable"
pin symmetric to the hook-missing test); `FOCUS_DIM=3` pinned in test_focus.py as a
semantic invariant (`Focus(...).tensor.shape == (FOCUS_DIM,)`), not a bare literal.
