---
name: sequence-model-spec
description: SequenceModel[THidden] ABC (exp/models/base.py) spec→test patterns — @final validation wrappers + abstract hooks
metadata:
  type: project
---

`exp/models/base.py` defines `SequenceModel[THidden](nn.Module, abc.ABC)` —
the swappable sequence-model base (minGRU / Transformer subclass it).

**Why:** it is a project-owned ABC, so the testing strategy allows a minimal
concrete fake (integration-with-fakes) rather than forcing a real model.

**How to apply (tests in `tests/models/test_base.py`):**
- Public `forward`/`step` are `@final` validation wrappers owned by the base;
  subclasses implement only the abstract hooks `_forward`/`_step`. Fakes must
  NOT redefine `forward`/`step` (they'd hit the @final restriction and also
  bypass the contract under test).
- Contract: validate input ndim (`forward` needs `(*, seq, dim)` ndim>=2,
  `step` needs `(*, dim)` ndim>=1) then call the hook, then validate
  `out.shape == x.shape` (`"must return"` on mismatch). Substring-match the
  shape hint via raw-string regex, e.g. `match=r"\(\*, seq, dim\)"`.
- Observe hidden threading behaviourally: echo fake adds `hidden.unsqueeze(-2)`
  in `_forward` so `model(x)` vs `model(x, h)` differ — pins that hidden
  reaches the hook without inspecting internals.
- Gradient/hook liveness: `model(x)[0].sum().backward()` then
  `model.lin.weight.grad is not None` (confirms `nn.Module.__call__` dispatch).
- `SequenceModel[None]` is valid: a fake returning `(x, None)` from both hooks;
  assert `hidden is None`.
- Abstract pin: `with pytest.raises(TypeError): SequenceModel()  # type: ignore[abstract]`
  and a subclass implementing only `_forward` (missing `_step`) likewise raises.
- Do NOT test `@final` override-prohibition (pyright static guarantee) or
  issubclass/import-existence. See [[torch-module-tests]] [[testing-conventions]].

As of 2026-06-29 the base implementation already exists and conforms — the
suite is green (16 tests pass), not red.
