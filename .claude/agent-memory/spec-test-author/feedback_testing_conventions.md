---
name: testing-conventions
description: Recurring spec→test conventions for saccade-world-model-exp (layout, markers, what not to test, torch usage)
metadata:
  type: feedback
---

Conventions that hold across this project's test suite.

**Why:** the testing-strategy skill + per-task specs repeat these; encoding them
avoids re-deriving each time.

**How to apply:**
- `tests/` mirrors `exp/` 1:1 (`exp/types/image.py` ↔ `tests/types/test_image.py`).
  Sub-package test dirs get NO `__init__.py` when filenames are unique.
- Test functions take NO return annotation (project convention; even though the
  pre-existing `tests/test_smoke.py` uses `-> None`, the testing-strategy skill
  says none is needed — follow the skill).
- Real CPU torch always; never mock torch/numpy/torchvision surfaces. Device-/IO-
  touching tests are integration-real. Deterministic construction ⇒ no seed.
- Exception messages: substring checks only, never full-message equality.
- Do NOT write: import-existence asserts, getter round-trips, `__init__`-set-field
  asserts, inheritance asserts OUTSIDE api_contract, constant-literal asserts.
- `issubclass`/`__all__`/type-alias pins are the ONE exception, allowed only in
  `tests/test_api_contract.py` with `@pytest.mark.api_contract` and a comment
  saying "contract pin, not a behaviour test".
- ruff isort wants third-party imports (pytest, torch) grouped with no blank line
  among them, then a blank line, then first-party `exp` imports; and NO blank line
  before the first section-comment after the import block. Run
  `uv run ruff check --fix && uv run ruff format` before reporting.
- I cannot edit `exp/` or `pyproject.toml`. The `api_contract` marker is already
  registered in pyproject.toml. See [[image-types-spec]].
