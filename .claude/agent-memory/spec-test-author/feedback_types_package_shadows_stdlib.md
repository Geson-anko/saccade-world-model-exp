---
name: types-package-shadows-stdlib
description: An __init__.py under tests/types/ makes pytest import the package as top-level `types`, shadowing the stdlib and breaking collection suite-wide
metadata:
  type: feedback
---

Under pytest `import-mode=prepend` (the default here) with `tests/` having NO
top-level `tests/__init__.py`, adding an `__init__.py` at `tests/types/` makes
pytest name that package top-level `types`, which collides with the stdlib
`types` module. Result: EVERY test module fails collection with
`ModuleNotFoundError: No module named 'types.<x>'; 'types' is not a package`
(not just files under tests/types).

**Why:** pytest walks up from a test file collecting `__init__.py` dirs to form
the dotted module name; it stops at the first dir lacking `__init__.py`. With
`tests/` bare, `tests/types/__init__.py` becomes the package root -> package name
`types` -> shadows stdlib.

**How to apply:**
- If the suite suddenly fails collection with a `types.*` / stdlib-shadow error,
  check for a newly-added `tests/types/__init__.py` (often from a concurrent
  refactor moving `exp/types/*.py` into `exp/types/elements/` and mirroring the
  test tree). This is an environment/layout bug, NOT a bug in the test file.
- To self-verify a file in `tests/types/**` while the suite is broken by this,
  temporarily move the offending `tests/types/__init__.py` (and any nested one)
  aside, run `uv run pytest <file> -q`, then restore them byte-for-byte. Do this
  non-destructively in scratchpad; do not edit another task's files.
- The real fix belongs to whoever owns the layout: either add a
  `tests/__init__.py` (package becomes `tests.types`, no shadow) or drop the
  `tests/types/__init__.py`. Report it; don't silently change it.
- HEAD layout (verified 2026-07-01) has NO `__init__.py` under `tests/types/`
  and collects cleanly; `tests/models/` DOES have one but doesn't collide
  because `models` isn't a stdlib name.

Related: [[testing-conventions]]
