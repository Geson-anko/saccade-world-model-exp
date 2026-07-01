---
name: tests-types-stdlib-collision
description: tests/types/ package shadows stdlib `types`, breaking pytest collection under default prepend import mode
metadata:
  type: project
---

`tests/types/` (with `__init__.py` files) collides with the stdlib `types`
module. Under pytest's default `prepend` import mode this makes collection
fail repo-wide: `ModuleNotFoundError: No module named 'types.elements';
'types' is not a package`. It affects EVERY file under `tests/types/`
(test_base / test_image / test_latent / test_focus / test_device / test_size
/ test_tensor) plus others, not any single new file.

**Why:** the tests tree was reorganized into packages mirroring
`exp/types/elements/`, and the top-level package name `types` clashes with
the stdlib module of the same name. `pyproject.toml` sets no `importmode`
(defaults to `prepend`) and uses `pythonpath = "."`.

**How to apply:** when a `tests/types/**` file "fails" collection with the
`types` ModuleNotFound error, do NOT treat it as a bug in your test — verify
with `uv run pytest <file> --import-mode=importlib` (that resolves it and the
tests run). The real fix is a one-line config change
(`importmode = "importlib"` in `[tool.pytest.ini_options]`), which is outside
the test-author scope (pyproject.toml is not a test file). Report it to the
orchestrator rather than editing pyproject or renaming the package. See
[[testing-conventions]] for the layout mirror rule that produced this name.
