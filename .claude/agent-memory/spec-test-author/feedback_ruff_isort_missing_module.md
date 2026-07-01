---
name: ruff-isort-missing-module
description: red-first test files that import a not-yet-created exp.* module get an odd isort grouping that self-corrects once the module lands
metadata:
  type: feedback
---

When writing a red-first test that imports a not-yet-existent `exp.*`
module (e.g. `from exp.models.decoder import ImageDecoder` before
`exp/models/decoder.py` exists), `ruff check --fix` groups that import
with third-party (next to `torch`) instead of first-party, splitting the
`exp.*` imports with a blank line. The mirror test file (e.g.
test_encoder.py) looks cleaner because its source module already exists.

**Why:** ruff's isort classifies a module as first-party only if it can
resolve it on disk. A missing module is treated as unknown/third-party.

**How to apply:** do not fight it — run `ruff check --fix` and accept the
transient grouping; the file still passes the lint gate. Once
`spec-driven-implementer` creates the source module, a `just format`
run regroups the import correctly with no functional change. Mention this
in the handoff so the implementer isn't surprised by the reflow. Related:
this is normal for the red phase; see [[project_predictor_spec]] and other
mirror-of-encoder specs.
