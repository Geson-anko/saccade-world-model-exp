---
name: torch-module-tests
description: Patterns for spec-testing real torch nn.Module components (RoPE / ViT / weight init) on CPU
metadata:
  type: feedback
---

When spec-testing torch `nn.Module` components (e.g. `exp/models/components/`),
write against the public spec on real CPU tensors. No torch-internals mocking.

**Why:** project testing-strategy forbids 3rd-party (torch) surface mocking;
real tensors catch upstream/behaviour drift that fakes hide.

**How to apply:**
- Seed with `torch.manual_seed(0)` before any `torch.randn`.
- Use `torch.testing.assert_close` for tensor equality; `pytest.raises(ValueError, match="...")` (substring) for validation.
- Math-contract tests for rotary/orthogonal ops: norm preservation
  (`out.norm(dim=-1) == in.norm(dim=-1)`), identity at origin position,
  and relative-position invariance (joint translation leaves rotated q·k
  inner product unchanged). Row-major patch flatten -> `idx = y*grid_w + x`.
- RoPE relative-position test GOTCHA: the property is
  `<R(p)q, R(p')k> = <q, R(p'-p)k>` — it holds for the SAME q,k vectors
  rotated at different positions. Place one shared `q0`/`k0`
  (`q0.expand(...).clone()`) at every grid cell, NOT independent
  `torch.randn` per position (independent vectors make equal relative
  offsets give unequal dot products — a false red that looks like an
  implementation bug). Also assert a different offset gives a different
  dot, to pin the property both ways.
- Arbitrary-leading-batch forward `(*, C, H, W) -> (*, n_patches, embed_dim)`:
  pin empty / 1 / 2 leading dims as separate named tests (flatten/unflatten).
- Gradient flow: `out.sum().backward()` then assert at least one param has a
  non-None, non-zero grad (don't pin which param — implementation detail).
- Eval determinism: `model.eval()` + `torch.no_grad()`, two forwards close.
- `torch.compile` parity is an *executability* guarantee: wrap compile+forward
  in try/except and `pytest.skip` on Exception (CPU inductor may be missing).
  Avoids needing a new strict-marker; keeps the test self-contained.
- Don't register a `gpu`/compile marker unless GPU is actually required — the
  try/skip pattern is preferred for optional-backend tests.
