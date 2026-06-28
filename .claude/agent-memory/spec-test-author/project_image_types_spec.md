---
name: image-types-spec
description: Spec→test conventions established while testing the exp/types/ package (Image value object + DeviceTransferMixin ABC)
metadata:
  type: project
---

`exp/types/` holds shared value objects. Tested on branch
`feature/20260627/image-types` (tests written spec-first, impl by a parallel agent).

**Why:** these patterns recur for future value objects in this package.

**How to apply:**
- `Image` is attrs-built, immutable, `eq=False` (identity equality). Test the
  identity contract with two equal-valued instances that must compare `!=`, and
  assert `==` does NOT raise (guards against torch's "boolean value of Tensor is
  ambiguous"). Plain getter round-trips are forbidden; instead pin the CHW axis
  mapping with a NON-symmetric shape (e.g. `torch.zeros(3, 8, 16)` →
  channels=3/height=8/width=16) so a swapped axis would fail.
- `__attrs_post_init__` validates `ndim == 3`; error message contains substring
  `"C, H, W"` + the actual shape. Verify by substring only (`match=` /
  `in str(exc.value)`), never full-message equality.
- `DeviceTransferMixin` is an ABC; the one behavioural pin is that direct
  instantiation raises `TypeError` (use `# type: ignore[abstract]`).
- `to(device)` returns a NEW Image (`result is not img`), values equal via
  `torch.equal`. Classify `to`/`device` tests as integration-real (real CPU
  torch, no seed, no torch mocking).
- API contract pins live in `tests/test_api_contract.py` with
  `@pytest.mark.api_contract` (marker already registered in pyproject.toml).
  `exp.__all__` set-compared to `{"DeviceLike","DeviceTransferMixin","Image"}`;
  `issubclass(exp.Image, exp.DeviceTransferMixin)` is allowed ONLY here as a
  base-class contract pin. See [[testing-conventions]].
- `tests/types/` has NO `__init__.py` (test filenames are unique).
