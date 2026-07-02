---
name: data-pipeline-spec
description: exp.data glimpse pipeline (random_focus_sequence / GlimpseDataset / collate_glimpses) spec→test patterns
metadata:
  type: project
---

`exp.data` public面: `random_focus_sequence`, `GlimpseDataset`, `collate_glimpses`
(tests at `tests/data/test_dataset.py`, integration-real).

Key spec→test patterns:
- **real image I/O fixture**: helper writes PNG/JPG via `Image(uint8 tensor).save(path)`
  with a seeded `torch.Generator` per file for deterministic pixels. Sub-dirs via
  `path.parent.mkdir(parents=True)`. Ignored file: `notes.txt`.
- **path collection** verified only through `len(ds)` and cross-dataset determinism
  (`torch.equal` on ds[0] with two same-seed generators) — never peek `ds._paths`.
- **randomness contracts**: same-seed generators → `torch.equal`; consecutive draws
  from ONE generator → `not torch.equal`. point cols take negatives (x,y∈[-1,1]),
  zoom col never negative (asymmetric offset: point scale2/offset-1, zoom scale1/offset0).
- **collate**: empty list ValueError delegates to `BatchedFocusSequence.from_sequences`
  → message "at least one sequence"; spec asked match="at least one" (substring OK).
- Real `DataLoader(ds, batch_size=2, collate_fn=collate_glimpses, num_workers=0)` +
  `next(iter(loader))` for the DataLoader-integration test.
- image_size int→square (S,3,s,s); tuple (h,w)→(S,3,h,w). GRAY source promoted to RGB
  (3ch obs) via `as_channel_format(RGB).float()`.

Implementation already existed & complete when tests written; all 17 green.
