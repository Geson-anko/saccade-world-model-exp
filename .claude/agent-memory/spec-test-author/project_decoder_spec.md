---
name: project-decoder-spec
description: ImageDecoder (exp.models.decoder) contract-test spec — the read-out mirror of ImageEncoder
metadata:
  type: project
---

`ImageDecoder` in `exp/models/decoder.py` is the visualization/eval
read-out (CLAUDE.md decision #6). Contract tests live in
`tests/models/test_decoder.py`, mirroring `tests/models/test_encoder.py`.

**Why:** it is the exact mirror of `ImageEncoder`: rank-preserving
overloaded `__call__` mapping Latent family -> Image family (Latent->Image,
BatchedLatent->BatchedImage, LatentSequence->ImageSequence,
BatchedLatentSequence->BatchedImageSequence). Trunk = ConvDecoder
(transposed conv). Public attr `latent_dim`.

**How to apply** when re-testing / answering implementer questions:
- Output is LINEAR (no sigmoid/tanh) -> do NOT write output-range tests.
- Decoder does NOT detach its input; caller is responsible for the detach.
  This is pinned as a public contract in `TestDetachContract` (t.grad is
  not None after backward). Not a behaviour test — a decision pin.
- Internal norm is batch-independent (GroupNorm), so a lone `Latent` (N=1)
  passes in TRAIN mode. This is the key contrast vs encoder's BatchNorm
  N=1 ValueError. `test_latent_returns_image_in_train_mode` guards it.
- Construction validation (all ValueError): out_channels not in ChannelFormat
  {1,3,4}; image_size not expressible as init_spatial * 2**N; latent_dim<=0.
  Use substring/`pytest.raises(ValueError)`, never message equality.
- Test config: latent_dim=8, out_channels=3, image_size=32 (=4*2**3),
  base_channels=16, init_spatial=4. Value objects take a single positional
  tensor and expose `.tensor` / `.to(device)`.
- No loss computation (exp.loss.MSELoss is Latent-family-typed, not images).
  Contract = shape/type/gradient/detach/validation/device only.

**ConvDecoder (the trunk under ImageDecoder)** lives in
`exp/models/components/conv_decoder.py`, re-exported from
`exp.models.components` (`__all__=["ConvDecoder"]`) so it is a tested
public component. Tests: `tests/models/components/test_conv_decoder.py`
(mirror test_mingru.py style, raw tensors, no value objects). Contract:
`forward(x)` maps `(*, feature_dim) -> (*, out_channels, s', s')`,
arbitrary leading dims folded via `math.prod(lead) or 1` (so N=1 works,
GroupNorm is batch-independent). `s' = init_spatial * 2**N` (square only;
`init_spatial` scalar). `out_channels` is any positive int — NO
ChannelFormat constraint (that check lives only in ImageDecoder). Construct
ValueError when a side isn't `init_spatial*2**N` OR the per-side stage
counts disagree (non-square, e.g. out_size=(8,16) with init_spatial=4).
ValueError message contains "init_spatial" -> safe `match="init_spatial"`.
`out_size` accepts int or (h,w) tuple via `size_2d_to_tuple`. Output linear
(no range test). All green as of first write (no impl change needed).

Pyright gotcha: a `_make_*` factory helper whose `out_size` (or any
`Size2d`) default is an int gets inferred as `out_size: int`, so a test
passing a `(n, n)` tuple fails `just type`. Annotate the helper param as
`out_size: Size2d` (`from exp.types import Size2d`). tests/ isn't strict
pyright but `just type` still runs over it.
Related: [[project_encoder_overload_spec]], [[ruff-isort-missing-module]].
