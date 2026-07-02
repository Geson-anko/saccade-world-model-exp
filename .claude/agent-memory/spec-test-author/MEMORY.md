# spec-test-author memory

- [Testing conventions](feedback_testing_conventions.md) — layout/markers/torch/what-not-to-test rules that recur across the suite
- [exp/types/ spec](project_image_types_spec.md) — Image value object + DeviceTransferMixin ABC spec→test patterns
- [Focus types spec](project_focus_types_spec.md) — Focus zoom [0,1] + FocusSequence/BatchedFocusSequence (rank-only post_init, is_valid/validate, apply uniform shape)
- [Latent types spec](project_latent_types_spec.md) — Latent family on ElementArray base; ndim-only; iter_batch/iter_sequence/from_batches; import moved to exp.types.elements.latent
- [torch module tests](feedback_torch_module_tests.md) — patterns for spec-testing nn.Module (RoPE/ViT/weight init) on real CPU tensors
- [SequenceModel spec](project_sequence_model_spec.md) — SequenceModel[THidden] ABC: @final validation wrappers + abstract hooks, fake patterns
- [Predictor spec](project_predictor_spec.md) — Predictor[THidden]: DI-composed, focus||latent fusion, focus-type→output-type dispatch (step/forward), inject real MinGRU
- [tests/types stdlib collision](project_tests_types_stdlib_collision.md) — tests/types/ shadows stdlib `types`; collection breaks under default prepend mode, needs importmode=importlib
- [tests/types package collision](feedback_tests_types_package_collision.md) — tests/types/** が stdlib `types` と衝突し collection error になる harness 問題と切り分け
- [tests/types/ shadows stdlib](feedback_types_package_shadows_stdlib.md) — an __init__.py under tests/types/ names the pkg `types`, breaking collection suite-wide
- [ImageEncoder overload spec](project_encoder_overload_spec.md) — rank-preserving __call__ (Image/Batched/Sequence/BatchedSequence); BatchNorm N=1 train constraint = ValueError
- [ImageDecoder spec](project_decoder_spec.md) — encoder-mirror Latent→Image decoder; linear out (no range test), no detach (contract pin), GroupNorm so N=1 train OK
- [Data pipeline spec](project_data_pipeline_spec.md) — exp.data glimpse pipeline (random_focus_sequence/GlimpseDataset/collate_glimpses) real-file-IO test patterns
- [ruff isort missing module](feedback_ruff_isort_missing_module.md) — red-first tests importing a not-yet-created exp.* module get odd isort grouping that self-corrects once the module lands
