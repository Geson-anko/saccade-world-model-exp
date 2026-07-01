# Code Quality Reviewer — Memory Index

- [exp/types package](project_types_package.md) — 最基盤の型パッケージ。elements/ 値オブジェクト群 + device 抽象。public API は契約テストで固定、不変条件あり
- [element base](project_element_base.md) — elements/base.py の内部共通基底 (Element/ElementArray/BatchedElementSequence)。shape 検証・collection・stack・device を DRY 集約
- [focus types](project_focus_types.md) — Focus 行動値オブジェクト + Focus 収集型。zoom 閉区間 [0,1]・_FocusValidation mixin・elements/ 移設後の refactor 境界メモ
- [vit component](project_vit_component.md) — VisionTransformer / weight.py の public 面と RoPE 不変条件。refactor 境界メモ (components flatten 済み・Mlp は mlp.py へ移設)
- [mingru component](project_mingru_component.md) — minGRU の public 面・log-space scan 数値不変条件・実施済み refactor の記録
- [loss module](project_loss_module.md) — MSELoss / SIGReg functor の public 面・SIGReg 数値不変条件・実施済み refactor の記録
- [predictor model](project_predictor_model.md) — Predictor 上位モデル + SequenceModel.dim 契約 + FOCUS_DIM 定数の public 面・forward match は明示のまま維持する判断
