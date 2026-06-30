# Code Quality Reviewer — Memory Index

- [exp/types package](project_types_package.md) — 最基盤の型パッケージ。Image / DeviceTransferMixin。public API は契約テストで固定、不変条件あり
- [focus types](project_focus_types.md) — Focus 行動値オブジェクト + FocusSequence / BatchedFocusSequence。zoom 閉区間 [0,1]・共有 helper・refactor 境界メモ
- [vit component](project_vit_component.md) — VisionTransformer / weight.py の public 面と RoPE 不変条件。refactor 境界メモ (components flatten 済み・Mlp は mlp.py へ移設)
- [mingru component](project_mingru_component.md) — minGRU の public 面・log-space scan 数値不変条件・実施済み refactor の記録
