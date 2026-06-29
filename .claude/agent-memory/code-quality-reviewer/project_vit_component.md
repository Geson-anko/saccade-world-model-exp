---
name: vit-component
description: exp/models/components の vit.py / weight.py の public 面と RoPE 不変条件。refactor 時の触れてよい境界
metadata:
  type: project
---

`exp/models/vit.py` = 画像エンコーダ VisionTransformer と内部 building block (`components/` は flatten 済みで存在しない)。`Mlp` は `exp/models/mlp.py` へ移設され、`vit.py` は `from .mlp import Mlp` で import して `__all__` に再 export 維持。

public 面 (変更不可):
- `__all__`: AxialRoPE, Attention, Block, Mlp, PatchEmbed, VisionTransformer
- 各クラスの `__init__` シグネチャ・引数名・既定値
- VisionTransformer.forward 契約: `(*, C, H, W) -> (*, n_patches, embed_dim)` (任意の leading batch を flatten/unflatten)
- 公開属性: grid_hw, n_patches, embed_dim (PatchEmbed は num_patches)
- module-private `_fix_init_weight(attn_proj, mlp_out, layer_id)` はテストが直接触る → 実質 public 扱い

RoPE 不変条件 (数学的振る舞いを変えない):
- 2D axial: head_dim を y軸(前半)/x軸(後半) に 2分割。head_dim % 4 == 0 必須
- half-split 配置、行優先パッチ順序 (y外側/x内側 flatten)、相対位置性
- cos/sin は persistent=False buffer に precompute

`weight.py` = `init_weights` (iJEPA系 trunc_normal、match で Linear/Conv/LayerNorm 分岐)。最小・clean。

**Why:** spec-driven-implementer が初版を書いた直後の refactor 対象。コードは元から簡潔。
**How to apply:** この module は既にシンプル。`_rotate_half` の y/x ブロック処理に half_split 局所ヘルパを入れて 2x 重複を解消済み。それ以外の 2x 重複 (axis_angle_y/x の cat 等) は変数名が y/x を文書化しているので抽象化しない。torch.compile parity テストがあるので nested closure 等は compile 可否に注意。

関連: [[project-types-package]]
