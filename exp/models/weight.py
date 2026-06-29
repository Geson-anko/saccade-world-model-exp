import torch.nn as nn

__all__ = ["init_weights"]


def init_weights(m: nn.Module, init_std: float = 0.02) -> None:
    """Truncated normal で重みを、0 で bias を初期化する (iJEPA 系)。

    Linear / Conv2d / ConvTranspose2d の weight は std=init_std の切断正規分布、
    LayerNorm は weight=1・bias=0。`nn.Module.apply` に渡して再帰適用する想定。

    Args:
        m: 初期化するモジュール (子モジュールには apply 経由で個別に適用される)。
        init_std: 切断正規分布の標準偏差。
    """
    match m:
        case nn.Linear() | nn.Conv2d() | nn.ConvTranspose2d():
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        case nn.LayerNorm():
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        case _:
            pass
