"""`example.Standardizer` の振る舞い仕様 (テスト方針の説明用サンプル)。

public 面 (`Standardizer`) だけを import し、正常系・エッジ・異常系を押さえる。 real CPU tensor
を実ライブラリへ通し、乱数は seed 固定。private `_safe_std` は 直接テストしない
(その振る舞いはエッジケースのテストが public 越しに担保する)。
"""

import pytest
import torch
from exp.models.components.example import Standardizer  # public だけを import


class TestStandardizer:
    def test_outputs_zero_mean_unit_std(self):
        # 正常系: 標準化後は平均≈0・標準偏差≈1 (実 CPU tensor を実ライブラリへ)。
        torch.manual_seed(0)  # 乱数は seed 固定で決定的に
        out = Standardizer()(torch.randn(256) * 5 + 3)

        torch.testing.assert_close(out.mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
        torch.testing.assert_close(out.std(), torch.tensor(1.0), atol=1e-2, rtol=0)

    def test_constant_input_does_not_divide_by_zero(self):
        # エッジケース: 分散 0 でも eps で割り、NaN/Inf を出さない。
        out = Standardizer()(torch.full((16,), 7.0))

        assert torch.isfinite(out).all()

    def test_non_positive_eps_raises(self):
        # 異常系: 不正な eps は弾く (メッセージは substring 検証、完全一致しない)。
        with pytest.raises(ValueError, match="eps"):
            Standardizer(eps=0.0)
