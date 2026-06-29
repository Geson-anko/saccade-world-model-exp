---
name: testing-strategy
description: saccade-world-model-exp のテスト方針。real resource を最優先し、自前 ABC の fake のみ許可、3rd-party ライブラリ表面 (torch / numpy / torchvision の internals 等) や自分のコードの内部関数のモックは禁止。テスト 3 区分 (unit / integration-with-fakes / integration-real)、tests は exp/ を 1 対 1 でミラー、書いてはいけないテスト、公開 API 契約ピン例外、ソース／テストの具体サンプル、mutation testing。テストコードを書く／pytest 周りを設定する前に読む。
---

# saccade-world-model-exp テスト方針リファレンス

手元で書き始める前にざっと読む「実行可能なまとめ」。まだソース・テストは無いので、**書き始めるときの方針** として位置づける。背景の学びは [memory/](../../../memory/) に蓄積する。

## 哲学

「動くテスト」ではなく「**実環境の振る舞いを保証するテスト**」を優先する。3rd-party ライブラリ表面をミラーした fake は、その挙動に対する **自分の仮定** をテストするだけで上流変更を検出できない (GOOS / Freeman & Pryce: "Don't mock what you don't own")。fake が drift して緑でも実機で死ぬ事故を防ぐため、**実 resource を最優先** する。

### 検証対象の優先順位

1. **実 resource** — `tmp_path` で実 file I/O、実 subprocess (`sleep` / `echo`)、loopback socket、実画像/実 tensor fixture を実ライブラリ (numpy / torch / torchvision) に通す。GPU 不要なものは CPU tensor で実行する。これが第一選択
2. **自前 ABC の fake** — このプロジェクトが自分で定義した抽象 (例: データソース IF、Encoder / SequenceModel / Decoder の base IF など) の差し替え。**所有しているもの** は fake OK
3. **3rd-party 表面のモック → 禁止**: `torch.*` / `numpy.*` / `torchvision.*` の internals、`time.sleep`、`os` の I/O、`socket` 直叩き等。必要なら integration-real に分類する
4. **自分のコードの内部関数モック → 禁止**: 内部 private 関数を `mocker.patch` で直接置き換える行為。リファクタで壊れるだけで何も保証しない

## 基本原則

- 必要十分なテストのみを記述する。過剰なテストは避ける
- 内部実装の詳細はテストしない。公開インターフェースと振る舞いをテストする
- Python のテスト関数に戻り値の型アノテーションは不要
- **コードカバレッジは診断であり目標ではない**。Fowler: *"high coverage numbers are too easy to reach with low quality testing"*。100% は赤信号
- 乱数を使うテストは seed を固定し決定的にする (torch / numpy / random の seed)

## テストレイアウト — `tests/` を `exp/` に 1 対 1 でミラー

- `exp/saccade.py` ↔ `tests/test_saccade.py`
- `exp/data.py` ↔ `tests/test_data.py`
- `exp/__init__.py` の公開面 ↔ `tests/test_api_contract.py` (公開 API 契約ピン)
- `tests/` 直下に置くのは `conftest.py` / `helpers.py` / (必要なら) `fakes/` のみ
- 1 ファイル 1 モジュールのミラーを原則とする

## テスト 3 区分

| 区分 | 検証対象 | モック許容 |
|---|---|---|
| **unit** | 純粋ロジック (座標変換、padding サイズ計算、tensor shape の不変条件 等) | なし |
| **integration-with-fakes** | モジュール間結合 (自前 ABC 越しの契約) | **自前 ABC のみ** |
| **integration-real** | 実 file I/O・実ライブラリ (numpy/torch/torchvision)・実 subprocess・loopback 等の結合点 | 原則なし。実 resource を立てる |

新規テストを書く前に区分を決める。3rd-party モックが必要に見えたら integration-real に分類できないか先に検討する。本質的に人間しかできない確認 (出力画像の視覚的品質判断など) のみ manual に残す。

### GPU まわり

- 既定は CPU tensor で書く (CI も無く、ローカルでも GPU を専有しないため)。
- GPU 必須のテストを書く場合は `@pytest.mark.gpu` のような marker を **`pyproject.toml` に登録した上で** 付け、`torch.cuda.is_available()` が False なら skip する。`--strict-markers` 有効なので未登録 marker はエラーになる。

## 何をテストするか / しないか

### 書く

- 正常系: 期待入力に対する期待出力 (crop の shape、z=1 で全体が入る、padding 後サイズ 等)
- 異常系: エラー時の例外・メッセージ (**substring** 検証、完全一致は不可)
- 警告: 設定不正時の `RuntimeWarning` 等
- エッジケース: 境界値・空入力・巨大入力 (z→0、p が端、画像が非正方 等)

### 書かない (marginal value ゼロ — 削除対象)

- **継承の追試**: `assert issubclass(MyError, RuntimeError)` を `class MyError(RuntimeError):` のために書く。pyright と言語仕様が既に保証
- **import 可能性の追試**: import 直後の `assert X is not None`。import が失敗すれば collection で死ぬので冗長
- **定数 literal の追試**: `assert TIMEOUT == 5`。意味的不変条件 (`assert s_prime <= size`) なら OK
- **getter/setter のラウンドトリップ**: `obj.foo = x; assert obj.foo == x`
- **`__init__` でフィールド設定されたことだけの確認**
- **framework / stdlib の動作追試**: `assert json.loads("{}") == {}`
- **例外メッセージの完全一致**: `assert str(err) == "exact text"`。`"keyword" in str(err)` の意味性検証に留める
- **private 関数を直接 import してのテスト**: `_prefix` の関数・メソッドを test から import して叩く。public 面越しに振る舞いを検証する ([memory/no-testing-private-functions](../../../memory/no-testing-private-functions.md))
- **モックの戻り値をそのまま検証するだけ**

### 例外: 公開 API 契約ピン

外部から依存される公開 API 名・基底クラス・型エイリアスは契約として固定する価値あり (Hyrum's law mitigation)。**唯一の例外**:

- 集約場所: `tests/test_api_contract.py`
- マーカー: `@pytest.mark.api_contract` (要 `pyproject.toml` の `[tool.pytest.ini_options] markers` 登録)
- 意図を明示: コメントで「これは契約ピンであり振る舞いテストではない」と書く
- 対象例: `exp` の公開シンボル整合性、公開例外の継承関係、公開型エイリアスの解決先

## 具体例: ソースとテストのサンプル

ここまでの原則を 1 セットにまとめた最小サンプルを `examples/` に同梱する (**説明用の縮約コード・実在モジュールではない**。`.claude/` 配下なので pytest には収集されない)。「public 面だけをテストする」「正常系・エッジ・異常系を押さえる」「real CPU tensor + seed 固定」「private は触らない」を 1 つの例で示す。

- [examples/example.py](examples/example.py) — ソース。public な `Standardizer` (nn.Module・`__all__` 明示) と private helper `_safe_std`。
- [examples/test_example.py](examples/test_example.py) — テスト。public だけを import し、**正常系** (平均0・std1) / **エッジ** (分散0で NaN を出さない) / **異常系** (不正 eps を substring match) の 3 本。

private `_safe_std` は **直接テストしない**。その「分散 0 を eps で下支えする」振る舞いは、エッジケースのテストが public 面越しに既に担保している。

### 反例 (書いてはいけない)

```python
# ❌ private helper を直接 import して叩く — _prefix を import した時点で規約違反。
#    リファクタで壊れるだけで公開契約を何も保証しない。
from exp.models.components.example import _safe_std

def test_safe_std_clamps():
    assert float(_safe_std(torch.zeros(4), 1e-5)) == 1e-5
```

### 数学的契約を pin するときの落とし穴

直交変換のノルム保存・相対位置不変性のような **数学的性質** を assert するときは、その性質が成り立つ前提を正しく固定すること。例えば RoPE の相対位置性 `<R(p)q, R(p')k> = <q, R(p'-p)k>` は **同一の q,k ベクトル**を異なる位置で回したときの不変性であり、各位置に独立乱数を置くと成り立たず false red になる (実際に踏んだ。共有ベクトルを broadcast して検証する。[memory/no-testing-private-functions](../../../memory/no-testing-private-functions.md) 周辺の知見)。

## モック (使用する場合)

- `pytest-mock` を使う (`unittest.mock` ではなく `mocker`)。複数テストで共有するなら `tests/conftest.py` の fixture に
- **モック対象は自前 ABC のみ**。3rd-party 表面 / 自分のコードの内部関数はモックしない (前述)

## doctest について

`pytest --doctest-modules` は **有効化していない**。docstring に `>>>` 例を書く場合も正しく保つべきだが、自動実行はされない前提。doctest を実行したくなったら明示的に設定を足す。

## 推奨ツール: mutation testing

「fake / テストが弱くないか」を経験的に検証する手段として `mutmut` ([github.com/boxed/mutmut](https://github.com/boxed/mutmut)) が有効。

- 仕組み: source を機械的に corrupt し (`+`→`-`, `>`→`>=`, `True`→`False` 等)、テストが mutant を検出するか測る。**survive した mutant** はその箇所のテストが弱い合図
- 常用しない。リリース前 / 節目に回す程度で十分。100% mutation score は狙わない (コストが super-linear)
- 適用優先: 純粋ロジック (座標変換・shape 計算) → モジュール結合層

## 参考文献

- [Martin Fowler: On the Diverse And Fantastical Shapes of Testing (2021)](https://martinfowler.com/articles/2021-test-shapes.html)
- [Kent C. Dodds: Write tests. Not too many. Mostly integration.](https://kentcdodds.com/blog/write-tests)
- [André Schaffer (Spotify): Testing of Microservices](https://engineering.atspotify.com/2018/01/testing-of-microservices)
- [Sebastian Bergmann: Do not mock what you do not own](https://thephp.cc/articles/do-not-mock-what-you-do-not-own)
- [James Shore: Testing Without Mocks: A Pattern Language](https://www.jamesshore.com/v2/projects/nullables/testing-without-mocks)
- [Kent Beck: Test Desiderata](https://testdesiderata.com/)
- [Hillel Wayne: Some tests are stronger than others](https://buttondown.com/hillelwayne/archive/some-tests-are-stronger-than-others/)
- [Martin Fowler: Test Coverage](https://martinfowler.com/bliki/TestCoverage.html) — 100% は赤信号
