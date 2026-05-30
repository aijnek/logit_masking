# logit-masking: 文字数制約付き日本語要約

ロジットマスキングによって、LLM の生成テキストを指定した文字数範囲（下限〜上限）に収める実験スクリプトです。Qwen3.5-9B を MacBook M3 (MPS) で動作させることを想定しています。

## 概要

LLM に「〇〇〜△△文字で要約して」とプロンプトだけで指示しても、文字数制約は守られないことが多い。本スクリプトでは生成中のロジット操作と後段トリムを組み合わせ、**一回の生成で制約を満たす確率を高める**3つの手法を比較します。

### 制約の実現方針

| レイヤー | 担当 | 手法 |
|---|---|---|
| 生成中（下限） | `CharRangeProcessor` | 現在文字数が下限未満のとき、ストップトークンを `-inf` で禁止 |
| 生成中（上限） | `CharRangeProcessor` | 残余バジェットを超えるトークンを `-inf` でハードマスク |
| 生成中（収束） | `CharRangeProcessor` | 帯後半に入ったらストップのロジットをソフトにブースト |
| 後段（最終保証） | `trim_to_range` | 文境界単位でトリムし、範囲外なら `None`（再生成） |

### 比較条件

| 条件 | 内容 |
|---|---|
| `baseline` | 介入なし（プロンプトのみ） |
| `hard_only` | 上限ハードマスク ＋ 下限ストップ禁止 |
| `hard_soft` | `hard_only` ＋ ソフト EOS ブースト |

## 文字数の定義

**空白を除く Unicode コードポイント数**。全角・半角の区別なし、記号も数える。`str.isspace()` を使うため全角スペース (U+3000) も空白扱い。

## 環境

- Python 3.11+
- Apple Silicon Mac (MPS) / NVIDIA GPU (CUDA) / CPU いずれも動作
- Qwen3.5-9B fp16 は約 18 GB の unified memory が必要。メモリが足りない場合は `MODEL_ID` を `"Qwen/Qwen3-1.7B"` 等に変更してください。

## セットアップ

```bash
uv sync
```

依存ライブラリ（`pyproject.toml` に記載）:

- `torch`
- `transformers`
- `python-dotenv`

## 実行

```bash
uv run python run_experiment.py
```

初回実行時にモデルのダウンロードとトークン別文字数テーブルの構築が走ります。テーブルは `~/.cache/char_budget/` にキャッシュされ、2回目以降はスキップされます。

## 出力例

```
condition   p_raw  p_accept  trim%   E[N]  P(N=2)  P(N=3)    MAE
--------------------------------------------------------------------------
baseline     0.12      0.45    18%   2.22    0.70    0.83   12.3
hard_only    0.55      0.88     8%   1.14    0.99    1.00    4.1
hard_soft    0.72      0.94     5%   1.06    1.00    1.00    2.8
```

| 指標 | 意味 |
|---|---|
| `p_raw` | トリム前から範囲内に着地した割合（自然収束の度合い） |
| `p_accept` | トリム後に範囲内の最終合格率 |
| `trim%` | 上限トリムが発火した割合 |
| `E[N]` | 期待生成回数 = 1 / p_accept |
| `P(N=k)` | k 回以内に 1 回合格する確率 |
| `MAE` | 帯中心からの平均絶対文字数ズレ |

## ファイル構成

```
run_experiment.py   # メインスクリプト（全処理が含まれる）
```

## カスタマイズ

`run_experiment.py` の以下を変更することで動作を調整できます。

```python
MODEL_ID = "Qwen/Qwen3.5-9B"   # モデル変更
```

```python
tasks = [
    (SAMPLE, 140, 200),   # (テキスト, 下限, 上限) を追加・変更
    (SAMPLE,  70, 100),
]
```

```python
summary = run_experiment(tasks, K=8)   # K: 1条件あたりのサンプル数
```

`CharRangeProcessor` の `soft_frac`（ソフトブーストを開始する帯内の位置比率）や `boost`（ロジット加算量）も調整可能です。
