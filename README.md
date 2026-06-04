# logit-masking: 文字数制約付き日本語要約

ロジットマスキングによって、LLM の生成テキストを指定した文字数範囲（下限〜上限）に収める実験スクリプトです。Qwen3.5-9B を MacBook M3 (MPS) で動作させることを想定しています。

## 概要

LLM に「〇〇〜△△文字で要約して」とプロンプトだけで指示しても、文字数制約は守られないことが多い。本スクリプトでは生成中のロジット操作と後段トリムを組み合わせ、**一回の生成で制約を満たす確率を高める**4つの手法を比較します。

### 制約の実現方針

| レイヤー | 担当 | 手法 |
|---|---|---|
| 生成中（下限） | `CharRangeProcessor` | 現在文字数が下限未満のとき、ストップトークンを `-inf` で禁止 |
| 生成中（上限） | `CharRangeProcessor` | 残余バジェットを超えるトークンを `-inf` でハードマスク |
| 生成中（収束） | `CharRangeProcessor` | 帯後半に入ったらストップのロジットをソフトにブースト |
| 生成中（文末強制） | `CharRangeProcessor` | テキストが `。！？` で終わったとき、ストップトークンを `+inf` で強制 |
| 後段（最終保証） | `trim_to_range` | 文境界単位でトリムし、範囲外なら `None`（再生成） |

### 比較条件

| 条件 | 内容 |
|---|---|
| `baseline` | 介入なし（プロンプトのみ） |
| `hard_only` | 上限ハードマスク ＋ 下限ストップ禁止 |
| `hard_soft` | `hard_only` ＋ ソフト EOS ブースト |
| `floor_sent` | 下限ストップ禁止 ＋ 文末強制 EOS ＋ ソフトブースト（ハードマスクなし） |
| `closing_inject` | 下限手前の窓内で一時停止し締め句を注入後に続きを生成（ハードマスクなし、短バンドは冒頭注入） |

### 合格判定

生成サンプルは以下の**3条件をすべて満たす**場合のみ `accepted`（合格）とカウントします。

1. `trim_to_range` によるトリム後に `[lower, upper]` 範囲内に収まること
2. raw テキストが文末記号（`。！？…」`）で終わっていること
3. `gpt-oss:20b`（Ollama）が「自然な終わり方」と判定すること（`はい` / `いいえ` で判定、曖昧回答はレポートにフラグ表示）

## 文字数の定義

**空白を除く Unicode コードポイント数**。全角・半角の区別なし、記号も数える。`str.isspace()` を使うため全角スペース (U+3000) も空白扱い。

## 環境

- Python 3.11+
- Apple Silicon Mac (MPS) / NVIDIA GPU (CUDA) / CPU いずれも動作
- Qwen3.5-9B fp16 は約 18 GB の unified memory が必要。メモリが足りない場合は `MODEL_ID` を `"Qwen/Qwen3-1.7B"` 等に変更してください。
- 自然さ判定に [Ollama](https://ollama.com) + `gpt-oss:20b` を使用します。事前に `ollama pull gpt-oss:20b` でモデルを取得し、Ollama サーバーを起動しておいてください。

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
# 全条件・全タスクを K=8 まで実行（既存分はスキップ）
uv run python run_experiment.py

# 特定条件・タスクのみ実行
uv run python run_experiment.py --condition baseline --condition hard_soft --task 1 --task 2

# K を 16 に増やす（k=9〜16 の不足分のみ生成し、k=1〜8 の既存結果を再利用）
uv run python run_experiment.py --k 16

# 生成せず、ストアから HTML レポートのみ再計算
uv run python run_experiment.py --report-only

# ストアを削除してゼロから実行（既存結果がすべて消えます）
uv run python run_experiment.py --reset
```

初回実行時にモデルのダウンロードとトークン別文字数テーブルの構築が走ります。テーブルは `~/.cache/char_budget/` にキャッシュされ、2回目以降はスキップされます。

結果はサンプル単位で `results.jsonl` に追記されます。クラッシュしても再実行で続きから再開できます（`checkpoint.json` 方式を廃止し、自動削除もなくなりました）。

### 再現性について

シードは `derive_seed(base_seed, task_i, k)` で決定論的に導出されます（`BASE_SEED = 1234`、`--seed` で変更可）。全条件が同じ `(task_i, k)` に同じシードを使う「共通乱数法」で、条件間の差を乱数ゆらぎではなく介入由来として切り分けやすくします。

> **注意**: MPS (Apple Silicon) は一部カーネルが非決定的なため、再現性はベストエフォートです。CUDA / CPU では完全再現できます。

### 主なオプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--k N` | 目標サンプル数/タスク（累積） | `8` |
| `--condition NAME` | 実行する条件名（複数指定可） | 全条件 |
| `--task IDX` | 実行するタスク番号 1〜4（複数指定可） | 全タスク |
| `--seed S` | 乱数のベースシード | `1234` |
| `--temperature` | 生成温度 | `0.7` |
| `--top-p` | top-p サンプリング | `0.8` |
| `--store PATH` | 結果ストアパス | `results.jsonl` |
| `--report PATH` | HTML レポートパス | `experiment_report.html` |
| `--report-only` | 生成せずサマリ + HTML のみ再計算 | — |
| `--reset` | ストア削除 | — |

## 出力例

```
condition          n  p_raw         p_accept        trim%  E[N]  P(N≤2)  P(N≤3)    MAE  p_punct  p_nat
baseline          32  0.12[0.05,0.25] 0.38[0.22,0.56]   18%  2.63   0.62    0.77   12.3     0.85   0.90
hard_only         32  0.55[0.38,0.71] 0.75[0.57,0.87]    8%  1.33   0.94    0.98    4.1     0.92   0.88
hard_soft         32  0.72[0.55,0.85] 0.85[0.69,0.94]    5%  1.18   0.98    1.00    2.8     0.95   0.95
```

| 指標 | 意味 |
|---|---|
| `n` | 累積サンプル数（全タスク合計） |
| `p_raw` | トリム前から範囲内に着地した割合（Wilson 95%CI 付き） |
| `p_accept` | 品質チェック込みの最終合格率（Wilson 95%CI 付き） |
| `trim%` | 上限トリムが発火した割合 |
| `E[N]` | 期待生成回数 = 1 / p_accept |
| `P(N≤k)` | k 回以内に 1 回合格する確率 |
| `MAE` | 帯中心からの平均絶対文字数ズレ |
| `p_punct` | raw テキストが文末記号で終わった割合 |
| `p_nat` | 自然さ判定で「はい」だった割合（文末記号合格サンプルのみ） |

## ファイル構成

```
run_experiment.py   # メインスクリプト（全処理が含まれる）
results.jsonl       # 永続累積ストア（サンプル単位 JSONL、自動削除されない）
```

## カスタマイズ

`run_experiment.py` の以下を変更することで動作を調整できます。

```python
MODEL_ID = "Qwen/Qwen3.5-9B"   # モデル変更（検証時は "Qwen/Qwen3-1.7B" 推奨）
BASE_SEED = 1234                 # ベースシード（--seed でも変更可）
```

タスク定義は `TASKS` リスト（`(text, lower, upper)` の組）を変更してください。タスクを追加・変更すると `task_hash` が変わり、旧レコードとは別の実験として蓄積されます。

`CharRangeProcessor` の `soft_frac`（ソフトブーストを開始する帯内の位置比率）や `boost`（ロジット加算量）も調整可能です。
