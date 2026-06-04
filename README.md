# logit-masking: 文字数制約付き日本語要約

ロジットマスキングによって、LLM の生成テキストを指定した文字数範囲（下限〜上限）に収める実験スクリプトです。Qwen3.5-9B を MacBook M3 (MPS) で動作させることを想定しています。

## 概要

LLM に「〇〇〜△△文字で要約して」とプロンプトだけで指示しても、文字数制約は守られないことが多い。本スクリプトでは生成中のロジット操作と後段トリムを組み合わせ、**一回の生成で制約を満たす確率を高める**8条件を比較します。

文字数制約は `(lower, upper) = (upper × 0.8, upper)` の帯として設定します。

### 制約の実現方針

| レイヤー | 担当 | 手法 |
|---|---|---|
| 生成中（下限） | `CharRangeProcessor` | 現在文字数が下限未満のとき、ストップトークンを `-inf` で禁止 |
| 生成中（上限） | `CharRangeProcessor` | 残余バジェットを超えるトークンを `-inf` でハードマスク（`hard_hard` のみ） |
| 生成中（収束） | `CharRangeProcessor` | 帯後半に入ったらストップのロジットをソフトにブースト |
| 生成中（文末強制） | `CharRangeProcessor` | テキストが `。！？` で終わったとき、ストップトークンを `+inf` で強制（`hard_force` のみ） |
| 生成中（締め句注入） | `ClosingInjectProcessor` + `ClosingInjectStopping` | 下限手前の文末窓内で一時停止し締め句を注入、続きを生成（`closing_inject` のみ） |
| 後段（最終保証） | `trim_to_range` | 文境界単位でトリムし、範囲外なら `None`（`*_trim` 条件のみ） |

### 比較条件（8条件）

下限制御は全条件で hard EOS フロア（下限未満での EOS 禁止）を使用。上限制御方式で命名。

| # | 条件 | 上限制御 | trim | 備考 |
|---|---|---|---|---|
| 1 | `baseline` | なし | N/A | 素のモデル出力で評価 |
| 2 | `baseline_trim` | なし | ✓ | #1 の出力を再利用（要約再生成なし） |
| 3 | `hard_hard` | ハードマスク | N/A | 上限を構造的に保証、trim は理論上不要 |
| 4 | `hard_soft` | ソフトブースト | — | ハードマスクなし、素出力で評価 |
| 5 | `hard_soft_trim` | ソフトブースト | ✓ | #4 の出力を再利用（要約再生成なし） |
| 6 | `hard_force` | 文末 EOS 強制 | N/A | 文末到達時に EOS を強制、trim は理論上不要 |
| 7 | `closing_inject` | 締め句注入 | — | 可視テキストの素出力で評価 |
| 8 | `closing_inject_trim` | 締め句注入 | ✓ | #7 の出力を再利用（要約再生成なし） |

`*_trim` 条件は対応する base 条件の生成テキストを再利用し、`trim_to_range` 適用後のテキストで品質判定のみ再実行します（要約の再生成なし）。

### 合格判定

`evaluate_output(eval_text, lower, upper, *, use_trim)` が統一的に合否を計算します。

- **`use_trim=False`（trim N/A・なし条件）**: raw テキストが `[lower, upper]` 範囲内 ＋ 文末記号 ＋ 自然さ OK
- **`use_trim=True`（`*_trim` 条件）**: `trim_to_range` 成功 ＋ trim 後テキストで文末記号・自然さ再評価

自然さ判定は Qwen3.5-9B 自身に文末の文法的完結性を問い合わせます（`はい` / `いいえ` で判定、曖昧回答はレポートにオレンジでフラグ表示）。

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

HuggingFace のゲートモデルへのアクセスには `.env` に `HF_TOKEN` が必要です。

## 実行

```bash
# 全条件・全タスクを K=10 まで実行（既存分はスキップ）
uv run python run_experiment.py --k 10

# 特定条件・タスクのみ実行
uv run python run_experiment.py --condition baseline --condition hard_soft --task 1 --task 2

# K を増やす（不足分のみ生成し、既存結果を再利用）
uv run python run_experiment.py --k 16

# 生成せず、ストアから HTML レポートのみ再計算
uv run python run_experiment.py --report-only

# ストアを削除してゼロから実行（既存結果がすべて消えます）
uv run python run_experiment.py --reset
```

初回実行時にモデルのダウンロードとトークン別文字数テーブルの構築が走ります。テーブルは `~/.cache/char_budget/` にキャッシュされ、2回目以降はスキップされます。

結果はサンプル単位で `results.jsonl` に追記されます。クラッシュしても再実行で続きから再開できます。

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
condition              n         p_raw        p_accept  trim%   E[N]  P(N≤2)  P(N≤3)    MAE  p_punct   p_nat
baseline              40  0.25[0.13,0.42] 0.20[0.10,0.36]     0%   5.00    0.36    0.49   64.2     0.85    0.94
baseline_trim         40  0.25[0.13,0.42] 0.35[0.22,0.51]    30%   2.86    0.63    0.76   38.7     0.85    0.94
hard_hard             40  1.00[0.91,1.00] 0.50[0.35,0.65]     0%   2.00    0.75    0.88    8.1     0.50    1.00
hard_soft             40  0.30[0.18,0.46] 0.25[0.14,0.40]     0%   4.00    0.44    0.58   57.3     0.90    0.97
hard_soft_trim        40  0.30[0.18,0.46] 0.43[0.28,0.58]    28%   2.35    0.68    0.82   33.8     0.90    0.97
hard_force            40  0.65[0.50,0.78] 0.58[0.42,0.72]     0%   1.73    0.81    0.93   18.5     0.95    0.97
closing_inject        40  0.40[0.26,0.56] 0.35[0.22,0.51]     0%   2.86    0.63    0.76   42.1     0.88    0.96
closing_inject_trim   40  0.40[0.26,0.56] 0.48[0.33,0.63]    22%   2.09    0.73    0.87   29.6     0.88    0.96
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
| `p_punct` | 配信テキストが文末記号で終わった割合 |
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
