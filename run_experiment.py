# -*- coding: utf-8 -*-
"""
文字数制約付き要約 End-to-End（Qwen3.5-9B / MacBook M3 / MPS）

役割分担:
  - 下限 L : 生成時に EOS を禁止して止まらせない（後処理では作れないため必須）
  - 上限 U : nonspace_len によるハードマスクで構造保証 ＋ ソフト EOS ブーストで自然収束
  - 後段   : 文境界トリムで上限を最終保証。範囲外なら再生成。

比較できる条件:
  baseline   : 介入なし（プロンプトのみ＝現状の素の best-of-N 相当）
  hard_only  : 上限ハードマスク ＋ 下限 EOS 禁止（ソフトブーストなし）
  hard_soft  : hard_only ＋ ソフト EOS ブースト

文字数定義: 空白を除く Unicode コードポイント数（全角/半角の区別なし・記号は数える）。

注意(M3):
  Qwen3.5-9B fp16 は約18GBのunified memoryを要求します。メモリが苦しい場合は
  まず MODEL_ID を "Qwen/Qwen3-1.7B" などに変えてハーネスを検証してください。
"""

import os
import re
import time
import hashlib
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

# ----------------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3.5-9B"     # 検証時は "Qwen/Qwen3-1.7B" 等に差し替え推奨
CACHE_DIR = os.path.expanduser("~/.cache/char_budget")
os.makedirs(CACHE_DIR, exist_ok=True)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = pick_device()
DTYPE = torch.float16 if DEVICE.type in ("mps", "cuda") else torch.float32


# ----------------------------------------------------------------------------
# 文字数カウント（外側の検証と LogitsProcessor 内で同一ロジックを共有）
# ----------------------------------------------------------------------------
def count_chars(text: str) -> int:
    """空白を除いた文字数。str.isspace() は全角スペース U+3000 も True。"""
    return sum(1 for c in text if not c.isspace())


# ----------------------------------------------------------------------------
# 文境界トリム（上限の最終保証・下限を尊重）
# ----------------------------------------------------------------------------
def trim_to_range(text: str, lower: int, upper: int):
    """範囲内に収まる最大の文集合に切り詰める。収まらなければ None（=要再生成）。"""
    # 末尾の「。」を保持したまま文単位に分割
    parts = re.findall(r"[^。]*。|[^。]+$", text)
    out = ""
    for p in parts:
        if count_chars(out + p) > upper:
            break
        out += p
    out = out or text  # 1文目すら上限超過なら全文を候補に（→下で弾かれる）
    n = count_chars(out)
    return out if lower <= n <= upper else None


# ----------------------------------------------------------------------------
# トークン別 非空白文字数テーブル（保守的上限）。重いのでディスクキャッシュ。
# ----------------------------------------------------------------------------
def build_nonspace_len(tokenizer) -> torch.Tensor:
    key = hashlib.md5(MODEL_ID.encode()).hexdigest()[:12]
    path = os.path.join(CACHE_DIR, f"nonspace_len_{key}.pt")
    if os.path.exists(path):
        return torch.load(path)

    vocab_size = len(tokenizer)
    lens = torch.zeros(vocab_size, dtype=torch.int32)
    for tid in range(vocab_size):
        s = tokenizer.decode([tid])
        lens[tid] = sum(1 for c in s if not c.isspace())

    # 特殊/ストップトークンは必ず 0 にする。
    # さもないと "<|im_end|>" 等が長文字列として数えられ、残予算が小さいとき
    # ストップトークンまでハードマスクで禁止され構造保証が壊れる。
    for sid in tokenizer.all_special_ids:
        if 0 <= sid < vocab_size:
            lens[sid] = 0

    torch.save(lens, path)
    return lens


# ----------------------------------------------------------------------------
# 文字数レンジ用 LogitsProcessor（各フラグで挙動を切替）
# ----------------------------------------------------------------------------
class CharRangeProcessor(LogitsProcessor):
    def __init__(self, lower, upper, prompt_len, tokenizer, nonspace_len,
                 stop_ids, *, use_hard_mask=True, use_soft_boost=True,
                 use_lower_floor=True, soft_frac=0.7, boost=4.0):
        self.lower = lower
        self.upper = upper
        self.prompt_len = prompt_len
        self.tok = tokenizer
        self.nonspace_len = nonspace_len
        self.stop_ids = list(stop_ids)
        self.use_hard_mask = use_hard_mask
        self.use_soft_boost = use_soft_boost
        self.use_lower_floor = use_lower_floor
        # ソフトブーストは [lower, upper] 帯の後半から効かせる
        self.soft_start = lower + soft_frac * (upper - lower)
        self.boost = boost

    def __call__(self, input_ids, scores):
        nslen = self.nonspace_len.to(scores.device)
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:]
            # 毎ステップ生成部を丸ごとデコードして数える（byte-fallback でも正確）
            text = self.tok.decode(gen, skip_special_tokens=True)
            used = sum(1 for c in text if not c.isspace())
            remaining = self.upper - used

            # 下限: L 未満ではストップを禁止して止まらせない
            if self.use_lower_floor and used < self.lower:
                for sid in self.stop_ids:
                    scores[b, sid] = float("-inf")
            # ソフト: 帯後半でストップのロジットを押し上げ自然に締めさせる
            elif self.use_soft_boost and used >= self.soft_start:
                for sid in self.stop_ids:
                    scores[b, sid] += self.boost

            # 上限: オーバーしうるトークンを禁止（stop は nslen=0 なので生き残る）
            if self.use_hard_mask:
                vocab_size = scores.shape[1]
                if nslen.shape[0] < vocab_size:
                    nslen = torch.nn.functional.pad(nslen, (0, vocab_size - nslen.shape[0]))
                scores[b, nslen[:vocab_size] > remaining] = float("-inf")
        return scores


# ----------------------------------------------------------------------------
# モデル読み込み
# ----------------------------------------------------------------------------
print(f"[load] device={DEVICE} dtype={DTYPE} model={MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
).to(DEVICE)
model.eval()

NONSPACE_LEN = build_nonspace_len(tokenizer)

# ストップトークン集合（eos と <|im_end|> を両方カバー）
STOP_IDS = set()
if tokenizer.eos_token_id is not None:
    STOP_IDS.add(tokenizer.eos_token_id)
_im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
if isinstance(_im_end, int) and _im_end >= 0:
    STOP_IDS.add(_im_end)
STOP_IDS = sorted(STOP_IDS)


def build_prompt(text: str, lower: int, upper: int) -> str:
    messages = [
        {"role": "system", "content": "あなたは優秀な要約者です。"},
        {"role": "user", "content":
            f"次の文章を{lower}〜{upper}文字（空白を除く）で要約してください。"
            f"結論を先に書き、指定文字数に収めてください。\n\n{text}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,  # <think> を抑止（カウント破壊を防ぐ）
    )


@dataclass
class GenResult:
    text: str
    raw_chars: int
    raw_in_range: bool
    trimmed: str
    trimmed_chars: int
    accepted: bool
    needed_trim: bool


@dataclass
class SampleRecord:
    cond: str
    task_i: int
    lower: int
    upper: int
    k: int
    result: GenResult
    elapsed: float


@torch.no_grad()
def generate_one(text, lower, upper, *, hard, soft, floor,
                 temperature=0.7, top_p=0.8) -> GenResult:
    prompt = build_prompt(text, lower, upper)
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = enc.input_ids.shape[1]

    processors = []
    if hard or soft or floor:
        processors.append(CharRangeProcessor(
            lower, upper, prompt_len, tokenizer, NONSPACE_LEN, STOP_IDS,
            use_hard_mask=hard, use_soft_boost=soft, use_lower_floor=floor,
        ))

    out = model.generate(
        **enc,
        max_new_tokens=upper + 64,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        logits_processor=processors if processors else None,
        eos_token_id=STOP_IDS if STOP_IDS else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, prompt_len:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    raw_n = count_chars(raw)
    raw_in_range = lower <= raw_n <= upper

    trimmed = trim_to_range(raw, lower, upper)
    accepted = trimmed is not None
    trimmed_text = trimmed if accepted else raw
    return GenResult(
        text=raw,
        raw_chars=raw_n,
        raw_in_range=raw_in_range,
        trimmed=trimmed_text,
        trimmed_chars=count_chars(trimmed_text),
        accepted=accepted,
        needed_trim=accepted and (raw_n > upper),
    )


# ----------------------------------------------------------------------------
# 実験ハーネス: 条件ごとに K サンプルを取り、合格率と必要 N を比較
# ----------------------------------------------------------------------------
CONDITIONS = {
    "baseline":  dict(hard=False, soft=False, floor=False),
    "hard_only": dict(hard=True,  soft=False, floor=True),
    "hard_soft": dict(hard=True,  soft=True,  floor=True),
}


def expected_n(p):
    return float("inf") if p == 0 else 1.0 / p


def best_of_n_success(p, n):
    return 1.0 - (1.0 - p) ** n


def run_experiment(tasks, K=8):
    """tasks: List[Tuple[text, lower, upper]]"""
    n_cond = len(CONDITIONS)
    n_tasks = len(tasks)
    total_per_cond = n_tasks * K
    grand_total = n_cond * total_per_cond
    grand_done = 0

    summary = {}
    records: list[SampleRecord] = []

    for cond_i, (cond, flags) in enumerate(CONDITIONS.items(), 1):
        n_accept = 0       # トリム後に範囲内（=最終合格）
        n_raw_in = 0       # トリム前から範囲内（=自然収束）
        n_trim = 0         # トリムが発火
        n_total = 0
        char_err = 0.0     # 中心からのズレ（参考）
        t0 = time.time()
        print(f"\n[{cond_i}/{n_cond}] condition={cond}  ({total_per_cond} samples)")
        for task_i, (text, lower, upper) in enumerate(tasks, 1):
            center = (lower + upper) / 2
            for k in range(1, K + 1):
                t_s = time.time()
                r = generate_one(text, lower, upper, **flags)
                elapsed = time.time() - t_s
                n_total += 1
                grand_done += 1
                n_accept += int(r.accepted)
                n_raw_in += int(r.raw_in_range)
                n_trim += int(r.needed_trim)
                char_err += abs(r.trimmed_chars - center)
                records.append(SampleRecord(cond, task_i, lower, upper, k, r, elapsed))
                status = "OK" if r.accepted else "NG"
                trim_flag = " trim" if r.needed_trim else "     "
                print(
                    f"  task{task_i} k={k}/{K}  chars={r.trimmed_chars:>4} [{lower},{upper}]"
                    f"  {status}{trim_flag}  {elapsed:.1f}s"
                    f"  [overall {grand_done}/{grand_total}]"
                )
        dt = time.time() - t0
        p_accept = n_accept / n_total
        p_raw = n_raw_in / n_total
        summary[cond] = dict(
            p_accept=p_accept,
            p_raw=p_raw,
            trim_rate=n_trim / n_total,
            E_N=expected_n(p_accept),
            mae_center=char_err / n_total,
            n=n_total,
            sec=dt,
        )
    return summary, records


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_html_report(summary, records: list[SampleRecord], path: str):
    cond_order = list(CONDITIONS.keys())
    # task keys
    task_keys = sorted({(r.task_i, r.lower, r.upper) for r in records})

    rows_html = []
    for rec in records:
        r = rec.result
        ok_cls = "ok" if r.accepted else "ng"
        trim_badge = '<span class="badge trim">trim</span>' if r.needed_trim else ""
        raw_diff = "" if r.raw_in_range else f'<span class="raw-out">(raw {r.raw_chars})</span>'
        rows_html.append(f"""
      <tr class="{ok_cls}">
        <td>{rec.cond}</td>
        <td>task{rec.task_i} [{rec.lower},{rec.upper}]</td>
        <td>{rec.k}</td>
        <td class="chars">{rec.result.trimmed_chars} {raw_diff}</td>
        <td><span class="badge {ok_cls}">{"OK" if r.accepted else "NG"}</span>{trim_badge}</td>
        <td class="elapsed">{rec.elapsed:.1f}s</td>
        <td class="output">{_esc(r.trimmed)}</td>
        <td class="output raw">{_esc(r.text)}</td>
      </tr>""")

    summary_rows = []
    for cond, s in summary.items():
        summary_rows.append(f"""
      <tr>
        <td>{cond}</td>
        <td>{s['p_raw']:.2f}</td>
        <td>{s['p_accept']:.2f}</td>
        <td>{s['trim_rate']*100:.0f}%</td>
        <td>{s['E_N']:.2f}</td>
        <td>{best_of_n_success(s['p_accept'], 2):.2f}</td>
        <td>{best_of_n_success(s['p_accept'], 3):.2f}</td>
        <td>{s['mae_center']:.1f}</td>
        <td>{s['sec']:.0f}s</td>
      </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>logit masking experiment</title>
<style>
  body {{ font-family: sans-serif; font-size: 13px; margin: 20px; }}
  h2 {{ margin-top: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }}
  th {{ background: #f0f0f0; position: sticky; top: 0; }}
  tr.ok {{ background: #f6fff6; }}
  tr.ng {{ background: #fff4f4; }}
  .chars {{ text-align: right; white-space: nowrap; }}
  .elapsed {{ text-align: right; white-space: nowrap; }}
  .badge {{ border-radius: 3px; padding: 1px 5px; font-size: 11px; font-weight: bold; }}
  .badge.ok {{ background: #4caf50; color: white; }}
  .badge.ng {{ background: #f44336; color: white; }}
  .badge.trim {{ background: #ff9800; color: white; margin-left: 4px; }}
  .raw-out {{ color: #888; font-size: 11px; }}
  .output {{ max-width: 420px; line-height: 1.5; word-break: break-all; }}
  .raw {{ color: #555; font-size: 11px; }}
  details summary {{ cursor: pointer; color: #555; font-size: 11px; }}
</style>
</head>
<body>
<h1>Logit Masking Experiment</h1>

<h2>Summary</h2>
<table>
  <thead><tr>
    <th>condition</th><th>p_raw</th><th>p_accept</th><th>trim%</th>
    <th>E[N]</th><th>P(N≤2)</th><th>P(N≤3)</th><th>MAE</th><th>time</th>
  </tr></thead>
  <tbody>{"".join(summary_rows)}</tbody>
</table>
<p style="font-size:11px;color:#555">
  p_raw: トリム前から範囲内 / p_accept: トリム後に範囲内 / trim%: 上限トリム発火率 /
  E[N]: 期待生成回数 / MAE: 帯中心からの平均絶対文字数ズレ
</p>

<h2>Samples</h2>
<table>
  <thead><tr>
    <th>condition</th><th>task</th><th>k</th><th>chars</th>
    <th>status</th><th>time</th>
    <th>trimmed output</th><th>raw output</th>
  </tr></thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[report] {path}")


def print_summary(summary):
    print("\n" + "=" * 78)
    print(f"{'condition':<11} {'p_raw':>7} {'p_accept':>9} {'trim%':>7} "
          f"{'E[N]':>6} {'P(N=2)':>7} {'P(N=3)':>7} {'MAE':>6}")
    print("-" * 78)
    for cond, s in summary.items():
        print(f"{cond:<11} {s['p_raw']:>7.2f} {s['p_accept']:>9.2f} "
              f"{s['trim_rate']*100:>6.0f}% {s['E_N']:>6.2f} "
              f"{best_of_n_success(s['p_accept'], 2):>7.2f} "
              f"{best_of_n_success(s['p_accept'], 3):>7.2f} "
              f"{s['mae_center']:>6.1f}")
    print("=" * 78)
    print("p_raw    : トリム前から範囲内に着地した割合（自然収束＝ソフトの効果はここ）")
    print("p_accept : トリム後に範囲内（最終合格率）")
    print("trim%    : 上限トリムが発火した割合")
    print("E[N]     : 期待生成回数 = 1/p_accept   P(N=k): k回以内に1回成功する確率")
    print("MAE      : 帯中心からの平均絶対文字数ズレ（参考）")


# ----------------------------------------------------------------------------
# 実行例
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE = (
        "近年の機械学習システムでは、大規模言語モデルを用いた要約や情報抽出が"
        "実運用に投入されつつある。GPT系やLLaMA系をはじめとする多様なアーキテクチャが"
        "公開され、日本語処理においてもQwenやMistralなど多言語対応モデルが"
        "実用水準に達したことで、企業内文書の自動要約やカスタマーサポートへの応用が"
        "急速に広がっている。しかし実運用に際しては、出力の文字数を所定の範囲に"
        "収めることが依然として難しい課題として残っている。"
        "たとえば広告コピーや法的な注意書き、SNS投稿など、厳密な文字数制約が"
        "課されるユースケースでは、モデルが自由に生成した結果が上限を超えたり、"
        "下限を下回ったりすることが頻繁に発生する。"
        "現在の主流なアプローチは、生成後に外部でカウントして範囲外なら再生成する"
        "確率的な運用であり、最悪の場合は数十回の試行が必要になることもある。"
        "この非効率さはレイテンシとコストの両面でシステム設計上の足かせとなっている。"
        "本稿では、生成中にリアルタイムでロジットを操作することで、"
        "一回の生成で文字数制約を満たす確率を高める手法を検討する。"
        "具体的には三つの機構を組み合わせる。"
        "第一に、ロジットマスキングによる上限の構造保証である。"
        "各ステップで残余文字数を超えるトークンを確率ゼロに落とすことで、"
        "物理的に上限を超えられない生成過程を実現する。"
        "第二に、下限を割らないためのストップトークン抑制である。"
        "消費済み文字数が下限に到達するまでEOSや句点などの終端トークンを"
        "マスクすることで、短すぎる出力を構造的に防ぐ。"
        "第三に、自然な収束を促すソフトなブーストである。"
        "消費済み文字数がバンド幅の一定割合（デフォルト七割）を超えた時点で"
        "終端トークンのロジットに正のバイアスを加え、目標範囲内での自然な文末を"
        "誘導する。さらに後段の文境界トリムを最終保証として置くことで、"
        "上限超過を構造的に排除しつつ、品質の低下を抑える設計を示す。"
        "実験ではQwen3.5-9Bを用い、複数の文字数バンドに対してベースライン、"
        "ハードマスクのみ、ハードマスクとソフトブーストの三条件を比較する。"
        "評価指標としては、バンド内収録率、期待生成回数、バンド中心からの平均絶対誤差を用いる。"
        "結果として、ハードマスクとソフトブーストを組み合わせた条件では、"
        "ベースラインと比較してバンド内収録率が大幅に改善されることが確認された。"
        "本手法は追加の学習を要さず、推論時のロジット操作のみで実現できる点で実用的である。"
    )
    # SAMPLE: 1028文字（非空白）
    # (text, lower, upper)。下限は上限の0.7倍を想定（例: 200→140）
    tasks = [
        (SAMPLE, 280, 400),
        (SAMPLE, 140, 200),
        (SAMPLE, 70, 100),
    ]

    summary, records = run_experiment(tasks, K=8)
    print_summary(summary)
    write_html_report(summary, records, "experiment_report.html")