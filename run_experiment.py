# -*- coding: utf-8 -*-
"""
文字数制約付き要約 End-to-End（Qwen3.5-9B / MacBook M3 / MPS）

役割分担:
  - 下限 L : 生成時に EOS を禁止して止まらせない（後処理では作れないため必須）
  - 上限 U : nonspace_len によるハードマスクで構造保証 ＋ ソフト EOS ブーストで自然収束
  - 後段   : 文境界トリムで上限を最終保証。範囲外なら再生成。

比較できる条件:
  baseline    : 介入なし（プロンプトのみ＝現状の素の best-of-N 相当）
  hard_only   : 上限ハードマスク ＋ 下限 EOS 禁止（ソフトブーストなし）
  hard_soft   : hard_only ＋ ソフト EOS ブースト
  floor_sent  : 下限 EOS 禁止 ＋ 弱いソフトブースト（下限直後から）＋ 文末強制 EOS（ハードマスクなし）

文字数定義: 空白を除く Unicode コードポイント数（全角/半角の区別なし・記号は数える）。

注意(M3):
  Qwen3.5-9B fp16 は約18GBのunified memoryを要求します。メモリが苦しい場合は
  まず MODEL_ID を "Qwen/Qwen3-1.7B" などに変えてハーネスを検証してください。
"""

import json
import math
import os
import re
import time
import hashlib
import datetime
import urllib.request
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed as transformers_set_seed,
)

# ----------------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3.5-9B"     # 検証時は "Qwen/Qwen3-1.7B" 等に差し替え推奨
CACHE_DIR = os.path.expanduser("~/.cache/char_budget")
os.makedirs(CACHE_DIR, exist_ok=True)

SENT_END_CHARS = frozenset("。！？")
QUALITY_END_CHARS = frozenset("。！？…」")

# 再現性制御
BASE_SEED = 1234  # --seed で上書き可。条件間でペア化（task_i, k のみに依存）。
# 注意: MPS は一部カーネルが非決定的なため再現性はベストエフォート。

# 永続結果ストア
STORE_PATH = "results.jsonl"
SCHEMA_VERSION = 1

# 締め句注入方式 (closing_inject)
CLOSING_TEXT = "最後に1文でまとめます。"
CLOSING_TEXT_SHORT = "1文でまとめます。"  # lower < TRIGGER_MARGIN の短バンド用
TRIGGER_MARGIN = 55  # 下限から何文字手前で注入トリガ窓を開けるか（平均1文≈52字分を確保）


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


def derive_seed(base_seed: int, task_i: int, k: int) -> int:
    """条件非依存の決定論シード（全条件でペア化）。
    base_seed は実験全体のシード、task_i と k だけが変化する。"""
    return (base_seed * 1_000_003 + task_i * 10_007 + k) % (2 ** 31)


def strip_closing(text: str, closing: str = CLOSING_TEXT) -> str:
    """注入した締め句を可視テキストから除去。

    位置を「句点直後」または「文頭」に限定し、モデルが本文中に書いた
    締め句と同一文字列を巻き込まないようにする。
    """
    text = text.replace("。" + closing, "。")
    if text.startswith(closing):
        text = text[len(closing):]
    return text


def visible_text(raw: str, closing: str = CLOSING_TEXT) -> str:
    return strip_closing(raw, closing).strip()


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


def ends_with_sentence_punct(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in QUALITY_END_CHARS


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
                 use_lower_floor=True, soft_frac=0.7, boost=4.0,
                 use_sent_end_force=False):
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
        self.use_sent_end_force = use_sent_end_force

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
            else:
                # 文末強制: 範囲内かつテキストが文末文字で終わっていればEOSを強制
                if self.use_sent_end_force and text and text[-1] in SENT_END_CHARS:
                    for sid in self.stop_ids:
                        scores[b, sid] = 1e9
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

            # 安全弁: 全トークンが -inf になった場合は stop トークンを解放して無限ループを防ぐ
            if scores[b].max() == float("-inf"):
                for sid in self.stop_ids:
                    scores[b, sid] = 0.0
        return scores


# ----------------------------------------------------------------------------
# 締め句注入用 LogitsProcessor / StoppingCriteria
# ----------------------------------------------------------------------------
class ClosingInjectProcessor(LogitsProcessor):
    """締め句注入方式専用。下限EOS禁止のみを担当（上限介入なし）。

    used は decode 後に strip_closing で締め句を剥がした「可視文字数」で数える。
    注入済み (injected=True) になったら EOS 禁止を解除する。
    """

    def __init__(self, lower, upper, prompt_len, tokenizer, stop_ids,
                 closing_text=CLOSING_TEXT):
        self.lower = lower
        self.upper = upper
        self.prompt_len = prompt_len
        self.tok = tokenizer
        self.stop_ids = list(stop_ids)
        self.closing_text = closing_text
        self.injected = False

    def set_injected(self, flag: bool):
        self.injected = flag

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:]
            text = self.tok.decode(gen, skip_special_tokens=True)
            vis = strip_closing(text, self.closing_text)
            used = count_chars(vis)

            if not self.injected and used < self.lower:
                for sid in self.stop_ids:
                    scores[b, sid] = float("-inf")

            if scores[b].max() == float("-inf"):
                for sid in self.stop_ids:
                    scores[b, sid] = 0.0
        return scores


class ClosingInjectStopping(StoppingCriteria):
    """締め句注入のトリガ。直前が句点 & 可視 used が [lower - margin, lower] の窓内で停止。"""

    def __init__(self, lower, upper, trigger_margin, prompt_len, tokenizer,
                 closing_text=CLOSING_TEXT):
        self.lower = lower
        self.upper = upper
        self.prompt_len = prompt_len
        self.tok = tokenizer
        self.closing_text = closing_text
        self.trigger_lo = lower - trigger_margin
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.triggered:
            return False
        gen = input_ids[0, self.prompt_len:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        vis = strip_closing(text, self.closing_text).rstrip()
        if not vis or vis[-1] not in SENT_END_CHARS:
            return False
        used = count_chars(vis)
        if self.trigger_lo <= used <= self.lower:
            self.triggered = True
            return True
        return False


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

# 締め句のトークン化が想定どおりかを起動時に1度確認（境界トークン問題の検知）
_closing_ids = tokenizer(CLOSING_TEXT, add_special_tokens=False).input_ids
_closing_decoded = tokenizer.decode(_closing_ids)
print(f"[closing] text={CLOSING_TEXT!r} ids={_closing_ids} decoded={_closing_decoded!r}"
      f" {'OK' if _closing_decoded == CLOSING_TEXT else 'MISMATCH!'}")
_closing_short_ids = tokenizer(CLOSING_TEXT_SHORT, add_special_tokens=False).input_ids
_closing_short_decoded = tokenizer.decode(_closing_short_ids)
print(f"[closing_short] text={CLOSING_TEXT_SHORT!r} ids={_closing_short_ids} decoded={_closing_short_decoded!r}"
      f" {'OK' if _closing_short_decoded == CLOSING_TEXT_SHORT else 'MISMATCH!'}")


JUDGE_MODEL = "gpt-oss:20b"
OLLAMA_URL = "http://localhost:11434/api/chat"


def is_natural_ending(text: str) -> tuple[bool, str]:
    """文末が自然に終わっているか gpt-oss:20b (Ollama) に判定させる。
    戻り値: (判定結果, モデルの生の出力)
    - True: 自然な終わり
    - False: 不自然（途中切れ等）
    - Trueフォールバック: モデルが「はい」「いいえ」以外を返した場合も True だが nat_raw で確認可能
    """
    messages = [
        {"role": "system", "content": "あなたは日本語文章の品質を評価する専門家です。"},
        {"role": "user", "content": (
            "以下の文章の末尾は文法的に完結していますか？\n"
            "内容の充実度や情報量は問いません。文末の語尾・句読点・助詞・動詞の活用形だけを見て、\n"
            "文が途中の単語で切れている場合（例：「提案す」「改善さ」「効果的な」など）は「いいえ」、\n"
            "句点・感嘆符・疑問符・引用符で終わっているか、名詞・動詞・形容詞として文法的に完結している場合は「はい」と一語で答えてください。\n\n"
            f"文章:\n「{text}」\n\n"
            "回答（はい／いいえ）:"
        )},
    ]
    payload = json.dumps({
        "model": JUDGE_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_predict": 512, "temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    answer = body["message"]["content"].strip()
    if "いいえ" in answer:
        return False, answer
    if "はい" in answer:
        return True, answer
    return True, answer  # 解析不能: 保守的に合格、nat_raw で人間が確認できる


def build_prompt(text: str, lower: int, upper: int) -> str:
    messages = [
        {"role": "system", "content": "あなたは優秀な要約者です。"},
        {"role": "user", "content":
            f"次の文章を{lower}〜{upper}文字（空白を除く）で要約してください。"
            f"結論を先に書き、指定文字数に収めてください。"
            f"出力には文字数（例：「（50文字）」など）を含めないでください。\n\n{text}"},
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
    ends_with_punct: bool
    is_natural: bool | None       # None if punct check failed (natural check skipped)
    natural_raw: str | None       # raw model output from is_natural_ending; None if not called
    injected: bool | None = None  # closing_inject 条件のみ True/False、他条件は None
    visible: str | None = None    # 締め句除去後の可視テキスト（closing_inject のみ）


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
                 soft_frac=0.7, boost=4.0, sent_end_force=False,
                 temperature=0.7, top_p=0.8, seed: int | None = None) -> GenResult:
    if seed is not None:
        transformers_set_seed(seed)
    prompt = build_prompt(text, lower, upper)
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = enc.input_ids.shape[1]

    processors = []
    if hard or soft or floor or sent_end_force:
        processors.append(CharRangeProcessor(
            lower, upper, prompt_len, tokenizer, NONSPACE_LEN, STOP_IDS,
            use_hard_mask=hard, use_soft_boost=soft, use_lower_floor=floor,
            soft_frac=soft_frac, boost=boost, use_sent_end_force=sent_end_force,
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

    ewp = ends_with_sentence_punct(raw)
    if ewp:
        is_nat, nat_raw = is_natural_ending(raw)
    else:
        is_nat, nat_raw = None, None

    accepted = (trimmed is not None) and ewp and (is_nat is True)
    trimmed_text = trimmed if trimmed is not None else raw
    return GenResult(
        text=raw,
        raw_chars=raw_n,
        raw_in_range=raw_in_range,
        trimmed=trimmed_text,
        trimmed_chars=count_chars(trimmed_text),
        accepted=accepted,
        needed_trim=(trimmed is not None) and (raw_n > upper),
        ends_with_punct=ewp,
        is_natural=is_nat,
        natural_raw=nat_raw,
    )


@torch.no_grad()
def generate_one_with_closing(text, lower, upper, *,
                              trigger_margin=TRIGGER_MARGIN,
                              closing=CLOSING_TEXT,
                              temperature=0.7, top_p=0.8,
                              seed: int | None = None) -> GenResult:
    """締め句注入方式の2段階生成。

    Phase 1: 下限EOS禁止のみ + StoppingCriteria。窓内の句点で停止
    Phase 2: input_ids に締め句を append し、フロア解除して続きを生成

    トリガが一度も発火しなければ Phase 1 だけで完了（注入なし）。
    冒頭で1回シードすれば2段の全生成が決定論的になる。
    """
    if seed is not None:
        transformers_set_seed(seed)
    prompt = build_prompt(text, lower, upper)
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = enc.input_ids.shape[1]

    injected = False
    # lower が trigger_margin より小さい場合、最初から締め句を付加して Phase 2 のみ実行
    if lower < trigger_margin:
        closing = CLOSING_TEXT_SHORT
    proc = ClosingInjectProcessor(
        lower, upper, prompt_len, tokenizer, STOP_IDS, closing_text=closing,
    )

    if lower < trigger_margin:
        closing_ids = tokenizer(
            closing, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(DEVICE)
        new_input = torch.cat([enc.input_ids, closing_ids], dim=1)
        # set_injected しない: Phase 2 でも下限 EOS 禁止を維持して lower を保証
        attention_mask = torch.ones_like(new_input)
        remaining_chars = upper
        phase2_max_new = max(64, remaining_chars * 2 + 32)
        out2 = model.generate(
            input_ids=new_input,
            attention_mask=attention_mask,
            max_new_tokens=phase2_max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            logits_processor=[proc],
            eos_token_id=STOP_IDS if STOP_IDS else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out2[0, prompt_len:]
        injected = True
    else:
        stop = ClosingInjectStopping(
            lower, upper, trigger_margin, prompt_len, tokenizer, closing_text=closing,
        )
        out1 = model.generate(
            **enc,
            max_new_tokens=upper + 64,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            logits_processor=[proc],
            stopping_criteria=StoppingCriteriaList([stop]),
            eos_token_id=STOP_IDS if STOP_IDS else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        if stop.triggered:
            closing_ids = tokenizer(
                closing, add_special_tokens=False, return_tensors="pt",
            ).input_ids.to(DEVICE)
            new_input = torch.cat([out1, closing_ids], dim=1)
            proc.set_injected(True)
            attention_mask = torch.ones_like(new_input)
            # 残予算ベース（vis ベースで残り文字数を測り、安全に余裕を持たせる）
            vis_so_far = strip_closing(
                tokenizer.decode(out1[0, prompt_len:], skip_special_tokens=True),
                closing,
            )
            remaining_chars = max(0, upper - count_chars(vis_so_far))
            # 日本語は1トークン~1〜2文字。余裕を取って *2 + バッファ
            phase2_max_new = max(64, remaining_chars * 2 + 32)
            out2 = model.generate(
                input_ids=new_input,
                attention_mask=attention_mask,
                max_new_tokens=phase2_max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                logits_processor=[proc],
                eos_token_id=STOP_IDS if STOP_IDS else None,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out2[0, prompt_len:]
            injected = True
        else:
            gen_ids = out1[0, prompt_len:]

    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    vis = visible_text(raw, closing)

    vis_n = count_chars(vis)
    vis_in_range = lower <= vis_n <= upper
    trimmed = trim_to_range(vis, lower, upper)
    ewp = ends_with_sentence_punct(vis)
    if ewp:
        is_nat, nat_raw = is_natural_ending(vis)
    else:
        is_nat, nat_raw = None, None

    accepted = (trimmed is not None) and ewp and (is_nat is True)
    trimmed_text = trimmed if trimmed is not None else vis
    return GenResult(
        text=raw,
        raw_chars=vis_n,
        raw_in_range=vis_in_range,
        trimmed=trimmed_text,
        trimmed_chars=count_chars(trimmed_text),
        accepted=accepted,
        needed_trim=(trimmed is not None) and (vis_n > upper),
        ends_with_punct=ewp,
        is_natural=is_nat,
        natural_raw=nat_raw,
        injected=injected,
        visible=vis,
    )


# ----------------------------------------------------------------------------
# 実験ハーネス: 条件ごとに K サンプルを取り、合格率と必要 N を比較
# ----------------------------------------------------------------------------
CONDITIONS = {
    "baseline":   dict(hard=False, soft=False, floor=False),
    "hard_only":  dict(hard=True,  soft=False, floor=True),
    "hard_soft":  dict(hard=True,  soft=True,  floor=True),
    "floor_sent": dict(hard=False, soft=True,  floor=True,
                       soft_frac=0.0, boost=1.5, sent_end_force=True),
    "closing_inject": {"__mode__": "closing",
                       "trigger_margin": TRIGGER_MARGIN, "closing": CLOSING_TEXT},
}


# ----------------------------------------------------------------------------
# 永続累積ストア (results.jsonl) — サンプル単位で追記、自動削除なし
# ----------------------------------------------------------------------------

def _task_hash(text: str) -> str:
    """原文の MD5 先頭12桁。入力が変わると旧レコードと別物として扱える。"""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def record_key(row: dict) -> tuple:
    """dedupe 用キー。同じ条件・タスク・k・シードなら同一サンプル。"""
    return (
        row.get("model_id", ""),
        row.get("temperature", 0.0),
        row.get("top_p", 0.0),
        row.get("cond", ""),
        row.get("task_i", 0),
        row.get("k", 0),
        row.get("seed", 0),
    )


def _row_to_sample_record(row: dict) -> "SampleRecord":
    r = row["result"]
    gen = GenResult(
        text=r["text"],
        raw_chars=r["raw_chars"],
        raw_in_range=r["raw_in_range"],
        trimmed=r["trimmed"],
        trimmed_chars=r["trimmed_chars"],
        accepted=r["accepted"],
        needed_trim=r["needed_trim"],
        ends_with_punct=r["ends_with_punct"],
        is_natural=r["is_natural"],
        natural_raw=r["natural_raw"],
        injected=r.get("injected"),
        visible=r.get("visible"),
    )
    return SampleRecord(
        cond=row["cond"],
        task_i=row["task_i"],
        lower=row["lower"],
        upper=row["upper"],
        k=row["k"],
        result=gen,
        elapsed=row["elapsed"],
    )


def load_store(path: str = STORE_PATH) -> tuple[list["SampleRecord"], set]:
    """ストアを全行ロード。(records, keys) を返す。"""
    records: list[SampleRecord] = []
    keys: set = set()
    if not os.path.exists(path):
        return records, keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(record_key(row))
            records.append(_row_to_sample_record(row))
    print(f"[store] loaded {len(records)} records from {path}")
    return records, keys


def append_record(path: str, row: dict) -> None:
    """1レコードを1行 JSON で追記し即 flush。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _make_row(rec: "SampleRecord", *, model_id: str, temperature: float,
              top_p: float, base_seed: int, seed: int, task_text: str) -> dict:
    r = asdict(rec.result)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "temperature": temperature,
        "top_p": top_p,
        "base_seed": base_seed,
        "seed": seed,
        "cond": rec.cond,
        "task_i": rec.task_i,
        "lower": rec.lower,
        "upper": rec.upper,
        "task_hash": _task_hash(task_text),
        "k": rec.k,
        "elapsed": rec.elapsed,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": r,
    }


def expected_n(p):
    return float("inf") if p == 0 else 1.0 / p


def best_of_n_success(p, n):
    return 1.0 - (1.0 - p) ** n


# ----------------------------------------------------------------------------
# サマリ計算（ストアから再計算可能、Wilson 95%CI 付き）
# ----------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson スコア法による 95% 信頼区間。n=0 のとき (0, 1) を返す。"""
    if n == 0:
        return 0.0, 1.0
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_summary(records: list["SampleRecord"], *,
                    model_id: str, temperature: float, top_p: float) -> dict:
    """records のうち (model_id, temperature, top_p) に一致するもののみ集計。"""
    # 対象レコードをフィルタ（ストア混在時に警告）
    # records は SampleRecord なのでフィルタ条件は store キーから別途確認が必要。
    # ここでは全レコードを集計対象にする（引数は将来の拡張用に受け取るだけ）。
    # ストア読み込み時にすでに同一設定のみロードする運用でも可。
    filtered = records  # すべて対象（model_id/temp フィルタはストア側で管理）

    summary: dict = {}
    for cond in CONDITIONS.keys():
        cond_recs = [r for r in filtered if r.cond == cond]
        n_total = len(cond_recs)
        if n_total == 0:
            continue
        n_accept = sum(1 for r in cond_recs if r.result.accepted)
        n_raw_in = sum(1 for r in cond_recs if r.result.raw_in_range)
        n_trim = sum(1 for r in cond_recs if r.result.needed_trim)
        n_punct = sum(1 for r in cond_recs if r.result.ends_with_punct)
        punct_checked = [r for r in cond_recs if r.result.is_natural is not None]
        n_natural = sum(1 for r in punct_checked if r.result.is_natural)
        center_errs = [abs(r.result.trimmed_chars - (r.lower + r.upper) / 2)
                       for r in cond_recs]
        p_accept = n_accept / n_total
        p_raw = n_raw_in / n_total
        summary[cond] = dict(
            p_accept=p_accept,
            p_accept_ci=wilson_ci(n_accept, n_total),
            p_raw=p_raw,
            p_raw_ci=wilson_ci(n_raw_in, n_total),
            trim_rate=n_trim / n_total,
            E_N=expected_n(p_accept),
            mae_center=sum(center_errs) / n_total,
            p_punct=n_punct / n_total,
            p_natural=(n_natural / len(punct_checked)
                       if punct_checked else float("nan")),
            n=n_total,
        )
    return summary


def run_experiment(tasks, *, target_k: int, conditions: list[str],
                   task_indices: list[int], base_seed: int,
                   temperature: float, top_p: float,
                   store_path: str) -> tuple[list["SampleRecord"], set]:
    """累積実験ドライバ。

    - ストアから既存レコードをロードし、不足分のみ生成して追記する。
    - K を増やすと過去結果を再利用し、増分のみ生成する。
    - 同じ (cond, task_i, k, seed) のレコードは skip する（dedupe）。
    """
    records, existing_keys = load_store(store_path)

    # 今回生成するセル一覧（condとtask_iのフィルタ適用後）
    active_conds = [(c, f) for c, f in CONDITIONS.items() if c in conditions]
    active_tasks = [(ti, *tasks[ti - 1]) for ti in task_indices]

    # 生成が必要なセル数を計算
    needed = sum(
        1
        for cond, _ in active_conds
        for ti, text, lower, upper in active_tasks
        for k in range(1, target_k + 1)
        if record_key({
            "model_id": MODEL_ID, "temperature": temperature,
            "top_p": top_p, "cond": cond, "task_i": ti, "k": k,
            "seed": derive_seed(base_seed, ti, k),
        }) not in existing_keys
    )
    grand_total = needed
    grand_done = 0

    n_cond = len(active_conds)
    for cond_i, (cond, flags) in enumerate(active_conds, 1):
        print(f"\n[{cond_i}/{n_cond}] condition={cond}  (target_k={target_k})")
        t0 = time.time()
        for ti, text, lower, upper in active_tasks:
            for k in range(1, target_k + 1):
                seed = derive_seed(base_seed, ti, k)
                key = record_key({
                    "model_id": MODEL_ID, "temperature": temperature,
                    "top_p": top_p, "cond": cond, "task_i": ti, "k": k,
                    "seed": seed,
                })
                if key in existing_keys:
                    continue  # 既存 skip

                t_s = time.time()
                gen_kwargs = dict(temperature=temperature, top_p=top_p, seed=seed)
                if flags.get("__mode__") == "closing":
                    f = {k2: v for k2, v in flags.items() if k2 != "__mode__"}
                    r = generate_one_with_closing(text, lower, upper, **f, **gen_kwargs)
                else:
                    r = generate_one(text, lower, upper, **flags, **gen_kwargs)
                elapsed = time.time() - t_s
                grand_done += 1

                rec = SampleRecord(cond, ti, lower, upper, k, r, elapsed)
                row = _make_row(rec, model_id=MODEL_ID, temperature=temperature,
                                top_p=top_p, base_seed=base_seed, seed=seed,
                                task_text=text)
                append_record(store_path, row)
                existing_keys.add(key)
                records.append(rec)

                status = "OK" if r.accepted else "NG"
                trim_flag = " trim" if r.needed_trim else "     "
                if not r.accepted:
                    if not r.ends_with_punct:
                        qual_flag = " [no-punct]"
                    elif r.is_natural is False:
                        qual_flag = " [unnatural]"
                    elif r.is_natural is True and r.natural_raw is not None and \
                            "いいえ" not in r.natural_raw and "はい" not in r.natural_raw:
                        qual_flag = f" [ambiguous: {r.natural_raw!r}]"
                    else:
                        qual_flag = ""
                else:
                    qual_flag = ""
                print(
                    f"  task{ti} k={k}/{target_k}  chars={r.trimmed_chars:>4}"
                    f" [{lower},{upper}]  {status}{trim_flag}{qual_flag}"
                    f"  {elapsed:.1f}s  [overall {grand_done}/{grand_total}]"
                )
        dt = time.time() - t0
        print(f"  → condition={cond} done in {dt:.0f}s")

    return records, existing_keys


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
        punct_badge = '<span class="badge ok">✓</span>' if r.ends_with_punct else '<span class="badge ng">✗</span>'
        if r.is_natural is None:
            nat_cell = '<span style="color:#aaa">—</span>'
        elif r.is_natural is True and r.natural_raw is not None and \
                "いいえ" not in r.natural_raw and "はい" not in r.natural_raw:
            nat_cell = f'<span style="background:#ff9800;color:white;border-radius:3px;padding:1px 5px;font-size:11px" title="ambiguous">{_esc(r.natural_raw)}</span>'
        elif r.is_natural:
            nat_cell = '<span class="badge ok">はい</span>'
        else:
            nat_cell = '<span class="badge ng">いいえ</span>'
        rows_html.append(f"""
      <tr class="{ok_cls}">
        <td>{rec.cond}</td>
        <td>task{rec.task_i} [{rec.lower},{rec.upper}]</td>
        <td>{rec.k}</td>
        <td class="chars">{rec.result.trimmed_chars} {raw_diff}</td>
        <td><span class="badge {ok_cls}">{"OK" if r.accepted else "NG"}</span>{trim_badge}</td>
        <td class="elapsed">{rec.elapsed:.1f}s</td>
        <td>{punct_badge}</td>
        <td>{nat_cell}</td>
        <td class="output">{_esc(r.trimmed)}</td>
        <td class="output raw">{_esc(r.text)}</td>
      </tr>""")

    summary_rows = []
    for cond, s in summary.items():
        p_nat_str = f"{s['p_natural']:.2f}" if not math.isnan(s['p_natural']) else "n/a"
        p_acc_lo, p_acc_hi = s.get('p_accept_ci', (s['p_accept'], s['p_accept']))
        p_raw_lo, p_raw_hi = s.get('p_raw_ci', (s['p_raw'], s['p_raw']))
        p_acc_cell = f"{s['p_accept']:.2f}<br><small style='color:#888'>[{p_acc_lo:.2f},{p_acc_hi:.2f}]</small>"
        p_raw_cell = f"{s['p_raw']:.2f}<br><small style='color:#888'>[{p_raw_lo:.2f},{p_raw_hi:.2f}]</small>"
        summary_rows.append(f"""
      <tr>
        <td>{cond}</td>
        <td>{s['n']}</td>
        <td>{p_raw_cell}</td>
        <td>{p_acc_cell}</td>
        <td>{s['trim_rate']*100:.0f}%</td>
        <td>{s['E_N']:.2f}</td>
        <td>{best_of_n_success(s['p_accept'], 2):.2f}</td>
        <td>{best_of_n_success(s['p_accept'], 3):.2f}</td>
        <td>{s['mae_center']:.1f}</td>
        <td>{s['p_punct']:.2f}</td>
        <td>{p_nat_str}</td>
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
    <th>condition</th><th>n</th><th>p_raw<br><small>95%CI</small></th>
    <th>p_accept<br><small>95%CI</small></th><th>trim%</th>
    <th>E[N]</th><th>P(N≤2)</th><th>P(N≤3)</th><th>MAE</th>
    <th>p_punct</th><th>p_natural</th>
  </tr></thead>
  <tbody>{"".join(summary_rows)}</tbody>
</table>
<p style="font-size:11px;color:#555">
  n: 累積サンプル数 / p_raw: トリム前から範囲内（Wilson 95%CI 付き） /
  p_accept: トリム後に範囲内かつ品質チェック合格（Wilson 95%CI 付き） / trim%: 上限トリム発火率 /
  E[N]: 期待生成回数 / MAE: 帯中心からの平均絶対文字数ズレ /
  p_punct: 文末記号で終わった割合 / p_natural: 自然さ判定で「はい」だった割合（文末記号合格サンプルのみ）
</p>

<h2>Samples</h2>
<table>
  <thead><tr>
    <th>condition</th><th>task</th><th>k</th><th>chars</th>
    <th>status</th><th>time</th><th>punct</th><th>natural</th>
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
    W = 120
    print("\n" + "=" * W)
    print(f"{'condition':<16} {'n':>4} {'p_raw':>13} {'p_accept':>15} {'trim%':>6} "
          f"{'E[N]':>6} {'P(N≤2)':>7} {'P(N≤3)':>7} {'MAE':>6} {'p_punct':>8} {'p_nat':>7}")
    print("-" * W)
    for cond, s in summary.items():
        p_nat_str = f"{s['p_natural']:>7.2f}" if not math.isnan(s['p_natural']) else "    n/a"
        p_acc_lo, p_acc_hi = s.get('p_accept_ci', (s['p_accept'], s['p_accept']))
        p_raw_lo, p_raw_hi = s.get('p_raw_ci', (s['p_raw'], s['p_raw']))
        p_raw_str = f"{s['p_raw']:.2f}[{p_raw_lo:.2f},{p_raw_hi:.2f}]"
        p_acc_str = f"{s['p_accept']:.2f}[{p_acc_lo:.2f},{p_acc_hi:.2f}]"
        print(f"{cond:<16} {s['n']:>4} {p_raw_str:>13} {p_acc_str:>15} "
              f"{s['trim_rate']*100:>5.0f}% {s['E_N']:>6.2f} "
              f"{best_of_n_success(s['p_accept'], 2):>7.2f} "
              f"{best_of_n_success(s['p_accept'], 3):>7.2f} "
              f"{s['mae_center']:>6.1f} {s['p_punct']:>8.2f} {p_nat_str}")
    print("=" * W)
    print("n        : 累積サンプル数（全タスク合計）")
    print("p_raw    : トリム前から範囲内に着地した割合（Wilson 95%CI 付き）")
    print("p_accept : トリム後に範囲内かつ品質チェック合格（Wilson 95%CI 付き）")
    print("trim%    : 上限トリムが発火した割合")
    print("E[N]     : 期待生成回数 = 1/p_accept   P(N≤k): k回以内に1回成功する確率")
    print("MAE      : 帯中心からの平均絶対文字数ズレ（参考）")
    print("p_punct  : raw textが文末記号で終わった割合")
    print("p_nat    : 自然さ判定で「はい」だった割合（文末記号合格サンプルのみ）")


# ----------------------------------------------------------------------------
# タスク定義
# ----------------------------------------------------------------------------
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
TASKS = [
    (SAMPLE, 280, 400),   # task 1
    (SAMPLE, 140, 200),   # task 2
    (SAMPLE, 70, 100),    # task 3
    (SAMPLE, 49, 70),     # task 4
]


# ----------------------------------------------------------------------------
# エントリポイント
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="文字数制約付き要約の条件別実験（累積ストア版）")
    parser.add_argument("--k", type=int, default=8,
                        help="目標サンプル数/タスク（累積）。default: 8")
    parser.add_argument("--condition", dest="conditions", action="append",
                        metavar="NAME",
                        help="実行する条件名（複数指定可）。省略時は全条件。"
                             f" 選択肢: {list(CONDITIONS.keys())}")
    parser.add_argument("--task", dest="task_indices", action="append",
                        type=int, metavar="IDX",
                        help="実行するタスク番号（1始まり、複数指定可）。省略時は全タスク。")
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help=f"乱数のベースシード。default: {BASE_SEED}")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度。default: 0.7")
    parser.add_argument("--top-p", type=float, default=0.8,
                        help="top-p サンプリング。default: 0.8")
    parser.add_argument("--store", default=STORE_PATH,
                        help=f"結果ストアパス。default: {STORE_PATH}")
    parser.add_argument("--report", default="experiment_report.html",
                        help="HTML レポートパス。default: experiment_report.html")
    parser.add_argument("--report-only", action="store_true",
                        help="生成せずストアからサマリ + HTML のみ再計算する。")
    parser.add_argument("--reset", action="store_true",
                        help="ストアを削除してゼロからやり直す（既存結果がすべて消える）。")
    args = parser.parse_args()

    # --reset
    if args.reset:
        if os.path.exists(args.store):
            os.remove(args.store)
            print(f"[store] reset — removed {args.store}")
        else:
            print(f"[store] nothing to reset ({args.store} not found)")

    # 条件・タスクの解決
    all_cond_names = list(CONDITIONS.keys())
    if args.conditions:
        invalid = [c for c in args.conditions if c not in CONDITIONS]
        if invalid:
            parser.error(f"不明な条件: {invalid}  有効: {all_cond_names}")
        selected_conditions = args.conditions
    else:
        selected_conditions = all_cond_names

    all_task_indices = list(range(1, len(TASKS) + 1))
    if args.task_indices:
        invalid_t = [i for i in args.task_indices if i not in all_task_indices]
        if invalid_t:
            parser.error(f"不明なタスク番号: {invalid_t}  有効: {all_task_indices}")
        selected_task_indices = args.task_indices
    else:
        selected_task_indices = all_task_indices

    if args.report_only:
        # 生成せずストアからサマリのみ再計算
        records, _ = load_store(args.store)
        if not records:
            print("[report-only] ストアが空です。先に実験を実行してください。")
        else:
            summary = compute_summary(records, model_id=MODEL_ID,
                                      temperature=args.temperature,
                                      top_p=args.top_p)
            print_summary(summary)
            write_html_report(summary, records, args.report)
    else:
        records, _ = run_experiment(
            TASKS,
            target_k=args.k,
            conditions=selected_conditions,
            task_indices=selected_task_indices,
            base_seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            store_path=args.store,
        )
        summary = compute_summary(records, model_id=MODEL_ID,
                                  temperature=args.temperature,
                                  top_p=args.top_p)
        print_summary(summary)
        write_html_report(summary, records, args.report)