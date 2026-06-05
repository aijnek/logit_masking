"""
visualize_results.py — k=50 実験結果の可視化
出力: experiment_viz.html（自己完結 HTML、外部依存なし）

Usage:
    uv run python visualize_results.py [--store results.jsonl] [--out experiment_viz.html]
"""

import argparse
import json
import math
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

COND_ORDER = [
    "baseline",
    "baseline_trim",
    "lower_only",
    "hard_hard",
    "hard_soft",
    "hard_soft_trim",
    "hard_force",
    "closing_inject",
    "closing_inject_trim",
    "closing_inject_force",
    "closing_inject_regen",
]

# NG カテゴリの優先順位・表示名・色
NG_CATS = [
    ("over",      "上限超過",   "#e65100"),  # 濃オレンジ
    ("under",     "下限不足",   "#1565c0"),  # 青
    ("no-punct",  "句点なし",   "#c62828"),  # 赤
    ("unnatural", "不自然",     "#6a1b9a"),  # 紫
    ("other",     "その他",     "#757575"),  # グレー
]
NG_CAT_NAMES  = {k: name  for k, name, _ in NG_CATS}
NG_CAT_COLORS = {k: color for k, name, color in NG_CATS}
OK_COLOR = "#2e7d32"  # 緑


# ---------------------------------------------------------------------------
# データ読み込みと集計
# ---------------------------------------------------------------------------

def ng_reason(row: dict) -> str:
    r = row["result"]
    if r["accepted"]:
        return "OK"
    cond = row["cond"]
    lo, up = row["lower"], row["upper"]
    is_trim = cond.endswith("_trim")
    # 優先1: 範囲外
    if is_trim:
        tc = r["trimmed_chars"]
        if tc < lo:
            return "under"
        if tc > up:
            return "over"
    else:
        if not r["raw_in_range"]:
            return "under" if r["raw_chars"] < lo else "over"
    # 優先2: 句点なし
    if not r["ends_with_punct"]:
        return "no-punct"
    # 優先3: 不自然
    if r["is_natural"] is False:
        return "unnatural"
    return "other"


def load_data(store_path: str):
    rows = [json.loads(line) for line in open(store_path, encoding="utf-8")]
    task_keys = sorted({r["task_i"] for r in rows})  # [1,2,3,4]
    task_range = {r["task_i"]: (r["lower"], r["upper"]) for r in rows}

    # (cond, task_i) → stats dict
    stats: dict[tuple, dict] = {}
    for row in rows:
        key = (row["cond"], row["task_i"])
        if key not in stats:
            stats[key] = {
                "n": 0,
                "ok": 0,
                "ng": Counter(),
                "samples": defaultdict(list),  # reason → [row, ...]
            }
        st = stats[key]
        st["n"] += 1
        reason = ng_reason(row)
        if reason == "OK":
            st["ok"] += 1
        else:
            st["ng"][reason] += 1
            st["samples"][reason].append(row)

    return rows, task_keys, task_range, stats


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------

def ok_rate(st: dict) -> float:
    return st["ok"] / st["n"] if st["n"] > 0 else 0.0


def ok_color(rate: float) -> str:
    """OK率 0→1 を赤→緑のグラデーションに変換。"""
    r_start, g_start, b_start = 244, 67, 54   # #f44336 (red)
    r_end,   g_end,   b_end   = 76, 175, 80   # #4caf50 (green)
    r = int(r_start + (r_end - r_start) * rate)
    g = int(g_start + (g_end - g_start) * rate)
    b = int(b_start + (b_end - b_start) * rate)
    return f"rgb({r},{g},{b})"


def text_color(rate: float) -> str:
    return "#fff" if rate < 0.55 else "#000"


def esc(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# Graph 1: ヒートマップ（条件 × タスク）
# ---------------------------------------------------------------------------

def build_heatmap(stats, task_keys):
    cells_per_row = len(task_keys) + 1  # タスク列 + 合計列

    header_cols = "".join(f'<th>Task {t}</th>' for t in task_keys)
    header = f"<tr><th>条件</th>{header_cols}<th>合計</th></tr>"

    body_rows = []
    col_ok = Counter()
    col_n  = Counter()

    for cond in COND_ORDER:
        total_ok = total_n = 0
        cells = []
        for t in task_keys:
            st = stats.get((cond, t), {"n": 0, "ok": 0})
            n, ok = st["n"], st["ok"]
            rate = ok_rate(st)
            bg = ok_color(rate)
            fg = text_color(rate)
            pct = f"{rate*100:.0f}%"
            cells.append(
                f'<td style="background:{bg};color:{fg};text-align:center;'
                f'font-weight:bold;min-width:80px">'
                f'{pct}<br><span style="font-size:11px;font-weight:normal">'
                f'({ok}/{n})</span></td>'
            )
            total_ok += ok
            total_n  += n
            col_ok[t] += ok
            col_n[t]  += n

        total_rate = total_ok / total_n if total_n else 0
        bg_t = ok_color(total_rate)
        fg_t = text_color(total_rate)
        cells.append(
            f'<td style="background:{bg_t};color:{fg_t};text-align:center;'
            f'font-weight:bold">'
            f'{total_rate*100:.0f}%<br><span style="font-size:11px;font-weight:normal">'
            f'({total_ok}/{total_n})</span></td>'
        )
        body_rows.append(f"<tr><td><b>{esc(cond)}</b></td>{''.join(cells)}</tr>")

    # 合計行
    footer_cells = []
    grand_ok = grand_n = 0
    for t in task_keys:
        ok, n = col_ok[t], col_n[t]
        rate = ok / n if n else 0
        bg = ok_color(rate)
        fg = text_color(rate)
        footer_cells.append(
            f'<td style="background:{bg};color:{fg};text-align:center;font-weight:bold">'
            f'{rate*100:.0f}%<br><span style="font-size:11px;font-weight:normal">'
            f'({ok}/{n})</span></td>'
        )
        grand_ok += ok
        grand_n  += n
    grand_rate = grand_ok / grand_n if grand_n else 0
    bg_g = ok_color(grand_rate)
    fg_g = text_color(grand_rate)
    footer_cells.append(
        f'<td style="background:{bg_g};color:{fg_g};text-align:center;font-weight:bold">'
        f'{grand_rate*100:.0f}%<br><span style="font-size:11px;font-weight:normal">'
        f'({grand_ok}/{grand_n})</span></td>'
    )
    body_rows.append(f"<tr><td><b>合計</b></td>{''.join(footer_cells)}</tr>")

    return f"""
<table class="heatmap">
  <thead>{header}</thead>
  <tbody>{''.join(body_rows)}</tbody>
</table>"""


# ---------------------------------------------------------------------------
# Graph 2: NG 内訳の積み上げ棒
# ---------------------------------------------------------------------------

BAR_MAX_PX = 400  # 50件 = 400px → 1件 = 8px
BAR_SCALE  = BAR_MAX_PX / 50  # px per sample


def build_stacked_bars(stats, task_keys):
    # 凡例
    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px">'
        f'<span style="display:inline-block;width:14px;height:14px;background:{color};'
        f'border-radius:2px;margin-right:4px"></span>{name}</span>'
        for _, name, color in NG_CATS
    )
    legend = f'<div style="margin-bottom:8px">{legend_items}</div>'

    rows_html = []
    for cond in COND_ORDER:
        group_rows = []
        for t in task_keys:
            st = stats.get((cond, t), {"n": 0, "ok": 0, "ng": Counter()})
            ng = st["ng"]
            ng_total = sum(ng.values())

            # 積み上げセグメント
            segs = []
            for cat, _, color in NG_CATS:
                cnt = ng.get(cat, 0)
                if cnt == 0:
                    continue
                px = max(cnt * BAR_SCALE, 1)
                label = str(cnt) if px >= 18 else ""
                segs.append(
                    f'<span style="display:inline-block;width:{px:.1f}px;height:22px;'
                    f'background:{color};line-height:22px;text-align:center;'
                    f'font-size:11px;color:white;overflow:hidden;vertical-align:middle;'
                    f'title="{esc(NG_CAT_NAMES[cat])}: {cnt}">{label}</span>'
                )

            # OK バー（緑、右端に小さく）
            ok_cnt = st["ok"]
            bar_inner = "".join(segs)
            n = st["n"]
            rate_str = f"{ok_cnt/n*100:.0f}% OK" if n else "-"

            group_rows.append(
                f'<tr>'
                f'<td style="white-space:nowrap;text-align:right;padding-right:8px;'
                f'font-size:12px;color:#555">Task {t}</td>'
                f'<td><div style="display:flex;align-items:center">'
                f'{bar_inner}'
                f'<span style="margin-left:8px;font-size:11px;color:#666">'
                f'NG={ng_total}件 / {rate_str}</span>'
                f'</div></td>'
                f'</tr>'
            )

        rows_html.append(
            f'<tr><td colspan="2" style="padding:6px 0 2px;font-weight:bold;'
            f'border-top:2px solid #ddd">{esc(cond)}</td></tr>'
            + "".join(group_rows)
        )

    return f"""
{legend}
<table style="border-collapse:collapse;font-size:13px">
  {''.join(rows_html)}
</table>"""


# ---------------------------------------------------------------------------
# Graph 3: 最頻 NG の実出力例
# ---------------------------------------------------------------------------

def _char_badge(chars: int, lo: int, up: int, is_trim: bool = False) -> str:
    label = "trimmed " if is_trim else ""
    if chars < lo:
        return (f'<span style="background:#1565c0;color:white;border-radius:3px;'
                f'padding:1px 5px;font-size:11px">{label}{chars}字 (下限{lo}に不足)</span>')
    elif chars > up:
        return (f'<span style="background:#e65100;color:white;border-radius:3px;'
                f'padding:1px 5px;font-size:11px">{label}{chars}字 (上限{up}を超過)</span>')
    else:
        return (f'<span style="background:#2e7d32;color:white;border-radius:3px;'
                f'padding:1px 5px;font-size:11px">{label}{chars}字 ✓</span>')


def _sample_card(row: dict, is_trim_cond: bool) -> str:
    r = row["result"]
    lo, up = row["lower"], row["upper"]
    k = row["k"]

    # 文字数バッジ
    raw_badge = _char_badge(r["raw_chars"], lo, up, is_trim=False)
    trim_badge = ""
    if is_trim_cond and r["raw_chars"] != r["trimmed_chars"]:
        trim_badge = " → " + _char_badge(r["trimmed_chars"], lo, up, is_trim=True)

    # 句点バッジ
    punct_badge = (
        '<span style="background:#2e7d32;color:white;border-radius:3px;padding:1px 5px;font-size:11px">句点あり ✓</span>'
        if r["ends_with_punct"] else
        '<span style="background:#c62828;color:white;border-radius:3px;padding:1px 5px;font-size:11px">句点なし ✗</span>'
    )

    # 自然さバッジ
    if r["is_natural"] is None:
        nat_badge = '<span style="color:#aaa;font-size:11px">判定なし</span>'
    elif r["is_natural"]:
        nat_badge = '<span style="background:#2e7d32;color:white;border-radius:3px;padding:1px 5px;font-size:11px">自然 ✓</span>'
    else:
        nat_badge = '<span style="background:#6a1b9a;color:white;border-radius:3px;padding:1px 5px;font-size:11px">不自然 ✗</span>'

    # outcome バッジ（regen 条件のみ存在）
    outcome_badge = ""
    outcome = r.get("outcome")
    if outcome is not None:
        regen_k_val = r.get("regen_k") or 0
        if outcome == "first_pass":
            oc_style = "background:#2e7d32;color:white"
            oc_label = "outcome: first_pass"
        elif outcome == "trim":
            oc_style = "background:#00796b;color:white"
            oc_label = "outcome: trim"
        elif outcome == "fallback":
            oc_style = "background:#c62828;color:white"
            oc_label = "outcome: fallback"
        elif outcome.startswith("regen_"):
            oc_style = "background:#e65100;color:white"
            oc_label = f"outcome: {outcome}（引き直し{regen_k_val}回）"
        else:
            oc_style = "background:#757575;color:white"
            oc_label = f"outcome: {esc(outcome)}"
        outcome_badge = (
            f' &nbsp; <span style="{oc_style};border-radius:3px;'
            f'padding:1px 5px;font-size:11px">{oc_label}</span>'
        )

    # 出力テキスト
    display_text = r.get("visible") or r.get("trimmed") or r["text"]
    raw_section = ""
    if r["text"] != display_text:
        raw_section = (
            f'<div style="margin-top:4px;font-size:11px;color:#888">'
            f'<b>生出力:</b> {esc(r["text"])}</div>'
        )

    return f"""
<div style="border:1px solid #ddd;border-radius:4px;padding:10px;margin-bottom:8px;background:#fafafa">
  <div style="margin-bottom:6px;font-size:12px">
    <b>k={k}</b> &nbsp;
    {raw_badge}{trim_badge} &nbsp; {punct_badge} &nbsp; {nat_badge}{outcome_badge}
  </div>
  <div style="font-size:13px;line-height:1.6;word-break:break-all">{esc(display_text)}</div>
  {raw_section}
</div>"""


def build_ng_examples(stats, task_keys):
    sections = []

    for cond in COND_ORDER:
        is_trim_cond = cond.endswith("_trim")
        cond_sections = []

        for t in task_keys:
            st = stats.get((cond, t))
            if st is None or not st["ng"]:
                # NG なし
                cond_sections.append(
                    f'<details><summary>Task {t} — <b style="color:#2e7d32">NG なし（全件 OK）</b></summary></details>'
                )
                continue

            top_cat, top_cnt = st["ng"].most_common(1)[0]
            top_name = NG_CAT_NAMES[top_cat]
            top_color = NG_CAT_COLORS[top_cat]
            top_ng_total = sum(st["ng"].values())

            # 代表例: k が最小のサンプルを 2 件まで
            samples_for_cat = sorted(st["samples"][top_cat], key=lambda r: r["k"])
            representative = samples_for_cat[:2]
            cards = "".join(_sample_card(row, is_trim_cond) for row in representative)

            # NG 内訳サマリ
            ng_summary = " / ".join(
                f'<span style="color:{NG_CAT_COLORS[c]}">{NG_CAT_NAMES[c]}: {cnt}</span>'
                for c, cnt in st["ng"].most_common()
            )

            cond_sections.append(f"""
<details open>
  <summary style="cursor:pointer;padding:4px 0">
    Task {t} —
    最頻 NG: <b style="color:{top_color}">{top_name} ({top_cnt}/{top_ng_total} NG件)</b>
    &nbsp;|&nbsp; NG内訳: {ng_summary}
  </summary>
  <div style="padding:8px 0 4px;font-size:12px;color:#555">
    最頻カテゴリ「{top_name}」の代表例 (最大2件、k が小さい順):
  </div>
  {cards}
</details>""")

        sections.append(f"""
<details open>
  <summary style="cursor:pointer;font-size:15px;font-weight:bold;
    padding:8px 0;border-bottom:2px solid #e0e0e0">
    {esc(cond)}
  </summary>
  <div style="padding:12px 0 4px 16px">
    {''.join(cond_sections)}
  </div>
</details>""")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Graph 4: closing_inject_regen の outcome 内訳
# ---------------------------------------------------------------------------

OUTCOME_CATS = [
    ("first_pass", "first_pass（帯内）",  "#2e7d32"),   # 緑
    ("trim",       "trim（トリム成功）",   "#00796b"),   # ティール
    ("regen",      "regen（引き直し）",    "#e65100"),   # オレンジ
    ("fallback",   "fallback（全失敗）",   "#c62828"),   # 赤
]


def build_regen_outcomes(rows: list, task_keys: list) -> str:
    """closing_inject_regen の outcome 内訳を積み上げ棒で描画する。
    regen 条件がストアに存在しない場合は空文字を返す（後方互換）。"""
    regen_rows = [r for r in rows if r["cond"] == "closing_inject_regen"]
    if not regen_rows:
        return ""

    # タスク別集計
    from collections import defaultdict
    task_data: dict[int, dict] = {t: {"n": 0, "ok": 0, "oc": Counter(), "rk": Counter()}
                                   for t in task_keys}
    for row in regen_rows:
        t = row["task_i"]
        res = row["result"]
        task_data[t]["n"] += 1
        if res["accepted"]:
            task_data[t]["ok"] += 1
        oc_raw = res.get("outcome") or "fallback"
        # regen_1, regen_2, ... → "regen" にまとめ、regen_k 別も別途集計
        if oc_raw.startswith("regen_"):
            task_data[t]["oc"]["regen"] += 1
            rk = int(oc_raw.split("_")[1])
            task_data[t]["rk"][rk] += 1
        else:
            task_data[t]["oc"][oc_raw] += 1

    # 凡例
    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px">'
        f'<span style="display:inline-block;width:14px;height:14px;background:{color};'
        f'border-radius:2px;margin-right:4px"></span>{name}</span>'
        for _, name, color in OUTCOME_CATS
    )
    legend = f'<div style="margin-bottom:8px">{legend_items}</div>'

    rows_html = []
    for t in task_keys:
        td = task_data[t]
        n, ok = td["n"], td["ok"]
        oc = td["oc"]
        rk = td["rk"]

        segs = []
        for cat, _, color in OUTCOME_CATS:
            cnt = oc.get(cat, 0)
            if cnt == 0:
                continue
            px = max(cnt * BAR_SCALE, 1)
            label = str(cnt) if px >= 18 else ""
            # regen セグメントは regen_k 別件数をツールチップ代わりにラベルに足す
            if cat == "regen" and rk:
                rk_detail = " ".join(f"k{k}:{v}" for k, v in sorted(rk.items()))
                title = f"regen({cnt}): {rk_detail}"
            else:
                title = f"{cat}: {cnt}"
            segs.append(
                f'<span style="display:inline-block;width:{px:.1f}px;height:22px;'
                f'background:{color};line-height:22px;text-align:center;'
                f'font-size:11px;color:white;overflow:hidden;vertical-align:middle;'
                f'" title="{esc(title)}">{label}</span>'
            )

        rate_str = f"{ok/n*100:.1f}% accepted ({ok}/{n})" if n else "-"
        # regen_k 別サマリ
        rk_summary = ""
        if rk:
            rk_summary = (
                ' &nbsp; <span style="font-size:11px;color:#888">引き直し件数: '
                + " / ".join(f'k={k}:{v}件' for k, v in sorted(rk.items()))
                + "</span>"
            )

        rows_html.append(
            f'<tr>'
            f'<td style="white-space:nowrap;text-align:right;padding-right:8px;'
            f'font-size:12px;color:#555">Task {t}</td>'
            f'<td><div style="display:flex;align-items:center">'
            f'{"".join(segs)}'
            f'<span style="margin-left:8px;font-size:11px;color:#555">{rate_str}</span>'
            f'{rk_summary}'
            f'</div></td>'
            f'</tr>'
        )

    return f"""
{legend}
<table style="border-collapse:collapse;font-size:13px">
  {''.join(rows_html)}
</table>"""


# ---------------------------------------------------------------------------
# HTML 組み立て
# ---------------------------------------------------------------------------

def build_html(heatmap: str, bars: str, regen_outcomes: str, examples: str, store_path: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>実験結果ビジュアライザ — k=50</title>
<style>
  body {{ font-family: -apple-system, "Hiragino Sans", "Yu Gothic", sans-serif;
          font-size: 14px; margin: 24px; color: #212121; }}
  h1 {{ font-size: 20px; border-bottom: 2px solid #1565c0; padding-bottom: 6px; }}
  h2 {{ font-size: 17px; margin-top: 2em; color: #1565c0; }}
  h3 {{ font-size: 14px; margin-top: 1.5em; color: #444; }}
  table.heatmap {{ border-collapse: collapse; }}
  table.heatmap th, table.heatmap td {{
    border: 1px solid #bbb; padding: 6px 12px; }}
  table.heatmap th {{ background: #e8eaf6; text-align: center; }}
  summary {{ outline: none; }}
  summary::-webkit-details-marker {{ color: #1565c0; }}
</style>
</head>
<body>
<h1>実験結果ビジュアライザ <small style="font-size:13px;color:#888">({store_path})</small></h1>

<h2>📊 グラフ1: OK 率ヒートマップ（条件 × タスク）</h2>
<p style="color:#555;font-size:13px">
  各セルの背景色は OK 率を示します（<span style="color:#c62828">■赤=0%</span> → <span style="color:#2e7d32">■緑=100%</span>）。
</p>
{heatmap}

<h2>📊 グラフ2: NG 内訳の積み上げ棒（条件 × タスク）</h2>
<p style="color:#555;font-size:13px">
  棒の長さ＝NG件数（最大50）。各色のセグメントが失敗要因の件数を示します。
</p>
{bars}

{"" if not regen_outcomes else f"""<h2>📊 グラフ3: closing_inject_regen — outcome 内訳（タスク別）</h2>
<p style="color:#555;font-size:13px">
  regen 条件固有の処理フロー（first_pass / trim / regen引き直し / fallback）の件数内訳と、引き直し試行回数別の件数を示します。
</p>
""" + regen_outcomes}

<h2>📋 {"グラフ4" if regen_outcomes else "グラフ3"}: 最頻 NG の実出力例（条件 × タスク）</h2>
<p style="color:#555;font-size:13px">
  各条件・タスクで最も多い NG カテゴリの代表例（k が最小のサンプル最大2件）を掲載します。
</p>
{examples}

</body>
</html>"""


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="results.jsonl")
    parser.add_argument("--out",   default="experiment_viz.html")
    args = parser.parse_args()

    print(f"Loading {args.store} ...")
    rows, task_keys, task_range, stats = load_data(args.store)
    print(f"  {len(rows)} records, {len(task_keys)} tasks, "
          f"{len({r['cond'] for r in rows})} conditions")

    print("Building heatmap ...")
    heatmap = build_heatmap(stats, task_keys)

    print("Building stacked bars ...")
    bars = build_stacked_bars(stats, task_keys)

    print("Building regen outcomes ...")
    regen_outcomes = build_regen_outcomes(rows, task_keys)

    print("Building NG examples ...")
    examples = build_ng_examples(stats, task_keys)

    html = build_html(heatmap, bars, regen_outcomes, examples, args.store)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written → {args.out}")


if __name__ == "__main__":
    main()
