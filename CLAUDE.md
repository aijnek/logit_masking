# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`run_experiment.py` is a single-file experiment comparing four logit-masking strategies for constraining Japanese summarization output to a character-count range [L, U]. It runs Qwen3.5-9B on MPS/CUDA/CPU and reports acceptance rates, expected generation counts, and MAE from the band center.

## Commands

```bash
# Install dependencies
uv sync

# Run the experiment (all conditions, all tasks, K=8)
uv run python run_experiment.py

# Run specific condition/task only
uv run python run_experiment.py --condition baseline --task 1

# Increase K to 16 (reuses k=1..8 from store, generates only k=9..16)
uv run python run_experiment.py --k 16

# Regenerate summary + HTML report without running new samples
uv run python run_experiment.py --report-only

# Delete the store and start over
uv run python run_experiment.py --reset
```

Requires a `HF_TOKEN` in `.env` for gated Hugging Face model access.

Results are appended to `results.jsonl` one line per sample immediately after generation (crash-safe). Re-running resumes from where it left off. The store is **never deleted automatically** — only `--reset` removes it.

## Architecture

Everything lives in `run_experiment.py`. The key components:

**`CharRangeProcessor(LogitsProcessor)`** — Applied at generation time per token step:
- Lower bound (`use_lower_floor`): masks stop tokens to `-inf` while `used < lower`
- Upper bound (`use_hard_mask`): masks any token whose `nonspace_len > remaining` to `-inf`; stop tokens survive because their `nonspace_len` is forced to 0
- Soft convergence (`use_soft_boost`): adds `boost` to stop token logits once `used >= soft_start` (default: 70% into the band)
- Sentence-end force (`use_sent_end_force`): sets stop token logits to `+inf` when the current text ends with `。！？`
- Safety valve: if all logits are `-inf` after masking, stop tokens are released to prevent a crash

**`build_nonspace_len(tokenizer)`** — Precomputes a `vocab_size` tensor mapping each token ID to its non-whitespace character count. Special token IDs are zeroed so they are never blocked by the hard mask. Cached to `~/.cache/char_budget/nonspace_len_<md5>.pt`.

**`trim_to_range(text, lower, upper)`** — Post-generation sentence-boundary trim (splits on `。`). Returns `None` if the result still falls outside `[lower, upper]`, triggering re-generation in production use.

**`count_chars(text)`** — Shared character-counting function (non-whitespace Unicode code points, including `str.isspace()` which catches U+3000 full-width space). Used identically in `CharRangeProcessor.__call__`, `trim_to_range`, and result validation to ensure consistency.

**`ends_with_sentence_punct(text)`** — Returns `True` if the raw output ends with a sentence-ending character (`QUALITY_END_CHARS = 。！？…」`). Outputs that fail this check are marked as rejected.

**`is_natural_ending(text)`** — Calls `gpt-oss:20b` (Ollama) to judge whether the ending is natural. Returns `(bool, raw_answer_str)`. Ambiguous responses (neither `はい` nor `いいえ`) are treated as passing but flagged in the terminal output and HTML report (shown in orange) for human review.

**Five conditions** defined in `CONDITIONS`:
- `baseline`: no intervention
- `hard_only`: hard mask + lower floor, no soft boost
- `hard_soft`: hard mask + lower floor + soft boost
- `floor_sent`: lower floor + soft boost from 0% of band + sentence-end EOS force (no hard mask)
- `closing_inject`: lower floor only + closing phrase injection near the lower boundary

**Acceptance criteria** — a sample is `accepted` only if all three hold:
1. `trim_to_range` returns a result within `[lower, upper]`
2. The raw output ends with a sentence-ending punctuation mark
3. `is_natural_ending` returns `True`

**Reproducibility** — `derive_seed(base_seed, task_i, k)` produces a condition-independent seed (paired across conditions). `transformers_set_seed(seed)` is called at the start of each `generate_one` / `generate_one_with_closing`. MPS reproducibility is best-effort.

**Persistent store** — `results.jsonl` accumulates one JSON line per sample. Schema: `schema_version, model_id, temperature, top_p, base_seed, seed, cond, task_i, lower, upper, task_hash, k, elapsed, timestamp, result`. `record_key(row)` deduplicates by `(model_id, temperature, top_p, cond, task_i, k, seed)`. `compute_summary(records, ...)` re-derives statistics (including Wilson 95%CI) from the store at any time.

## Key parameters to tune

In `run_experiment.py`:
- `MODEL_ID` — swap to `"Qwen/Qwen3-1.7B"` to reduce memory (~18 GB needed for 9B fp16 on M3)
- `BASE_SEED` — default `1234`, also settable via `--seed`
- `CharRangeProcessor.__init__`: `soft_frac` (default 0.7), `boost` (default 4.0)
- CLI `--k N` — target samples per condition per task (cumulative)
- `TASKS` list — each entry is `(text, lower, upper)`; changing text changes `task_hash` and separates old/new records in the store
