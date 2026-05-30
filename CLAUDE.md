# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`run_experiment.py` is a single-file experiment comparing four logit-masking strategies for constraining Japanese summarization output to a character-count range [L, U]. It runs Qwen3.5-9B on MPS/CUDA/CPU and reports acceptance rates, expected generation counts, and MAE from the band center.

## Commands

```bash
# Install dependencies
uv sync

# Run the experiment
uv run python run_experiment.py
```

Requires a `HF_TOKEN` in `.env` for gated Hugging Face model access.

Intermediate results are saved to `checkpoint.json` after each condition completes. If the run is interrupted, re-running will skip completed conditions and resume from where it left off. The checkpoint is deleted automatically when all conditions finish.

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

**`is_natural_ending(text)`** — Calls Qwen3.5-9B itself to judge whether the ending is natural. Returns `(bool, raw_answer_str)`. Ambiguous responses (neither `はい` nor `いいえ`) are treated as passing but flagged in the terminal output and HTML report (shown in orange) for human review.

**Four conditions** defined in `CONDITIONS`:
- `baseline`: no intervention
- `hard_only`: hard mask + lower floor, no soft boost
- `hard_soft`: hard mask + lower floor + soft boost
- `floor_sent`: lower floor + soft boost from 0% of band + sentence-end EOS force (no hard mask)

**Acceptance criteria** — a sample is `accepted` only if all three hold:
1. `trim_to_range` returns a result within `[lower, upper]`
2. The raw output ends with a sentence-ending punctuation mark
3. `is_natural_ending` returns `True`

## Key parameters to tune

In `run_experiment.py`:
- `MODEL_ID` — swap to `"Qwen/Qwen3-1.7B"` to reduce memory (~18 GB needed for 9B fp16 on M3)
- `CharRangeProcessor.__init__`: `soft_frac` (default 0.7), `boost` (default 4.0)
- `run_experiment(tasks, K=8)` — `K` is samples per condition per task
- `tasks` list — each entry is `(text, lower, upper)`
