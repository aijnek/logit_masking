# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`run_experiment.py` is a single-file experiment comparing three logit-masking strategies for constraining Japanese summarization output to a character-count range [L, U]. It runs Qwen3.5-9B on MPS/CUDA/CPU and reports acceptance rates, expected generation counts, and MAE from the band center.

## Commands

```bash
# Install dependencies
uv sync

# Run the experiment
uv run python run_experiment.py
```

Requires a `HF_TOKEN` in `.env` for gated Hugging Face model access.

## Architecture

Everything lives in `run_experiment.py`. The key components:

**`CharRangeProcessor(LogitsProcessor)`** — Applied at generation time per token step:
- Lower bound (`use_lower_floor`): masks stop tokens to `-inf` while `used < lower`
- Upper bound (`use_hard_mask`): masks any token whose `nonspace_len > remaining` to `-inf`; stop tokens survive because their `nonspace_len` is forced to 0
- Soft convergence (`use_soft_boost`): adds `boost` to stop token logits once `used >= soft_start` (default: 70% into the band)

**`build_nonspace_len(tokenizer)`** — Precomputes a `vocab_size` tensor mapping each token ID to its non-whitespace character count. Special token IDs are zeroed so they are never blocked by the hard mask. Cached to `~/.cache/char_budget/nonspace_len_<md5>.pt`.

**`trim_to_range(text, lower, upper)`** — Post-generation sentence-boundary trim (splits on `。`). Returns `None` if the result still falls outside `[lower, upper]`, triggering re-generation in production use.

**`count_chars(text)`** — Shared character-counting function (non-whitespace Unicode code points, including `str.isspace()` which catches U+3000 full-width space). Used identically in `CharRangeProcessor.__call__`, `trim_to_range`, and result validation to ensure consistency.

**Three conditions** defined in `CONDITIONS`:
- `baseline`: no intervention
- `hard_only`: hard mask + lower floor, no soft boost
- `hard_soft`: hard mask + lower floor + soft boost

## Key parameters to tune

In `run_experiment.py`:
- `MODEL_ID` — swap to `"Qwen/Qwen3-1.7B"` to reduce memory (~18 GB needed for 9B fp16 on M3)
- `CharRangeProcessor.__init__`: `soft_frac` (default 0.7), `boost` (default 4.0)
- `run_experiment(tasks, K=8)` — `K` is samples per condition per task
- `tasks` list — each entry is `(text, lower, upper)`
