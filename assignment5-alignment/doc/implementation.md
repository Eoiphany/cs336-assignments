# Assignment 5 TODO implementation

Date: 2026-10-16 (Asia/Shanghai).

Scope: only this assignment. Reference source is the user-supplied
`cs336-assignments-refer/assignment5-alignment`; it targets 2025, while this
project and its handouts target 2026. Original tests, snapshots, model fixtures,
project configuration, starter utilities, and personal SUNET_ID are unchanged.

## Implemented

- `cs336_alignment/sft_utils.py`: migrated the reference's separate prompt/response
  tokenization, next-token masks, entropy, and causal-LM scoring. Removed unused
  intermediate lists; corrected an empty-prompt mask boundary and empty-token
  tensor handling.
- `cs336_alignment/grpo.py`: implemented the 2026 reward/advantage API, mean/no
  baseline, sample-standard-deviation/no/mean normalization, on-policy REINFORCE,
  unclipped importance ratios, clipped GRPO, sequence-ratio GSPO, sequence/fixed
  constant loss aggregation, gradient accumulation, optional gradient clipping,
  zero-advantage pruning, one optimizer update, and gradient cleanup.
  The handout explicitly requests `torch.std`'s default sample correction.
  Constant aggregation sums all masked token losses and divides once by the
  supplied constant; sequence aggregation averages within and across sequences.
  Pruning retains the original batch denominator. Entropy is logged over retained
  response tokens; clipping fractions accumulate over all processed microbatches.
- `cs336_alignment/safety.py`: implemented optional packed SFT Dataset/DataLoader,
  MMLU/GSM8K parsers, and per-instance DPO with the provided Alpaca template/EOS.
  The reference model runs without gradients and may reside on another device.
- `tests/adapters.py`: completed all 12 existing TODO adapters by connecting them
  to the above implementations. Function implementation comments identify Codex.

## Validation

Commands from this directory, identical on macOS and Windows CMD:

```text
uv run pytest -q --tb=short
```

Final result: **26 passed in 1.50s**. Full original output: `pytest-final.txt`.
Initial diagnostic failures were fixed by
matching constant-loss aggregation and removing the template's trailing newline
before SFT EOS packing; no expected test data was changed.

An additional in-memory `uv run python` check used the existing test tokenizer and
small random GPT2 fixture (no downloads) to compare one microbatch against four
uneven microbatches with zero-reward pruning. Both sequence and constant
normalization produced matching losses and updated weights. It also checked that
an empty prompt does not mark a padded next-token label as response content.
Output is in `edge-checks.txt`.

`git diff --check -- .` passed.

## Remaining work requiring external resources or user-specific input

- `modal_utils.py` retains the starter `SUNET_ID = "TODO"`: this is the user's
  account configuration and cannot be inferred.
- Handout GPU generation/training sweeps, GSM8K/prompting baselines, multi-seed
  GRPO/Dr.GRPO/RFT/MaxRL/GSPO experiments, custom-estimator experiments, and optional
  full SFT/DPO training/evaluation are not executed or claimed complete here.
  No GPU models/datasets were downloaded; no long training or external submission
  was started. Their metrics, plots, and empirical write-ups remain unmeasured.
- Theoretical derivations and experimental answer prose are not represented as
  executable code TODOs and were not fabricated.

This is an implementation/verification change, not a completed training experiment;
no important-experiment result is claimed. No Git commit or push was performed.


## 建议执行的 Git 命令

在本作业目录执行。以下仅为建议，本次没有执行暂存、提交或推送；未纳入无关 `.DS_Store` 文件。

```sh
git status --short
git diff
git add -- cs336_alignment/grpo.py cs336_alignment/safety.py cs336_alignment/sft_utils.py doc/edge-checks.txt doc/implementation.md doc/pytest-final.txt tests/adapters.py
git diff --cached
git commit -m 'feat: implement assignment5-alignment TODOs'
```
