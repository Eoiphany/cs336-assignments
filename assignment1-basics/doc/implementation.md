# Assignment 1 implementation and verification

Updated: 2026-10-16 (Asia/Shanghai). Codex adapted the user-designated reference.

## Core TODO implementation

Codex migrated the minimal implementation from the user-specified reference: BPE training, tokenizer, Linear/Embedding/RMSNorm/SwiGLU/RoPE/attention/Transformer, AdamW/scheduler, data sampling, cross-entropy/gradient clipping/checkpoint helpers, generation and logging. All **21** adapter stubs are connected while preserving target signatures and original test assertions. The staff pretokenization example needed only a main guard: importing its helper previously executed `open(...)` on the placeholder.

Original full test suite: **46 passed, 2 skipped** (the original memory-limit tests skip on this platform). Command: `uv run pytest -q`; full output: `pytest-final.txt`. No tests were rewritten to change expected results. The complete-epoch training smoke below additionally validates the training entrypoint, which these original unit tests do not exercise.

## Training and decoding entrypoints

The entrypoint work changes only `cs336_basics/config.py`, `train.py`,
`experiment_tracking.py` and `generate.py`. The existing model, optimizer,
mathematical helpers, tokenizer and adapter interfaces remain separate. In
particular, the assignment's iteration-based `nn_utils.save_checkpoint` /
`load_checkpoint` APIs were preserved; the training entrypoint has its own richer
complete-epoch checkpoint format.

The JSON config contains `run`, `data`, `model`, `optim`, `train` and `wandb`
objects. All fields/defaults are declared in `config.py`. Unknown groups/keys
raise an error. Dataset paths must be supplied, CPU is the default, the default
is one complete epoch, and W&B is disabled unless explicitly requested. No
training starts on import. These commands work on macOS/Linux and Windows CMD:

```sh
uv run python cs336_basics/train.py --config train.json
uv run python cs336_basics/train.py --resume runs/my_experiment/2_0.001_1_8_4_123/last.pt --epochs 2
```

Example small configuration (replace the two data file paths with actual
**pretokenized raw uint16 files**, not text documents):

```json
{
  "run": {"runs_dir": "runs", "experiment_name": "my_experiment"},
  "data": {
    "train_data_path": "train.uint16.bin",
    "val_data_path": "valid.uint16.bin",
    "np_dtype": "uint16", "context_length": 4, "device": "cpu"
  },
  "model": {
    "vocab_size": 257, "context_length": 4, "d_model": 8,
    "num_layers": 1, "num_heads": 2, "d_ff": 16
  },
  "optim": {"lr_max": 0.001, "lr_min": 0.0001, "warmup_iters": 0, "cosine_cycle_iters": 32},
  "train": {"epochs": 1, "batch_size": 2, "eval_batches": 2, "log_interval": 2, "seed": 123}
}
```

Each epoch shuffles all nonoverlapping context windows of the training memmap
once. A final incomplete context window is discarded. Validation uses the
separately supplied validation memmap; training never creates a new data split.
Data paths resolve at launch and are saved along with byte sizes, token counts,
modification times and window rules, which are checked on resume.

The output layout is:

```text
runs/<experiment_name>/<batch_lr_layers_dmodel_context_seed>/
  checkpoints/epoch_0001.pt
  checkpoints/epoch_0002.pt
  best.pt
  last.pt
  config.json
  history.json
  metrics.jsonl
```

An explicit `run.parameter_name` may replace the automatically generated
underscore-separated parameter name. New runs refuse to reuse an existing
parameter directory. Epoch archives are atomically published without overwriting
existing files. `best.pt` contains only the best model's `state_dict`; `last.pt`
and each archive contain model/buffer weights, optimizer state, configuration,
Python/NumPy/Torch and available accelerator RNG states, data identity/window
split, history, completed epoch, global step, best loss/epoch and elapsed time.
The stateless cosine scheduler's step/config are saved as `scheduler_state`.
AMP, EMA and early stopping are not enabled; their saved state is explicitly
`null`. Checkpoints are saved only after complete epochs, never by step.

Resume restores all active state and permits increasing the total target epoch
count. It does not change model/optimizer/batch/seed settings or data identity.
An interrupted epoch restarts from the preceding complete checkpoint; metric
lines after that checkpoint are removed before replay. Newer epoch archives
prevent resuming an older checkpoint over later work. Accelerator-specific RNG
restoration requires that accelerator backend; only CPU replay was verified.
`config.json` preserves the initial launch config, while resumed complete
checkpoints contain the updated total epoch target.

Generate from a complete checkpoint (the config is embedded):

```sh
uv run python cs336_basics/generate.py --checkpoint runs/my_experiment/2_0.001_1_8_4_123/last.pt --vocab vocab.pkl --merges merges.pkl --prompt "A" --max-new-tokens 128 --temperature 1 --top-p 0.9
```

For a pure `best.pt`, also pass `--config <run-directory>/config.json`. The
supported tokenizer format is the existing `Tokenizer.from_files` pair of
pickle vocabulary/merges files. `--device cpu` is the default. Temperature zero
uses greedy decoding. Generation keeps the prompt, uses a context window and
stops on the configured end token. The nucleus sampler was corrected to include
the token that first takes cumulative probability across `top_p`; the reference
incorrectly discarded that token. Decoding needs no dummy optimizer.

## Verified entrypoint checks

The following were actually executed with local synthetic data only:

```sh
uv run python cs336_basics/train.py --help
uv run python cs336_basics/generate.py --help
uv run ruff check cs336_basics/config.py cs336_basics/train.py cs336_basics/experiment_tracking.py cs336_basics/generate.py
```

An inline CPU smoke test used temporary raw uint16 files (65 training tokens,
33 validation tokens, token IDs `arange % 32`), the example 257-token model,
context 4, batch 2, seed 123 and two validation batches. It launched the actual
training CLI from outside the project directory for one epoch, loaded `last.pt`,
and launched the resume CLI for the second epoch. A separate uninterrupted
2-epoch run served as a deterministic control. Observed outputs:

| Epoch | Global step | Training loss | Validation loss |
| --- | --- | --- | --- |
| 1 | 8 | 5.468078970909119 | 5.515014171600342 |
| 2 | 16 | 5.419537723064423 | 5.341468811035156 |

Both runs produced **exactly equal model tensors and history** after epoch 2.
The smoke also checked nonempty optimizer state, Python/NumPy/Torch RNG state,
stable data/window metadata, two retained epoch archives, pure tensor-only
`best.pt`, refusal to overwrite a new run directory, and removal of a simulated
unfinished `step=999` metric during recovery. Actual generation CLI calls passed
for `best.pt`, `last.pt` and `checkpoints/epoch_0001.pt`. The pure-state call used
`--config`; complete-state calls used embedded configs. All temporary smoke
files were removed when the test finished.

A further CLI smoke passed with relative token paths at initial launch and
resume from a different current directory; saved data paths were absolute and
epoch 2 completed.

The top-p checks passed for probabilities `[0.6, 0.3, 0.1]` with `p=0.7`
(returning `[2/3, 1/3, 0]`) and `[0.5, 0.25, 0.25]` with `p=0.5`
(returning `[1, 0, 0]`). Ruff passed for the four owned entrypoint files.

These are implementation recovery/decoding checks, not a meaningful training
experiment or evidence of language-model quality. No real dataset training,
GPU/MPS run, W&B connection or long experiment was performed. Full-dataset
training and assignment write-up/results remain empirical work. The reference
source marker is in the entrypoint module/function documentation; no user-level
rules file was modified and no Git commit/push was performed.


## 建议执行的 Git 命令

在本作业目录执行。以下仅为建议，本次没有执行暂存、提交或推送；未纳入无关 `.DS_Store` 文件。

```sh
git status --short
git diff
git add -- cs336_basics/config.py cs336_basics/data.py cs336_basics/experiment_tracking.py cs336_basics/generate.py cs336_basics/modules.py cs336_basics/nn_utils.py cs336_basics/optimizer.py cs336_basics/pretokenization_example.py cs336_basics/tokenizer.py cs336_basics/train.py cs336_basics/train_bpe.py cs336_basics/transformer_lm.py doc/implementation.md doc/pytest-final.txt tests/adapters.py
git diff --cached
git commit -m 'feat: implement assignment1-basics TODOs'
```
