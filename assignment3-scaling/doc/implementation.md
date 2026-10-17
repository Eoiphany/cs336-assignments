# Assignment 3 implementation and verification

Updated: 2026-10-16 (Asia/Shanghai). Implemented/migrated by Codex.

## Scope and reference differences

The target is the 2026 assignment. Its existing JAX model, training service,
client, database, scheduler, API schemas and server tests were preserved. The
reference repository is the 2025 assignment, with a different API and fixed-FLOPs
queries. Copying its client, sweep or hardcoded architecture ranges would not
match the current handout. No credentials, student ID or external API requests
were supplied or executed.

The 2026 repository has no missing student adapter functions. Its handout asks
students to write scaling analysis scripts. The following modules implement that
code while leaving experimental measurements and personal report writing to
actual observed runs:

- `cs336_scaling/scaling_data.py`: handout JSON loader, observed optimum selector,
  2026 completed-experiment loader and model parameter accounting. The latter
  accounts for attention projections, SwiGLU, layer/QK norms and optional attention
  bias, excluding token embeddings and the output vocabulary projection. Only
  `status_type == "completed"` is used; final loss is `val_losses[-1]`. Failed,
  timed-out, running and queued jobs do not become completed observations.
- `cs336_scaling/chinchilla_isoflops.py`: migrates the reference's minimum observed
  loss per compute budget, log-log power-law fits, extrapolation and PNG plots.
  Adds finite/positive input validation, distinct-scale checks, JSON evidence and
  reliable direct-script execution. Default inputs resolve relative to the project.
- `cs336_scaling/fit_scaling_laws.py`: new offline 2026 experiment analysis. Fits
  `L = E + A(N/N0)^(-alpha) + B(D/D0)^(-beta)` with bounded multistart least squares,
  requiring at least six observations and variation in both N and D. Fits measured
  seconds/token as `k N^gamma`. Ranks explicitly supplied or previously observed
  configurations under a target runtime, aligns tokens to complete evaluation
  chunks, respects the service's 500B-token limit, and validates candidate configs
  with the existing `TrainingConfig`. Writes predictions locally only.
- `test/offline/test_scaling_analysis.py`: standalone analysis regression tests,
  independent of PostgreSQL integration fixtures. Synthetic test measurements are
  explicitly labeled and are never presented as API results.
- `pyproject.toml` and `uv.lock`: optional `analysis` dependencies (NumPy,
  Matplotlib, SciPy); existing server dependencies stay intact.
- `README.md`: analysis invocation instructions. `.gitignore`: excludes generated
  `runs/` analysis outputs from source commits.

## Verified commands and observed results

All commands below were run in the target assignment3 directory. No training or
hosted API request was run.

```sh
uv lock
uv run --extra analysis python cs336_scaling/chinchilla_isoflops.py --outdir runs/isoflops
uv run --extra analysis python cs336_scaling/fit_scaling_laws.py --help
uv run --extra analysis python -m pytest --confcutdir=test/offline test/offline -q
uv run --extra analysis ruff check cs336_scaling/scaling_data.py cs336_scaling/chinchilla_isoflops.py cs336_scaling/fit_scaling_laws.py test/offline/test_scaling_analysis.py
git diff --check
```

- `uv lock` resolved successfully and the analysis environment installed.
- The bundled handout data generated both PNG plots and
  `runs/isoflops/isoflops_fit.json`. Fitted model-size exponent: **0.4686826677**;
  token-count exponent: **0.5313173323**. Predicted optima:

  | FLOPs | Parameters | Training tokens |
  | --- | --- | --- |
  | 1e23 | 7.0054233905e10 | 2.3791091184e11 |
  | 1e24 | 2.0611853919e11 | 8.0859619579e11 |

  These are computed from the handout's explicitly synthetic dataset, not from
  newly trained models or the hosted 2026 service.
- Offline regression suite: **10 passed**. It checks selected minima, N/D
  extrapolation consistency (`6ND = C`), invalid observations, exact parameter
  accounting, exclusion of incomplete jobs, measured runtime use, recovery of
  known synthetic loss/runtime laws, budget and token-alignment constraints, and
  direct CLI execution from outside the project directory. The CLI test writes
  only temporary synthetic input/output files.
- Ruff checks and `git diff --check` passed. Source formatting was applied with
  Ruff to the four new Python files.

Use `--confcutdir=test/offline`: the repository's existing `tests/conftest.py`
imports the server stack and creates PostgreSQL databases. Those unrelated
integration tests were not run for this offline analysis change.

## Remaining experimental work and limitations

There are no unimplemented code stubs in the newly added modules. Handout §3's
real experiment design, budgeted API runs, actual 48-B200-hour validation loss,
choice of final architecture/learning rate and `writeup.pdf` remain empirical
work; no observations or conclusions were fabricated. No remote submission was
made and no Git commit/push was run.

The runtime fit uses **actual reported runtime**, not `max_runtime_seconds`, which
is only a reservation/timeout cap. It assumes similar hardware, batch sizes,
evaluation settings and efficiencies. The loss model pools runs by N and D and
does not separately identify learning-rate or architecture effects. Candidate
ranking searches only the supplied configurations; it does not invent a new
architecture or claim a globally optimal one. The reported training fit errors
are in-sample diagnostics, not confidence intervals on extrapolation. Large
extrapolations and coefficients on fit boundaries need real follow-up evidence.

Approximate `C = 6ND` omits attention context/vocabulary overhead. It is used as a
FLOPs proxy for analysis, never to treat the service's runtime budget as FLOPs.

This is implementation/verification work, not an important model experiment.
The test data and bundled synthetic handout data do not establish new empirical
research findings. No user-level rules document was modified.


## 建议执行的 Git 命令

在本作业目录执行。以下仅为建议，本次没有执行暂存、提交或推送；未纳入无关 `.DS_Store` 文件。

```sh
git status --short
git diff
git add -- .gitignore README.md cs336_scaling/chinchilla_isoflops.py cs336_scaling/fit_scaling_laws.py cs336_scaling/scaling_data.py doc/implementation.md pyproject.toml test/offline/test_scaling_analysis.py uv.lock
git diff --cached
git commit -m 'feat: implement assignment3-scaling TODOs'
```
