"""Fit local 2026 experiment observations and rank explicit candidate configs.

Codex implementation informed by the reference scaling-law analysis. This file
uses the 2026 schema and measured seconds, not the reference's 2025 FLOPs API.
It never sends requests, spends API budget, or generates experimental results.

macOS/Linux:
  uv run --extra analysis python cs336_scaling/fit_scaling_laws.py --experiments experiments.json --candidates candidates.json
Windows CMD:
  uv run --extra analysis python cs336_scaling/fit_scaling_laws.py --experiments experiments.json --candidates candidates.json
--experiments is a required local /experiments JSON array. --candidates is an
optional JSON array of complete TrainingConfig dictionaries; if omitted, use
unique observed configs. --target-seconds defaults to 172800 (48 B200 hours);
--outdir defaults to runs/scaling. Outputs scaling_fit.json and a LOCAL
final_submission.local.json, never a submission to the server.

Assumptions: loss = E + A*(N/N0)^(-alpha) + B*(D/D0)^(-beta), and
seconds/token = k*N^gamma. N excludes both token input/output matrices.
The runtime fit assumes comparable hardware/evaluation settings. Hyperparameters
and architecture details other than N are not modeled; inspect fit diagnostics
and perform real validation before trusting an extrapolated configuration.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

if __package__:
    from .chinchilla_isoflops import fit_power_law
    from .scaling_data import (
        Run,
        load_completed_experiments,
        nonembedding_parameters,
        positive,
    )
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cs336_scaling.chinchilla_isoflops import fit_power_law
    from cs336_scaling.scaling_data import (
        Run,
        load_completed_experiments,
        nonembedding_parameters,
        positive,
    )

from cs336_scaling.schemas import TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def predict_loss(fit: dict, parameters, tokens):
    """Codex implementation: predict cross entropy (nats) from positive N and D."""
    return (
        fit["floor"]
        + fit["a"] * (np.asarray(parameters) / fit["n0"]) ** -fit["alpha"]
        + fit["b"] * (np.asarray(tokens) / fit["d0"]) ** -fit["beta"]
    )


def fit_loss_surface(runs: list[Run]) -> dict:
    """Codex implementation: multistart bounded least squares of a Chinchilla law.

    Requires varied N and D (at least six measurements) to avoid fitting five
    coefficients to an underdetermined sweep. RMSE is in validation-loss nats;
    it is in-sample residual error, not an extrapolation confidence interval.
    """
    n = np.array([r.parameters for r in runs])
    d = np.array([r.tokens for r in runs])
    loss = np.array([r.final_loss for r in runs])
    if len(runs) < 6 or len(np.unique(n)) < 3 or len(np.unique(d)) < 3:
        raise ValueError(
            "Loss fit needs at least six runs with at least three distinct N and D values"
        )
    coordinates = np.column_stack((np.ones(len(n)), np.log(n), np.log(d)))
    if np.linalg.matrix_rank(coordinates) < 3:
        raise ValueError(
            "N and D must vary independently; a single IsoFLOPs profile is insufficient"
        )
    n0, d0 = float(np.exp(np.log(n).mean())), float(np.exp(np.log(d).mean()))
    log_n, log_d = np.log(n / n0), np.log(d / d0)

    def residual(theta):
        """Codex implementation: normalized loss residuals for stable optimization."""
        e, log_a, log_b, alpha, beta = theta
        return e + np.exp(log_a - alpha * log_n) + np.exp(log_b - beta * log_d) - loss

    solutions = []
    for exponent in (0.2, 0.5, 0.9):
        start = [
            loss.min() / 2,
            math.log(loss.min() / 4),
            math.log(loss.min() / 4),
            exponent,
            exponent,
        ]
        solution = least_squares(
            residual,
            start,
            bounds=([0, -30, -30, 0.01, 0.01], [loss.min(), 30, 30, 2, 2]),
            max_nfev=10000,
        )
        if solution.success and np.linalg.matrix_rank(solution.jac) == 5:
            solutions.append(solution)
    if not solutions:
        raise ValueError(
            "Loss fit failed or is unidentifiable; collect more independent N/D observations"
        )
    best = min(solutions, key=lambda s: float(np.sum(s.fun**2)))
    e, log_a, log_b, alpha, beta = best.x
    return {
        "floor": float(e),
        "a": float(np.exp(log_a)),
        "b": float(np.exp(log_b)),
        "alpha": float(alpha),
        "beta": float(beta),
        "n0": n0,
        "d0": d0,
        "rmse_nats": float(np.sqrt(np.mean(best.fun**2))),
        "jacobian_condition": float(np.linalg.cond(best.jac)),
        "boundary_parameters": [int(i) for i in np.flatnonzero(best.active_mask)],
    }


def fit_runtime(runs: list[Run]) -> dict:
    """Codex implementation: fit measured seconds/token = k*N**gamma.

    Report residuals because startup/evaluation time, batch size and accelerator
    efficiency are not captured by this simple model. Runtime is never derived
    from max_runtime_seconds, which is only a timeout/reservation cap.
    """
    n = np.array([r.parameters for r in runs])
    seconds_per_token = np.array([r.runtime_seconds / r.tokens for r in runs])
    k, gamma = fit_power_law(n, seconds_per_token)
    fitted = k * n**gamma
    return {
        "k": k,
        "gamma": gamma,
        "log_rmse": float(
            np.sqrt(np.mean((np.log(fitted) - np.log(seconds_per_token)) ** 2))
        ),
    }


def rank_candidates(
    configs: list[dict], loss_fit: dict, runtime_fit: dict, target_seconds: float
) -> list[dict]:
    """Codex implementation: assign feasible whole-evaluation token budgets.

    Candidate configs are schema-validated before and after token adjustment.
    Token counts respect sequence length, batch size, n_evals and the 500B cap.
    Every prediction is an estimate; no config is automatically submitted.
    """
    target_seconds = positive(target_seconds, "target_seconds")
    if target_seconds < 1:
        raise ValueError("target_seconds must be at least one second")
    ranked = []
    for candidate in configs:
        config = TrainingConfig.model_validate(candidate)
        n = nonembedding_parameters(config.architecture_config.model_dump())
        per_token = positive(
            runtime_fit["k"] * n ** runtime_fit["gamma"], "predicted seconds/token"
        )
        quantum = config.seq_len * config.train_batch_size * config.n_evals
        tokens = (
            int(min(500_000_000_000, target_seconds / per_token)) // quantum * quantum
        )
        if tokens < quantum:
            continue
        updated = deepcopy(config.model_dump(mode="json"))
        updated["total_train_tokens"] = tokens
        updated["max_runtime_seconds"] = target_seconds
        validated = TrainingConfig.model_validate(updated)
        loss = positive(float(predict_loss(loss_fit, n, tokens)), "predicted loss")
        ranked.append(
            {
                "training_config": validated.model_dump(mode="json"),
                "parameters": n,
                "predicted_runtime_seconds": tokens * per_token,
                "predicted_final_loss": loss,
            }
        )
    if not ranked:
        raise ValueError(
            "None of the candidates can fit one complete evaluation chunk in this time budget"
        )
    return sorted(ranked, key=lambda row: row["predicted_final_loss"])


def main() -> None:
    """Codex implementation: read local inputs and write fits/candidate estimates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--target-seconds", type=float, default=48 * 3600)
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "runs/scaling")
    args = parser.parse_args()
    runs = load_completed_experiments(args.experiments)
    if args.candidates:
        configs = json.loads(args.candidates.read_text(encoding="utf-8"))
        if not isinstance(configs, list) or not configs:
            raise ValueError(
                "Candidates must be a nonempty JSON array of full TrainingConfig objects"
            )
    else:
        configs = list(
            {
                json.dumps(r.training_config, sort_keys=True): r.training_config
                for r in runs
            }.values()
        )
    loss_fit, runtime_fit = fit_loss_surface(runs), fit_runtime(runs)
    ranked = rank_candidates(configs, loss_fit, runtime_fit, args.target_seconds)
    report = {
        "completed_runs": len(runs),
        "loss_fit": loss_fit,
        "runtime_fit": runtime_fit,
        "target_seconds": args.target_seconds,
        "candidates": ranked,
        "observed_ranges": {
            "parameters": [
                min(r.parameters for r in runs),
                max(r.parameters for r in runs),
            ],
            "tokens": [min(r.tokens for r in runs), max(r.tokens for r in runs)],
        },
        "limitations": [
            "In-sample fit errors do not bound extrapolation error.",
            "The loss model does not identify learning-rate or architecture effects at fixed N.",
            "Runtime assumes comparable hardware, batch size and evaluation settings.",
            "Candidate ranking is restricted to supplied or previously observed configurations.",
        ],
    }
    submission = {
        "training_config": ranked[0]["training_config"],
        "predicted_final_loss": ranked[0]["predicted_final_loss"],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("scaling_fit.json", report),
        ("final_submission.local.json", submission),
    ):
        (args.outdir / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "completed_runs": len(runs),
                "loss_fit": loss_fit,
                "runtime_fit": runtime_fit,
                "best_prediction": ranked[0],
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(f"Local artifacts only: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
