"""Offline scaling inputs, adapted by Codex from the 2025 reference for the 2026 API.

All sizes are scalar counts; losses are mean validation cross entropy in nats.
No API calls or training are performed. Imported by the two analysis scripts.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Run:
    """One completed observation: parameter count N, FLOPs C, loss L, and tokens D."""

    parameters: float
    compute_budget: float
    final_loss: float
    tokens: float
    runtime_seconds: float | None = None
    training_config: dict | None = None


def positive(value: float, name: str) -> float:
    """Codex implementation: reject nonfinite/nonpositive scalar observations."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive; got {value}")
    return value


def nonembedding_parameters(architecture: dict) -> int:
    """Codex implementation: count 2026 transformer parameters excluding token I/O.

    Includes GQA attention projections, optional attention bias, SwiGLU weights,
    two layer RMSNorms, per-head Q/K RMSNorms, and the final RMSNorm. Excludes
    both the input embedding and output vocabulary projection, tied or untied.
    This is a parameter count, not an exact training FLOPs measurement.
    """
    h = int(positive(architecture["hidden_size"], "hidden_size"))
    f = int(positive(architecture["intermediate_size"], "intermediate_size"))
    layers = int(positive(architecture["num_hidden_layers"], "num_hidden_layers"))
    q = int(architecture["num_attention_heads"]) * int(architecture["head_dim"])
    kv = int(architecture["num_key_value_heads"]) * int(architecture["head_dim"])
    attention = h * (2 * q + 2 * kv)
    if architecture["attention_bias"]:
        attention += q + 2 * kv + h
    norms = 2 * h + 2 * int(architecture["head_dim"])
    return layers * (attention + 3 * h * f + norms) + h


def load_runs(path: str | Path) -> list[Run]:
    """Codex migration: load the handout JSON array; infer D = C / (6N)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a nonempty JSON array of IsoFLOPs runs")
    runs = []
    for row in data:
        n = positive(row["parameters"], "parameters")
        c = positive(row["compute_budget"], "compute_budget")
        runs.append(Run(n, c, positive(row["final_loss"], "final_loss"), c / (6 * n)))
    return runs


def select_opt_points(runs: Iterable[Run]) -> dict[float, Run]:
    """Codex migration: select the lowest observed loss at each shared FLOPs budget."""
    best: dict[float, Run] = {}
    for run in runs:
        if (
            run.compute_budget not in best
            or run.final_loss < best[run.compute_budget].final_loss
        ):
            best[run.compute_budget] = run
    if not best:
        raise ValueError("No observations available")
    return best


def load_completed_experiments(path: str | Path) -> list[Run]:
    """Codex implementation: read a local 2026 /experiments response JSON array.

    Queued, running and failed runs (including timeout partial losses) are
    excluded. The final validation loss is status.val_losses[-1]. C = 6ND is
    only an approximate FLOPs proxy; runtime uses the measured API value.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected the JSON array returned by /experiments")
    runs = []
    for row in data:
        status = row["status"]
        if status["status_type"] != "completed":
            continue
        config = row["training_config"]
        losses = status["val_losses"]
        if not losses:
            raise ValueError(
                f"Completed experiment {row.get('experiment_id')} has no loss"
            )
        n = nonembedding_parameters(config["architecture_config"])
        d = positive(config["total_train_tokens"], "total_train_tokens")
        runs.append(
            Run(
                n,
                6 * n * d,
                positive(losses[-1], "val_loss"),
                d,
                positive(status["used_runtime_seconds"], "runtime_seconds"),
                config,
            )
        )
    if not runs:
        raise ValueError("No completed experiments with validation losses found")
    return runs
