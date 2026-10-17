"""Codex regression tests for offline analysis, with explicitly synthetic inputs.

Run: uv run --extra analysis python -m pytest --confcutdir=tests/offline tests/offline -q
The confcutdir avoids the unrelated service/PostgreSQL integration fixtures.
"""

from copy import deepcopy
import json

import numpy as np
import pytest

from cs336_scaling.chinchilla_isoflops import analyze, fit_power_law
from cs336_scaling.fit_scaling_laws import (
    fit_loss_surface,
    fit_runtime,
    predict_loss,
    rank_candidates,
)
from cs336_scaling.scaling_data import (
    Run,
    load_completed_experiments,
    nonembedding_parameters,
)


def config(layers=2, tokens=262144):
    """Codex test fixture: valid small 2026 config, without training a model."""
    return {
        "architecture_config": {
            "attention_bias": False,
            "head_dim": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": layers,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "tie_word_embeddings": False,
            "dtype": "float32",
            "vocab_size": 32768,
        },
        "optimizer_config": {"lr_scheduler": {"peak_value": 0.0003}},
        "train_batch_size": 2,
        "val_batch_size": 2,
        "n_evals": 4,
        "total_train_tokens": tokens,
        "max_runtime_seconds": 100,
        "model_seed": 0,
    }


def synthetic_runs():
    """Codex test fixture: exact known loss/runtime laws; never real observations."""
    runs = []
    for layers in (2, 4, 8):
        for tokens in (262144, 524288, 1048576):
            cfg = config(layers, tokens)
            n = nonembedding_parameters(cfg["architecture_config"])
            loss = 1.2 + 1.8 * (n / 100000) ** -0.4 + 1.1 * (tokens / 500000) ** -0.3
            seconds = 1e-9 * n**0.9 * tokens
            runs.append(Run(n, 6 * n * tokens, loss, tokens, seconds, cfg))
    return runs


def test_isoflops_minima_and_extrapolation(tmp_path):
    """Codex regression: minima, N/D exponents and output plots match an exact law."""
    runs = []
    for c in (6e6, 6e8, 6e10):
        n = c**0.5
        runs.extend([Run(n * 2, c, 4, c / (12 * n)), Run(n, c, 2, c / (6 * n))])
    result = analyze(runs, [1e23, 1e24], tmp_path)
    assert result["n_fit"]["exponent"] == pytest.approx(0.5)
    assert result["d_fit"]["exponent"] == pytest.approx(0.5)
    for row in result["predictions"]:
        assert 6 * row["parameters"] * row["tokens"] == pytest.approx(
            row["compute_budget"]
        )
    assert (tmp_path / "n_opt_vs_compute.png").stat().st_size > 1000
    assert (
        json.loads((tmp_path / "isoflops_fit.json").read_text())["observations"][0][
            "loss"
        ]
        == 2
    )


@pytest.mark.parametrize(
    "x,y", [([1, 1], [2, 3]), ([1, 2], [2, float("nan")]), ([0, 1], [2, 3]), ([1], [2])]
)
def test_power_law_rejects_invalid_observations(x, y):
    """Codex regression: reject mathematically invalid/underdetermined fits."""
    with pytest.raises(ValueError):
        fit_power_law(x, y)


def test_exact_nonembedding_parameter_count():
    """Codex regression: count GQA, SwiGLU, Q/K norms and optional biases."""
    architecture = config()["architecture_config"]
    architecture["num_key_value_heads"] = 1
    expected = 2 * (64 * (2 * 64 + 2 * 32) + 3 * 64 * 128 + 2 * 64 + 2 * 32) + 64
    assert nonembedding_parameters(architecture) == expected
    architecture["attention_bias"] = True
    assert nonembedding_parameters(architecture) == expected + 2 * (64 + 2 * 32 + 64)


def test_completed_loader_excludes_partial_and_uses_measured_time(tmp_path):
    """Codex regression: partial/failed status never becomes a final-loss datapoint."""
    row = {
        "experiment_id": 1,
        "training_config": config(),
        "status": {
            "status_type": "completed",
            "val_losses": [4.0, 3.0],
            "used_runtime_seconds": 12,
        },
    }
    partial = deepcopy(row)
    partial["status"]["status_type"] = "failed"
    path = tmp_path / "experiments.json"
    path.write_text(json.dumps([partial, row]))
    runs = load_completed_experiments(path)
    assert len(runs) == 1
    assert runs[0].final_loss == 3.0
    assert runs[0].runtime_seconds == 12
    assert runs[0].compute_budget == 6 * runs[0].parameters * runs[0].tokens


def test_loss_runtime_fitting_and_budgeted_candidates():
    """Codex regression: recover known laws and keep proposed configs in budget."""
    runs = synthetic_runs()
    loss_fit = fit_loss_surface(runs)
    runtime_fit = fit_runtime(runs)
    assert loss_fit["rmse_nats"] < 1e-6
    assert runtime_fit["gamma"] == pytest.approx(0.9)
    assert runtime_fit["log_rmse"] < 1e-10
    predicted = predict_loss(
        loss_fit,
        np.array([r.parameters for r in runs]),
        np.array([r.tokens for r in runs]),
    )
    assert predicted == pytest.approx([r.final_loss for r in runs], abs=1e-6)
    candidates = rank_candidates([config(2), config(8)], loss_fit, runtime_fit, 172800)
    assert len(candidates) == 2
    for row in candidates:
        cfg = row["training_config"]
        assert row["predicted_runtime_seconds"] <= 172800
        assert (
            cfg["total_train_tokens"] % (512 * cfg["train_batch_size"] * cfg["n_evals"])
            == 0
        )
        assert cfg["max_runtime_seconds"] == 172800
    assert (
        candidates[0]["predicted_final_loss"] <= candidates[1]["predicted_final_loss"]
    )


def test_loss_fit_rejects_insufficient_measurements():
    """Codex regression: do not invent a scaling law from an insufficient sweep."""
    with pytest.raises(ValueError, match="six runs"):
        fit_loss_surface(synthetic_runs()[:3])


def test_analysis_cli_with_synthetic_2026_json(tmp_path):
    """Codex regression: the direct CLI works from outside the project directory."""
    from pathlib import Path
    import subprocess
    import sys

    rows = [
        {
            "experiment_id": i,
            "training_config": run.training_config,
            "status": {
                "status_type": "completed",
                "val_losses": [run.final_loss + 0.5, run.final_loss],
                "used_runtime_seconds": run.runtime_seconds,
            },
        }
        for i, run in enumerate(synthetic_runs())
    ]
    path = tmp_path / "synthetic_experiments.json"
    path.write_text(json.dumps(rows))
    script = Path(__file__).resolve().parents[2] / "cs336_scaling/fit_scaling_laws.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--experiments",
            str(path),
            "--outdir",
            str(tmp_path / "out"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Local artifacts only" in result.stdout
    prediction = json.loads((tmp_path / "out/final_submission.local.json").read_text())
    assert prediction["training_config"]["max_runtime_seconds"] == 172800
    assert prediction["predicted_final_loss"] > 0
