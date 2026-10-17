"""Reproduce handout §2.1 from local data; migrated and validated by Codex.

macOS/Linux: uv run --extra analysis python cs336_scaling/chinchilla_isoflops.py
Windows CMD: uv run --extra analysis python cs336_scaling/chinchilla_isoflops.py
Options: --data defaults to the bundled data/isoflops_curves.json, --outdir to
runs/isoflops, --targets to 1e23 1e24 FLOPs. Outputs two PNGs and a JSON fit.
Paths supplied on the CLI are relative to the current working directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .scaling_data import Run, load_runs, positive, select_opt_points
else:
    from scaling_data import Run, load_runs, positive, select_opt_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fit_power_law(xs, ys) -> tuple[float, float]:
    """Codex migration: fit y = k x**a in log space for positive 1D arrays."""
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("Need two or more paired one-dimensional observations")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(x <= 0)
        or np.any(y <= 0)
    ):
        raise ValueError("Power-law inputs must be finite and positive")
    if np.unique(x).size < 2:
        raise ValueError("Need at least two distinct input scales")
    a, log_k = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_k)), float(a)


def predict_power_law(k: float, a: float, x):
    """Codex migration: evaluate the fitted power law at scalar/array scales."""
    return k * np.asarray(x, dtype=float) ** a


def plot_scaling(x, y, k: float, a: float, path: Path, ylabel: str, targets) -> None:
    """Codex migration: save observed optima and extrapolation through all targets."""
    import matplotlib

    matplotlib.use("Agg")  # CLI runs also work without an interactive display.
    import matplotlib.pyplot as plt

    curve = np.geomspace(min(x), max(max(x), max(targets)), 300)
    fig, axis = plt.subplots()
    axis.loglog(x, y, "o", label="Observed optimum")
    axis.loglog(curve, predict_power_law(k, a, curve), label=f"Fit: exponent {a:.4f}")
    axis.set(xlabel="Training compute (FLOPs)", ylabel=ylabel)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyze(runs: list[Run], targets: list[float], outdir: Path) -> dict:
    """Codex migration: select observed minima, fit N/D laws, save evidence."""
    targets = [positive(t, "target FLOPs") for t in targets]
    if not targets:
        raise ValueError("At least one target FLOPs budget is required")
    best = select_opt_points(runs)
    budgets = np.array(sorted(best))
    n_opt = np.array([best[c].parameters for c in budgets])
    d_opt = budgets / (6 * n_opt)
    k_n, a_n = fit_power_law(budgets, n_opt)
    k_d, a_d = fit_power_law(budgets, d_opt)
    result = {
        "method": "Lowest observed loss per IsoFLOPs budget; log-log least squares",
        "n_fit": {"k": k_n, "exponent": a_n},
        "d_fit": {"k": k_d, "exponent": a_d},
        "observations": [
            {
                "compute_budget": float(c),
                "parameters": best[c].parameters,
                "tokens": best[c].tokens,
                "loss": best[c].final_loss,
            }
            for c in budgets
        ],
        "predictions": [
            {
                "compute_budget": c,
                "parameters": float(predict_power_law(k_n, a_n, c)),
                "tokens": float(predict_power_law(k_d, a_d, c)),
            }
            for c in targets
        ],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    plot_scaling(
        budgets,
        n_opt,
        k_n,
        a_n,
        outdir / "n_opt_vs_compute.png",
        "Model size (parameters)",
        targets,
    )
    plot_scaling(
        budgets,
        d_opt,
        k_d,
        a_d,
        outdir / "d_opt_vs_compute.png",
        "Dataset size (tokens)",
        targets,
    )
    (outdir / "isoflops_fit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    """Codex migration: offline CLI; never submits or trains an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data/isoflops_curves.json"
    )
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "runs/isoflops")
    parser.add_argument("--targets", nargs="+", type=float, default=[1e23, 1e24])
    args = parser.parse_args()
    result = analyze(load_runs(args.data), args.targets, args.outdir)
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Artifacts: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
