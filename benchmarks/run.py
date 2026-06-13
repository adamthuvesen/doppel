#!/usr/bin/env python
"""Fidelity benchmark: run doppel end-to-end on the California Housing dataset.

This drives the *shipped* CLI (``doppel gen`` then ``doppel diff --json``) exactly the
way a user would, and writes a self-describing result artifact to
``benchmarks/results/housing.json``. That artifact backs the fidelity table at the top
of the README.

California Housing ships with scikit-learn (already a dependency). The first call
downloads ~400 KB and caches it under ``~/scikit_learn_data``; subsequent runs are
offline. We fetch it as NumPy arrays and build a Polars frame directly so pandas is
not required.

The metrics (avg_marginal, corr_frobenius, dcr_p5) are deterministic for a fixed
``--seed`` — re-running reproduces them exactly. ``fit_gen_seconds`` is wall-clock and
therefore machine-specific.

Usage::

    uv run python benchmarks/run.py                 # defaults: 20640 rows, seed 42
    uv run python benchmarks/run.py --seed 7 --rows 10000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.datasets import fetch_california_housing

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "housing.json"

DEFAULT_SEED = 42


def load_california_housing() -> pl.DataFrame:
    """Return California Housing (8 features + median-value target) as a Polars frame."""
    bunch: Any = fetch_california_housing(as_frame=False)
    columns = {name: bunch.data[:, i] for i, name in enumerate(bunch.feature_names)}
    columns[bunch.target_names[0]] = bunch.target
    return pl.DataFrame(columns)


def run_cli(*args: str) -> None:
    """Invoke the doppel CLI in-process via ``python -m doppel`` and fail loudly."""
    subprocess.run([sys.executable, "-m", "doppel", *args], check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic RNG seed.")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Synthetic rows to generate (default: match the source row count).",
    )
    args = parser.parse_args()

    source = load_california_housing()
    synth_rows = args.rows if args.rows is not None else source.height

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="doppel-bench-") as tmp:
        tmp_dir = Path(tmp)
        source_path = tmp_dir / "housing_source.parquet"
        synth_path = tmp_dir / "housing_synth.parquet"
        diff_path = tmp_dir / "diff.json"
        source.write_parquet(source_path)

        started = time.perf_counter()
        run_cli(
            "gen",
            str(source_path),
            "-n",
            str(synth_rows),
            "-o",
            str(synth_path),
            "--seed",
            str(args.seed),
        )
        fit_gen_seconds = time.perf_counter() - started

        run_cli("diff", str(source_path), str(synth_path), "--json", str(diff_path))
        report = json.loads(diff_path.read_text(encoding="utf-8"))

    # The diff labels are the temp file paths; rewrite them to stable, machine-independent
    # names so the committed artifact is fully reproducible for a fixed seed.
    report["real_label"] = "california_housing (real)"
    report["synth_label"] = f"doppel synth (seed={args.seed})"

    artifact = {
        "dataset": {
            "name": "California Housing",
            "source": "sklearn.datasets.fetch_california_housing",
            "rows": source.height,
            "columns": source.width,
        },
        "run": {
            "seed": args.seed,
            "synth_rows": synth_rows,
            "doppel_version": version("doppeldata"),
            "scikit_learn_version": version("scikit-learn"),
            "fit_gen_seconds": round(fit_gen_seconds, 2),
            "gen_command": (
                f"doppel gen housing_source.parquet -n {synth_rows} "
                f"-o housing_synth.parquet --seed {args.seed}"
            ),
            "diff_command": (
                "doppel diff housing_source.parquet housing_synth.parquet "
                "--json benchmarks/results/housing.json"
            ),
            "note": "Reproduce with: uv run python benchmarks/run.py",
        },
        "report": report,
    }
    RESULTS_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    avg_marginal = report["avg_marginal"]
    corr = report["correlations"]["frobenius_distance"]
    dcr_p5 = report["privacy"]["percentile_5"]
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO_ROOT)}")
    print(f"  rows (source/synth): {source.height} / {synth_rows}")
    print(f"  avg_marginal:        {avg_marginal:.4f}")
    print(f"  corr_frobenius:      {corr:.4f}")
    print(f"  dcr_p5:              {dcr_p5:.4f}")
    print(f"  fit+gen seconds:     {fit_gen_seconds:.2f}  (seed {args.seed})")


if __name__ == "__main__":
    main()
