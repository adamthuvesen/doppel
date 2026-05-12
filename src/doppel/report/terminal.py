"""Terminal rendering of a QualityReport via rich tables."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from doppel.quality.aggregate import QualityReport


def render(report: QualityReport, console: Console) -> None:
    console.print(
        f"[bold]doppel quality report[/]  "
        f"{report.real_label} ({report.real_rows} rows) vs. "
        f"{report.synth_label} ({report.synth_rows} rows)"
    )
    console.print()

    marg_table = Table(title="Marginals (lower = closer)", show_header=True)
    marg_table.add_column("Column")
    marg_table.add_column("Type")
    marg_table.add_column("Metric")
    marg_table.add_column("Score", justify="right")
    marg_table.add_column("Null%real", justify="right")
    marg_table.add_column("Null%synth", justify="right")
    for m in report.marginals:
        marg_table.add_row(
            m.column,
            m.type.value,
            m.metric,
            f"{m.value:.4f}",
            f"{m.null_rate_real * 100:.1f}",
            f"{m.null_rate_synth * 100:.1f}",
        )
    marg_table.caption = f"average marginal score: {report.avg_marginal:.4f}"
    console.print(marg_table)
    console.print()

    corr_table = Table(title="Correlation structure", show_header=False)
    corr_table.add_column("metric")
    corr_table.add_column("value", justify="right")
    corr_table.add_row("columns compared", str(len(report.correlations.columns)))
    corr_table.add_row(
        "Frobenius distance (normalised)",
        f"{report.correlations.frobenius_distance:.4f}",
    )
    console.print(corr_table)
    console.print()

    priv_table = Table(
        title="Privacy: distance-to-closest-record (heuristic; not differential privacy)",
        show_header=False,
    )
    priv_table.add_column("metric")
    priv_table.add_column("value", justify="right")
    priv_table.add_row(
        "rows compared", f"{report.privacy.n_synth} synth vs {report.privacy.n_real} real"
    )
    priv_table.add_row("features", str(report.privacy.n_features))
    priv_table.add_row("min DCR", f"{report.privacy.min_distance:.4f}")
    priv_table.add_row("DCR p5", f"{report.privacy.percentile_5:.4f}")
    priv_table.add_row("DCR p25", f"{report.privacy.percentile_25:.4f}")
    priv_table.add_row("DCR p50 (median)", f"{report.privacy.percentile_50:.4f}")
    priv_table.add_row("DCR mean", f"{report.privacy.mean_distance:.4f}")
    console.print(priv_table)
