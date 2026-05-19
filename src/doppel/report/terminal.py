"""Terminal rendering of a QualityReport via rich tables."""

from __future__ import annotations

import math

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from doppel.quality.aggregate import QualityReport


def render(report: QualityReport, console: Console, *, top_n: int | None = None) -> None:
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
    marginals = sorted(
        report.marginals,
        key=lambda m: m.value if math.isfinite(m.value) else -1.0,
        reverse=True,
    )
    shown = marginals[:top_n] if top_n is not None else marginals
    for m in shown:
        marg_table.add_row(
            escape(m.column),
            m.type.value,
            m.metric,
            f"{m.value:.4f}",
            f"{m.null_rate_real * 100:.1f}",
            f"{m.null_rate_synth * 100:.1f}",
        )
    marg_table.caption = f"average marginal score: {report.avg_marginal:.4f}"
    if top_n is not None and len(report.marginals) > top_n:
        marg_table.caption += f" · showing worst {top_n} of {len(report.marginals)} columns"
    console.print(marg_table)

    text_warnings = [
        m for m in report.marginals if m.verbatim_rate is not None and m.verbatim_rate > 0
    ]
    if text_warnings:
        console.print(
            "[yellow]note:[/] TEXT columns are resampled from training values "
            "(sample-with-replacement):"
        )
        for m in text_warnings:
            vr = m.verbatim_rate
            assert vr is not None
            console.print(
                f"  [dim]{escape(m.column)}[/]  {vr:.1%} of synth values are verbatim copies"
            )

    if report.dtype_mismatches or report.invariant_issues:
        console.print()
        issue_table = Table(title="Likely issues", show_header=True)
        issue_table.add_column("Issue")
        issue_table.add_column("Detail")
        for issue in report.dtype_mismatches[:10]:
            issue_table.add_row(
                "dtype mismatch",
                escape(f"{issue.column}: real {issue.real_dtype}, synth {issue.synth_dtype}"),
            )
        for issue in report.invariant_issues[:10]:
            issue_table.add_row(
                "count invariant",
                escape(f"{issue.label}: {issue.synth_violations} synth violations"),
            )
        issue_table.caption = (
            f"{len(report.dtype_mismatches)} dtype mismatches, "
            f"{len(report.invariant_issues)} likely count invariant drifts"
        )
        console.print(issue_table)

    _render_calendar_fidelity(report, console)

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


def _render_calendar_fidelity(report: QualityReport, console: Console) -> None:
    """Compact per-column calendar fidelity table.

    Shown whenever any datetime column has resolvable calendar features (regardless of
    whether they were enabled at synthesis time — informative either way).
    """
    if not report.calendar_fidelity:
        return
    console.print()
    cal_table = Table(title="Calendar fidelity (KS, lower = closer)", show_header=True)
    cal_table.add_column("Column")
    cal_table.add_column("Feature")
    cal_table.add_column("KS", justify="right")
    for column_name, scores in report.calendar_fidelity.items():
        for score in scores:
            value = f"{score.value:.4f}" if math.isfinite(score.value) else "n/a"
            cal_table.add_row(escape(column_name), score.feature, value)
    console.print(cal_table)
