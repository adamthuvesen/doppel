"""JSON rendering of a QualityReport — machine-readable form for CI / dashboards."""

from __future__ import annotations

import json
from dataclasses import asdict

from doppel.quality.aggregate import QualityReport


def to_json(report: QualityReport, *, indent: int = 2) -> str:
    payload = {
        "real_label": report.real_label,
        "synth_label": report.synth_label,
        "real_rows": report.real_rows,
        "synth_rows": report.synth_rows,
        "avg_marginal": report.avg_marginal,
        "marginals": [{**asdict(m), "type": m.type.value} for m in report.marginals],
        "correlations": {
            "columns": report.correlations.columns,
            "frobenius_distance": report.correlations.frobenius_distance,
            "real_matrix": report.correlations.real_matrix,
            "synth_matrix": report.correlations.synth_matrix,
        },
        "privacy": asdict(report.privacy),
    }
    return json.dumps(payload, indent=indent, default=str)
