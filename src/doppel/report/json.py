"""JSON rendering of a QualityReport — machine-readable form for CI / dashboards."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

from doppel.quality.aggregate import QualityReport


def to_json(report: QualityReport, *, indent: int = 2) -> str:
    payload = {
        "real_label": report.real_label,
        "synth_label": report.synth_label,
        "real_rows": report.real_rows,
        "synth_rows": report.synth_rows,
        "avg_marginal": report.avg_marginal,
        "marginals": [
            _without_non_finite({**asdict(m), "type": m.type.value}) for m in report.marginals
        ],
        "correlations": {
            "columns": report.correlations.columns,
            "frobenius_distance": _finite_or_none(report.correlations.frobenius_distance),
            "real_matrix": _without_non_finite(report.correlations.real_matrix),
            "synth_matrix": _without_non_finite(report.correlations.synth_matrix),
        },
        "privacy": _without_non_finite(asdict(report.privacy)),
    }
    return json.dumps(payload, indent=indent, default=str, allow_nan=False)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _without_non_finite(value: Any) -> Any:
    if isinstance(value, float):
        return _finite_or_none(value)
    if isinstance(value, dict):
        return {k: _without_non_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_non_finite(v) for v in value]
    return value
