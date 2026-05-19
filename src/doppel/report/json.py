"""JSON rendering of a QualityReport — machine-readable form for CI / dashboards."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict
from typing import Any

import numpy as np

from doppel.quality.aggregate import QualityReport


def to_json(
    report: QualityReport,
    *,
    indent: int = 2,
    thresholds: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
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
        "dtype_mismatches": [asdict(issue) for issue in report.dtype_mismatches],
        "invariant_issues": [asdict(issue) for issue in report.invariant_issues],
    }
    if thresholds is not None:
        payload["thresholds"] = thresholds
    return json.dumps(payload, indent=indent, default=_json_default, allow_nan=False)


def _json_default(value: Any) -> Any:
    """Explicit converter for types stdlib json can't handle natively.

    Fails loudly on anything outside this list rather than silently stringifying — a
    silent fallback would produce e.g. quoted numbers and break downstream consumers.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


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
