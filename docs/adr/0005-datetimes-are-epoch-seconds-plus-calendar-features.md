# Datetimes are epoch seconds, with calendar features as predictors

Status: accepted (2026-11-12)

CART never sees a raw timestamp. `decompose` in `src/doppel/schema/datetime.py` turns a datetime column into `Int64` epoch seconds before the fit, and `recompose` rebuilds the original Polars type, including the time zone, at output. Sub-second precision is dropped.

A datetime also feeds other columns through calendar features (hour, day of week, month, and others), added to the feature matrix as predictors only (`src/doppel/schema/calendar_feature.py`). They are never targets and never appear in the output. This keeps patterns such as "orders peak on Friday at 9am" without modelling a timestamp as free text. A user can turn them off per column in `schema.toml` with `calendar_features = false`. Feature columns start with the reserved prefix `__dt_`, and a source column with that prefix is refused.

## Considered options

- Feed nanoseconds to the trees. Rejected: the split thresholds are meaningless at that scale, and float32 routing loses the low digits.
- Model only the date and drop the time. Rejected: it loses time-of-day patterns.

## Consequences

- Toggling calendar features can change the output, because the feature matrix changes. Each setting is still deterministic under `--seed`.
- A new modelling path for datetimes must go through `decompose` and `recompose`.
