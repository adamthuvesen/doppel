# doppel

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-doppeldata-orange)](https://pypi.org/project/doppeldata/)

Synthetic data generator for tabular datasets.

Point doppel at a tabular dataset (CSV, TSV, Parquet, JSON/NDJSON, Arrow/IPC) and it
generates a new dataset that matches the statistical fingerprint of the original:
distributions, correlations, null patterns, cardinality, and relational structure.
doppel is deterministic with `--seed` and reports quality/privacy heuristics, but it is
not a formal privacy system. The privacy posture is tiered:

- No differential privacy in v0.1 (no `--epsilon`).
- Detected PII (emails, phones, names, etc.) is regenerated via Faker when the optional
  `[pii]` extra is installed, so detected columns do not carry source values into output.
- Undetected free-text values are sampled with replacement and **may copy verbatim from
  the source**. Mitigate with `--text-policy hash|fake|drop` for any column that could
  identify the underlying record.
- Always run `doppel diff` before sharing synthetic output — the report includes a
  distance-to-closest-record (DCR) percentile and a per-column verbatim-text fraction.

See [SECURITY.md](SECURITY.md) for the full threat model.

**PII and `.doppel` artifacts**: `doppel fit` refuses any source data where
Presidio detects PII. The artifact format does not yet carry detection
metadata for round-trip regeneration, so use `doppel gen` (one-shot, regenerates
PII inline) when working with detected-PII columns. Storing + regenerating
detected PII from a saved artifact is on the v0.2 roadmap.

Useful for testing data pipelines, creating demo fixtures, and augmenting small datasets.

## Quickstart

```bash
# Install
uv tool install doppeldata

# Generate 1,000 synthetic rows from a real CSV with a quality check
doppel gen examples/saas_accounts.csv -n 1000 -o synth.csv --seed 1
# → progress bar during fit
# → ok wrote 1,000 rows x 15 cols -> synth.csv
# → quality | marginal=0.03 | corr=0.11 | dcr_p5=0.05 | text_leaks=1
# → tip: column 'company_domain' is 100% verbatim from source;
#        rerun with --text-policy hash to mitigate

# Quality gate for CI (exit 2 on any breach)
doppel diff real.parquet synth.parquet \
  --max-marginal 0.10 \
  --min-dcr-p5 0.05 \
  --fail-on-verbatim-text

# Environment + version check
doppel doctor
```

## Install

```bash
uv tool install doppeldata
```

(Distribution package: `doppeldata`. CLI binary and import name: `doppel`.)

## Usage

```bash
doppel gen sales.csv -n 100000 -o synth.csv
doppel gen big.parquet -n 100000 -o synth.parquet --fit-rows 25000
doppel gen customers.csv -n 10000 -o synth.csv --text-policy hash
doppel fit sales.parquet -o sales.doppel
doppel sample sales.doppel -n 1_000_000 -o synth.parquet
doppel diff sales.csv synth.csv -o report.html --sample-rows 50000
doppel schema infer sales.csv -o schema.toml
```

Run `doppel --help` for the full surface.

Helpful knobs for real datasets:

- `--fit-rows N` on `gen`/`fit` samples large source files before fitting.
- `--text-policy sample|hash|fake|drop` controls free-text output. Use `hash`,
  `fake`, or `drop` for identifying strings such as domains.
- `doppel diff --sample-rows N --top-n 20` keeps large quality checks fast and readable.
- doppel applies conservative soft repairs for exact missingness flags and count bounds
  learned from the source data, then prints a short repair summary.

## Recipes

- [examples/pytest_fixture/](examples/pytest_fixture/) — use a fitted
  `.doppel` artifact as a deterministic, session-scoped pytest fixture.
- [examples/dbt_seed/](examples/dbt_seed/) — generate dbt seeds from a
  real export and gate them with `doppel diff`.
- [examples/github-action/](examples/github-action/) — full PR-gate
  GitHub Actions workflow.

## CI gate

`doppel diff` accepts threshold flags and exits non-zero on breach, so it drops
straight into a CI pipeline:

```bash
doppel diff real.parquet synth.parquet \
  --max-marginal 0.10 \
  --min-dcr-p5 0.05 \
  --fail-on-verbatim-text \
  --json doppel-report.json
```

See [examples/github-action/](examples/github-action/) for a copy-pasteable
GitHub Actions workflow.

## Limitations (v0.1)

- Numeric integer columns now preserve source dtype, but other numeric
  subtypes may collapse — Float32 round-trips as Float64.
- Datetime modelling uses epoch-seconds only. Hour-of-day, day-of-week, and
  business-hours patterns are not preserved.
- Multi-table synthesis preserves FK referential integrity and per-table
  marginals, not cross-table correlations (e.g. "gold users place bigger
  orders" — opt-in `inherit_parent_features` flag on the roadmap).
- Free-text columns without detected PII are sampled with replacement and
  may copy verbatim from the source. Run `doppel diff` and use
  `--text-policy hash|fake|drop` for any identifying column.
- No differential privacy (`--epsilon` is v0.2 roadmap).

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
uv run pyright
```

## Security

`.doppel` artifact files contain pickled fitted models. doppel loads them through a
restricted unpickler that refuses anything outside an explicit allowlist (see
[SECURITY.md](SECURITY.md)), but you should still **only load `.doppel` files from
sources you trust**.

## License

MIT
