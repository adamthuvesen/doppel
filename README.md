# doppel

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-doppeldata-orange)](https://pypi.org/project/doppeldata/)

Generates synthetic tabular data that matches the statistical fingerprint of a real
dataset — distributions, correlations, null patterns, cardinality, and relational
structure. Reads CSV, TSV, Parquet, JSON/NDJSON, Arrow/IPC. Deterministic given a `--seed`.

## Privacy

doppel is not a formal privacy system. The posture:

- No differential privacy in v0.1 (no `--epsilon`).
- Detected PII (emails, phones, names) is regenerated via Faker when the optional
  `[pii]` extra is installed.
- Undetected free-text is sampled with replacement and **may copy verbatim from
  the source**. Use `--text-policy hash|fake|drop` for any identifying column.
- Always run `doppel diff` before sharing — it reports a distance-to-closest-record
  (DCR) percentile and a per-column verbatim-text fraction.

`doppel fit` refuses any source where Presidio detects PII (the artifact format
doesn't yet carry detection metadata for round-trip regeneration; v0.2 roadmap).
Use `doppel gen` for one-shot PII regeneration.

See [SECURITY.md](SECURITY.md) for the threat model.

## Quickstart

```bash
uv tool install doppeldata

doppel gen examples/saas_accounts.csv -n 1000 -o synth.csv --seed 1
# ok wrote 1000 rows x 15 cols -> synth.csv
# quality | marginal=0.0338 | corr=0.1062 | dcr_p5=0.0507 | text_leaks=1
# tip: column 'company_domain' is 100% verbatim from source;
#      rerun with --text-policy hash to mitigate

doppel diff examples/saas_accounts.csv synth.csv \
  --max-marginal 0.10 --min-dcr-p5 0.05 --fail-on-verbatim-text
# exit 2 on threshold breach

doppel doctor   # environment + dep versions
```

> PyPI distribution: `doppeldata`. CLI binary and import name: `doppel`.

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

Useful flags:

- `--fit-rows N` on `gen`/`fit` samples large source files before fitting.
  `gen` auto-caps to `min(rows*5, 100k)` when source > 100k rows; pass `--fit-rows 0`
  to disable.
- `--text-policy sample|hash|fake|drop` — use `hash|fake|drop` for identifying strings.
- `doppel diff --sample-rows N --top-n 20` speeds up large diffs.
- `--max-dcr-rows 50000` caps the nearest-neighbour search for the DCR percentile.
- Soft repairs (missingness flags, count bounds) are applied after sampling; a
  summary prints if any fired.

## CI gate

`doppel diff` exits non-zero on threshold breach:

```bash
doppel diff real.parquet synth.parquet \
  --max-marginal 0.10 \
  --min-dcr-p5 0.05 \
  --fail-on-verbatim-text \
  --json doppel-report.json
```

Exit codes: `0` pass, `2` threshold breach.

## Recipes

- [examples/pytest_fixture/](examples/pytest_fixture/) — fitted `.doppel` as a
  session-scoped pytest fixture.
- [examples/dbt_seed/](examples/dbt_seed/) — synth CSV for dbt seeds, gated by
  `doppel diff`.
- [examples/github-action/](examples/github-action/) — PR-gate GitHub Actions
  workflow.

## Limitations (v0.1)

- Datetime modelling uses epoch-seconds plus a small set of calendar features
  (`hour`, `dow`, `month` by default — `dow`, `month` for `pl.Date`) as
  predictors for downstream columns. Override per-column in `schema.toml` via
  `calendar_features = false` or `calendar_features = ["hour", "dow"]`.
  Sub-second precision is still dropped.
- Multi-table preserves FK integrity and per-table marginals, not cross-table
  correlations. `inherit_parent_features` schema flag is parsed but not yet
  wired (v0.2).
- Undetected free-text may copy verbatim from source; see Privacy above.
- No differential privacy (v0.2).

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
uv run pyright
```

## Security

`.doppel` artifacts contain pickled fitted models. Loading goes through a restricted
unpickler with an explicit allowlist (see [SECURITY.md](SECURITY.md)). Still: **only
load `.doppel` files from sources you trust.**

## License

MIT
