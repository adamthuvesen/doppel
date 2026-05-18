# doppel

Synthetic data generator for tabular datasets.

Point doppel at a tabular dataset (CSV, TSV, Parquet, JSON/NDJSON, Arrow/IPC) and it
generates a new dataset that matches the statistical fingerprint of the original:
distributions, correlations, null patterns, cardinality, and relational structure.
doppel is deterministic with `--seed` and reports quality/privacy heuristics, but it is
not a formal privacy system: detected PII can be regenerated when the optional `[pii]`
extra is installed, while undetected free-text values may be copied from the source.
Use `doppel diff` before sharing synthetic output.

Useful for testing data pipelines, creating demo fixtures, and augmenting small datasets.

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
