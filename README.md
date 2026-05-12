# doppel

> Synthetic data that looks real.

Point doppel at a tabular dataset (CSV, Parquet, JSON, Arrow, SQL) and it
generates a new dataset with the same statistical fingerprint — distributions,
correlations, null patterns, cardinality, and relational structure — without
any of the original rows.

Useful for testing data pipelines, sharing data you can't share, and
augmenting small datasets.

Not random noise. A statistical double.

## Status

🚧 Pre-release scaffold. Phase 0 (project skeleton, CLI plumbing) is in;
Phase 1 (single-table CART synthesis) is next.

## Install

```bash
uv tool install doppeldata
```

(Distribution package: `doppeldata`. CLI binary and import name: `doppel`.)

## Usage

```bash
doppel gen sales.csv -n 100000 -o synth.csv
doppel fit sales.parquet -o sales.doppel
doppel sample sales.doppel -n 1_000_000 -o synth.parquet
doppel diff sales.csv synth.csv -o report.html
doppel schema infer sales.csv -o schema.toml
```

Run `doppel --help` for the full surface.

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
