# doppel for dbt seeds

Generate a synthetic CSV for a dbt project's `seeds/` directory; gate it with
`doppel diff` to catch distribution drift.

## One-shot synthesis

```bash
# Generate a 5,000-row synthetic seed from your real export
doppel gen exports/users.parquet \
  -n 5000 \
  -o my_dbt_project/seeds/users_synthetic.csv \
  --seed 42 \
  --text-policy hash       # mask identifying strings (domains, emails) deterministically
```

Then in `dbt_project.yml`:

```yaml
seeds:
  my_dbt_project:
    users_synthetic:
      +schema: synthetic
      +column_types:
        # match doppel's output dtypes for your warehouse — Parquet seeds preserve dtype
        # but CSV seeds need explicit hints for date / numeric columns
        signup_date: date
        ltv_usd: numeric(12,2)
```

```bash
dbt seed --select users_synthetic
```

## Quality gate

Compare the synthetic seed against the real export in CI; exit non-zero on breach.

```bash
doppel diff exports/users.parquet my_dbt_project/seeds/users_synthetic.csv \
  --sample-rows 20000 \
  --max-marginal 0.10 \
  --min-dcr-p5 0.05 \
  --fail-on-verbatim-text \
  --json doppel-seed-quality.json
```

See [examples/github-action/](../github-action/) for the full workflow.

## Cadence

- Local dev: regenerate when the source schema changes.
- CI: regenerate on the branch that updates the export, then `doppel diff` to gate
  the merge.

## Parquet vs CSV

Prefer `.parquet` if your warehouse supports Parquet seeds — dtype round-trip is
exact and `doppel diff` numbers are cleaner.
