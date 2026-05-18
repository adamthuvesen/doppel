# doppel for dbt seeds

Generate a synthetic CSV that drops straight into a dbt project's
`seeds/` directory, then gate the seed with `doppel diff` so you know
when the synthetic distribution has drifted from the real source.

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

## Quality gate (optional but recommended)

Run `doppel diff` as a CI step that compares the synthetic seed against
the real export. The job exits non-zero if any quality threshold is
breached — surfacing schema drift, distribution shifts, or accidental
data leakage before they hit your pipeline.

```bash
doppel diff exports/users.parquet my_dbt_project/seeds/users_synthetic.csv \
  --sample-rows 20000 \
  --max-marginal 0.10 \
  --min-dcr-p5 0.05 \
  --fail-on-verbatim-text \
  --json doppel-seed-quality.json
```

See [examples/github-action/](../github-action/) for a full workflow.

## Recommended cadence

- **Local dev**: regenerate once when the source schema changes.
- **CI**: regenerate on the same branch that updates the real export,
  then run `doppel diff` to gate the merge.
- **Refresh policy**: at most weekly — synthetic seeds should be cheap
  to regenerate but expensive churn isn't worth it for stable tables.

## What about Parquet seeds?

If your warehouse / dbt setup supports Parquet seeds (most don't yet,
but external tables do), prefer `.parquet` over `.csv`: dtype round-trip
is exact, and `doppel diff` against the same Parquet file is metrically
cleaner.
