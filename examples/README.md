# doppel demo

Use this fixture when you want to try doppel without touching private data.

The demo starts from [saas_accounts.csv](saas_accounts.csv), writes a 200-row
synthetic CSV, builds HTML and JSON quality reports, and infers an editable
schema file. Everything lands under `/tmp/doppel-demo`.

```bash
mkdir -p /tmp/doppel-demo

uv run doppel gen examples/saas_accounts.csv \
  --rows 200 \
  --output /tmp/doppel-demo/saas_accounts_synth.csv \
  --seed 7 \
  --text-policy hash

uv run doppel diff examples/saas_accounts.csv \
  /tmp/doppel-demo/saas_accounts_synth.csv \
  --html /tmp/doppel-demo/saas_accounts_report.html \
  --json /tmp/doppel-demo/saas_accounts_report.json \
  --top-n 8

uv run doppel schema infer examples/saas_accounts.csv \
  --output /tmp/doppel-demo/saas_accounts.schema.toml
```

Expected output:

```text
ok wrote 200 rows x 15 cols -> /tmp/doppel-demo/saas_accounts_synth.csv
quality | marginal=0.1188 | corr=0.0946 | dcr_p5=0.0373 | text_leaks=0
ok wrote HTML report -> /tmp/doppel-demo/saas_accounts_report.html
ok wrote JSON report -> /tmp/doppel-demo/saas_accounts_report.json
ok wrote schema -> /tmp/doppel-demo/saas_accounts.schema.toml
```

The metric values are from the checked-in fixture with `--seed 7`. Small changes can
come from dependency updates, but `text_leaks=0` should hold because the demo hashes
the text column.

What this demo covers:

- `account_id` is a unique key.
- `company_domain` is high-cardinality text. Plain sampling can copy these values,
  so the command uses `--text-policy hash`.
- `region` and `tier` exercise categorical columns.
- `num_active_seats_l90d <= num_seats` exercises count invariants.
- Nullable features and paired missingness flags exercise null handling.
- A binary target flag exercises boolean-like categorical data.
