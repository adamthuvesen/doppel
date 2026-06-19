# doppel demo

A small SaaS-accounts CSV to try doppel against without using real data.

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

Exercised features:

- Unique `account_id` key.
- High-cardinality text (`company_domain`) — defaults will leak; use `--text-policy
  hash|fake|drop` for safer output.
- Categorical columns (`region`, `tier`).
- Integer count invariants (`num_active_seats_l90d <= num_seats`).
- Nullable feature + paired missingness flag.
- Binary target flag.
