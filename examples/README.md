# doppel demo

A small customer-health CSV to try doppel against without using real data.

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

Exercised features:

- Unique `org_id` key.
- High-cardinality text (`company_domain`) — defaults will leak; use `--text-policy
  hash|fake|drop` for safer output.
- Categorical columns (`region`, `plan`).
- Integer count invariants (`num_active_seats_l90d <= num_seats`).
- Nullable feature + paired missingness flag.
- Binary target flag.
