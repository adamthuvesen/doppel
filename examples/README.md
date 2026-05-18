# doppel demo

A small customer-health CSV to try doppel against without using real data.

```bash
mkdir -p /tmp/doppel-demo

uv run doppel gen examples/customer_health.csv \
  --rows 200 \
  --output /tmp/doppel-demo/customer_health_synth.csv \
  --seed 7 \
  --text-policy hash

uv run doppel diff examples/customer_health.csv \
  /tmp/doppel-demo/customer_health_synth.csv \
  --html /tmp/doppel-demo/customer_health_report.html \
  --json /tmp/doppel-demo/customer_health_report.json \
  --top-n 8

uv run doppel schema infer examples/customer_health.csv \
  --output /tmp/doppel-demo/customer_health.schema.toml
```

Exercised features:

- Unique `org_id` key.
- High-cardinality text (`ultimate_domain`) — defaults will leak; use `--text-policy
  hash|fake|drop` for safer output.
- Categorical columns (`region`, `plan`).
- Integer count invariants (`num_active_users_l90d <= num_users`).
- Nullable feature + paired missingness flag.
- Binary target flag.
