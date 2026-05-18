# doppel demo

This folder contains a small customer-health CSV you can use to try doppel without
private data or setup.

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

What this demo exercises:

- A unique `org_id` key.
- High-cardinality domain text via `ultimate_domain`.
- Categorical columns such as `region` and `plan`.
- Integer count relationships such as `num_active_users_l90d <= num_users`.
- A nullable feature plus exact missingness flag.
- A binary target flag.

Use `--text-policy sample` to see the default highest-fidelity behavior, or
`hash`, `fake`, and `drop` when you want safer text output.
