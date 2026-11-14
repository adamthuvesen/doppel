"""Seeded synthetic fixtures. No real data: invented names and `.example` domains only.

`python fixtures.py` rewrites the committed files in `fixtures/`. `verify` regenerates them in
memory on every run and fails when a committed file differs.
"""

import random
import sys
from datetime import date, timedelta
from pathlib import Path

SEED = 20261112
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
ACCOUNT_ROWS = 120
CUSTOMER_ROWS = 40
EMAIL_DOMAIN = "example.com"

FIRST_NAMES = (
    "Alva", "Bruno", "Cleo", "Dagny", "Emil", "Frida", "Gunnar", "Hedda", "Ivar", "Jonna",
    "Kasper", "Linnea", "Mats", "Nora", "Olle", "Pia", "Rurik", "Saga", "Tore", "Ulla",
)  # fmt: skip
LAST_NAMES = (
    "Almqvist", "Bergstrom", "Cedergren", "Dahlberg", "Ekdahl", "Forsell", "Granlund",
    "Hedlund", "Ingvarsson", "Jernberg", "Kullberg", "Lindqvist", "Mattsson", "Nordin",
)  # fmt: skip
PLANS = ("free", "pro", "team", "enterprise")
REGIONS = ("north", "south", "east", "west")
STATUSES = ("paid", "refunded", "pending")
COUNTRIES = ("SE", "NO", "DK", "FI")
WORDS = (
    "acorn",
    "birch",
    "cobalt",
    "delta",
    "ember",
    "fjord",
    "grove",
    "harbor",
    "iris",
    "juniper",
)

SCHEMA_TOML = """\
[tables.customers]
file = "customers.csv"
primary_key = "customer_id"

[tables.orders]
file = "orders.csv"
primary_key = "order_id"

[[foreign_keys]]
child_table = "orders"
child_column = "customer_id"
parent_table = "customers"
parent_column = "customer_id"
"""


def _csv(header: list[str], rows: list[list[object]]) -> str:
    lines = [",".join(header)]
    lines += [",".join(str(cell) for cell in row) for row in rows]
    return "\n".join(lines) + "\n"


def accounts_csv() -> str:
    """Account table with one free-text column (`company_domain`, unique per row)."""
    rng = random.Random(SEED)
    start = date(2025, 1, 1)
    rows: list[list[object]] = []
    for account_id in range(1, ACCOUNT_ROWS + 1):
        plan = rng.choices(PLANS, weights=[4, 3, 2, 1])[0]
        seats = rng.randint(1, 8) * {"free": 1, "pro": 3, "team": 10, "enterprise": 40}[plan]
        rows.append(
            [
                account_id,
                plan,
                rng.choice(REGIONS),
                seats,
                round(seats * rng.uniform(4.0, 9.0), 2),
                rng.choice(["true", "false"]),
                (start + timedelta(days=rng.randint(0, 600))).isoformat(),
                f"{rng.choice(WORDS)}-{account_id:03d}.example",
            ]
        )
    header = ["account_id", "plan", "region", "seats", "mrr", "active", "signup_date"]
    return _csv([*header, "company_domain"], rows)


def customers_csv() -> str:
    """Customers with a person name and an email: the PII the multi-table drive must regenerate."""
    rng = random.Random(SEED + 1)
    rows: list[list[object]] = []
    for customer_id in range(1, CUSTOMER_ROWS + 1):
        first, last = FIRST_NAMES[(customer_id - 1) % len(FIRST_NAMES)], rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first}.{last}.{customer_id}@{EMAIL_DOMAIN}".lower()
        rows.append([customer_id, name, email, rng.choice(COUNTRIES), rng.choice(PLANS)])
    return _csv(["customer_id", "full_name", "email", "country", "tier"], rows)


def orders_csv() -> str:
    rng = random.Random(SEED + 2)
    rows: list[list[object]] = []
    order_id = 1001
    for customer_id in range(1, CUSTOMER_ROWS + 1):
        for _ in range(rng.choices([0, 1, 2, 3, 4], weights=[1, 4, 3, 2, 1])[0]):
            rows.append(
                [order_id, customer_id, round(rng.uniform(5, 400), 2), rng.choice(STATUSES)]
            )
            order_id += 1
    return _csv(["order_id", "customer_id", "amount", "status"], rows)


def all_fixtures() -> dict[str, str]:
    return {
        "accounts.csv": accounts_csv(),
        "customers.csv": customers_csv(),
        "orders.csv": orders_csv(),
        "schema.toml": SCHEMA_TOML,
    }


def drifted_files() -> list[str]:
    """Names of committed fixture files that differ from what the generator writes."""
    return [
        name
        for name, text in all_fixtures().items()
        if not (FIXTURE_DIR / name).is_file() or (FIXTURE_DIR / name).read_text() != text
    ]


if __name__ == "__main__":
    FIXTURE_DIR.mkdir(exist_ok=True)
    for file_name, content in all_fixtures().items():
        (FIXTURE_DIR / file_name).write_text(content)
    print(f"wrote {len(all_fixtures())} files to {FIXTURE_DIR}", file=sys.stderr)
