"""Faker-backed generation for detected PII entity types.

Each `generate()` call builds and re-seeds a fresh `Faker` instance. We deliberately do
not cache the instance across calls: Faker's internal RNG advances on every draw, so a
cached instance would produce different values for the same seed on successive calls and
silently break the `--seed` determinism contract. `Faker()` construction is cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from doppel.synth.seed import Rng


def generate(entity_type: str, n: int, rng: Rng) -> list[str]:
    seed = int(rng.numpy.integers(0, 2**31 - 1))
    fake: Any = _build_faker(seed)
    generator = _GENERATORS.get(entity_type)
    if generator is None:
        # Unsupported entity types fall back to a stable opaque token.
        return [f"<{entity_type.lower()}_{i}>" for i in range(n)]
    return [generator(fake) for _ in range(n)]


def _build_faker(seed: int) -> Any:
    from faker import Faker

    f = Faker()
    f.seed_instance(seed)
    return f


_GENERATORS: dict[str, Callable[[Any], str]] = {
    "EMAIL_ADDRESS": lambda f: f.email(),
    "PERSON": lambda f: f.name(),
    "PHONE_NUMBER": lambda f: f.phone_number(),
    "LOCATION": lambda f: f.city(),
    "URL": lambda f: f.url(),
    "IP_ADDRESS": lambda f: f.ipv4(),
    "CREDIT_CARD": lambda f: f.credit_card_number(),
    "US_SSN": lambda f: f.ssn(),
    "IBAN_CODE": lambda f: f.iban(),
}
