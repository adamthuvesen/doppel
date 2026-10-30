"""Restricted pickle unpickling — refuse anything outside an exact allowlist.

Naked `pickle.loads` on an untrusted `.doppel` artifact is RCE: an attacker can ship a
`__reduce__` payload that runs arbitrary code (`os.system`, `subprocess`, ...) before any
type or value check runs. This module's `RestrictedUnpickler` overrides `find_class` and
refuses every `(module, name)` pair that is not listed below.

The allowlist is exact. Module prefixes are not enough: `numpy`, `sklearn`, and `scipy`
expose callables such as `numpy.load(..., allow_pickle=True)` and
`numpy.testing._private.utils.runstring`, and protocol 4 resolves a dotted name such as
`("sklearn", "os.system")` through attribute access. A dotted name is always refused.

The set was derived by recording every global that real artifacts reference (the test
suite plus `doppel fit` on `examples/saas_accounts.csv` and a frame covering every
supported dtype). An artifact that needs a class outside this set fails with a clear
error naming the class. That is the intended behaviour: a loud refusal, not a silent RCE.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

_ALLOWED_BUILTINS: frozenset[str] = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "bool",
        "int",
        "float",
        "str",
        "bytes",
        "bytearray",
        "complex",
        "NoneType",
        "slice",
    }
)

_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        *(("builtins", name) for name in _ALLOWED_BUILTINS),
        *(("collections", name) for name in ("OrderedDict", "defaultdict", "deque", "Counter")),
        ("_codecs", "encode"),
        ("doppel.schema.calendar_feature", "CalendarFeature"),
        ("doppel.schema.datetime", "CalendarFeature"),  # artifacts saved before the move
        ("doppel.schema.types", "Column"),
        ("doppel.schema.types", "ColumnType"),
        ("doppel.synth.cart", "CartSynthesizer"),
        ("doppel.synth.cart", "ColumnFitInfo"),
        ("doppel.synth.cart", "RepairSummary"),
        ("doppel.synth.cart", "_ColumnSynth"),
        ("doppel.synth.cart", "_CountBound"),
        ("doppel.synth.cart", "_Encoder"),
        ("doppel.synth.cart", "_MissingFlag"),
        ("doppel.synth.cart_repair", "RepairSummary"),
        ("doppel.synth.cart_repair", "CountBound"),
        ("doppel.synth.cart_repair", "MissingFlag"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        *(
            (f"numpy.{core}.{module}", name)
            for core in ("_core", "core")
            for module, name in (
                ("multiarray", "_reconstruct"),
                ("multiarray", "scalar"),
                ("numeric", "_frombuffer"),
            )
        ),
        ("sklearn.tree._classes", "DecisionTreeClassifier"),
        ("sklearn.tree._classes", "DecisionTreeRegressor"),
        ("sklearn.tree._tree", "Tree"),
        ("polars.series.series", "Series"),
        *(
            ("polars.datatypes.classes", name)
            for name in (
                "Boolean",
                "Categorical",
                "Categories",
                "Date",
                "Datetime",
                "Enum",
                "Float32",
                "Float64",
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "Int128",
                "Null",
                "String",
                "UInt8",
                "UInt16",
                "UInt32",
                "UInt64",
            )
        ),
    }
)


class UnsafeArtifactError(pickle.UnpicklingError):
    """Raised when an artifact references a class outside the doppel allowlist."""


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if "." not in name and (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise UnsafeArtifactError(
            f"refusing to unpickle {module}.{name}: not in doppel artifact allowlist. "
            "If this is a legitimate doppel artifact from a newer release, upgrade. "
            "If this is from an unknown source, do not load it."
        )


def safe_loads(blob: bytes) -> Any:
    return RestrictedUnpickler(io.BytesIO(blob)).load()
