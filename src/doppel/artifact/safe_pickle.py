"""Restricted pickle unpickling — refuse anything outside an explicit allowlist.

Naked `pickle.loads` on an untrusted `.doppel` artifact is RCE: an attacker can ship a
`__reduce__` payload that runs arbitrary code (`os.system`, `subprocess`, ...) before any
type or value check runs. This module's `RestrictedUnpickler` overrides `find_class` and
refuses anything that isn't in our explicit allowlist of modules / classes we actually
need to round-trip a fitted `CartSynthesizer`.

The allowlist covers:
  - `builtins.{dict,list,tuple,set,frozenset,bool,int,float,str,bytes,bytearray,
              complex,type,object,NoneType,slice}`
  - `collections.{OrderedDict,defaultdict,deque,Counter}`
  - the narrow set of `doppel` classes that make up a fitted artifact
  - any class under `numpy.*` (needed for ndarrays, dtypes, scalar types)
  - any class under `sklearn.*` (needed for DecisionTree* and their internal Tree class)
  - any class under `polars.*` (needed for datetime dtype round-trip)
  - `scipy.*` (sklearn pulls in scipy.sparse and related types via dependency)

If you load a `.doppel` file produced by an older/newer doppel version and it pulls in a
class outside this allowlist, the load will refuse with a clear error message naming the
offending class. That's the intended behaviour — better a loud refusal than a silent RCE.
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
        "type",
        "object",
        "NoneType",
        "slice",
    }
)

_ALLOWED_COLLECTIONS: frozenset[str] = frozenset({"OrderedDict", "defaultdict", "deque", "Counter"})

_ALLOWED_DOPPEL_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("doppel.schema.datetime", "CalendarFeature"),
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
    }
)

_ALLOWED_MODULE_PREFIXES: tuple[str, ...] = (
    "numpy",
    "sklearn",
    "polars",
    "scipy",
)


class UnsafeArtifactError(pickle.UnpicklingError):
    """Raised when an artifact references a class outside the doppel allowlist."""


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in _ALLOWED_BUILTINS:
            return super().find_class(module, name)
        if module == "collections" and name in _ALLOWED_COLLECTIONS:
            return super().find_class(module, name)
        if module == "_codecs" and name == "encode":
            # numpy uses _codecs.encode to pickle dtype names; required for ndarray roundtrip.
            return super().find_class(module, name)
        if (module, name) in _ALLOWED_DOPPEL_GLOBALS:
            return super().find_class(module, name)
        for prefix in _ALLOWED_MODULE_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                return super().find_class(module, name)
        raise UnsafeArtifactError(
            f"refusing to unpickle {module}.{name}: not in doppel artifact allowlist. "
            "If this is a legitimate doppel artifact from a newer release, upgrade. "
            "If this is from an unknown source, do not load it."
        )


def safe_loads(blob: bytes) -> Any:
    return RestrictedUnpickler(io.BytesIO(blob)).load()
