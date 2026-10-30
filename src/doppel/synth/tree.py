"""A fitted decision tree as plain arrays, with the sklearn routing rule.

sklearn fits the tree. Sampling then walks these arrays, so a saved model holds data
only and no sklearn object. `apply` reproduces `sklearn.tree._tree.Tree._apply_dense`:
the input is cast to float32, a NaN feature follows `missing_go_to_left`, and any other
value goes left when it is `<= threshold`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

_LEAF = -1


@dataclass(frozen=True)
class DecisionTree:
    left_child: np.ndarray
    right_child: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    missing_go_to_left: np.ndarray
    n_features: int

    @classmethod
    def from_sklearn(cls, model: DecisionTreeClassifier | DecisionTreeRegressor) -> DecisionTree:
        tree = model.tree_
        return cls.from_arrays(
            left_child=np.asarray(tree.children_left, dtype=np.int64),
            right_child=np.asarray(tree.children_right, dtype=np.int64),
            feature=np.asarray(tree.feature, dtype=np.int64),
            threshold=np.asarray(tree.threshold, dtype=np.float64),
            missing_go_to_left=np.asarray(tree.missing_go_to_left, dtype=np.bool_),
            n_features=int(model.n_features_in_),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        left_child: np.ndarray,
        right_child: np.ndarray,
        feature: np.ndarray,
        threshold: np.ndarray,
        missing_go_to_left: np.ndarray,
        n_features: int,
    ) -> DecisionTree:
        """Build a tree, and reject arrays that do not describe a finite binary tree.

        A child index is always larger than its parent's index (sklearn adds a parent
        before its children), so every walk ends at a leaf.
        """
        expected = {
            "left_child": (left_child, np.int64),
            "right_child": (right_child, np.int64),
            "feature": (feature, np.int64),
            "threshold": (threshold, np.float64),
            "missing_go_to_left": (missing_go_to_left, np.bool_),
        }
        n_nodes = left_child.shape[0] if left_child.ndim == 1 else 0
        for name, (array, dtype) in expected.items():
            if array.dtype != dtype or array.ndim != 1 or array.shape[0] != n_nodes:
                raise ValueError(
                    f"tree array {name!r} must be 1-D {np.dtype(dtype)} of length {n_nodes}"
                )
        if n_nodes == 0:
            raise ValueError("tree has no nodes")
        if n_features < 1:
            raise ValueError(f"tree has n_features={n_features}, expected at least 1")
        is_leaf = left_child == _LEAF
        if not np.array_equal(is_leaf, right_child == _LEAF):
            raise ValueError("tree node has exactly one child")
        index = np.arange(n_nodes)
        internal = ~is_leaf
        for child in (left_child, right_child):
            if np.any((child[internal] <= index[internal]) | (child[internal] >= n_nodes)):
                raise ValueError("tree child index is out of range")
        if np.any((feature[internal] < 0) | (feature[internal] >= n_features)):
            raise ValueError("tree feature index is out of range")
        return cls(
            left_child=left_child,
            right_child=right_child,
            feature=feature,
            threshold=threshold,
            missing_go_to_left=missing_go_to_left,
            n_features=n_features,
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Return the leaf index for each row of `x`."""
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected a matrix with {self.n_features} features, got {x.shape}")
        x32 = np.asarray(x, dtype=np.float32)
        node = np.zeros(x32.shape[0], dtype=np.int64)
        while True:
            rows = np.flatnonzero(self.left_child[node] != _LEAF)
            if rows.size == 0:
                return node
            at = node[rows]
            value = x32[rows, self.feature[at]]
            go_left = np.where(
                np.isnan(value), self.missing_go_to_left[at], value <= self.threshold[at]
            )
            node[rows] = np.where(go_left, self.left_child[at], self.right_child[at])
