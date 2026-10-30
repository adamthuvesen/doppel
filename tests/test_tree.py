"""The array tree routes every row to the same leaf as sklearn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doppel.synth.tree import DecisionTree


def _training_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    gen = np.random.default_rng(seed)
    x = np.column_stack(
        [
            gen.integers(0, 6, size=400).astype(np.float64),
            gen.normal(size=400),
            gen.random(size=400) * 1e9,
        ]
    )
    x[gen.random(size=400) < 0.1, 1] = np.nan
    y = (x[:, 0] > 2).astype(np.int64) + (np.nan_to_num(x[:, 1]) > 0).astype(np.int64)
    return x, y


def _probe_rows(model: DecisionTreeClassifier | DecisionTreeRegressor, x: np.ndarray) -> np.ndarray:
    thresholds = model.tree_.threshold[model.tree_.children_left != -1]
    features = model.tree_.feature[model.tree_.children_left != -1]
    probes = np.repeat(x[:1], thresholds.size * 3, axis=0)
    for i, (feature, threshold) in enumerate(zip(features, thresholds, strict=True)):
        probes[3 * i : 3 * i + 3, feature] = [
            threshold,
            np.nextafter(threshold, np.inf),
            np.float32(threshold),
        ]
    return np.vstack([x, probes])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["classifier", "regressor"])
def test_apply_matches_sklearn_on_ties_and_missing_values(seed: int, kind: str) -> None:
    x, y = _training_data(seed)
    model = (
        DecisionTreeClassifier(random_state=seed, min_samples_leaf=5)
        if kind == "classifier"
        else DecisionTreeRegressor(random_state=seed, min_samples_leaf=5)
    )
    model.fit(x, y)
    rows = _probe_rows(model, x)
    rows[::7, 1] = np.nan

    tree = DecisionTree.from_sklearn(model)

    np.testing.assert_array_equal(tree.apply(rows), model.apply(rows))
    if isinstance(model, DecisionTreeClassifier):
        leaf_proba = np.asarray(model.tree_.value)[tree.apply(rows), 0, :]
        np.testing.assert_array_equal(leaf_proba, model.predict_proba(rows))


def test_from_arrays_rejects_a_cycle() -> None:
    with pytest.raises(ValueError, match="child index is out of range"):
        DecisionTree.from_arrays(
            left_child=np.array([1, 0, -1], dtype=np.int64),
            right_child=np.array([2, 2, -1], dtype=np.int64),
            feature=np.array([0, 0, -2], dtype=np.int64),
            threshold=np.array([0.5, 0.5, -2.0]),
            missing_go_to_left=np.array([True, True, False]),
            n_features=1,
        )


def test_from_arrays_rejects_a_feature_out_of_range() -> None:
    with pytest.raises(ValueError, match="feature index is out of range"):
        DecisionTree.from_arrays(
            left_child=np.array([1, -1, -1], dtype=np.int64),
            right_child=np.array([2, -1, -1], dtype=np.int64),
            feature=np.array([3, -2, -2], dtype=np.int64),
            threshold=np.array([0.5, -2.0, -2.0]),
            missing_go_to_left=np.array([True, False, False]),
            n_features=2,
        )


def test_apply_rejects_the_wrong_feature_count() -> None:
    x, y = _training_data(0)
    tree = DecisionTree.from_sklearn(DecisionTreeRegressor(random_state=0).fit(x, y))
    with pytest.raises(ValueError, match="expected a matrix with 3 features"):
        tree.apply(x[:, :2])
