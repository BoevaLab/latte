from typing import Sequence

import numpy as np
import pytest

import latte.direction.api as di


true_directions = [
    (0.5, 1.0, -0.2),
    (0.1, 1.0),
    (-1.0, 2.0),
    (-1.0, 3.0, -0.1),
]


@pytest.mark.parametrize(
    "true_direction",
    true_directions,
)
@pytest.mark.parametrize(
    "metric",
    [
        di.SpearmanCorrelationMetric(),
    ],
)
def test_find_gradient(true_direction: Sequence[float], metric: di.AxisMetric, auxiliary) -> None:
    """Note `auxiliary` is the fixture with scoring method. See `conftest.py`."""
    np.random.seed(42)

    n_dim = len(true_direction)
    true_direction = np.asarray(true_direction)
    unit_true_direction = true_direction / np.linalg.norm(true_direction)

    # Square
    data = np.random.uniform(low=-1.0, high=+1.0, size=(100, n_dim))
    # Assign the scores according to the direction that should be
    scores = auxiliary.score_according_to_gradient(direction=true_direction, points=data)

    # Retrieve the vector yielding best scores
    optimization_result = di.optimization.find_best_direction(
        points=data, scores=scores, metric=metric, budget=70, seed=10
    )

    true_value = metric.score(axis=true_direction, points=data, scores=scores)

    # Test if the value is calculated properly
    assert optimization_result.value == pytest.approx(
        metric.score(axis=optimization_result.direction, points=data, scores=scores)
    )

    # Test whether the values at the true and found direction are comparable
    assert true_value == pytest.approx(optimization_result.value, rel=0.05, abs=0.05), (
        f"Value mismatch: {true_value} != {optimization_result.value}. "
        f"Vectors: true: {unit_true_direction} found: {optimization_result.direction}"
    )
