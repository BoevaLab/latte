from typing import Sequence

import numpy as np
import pytest
from sklearn.metrics import r2_score, mean_squared_error

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
def test_find_gradient_linear_regression(true_direction: Sequence[float], auxiliary) -> None:
    np.random.seed(42)

    n_dim = len(true_direction)
    true_direction = np.asarray(true_direction)
    unit_true_direction = true_direction / np.linalg.norm(true_direction)

    # Square
    data = np.random.uniform(low=-2.0, high=+2.0, size=(200, n_dim))
    # Assign the scores according to the direction that should be
    scores = auxiliary.score_according_to_gradient(direction=true_direction, points=data)

    # Retrieve the vector yielding best scores
    optimization_result = di.linear_regression.find_best_direction(points=data, scores=scores)

    true_r_sq = r2_score(scores, scores)
    true_mse = mean_squared_error(scores, scores)

    predicted_scores = np.einsum("ij,j->i", data, optimization_result.direction) + optimization_result.intercept

    assert optimization_result.r2 == pytest.approx(r2_score(scores, predicted_scores))

    assert true_mse == pytest.approx(mean_squared_error(scores, predicted_scores))

    assert true_r_sq == pytest.approx(
        optimization_result.r2, 0.02
    ), f"True vector: {unit_true_direction}. Found vector: {optimization_result.direction}."
