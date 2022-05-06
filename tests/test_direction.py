from typing import Sequence

import numpy as np
import pytest
from sklearn.metrics import r2_score, mean_squared_error

import latte.direction.api as di


def score_according_to_gradient(direction: Sequence[float], points: np.ndarray) -> np.ndarray:
    """Calculates the score by taking a scalar product with a given direction vector.

    Args:
        direction: shape (n_dim,)
        points: shape (n_points, n_dim)

    Returns
        scores, shape (n_points,)
    """
    direction_vector = np.asarray(direction)
    return np.einsum("ij,j -> i", points, direction_vector)


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
    "metric", [di.MutualInformationMetric(n_neighbors=50), di.SpearmanCorrelationMetric(), di.KendallTauMetric()]
)
def test_find_gradient(true_direction: Sequence[float], metric: di.AxisMetric) -> None:
    np.random.seed(42)

    n_dim = len(true_direction)
    true_direction = np.asarray(true_direction)
    unit_true_direction = true_direction / np.linalg.norm(true_direction)

    # Square
    data = np.random.uniform(low=-1.0, high=+1.0, size=(100, n_dim))
    # Assign the scores according to the direction that should be
    scores = score_according_to_gradient(direction=true_direction, points=data)

    # Retrieve the vector yielding best scores
    optimization_result = di.random.find_best_direction(
        points=data, scores=scores, metric=metric, budget=500, random_generator=5
    )

    true_value = metric.score(axis=true_direction, points=data, scores=scores)

    # Test if the value is calculated properly
    assert optimization_result.value == pytest.approx(
        metric.score(axis=optimization_result.direction, points=data, scores=scores)
    )

    # Test whether the values at the true and found direction are comparable
    assert true_value == pytest.approx(optimization_result.value, rel=0.1, abs=0.05), (
        f"Value mismatch: {true_value} != {optimization_result.value}. "
        f"Vectors: true: {unit_true_direction} found: {optimization_result.direction}"
    )


@pytest.mark.parametrize(
    "true_direction",
    true_directions,
)
def test_find_gradient_linear_regression(true_direction: Sequence[float]) -> None:
    np.random.seed(42)

    n_dim = len(true_direction)
    true_direction = np.asarray(true_direction)
    unit_true_direction = true_direction / np.linalg.norm(true_direction)

    # Square
    data = np.random.uniform(low=-2.0, high=+2.0, size=(200, n_dim))
    # Assign the scores according to the direction that should be
    scores = score_according_to_gradient(direction=true_direction, points=data)

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


class TestFindMetricDistribution:
    @pytest.mark.parametrize("n_samples", (10, 100))
    @pytest.mark.parametrize("dim", (2, 4))
    @pytest.mark.parametrize("metric", [di.SpearmanCorrelationMetric(), di.MutualInformationMetric()])
    def test_smoke(self, n_samples: int, dim: int, metric: di.AxisMetric) -> None:
        np.random.seed(42)
        n_points: int = 20
        points = np.random.rand(n_points, dim)
        scores = np.random.rand(n_points)

        results = di.random.find_metric_distribution(
            points=points,
            scores=scores,
            metric=metric,
            budget=n_samples,
            random_generator=42,
        )

        assert results.metrics.shape == (n_samples,)
        assert results.vectors.shape == (n_samples, dim)

        result = di.random.select_best_direction(results)
        assert result.value == pytest.approx(results.metrics.max())
