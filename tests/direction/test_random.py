from typing import Sequence

import numpy as np
import pytest

import latte.direction.api as di


class TestFindMetricDistribution:
    @pytest.mark.parametrize("n_samples", (10, 100))
    @pytest.mark.parametrize("dim", (2, 4))
    @pytest.mark.parametrize("metric", [di.SpearmanCorrelationMetric(), di.MutualInformationMetric()])
    @pytest.mark.parametrize("n_points", [20])
    def test_smoke(self, auxiliary, n_samples: int, dim: int, metric: di.AxisMetric, n_points: int) -> None:
        generator = np.random.default_rng(10)

        points = auxiliary.create_data(n_dim=dim, n_points=n_points, seed=generator)
        scores = generator.uniform(0, 1, size=n_points)

        results = di.random.find_metric_distribution(
            points=points,
            scores=scores,
            metric=metric,
            budget=n_samples,
            seed=42,
        )

        assert results.metrics.shape == (n_samples,)
        assert results.vectors.shape == (n_samples, dim)

        result = di.random.select_best_direction(results)
        assert result.value == pytest.approx(results.metrics.max())


true_directions = [
    (0.5, 1.0, -0.2),
    (0.1, 1.0),
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
    np.random.seed(42)

    n_dim = len(true_direction)
    true_direction = np.asarray(true_direction)
    unit_true_direction = true_direction / np.linalg.norm(true_direction)

    # Square
    data = np.random.uniform(low=-1.0, high=+1.0, size=(80, n_dim))
    # Assign the scores according to the direction that should be
    scores = auxiliary.score_according_to_gradient(direction=true_direction, points=data)

    # Retrieve the vector yielding best scores
    optimization_result = di.random.find_best_direction(points=data, scores=scores, metric=metric, budget=100, seed=10)

    true_value = metric.score(axis=true_direction, points=data, scores=scores)

    # Test if the value is calculated properly
    assert optimization_result.value == pytest.approx(
        metric.score(axis=optimization_result.direction, points=data, scores=scores)
    )

    # Test whether the values at the true and found direction are comparable
    assert true_value == pytest.approx(optimization_result.value, abs=0.05), (
        f"Value mismatch: {true_value} != {optimization_result.value}. "
        f"Vectors: true: {unit_true_direction} found: {optimization_result.direction}"
    )
