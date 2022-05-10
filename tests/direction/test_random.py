import numpy as np
import pytest

import latte.direction.api as di


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
            seed=42,
        )

        assert results.metrics.shape == (n_samples,)
        assert results.vectors.shape == (n_samples, dim)

        result = di.random.select_best_direction(results)
        assert result.value == pytest.approx(results.metrics.max())
