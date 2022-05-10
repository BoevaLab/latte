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
