from typing import Tuple

import numpy as np
import pytest

import latte.direction.metrics as metrics


def create_data(vector: np.ndarray, n_dim: int, seed, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)

    data = generator.normal(0, 1, size=(n_points, n_dim))
    scores = np.einsum("ij,j->i", data, vector)

    return data, scores


@pytest.mark.parametrize("metric", (metrics.SpearmanCorrelation(), metrics.KendallTau(), metrics.MutualInformation()))
@pytest.mark.parametrize("length", (10, 100))
def test_similarity_metric(metric: metrics.SimilarityMetric, length: int) -> None:
    """Basic test which compares whether a metric of a vector with itself has a greater score
    than with a noisy one."""
    generator = np.random.default_rng(5)
    vector = generator.random(size=length)
    noisy = vector + generator.normal(0, 1, size=length)

    true_score = metric.score(projections=vector, scores=vector)
    noisy_score = metric.score(projections=noisy, scores=vector)

    assert true_score > noisy_score


@pytest.mark.parametrize("n_points", (10, 100))
@pytest.mark.parametrize("dim", (2, 5))
@pytest.mark.parametrize(
    "metric", (metrics.MutualInformationMetric(), metrics.SpearmanCorrelationMetric(), metrics.KendallTauMetric())
)
def test_axis_metric(metric: metrics.AxisMetric, n_points: int, dim: int) -> None:
    generator = np.random.default_rng(5)
    vector = generator.normal(0, 1, size=dim)

    data, scores = create_data(vector=vector, n_dim=dim, n_points=n_points, seed=generator)

    true_score = metric.score(points=data, axis=vector, scores=scores)
    noisy_score = metric.score(points=data, axis=vector + generator.normal(0, 4, size=dim), scores=scores)

    assert true_score > noisy_score
