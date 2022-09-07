from typing import Callable, Tuple

import numpy as np
import pytest

from scipy import spatial

from latte.tools.ksg import ksg


@pytest.mark.parametrize("k", [0, 1, 2, 3, 5])
def test_kth_neighbour(k: int) -> None:
    # The distance vector from the 0th point (so that distances[0] = 0)
    distances = np.arange(10)
    assert ksg._find_kth_neighbour(distances, k=k) == k


def random_covariance(size: int, jitter: float = 1e1, rng=0):
    a = np.random.default_rng(rng).normal(0, 1, size=(size, size))

    return np.dot(a, a.T) + np.diag([jitter] * size)


def test_digamma():
    """Tests whether the digamma function used seems right."""
    assert ksg._DIGAMMA(1) == pytest.approx(-0.5772156, abs=1e-5)

    for k in range(2, 5):
        assert ksg._DIGAMMA(k + 1) == pytest.approx(ksg._DIGAMMA(k) + 1 / k, rel=0.01)


# Distance-matrix-based estimators
# For now the second one doesn't work
# ESTIMATORS = [ksg.estimate_mi_ksg1, ksg.estimate_mi_ksg2]
ESTIMATORS = [ksg.estimate_mi_ksg1]


@pytest.mark.parametrize("n_points", [1000])
@pytest.mark.parametrize("k", [5, 10])
@pytest.mark.parametrize("correlation", [0.0, 0.6, 0.8, 0.9])
@pytest.mark.parametrize("func", ESTIMATORS)
def test_estimate_mi_ksg_known(n_points: int, k: int, correlation: float, func: Callable) -> None:
    covariance = np.array(
        [
            [1.0, correlation],
            [correlation, 1.0],
        ]
    )
    distribution = ksg.SplitMultinormal(
        mean=np.zeros(2),
        covariance=covariance,
        split=1,
        rng=0,
    )
    points_x, points_y = distribution.sample_joint(n_points)
    distance_x = spatial.distance_matrix(points_x, points_x)
    distance_y = spatial.distance_matrix(points_y, points_y)

    estimated_mi = func(distance_x, distance_y, k=k, normalize=False)
    true_mi = distribution.mutual_information()

    assert estimated_mi == pytest.approx(true_mi, rel=0.1, abs=0.05)


@pytest.mark.parametrize("n_points", [500, 1000])
@pytest.mark.parametrize("k", [5, 10])
@pytest.mark.parametrize("dims", [(1, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("func", ESTIMATORS)
def test_estimate_mi_ksg(n_points: int, k: int, dims: Tuple[int, int], func: Callable) -> None:
    dim_x, dim_y = dims

    distribution = ksg.SplitMultinormal(
        mean=np.zeros(dim_x + dim_y),
        covariance=random_covariance(dim_x + dim_y),
        split=dim_x,
        rng=0,
    )

    points_x, points_y = distribution.sample_joint(n_points)
    distance_x = spatial.distance_matrix(points_x, points_x)
    distance_y = spatial.distance_matrix(points_y, points_y)

    estimated_mi = func(distance_x, distance_y, k=k, normalize=False)
    true_mi = distribution.mutual_information()

    # Approximate the MI to 10% and take correction for very small values
    assert estimated_mi == pytest.approx(true_mi, rel=0.1, abs=0.02)


@pytest.mark.parametrize("n_points", [500])
@pytest.mark.parametrize("correlation", [0.0, 0.6, 0.8, 0.9])
def test_estimate_mi_ksg_known_from_points(
    n_points: int, correlation: float, neighborhoods: Tuple[int, ...] = (5, 10)
) -> None:
    covariance = np.array(
        [
            [1.0, correlation],
            [correlation, 1.0],
        ]
    )
    distribution = ksg.SplitMultinormal(
        mean=np.zeros(2),
        covariance=covariance,
        split=1,
        rng=0,
    )
    points_x, points_y = distribution.sample_joint(n_points)

    true_mi = distribution.mutual_information()
    estimates = ksg.estimate_mi_ksg(points_x, points_y, neighborhoods=neighborhoods)

    for k in neighborhoods:
        assert estimates[k] == pytest.approx(true_mi, rel=0.08, abs=0.03)
