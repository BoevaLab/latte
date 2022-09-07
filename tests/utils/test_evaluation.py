import pytest
import torch
import geoopt
import numpy as np


from latte.evaluation import metrics, subspace_evaluation


def test_subspace_fit() -> None:

    U_1 = geoopt.Stiefel().random(10, 5)
    U_2 = geoopt.Stiefel().random(10, 5)
    R = geoopt.Stiefel().random(5, 5)
    U_3 = U_1 @ R

    d_12 = subspace_evaluation.subspace_fit(U_1.numpy(), U_2.numpy())
    d_13 = subspace_evaluation.subspace_fit(U_1.numpy(), U_3.numpy())
    d_23 = subspace_evaluation.subspace_fit(U_2.numpy(), U_3.numpy())

    assert pytest.approx(0, abs=1e-4) == d_13["Value"]["Difference between subspaces"]
    assert (
        pytest.approx(d_12["Value"]["Difference between subspaces"], abs=1e-4)
        == d_23["Value"]["Difference between subspaces"]
    )
    assert 0 < d_13["Value"]["Difference between subspaces"]


def test_mixing_matrix_fit() -> None:
    rng = np.random.default_rng(1)

    A_1 = rng.random((10, 5))
    A_2 = rng.random((10, 5))
    A_3 = A_1 + 1e-4 * np.ones((10, 5))

    assert subspace_evaluation.mixing_matrix_fit(A_1, A_2)["Value"]["Difference between matrices"] > 0
    assert (
        pytest.approx(0, abs=50 * 1e-4)
        == subspace_evaluation.mixing_matrix_fit(A_1, A_3)["Value"]["Difference between matrices"]
        > 0
    )


def test_distance_to_permutation_matrix() -> None:
    rng = np.random.default_rng(1)
    Id = torch.eye(10)

    for _ in range(10):
        p = rng.permutation(10)
        assert pytest.approx(0) == metrics.distance_to_permutation_matrix(Id[p][p])
