import pytest

import numpy as np
import torch

from latte.models.cca import estimation as cca_estimation
from latte.evaluation import subspace_evaluation
from latte.manifolds import utils as manifold_utils


rng = np.random.default_rng(42)
torch.manual_seed(42)


@pytest.mark.parametrize(
    "N",
    [16000],
)
@pytest.mark.parametrize(
    "n_1",
    [8, 16],
)
@pytest.mark.parametrize(
    "n_2",
    [8, 16],
)
@pytest.mark.parametrize(
    "d",
    [4, 6],
)
def test_ppca_data(N: int, n_1: int, n_2: int, d: int):

    Z = rng.normal(size=(N, d))

    A_1, A_2 = rng.normal(size=(n_1, d)), rng.normal(size=(n_2, d))

    X_1 = Z @ A_1.T + rng.normal(scale=0.05, size=(N, n_1))
    X_2 = Z @ A_2.T + rng.normal(scale=0.05, size=(N, n_2))

    estimation_result = cca_estimation.fit(X_1, X_2, subspace_size=d)

    for A in [estimation_result.A_1, estimation_result.A_2]:
        evaluation_result = subspace_evaluation.subspace_fit(np.linalg.svd(A, full_matrices=False)[0], A.numpy())

        assert manifold_utils.is_orthonormal(A)

        assert evaluation_result["Value"]["Difference between subspaces"] < 5e-2
