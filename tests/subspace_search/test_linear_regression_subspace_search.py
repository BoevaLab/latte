import pytest

import numpy as np
import torch

from latte.models.probabilistic_pca import probabilistic_pca as ppca
from latte.models.linear_regression import estimation as lr_estimation
from latte.evaluation import subspace_evaluation
from latte.manifolds import utils as manifold_utils


rng = np.random.default_rng(42)
torch.manual_seed(42)


@pytest.mark.parametrize(
    "N",
    [16000],
)
@pytest.mark.parametrize(
    "n",
    [4, 8, 16, 32],
)
@pytest.mark.parametrize(
    "d",
    [1],
)
def test_ppca_data(N: int, n: int, d: int):

    data = ppca.generate(N, n, d, d, sigma=0.01)

    estimation_result = lr_estimation.fit(data.X, data.Z.flatten())

    evaluation_result = subspace_evaluation.subspace_fit(
        np.linalg.svd(data.A, full_matrices=False)[0], estimation_result.A_1.numpy()
    )

    assert manifold_utils.is_orthonormal(estimation_result.A_1)

    assert evaluation_result["Value"]["Difference between subspaces"] < 5e-2
