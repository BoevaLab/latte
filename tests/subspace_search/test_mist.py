from typing import Callable, List
import pytest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from latte.modules import callbacks as cbs
from latte.models.probabilistic_pca import probabilistic_pca as ppca
from latte.models.mist import estimation as mist_estimation
from latte.evaluation import subspace_evaluation
from latte.dataset import synthetic
from latte.manifolds import utils as manifold_utils


rng = np.random.default_rng(42)
torch.manual_seed(42)


@pytest.mark.parametrize(
    "N",
    [4000],
)
@pytest.mark.parametrize(
    "n",
    [8],
)
@pytest.mark.parametrize(
    "d",
    [4],
)
def test_projection_matrix_changes(N: int, n: int, d: int):

    data = ppca.generate(N, n, d, d)

    cb = cbs.WeightsTracker()
    mist_estimation.fit(
        data.X, data.Z, model_kwargs=dict(x_size=n, z_max_size=d), trainer_kwargs=dict(callbacks=[cb], gpus=0)
    )

    # cb.epoch_weights will contain the weight values for each layer at each epoch
    # We assert that the values for each layer were changed at each epoch
    for ii, entry in cb.epoch_weights.items():
        for jj in range(1, len(entry)):
            assert (
                np.linalg.norm(entry[jj] - entry[jj - 1]) > 0
            )  # since the layer weight are stored as numpy arrays, use numpy linalg


@pytest.mark.parametrize(
    "N",
    [8000],
)
@pytest.mark.parametrize(
    "n",
    [8, 16],
)
@pytest.mark.parametrize(
    "d",
    [2, 3, 4],
)
def test_ppca_data(N: int, n: int, d: int):

    data = ppca.generate(N, n, d, d)

    data.X = StandardScaler().fit_transform(data.X)
    data.Z = StandardScaler().fit_transform(data.Z)

    estimation_result = mist_estimation.fit(
        data.X,
        data.Z,
        model_kwargs=dict(x_size=n, z_max_size=d, subspace_size=d, mine_network_width=128),
        trainer_kwargs=dict(max_epochs=512, gpus=0),
    )

    evaluation_result = subspace_evaluation.subspace_fit(
        np.linalg.svd(data.A, full_matrices=False)[0], estimation_result.A_1.numpy()
    )

    random_matrices_results = [
        subspace_evaluation.subspace_fit(
            np.linalg.svd(data.A, full_matrices=False)[0],
            np.linalg.svd(rng.random(size=data.A.shape), full_matrices=False)[0],
        )["Value"]["Difference between subspaces"]
        for _ in range(1000)
    ]

    assert manifold_utils.is_orthonormal(estimation_result.A_1, atol=1e-2)

    assert evaluation_result["Value"]["Difference between subspaces"] < np.quantile(random_matrices_results, 0.05)


transforms = [
    [lambda x: np.sum(x**3, axis=1) + rng.normal(0, 0.05, size=(x.shape[0])).astype(np.float32)],
]


@pytest.mark.parametrize(
    "N",
    [8000],
)
@pytest.mark.parametrize(
    "n",
    [8, 16],
)
@pytest.mark.parametrize(
    "d",
    [2, 3, 4],
)
@pytest.mark.parametrize(
    "transform",
    transforms,
)
def test_nonlinear_data(N: int, n: int, d: int, transform: List[Callable[[np.ndarray], np.ndarray]]):

    data = synthetic.generate(N, n, d, fs=transform)

    data.X = StandardScaler().fit_transform(data.X)
    data.Z = StandardScaler().fit_transform(data.Z)

    estimation_result = mist_estimation.fit(
        data.X,
        data.Z,
        model_kwargs=dict(x_size=n, z_max_size=len(transform), subspace_size=d),
        trainer_kwargs=dict(max_epochs=512, gpus=0),
    )

    evaluation_result = subspace_evaluation.subspace_fit(data.A, estimation_result.A_1.numpy())

    random_matrices_results = [
        subspace_evaluation.subspace_fit(
            np.linalg.svd(data.A, full_matrices=False)[0],
            np.linalg.svd(rng.random(size=data.A.shape), full_matrices=False)[0],
        )["Value"]["Difference between subspaces"]
        for _ in range(1000)
    ]

    assert manifold_utils.is_orthonormal(estimation_result.A_1, atol=1e-2)

    assert evaluation_result["Value"]["Difference between subspaces"] < np.quantile(random_matrices_results, 0.05)


@pytest.mark.parametrize(
    "N",
    [32000],
)
@pytest.mark.parametrize(
    "n",
    [3],
)
@pytest.mark.parametrize(
    "d",
    [1, 2],
)
def test_probabilistic_pca_data_mi_estimation_full_space(N: int, n: int, d: int) -> None:

    data = ppca.generate(N, n, d, d)
    true_mi = ppca.generative_model_mutual_information(data.A, data.sigma)

    data.X = StandardScaler().fit_transform(data.X)

    estimation_result = mist_estimation.fit(
        data.X,
        data.Z,
        model_kwargs=dict(x_size=n, z_max_size=d, subspace_size=(n, d), mine_network_width=128),
        trainer_kwargs=dict(max_epochs=512, gpus=0),
    )

    assert manifold_utils.is_orthonormal(estimation_result.A_1, atol=1e-2)

    assert estimation_result.mutual_information == pytest.approx(true_mi, 0.1)


@pytest.mark.parametrize(
    "N",
    [32000],
)
@pytest.mark.parametrize(
    "n",
    [8, 16, 32],
)
@pytest.mark.parametrize(
    "d",
    [2],
)
def test_probabilistic_pca_data_mi_estimation_correctly_specified(N: int, n: int, d: int) -> None:

    data = ppca.generate(N, n, d, d, sigma=1)
    true_mi = ppca.generative_model_mutual_information(data.A, data.sigma)

    data.X = StandardScaler().fit_transform(data.X)

    estimation_result = mist_estimation.fit(
        data.X,
        data.Z,
        model_kwargs=dict(x_size=n, z_max_size=d, subspace_size=(d, d), mine_network_width=128),
        trainer_kwargs=dict(max_epochs=512, gpus=0),
    )

    assert manifold_utils.is_orthonormal(estimation_result.A_1, atol=1e-2)

    assert estimation_result.mutual_information == pytest.approx(true_mi, 0.1)
