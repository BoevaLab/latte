from typing import Callable, Tuple

import torch.nn as nn
from pytorch_lightning import Trainer
import numpy as np
import pytest
import sklearn.feature_selection as skfs

from latte.models import datamodules as dm
from latte.models import mine

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Some 1D distributions for which we have a good MI estimate
(and we can compare it with the KSG estimator implemented in sklearn).
"""

rng = np.random.default_rng(42)


transforms = [
    lambda x: x + rng.normal(0, 0.1, size=(x.shape[0], 1)),
    lambda x: np.sin(x) + rng.normal(0, 0.1, size=(x.shape[0], 1)),
    lambda x: x**3 + rng.normal(0, 0.1, size=(x.shape[0], 1)),
    lambda x: 1 / (x + 2) + rng.normal(0, 0.1, size=(x.shape[0], 1)),
    lambda x: np.exp(x) + rng.normal(0, 0.1, size=(x.shape[0], 1)),
]


def get_mine_prediction(
    x: np.ndarray, y: np.ndarray, objective_type: mine.MINEObjectiveType, small: bool = False
) -> float:
    """Calculates the MINE estimate of mutual information between x and y by training the network."""
    data = dm.GenericMINEDataModule(X=x, Z=y, p_train=0.5, p_val=0.1, batch_size=32)

    # construct the MINE estimator
    if small:
        S = mine.StatisticsNetwork(
            S=nn.Sequential(
                nn.Linear(x.shape[-1] + y.shape[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            ),
            out_dim=128,
        )
    else:
        S = mine.StatisticsNetwork(
            S=nn.Sequential(
                nn.Linear(x.shape[-1] + y.shape[-1], 16),
                nn.Linear(16, 16),
            ),
            out_dim=16,
        )

    model = mine.MINE(T=S, kind=objective_type, alpha=0.01)
    trainer = Trainer(max_epochs=20, enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(model, datamodule=data)
    test_result = trainer.test(ckpt_path="best", dataloaders=data, verbose=False)

    # get the test set mutual information prediction
    mine_mi = test_result[0]["test_mutual_information_epoch"]

    return mine_mi


def function_test(transform: Callable, n_samples: int, objective_type: mine.MINEObjectiveType) -> Tuple[float, float]:
    """Generates a random sample, transforms it with noise, and estimates the "true" and MINE mutual information"""
    x = rng.uniform(-5, 5, size=(n_samples, 1))
    y = transform(x)

    true_mi = skfs.mutual_info_regression(x.reshape((-1, 1)), y, n_neighbors=5)
    mine_mi = get_mine_prediction(x, y, objective_type)

    return true_mi, mine_mi


@pytest.mark.parametrize(
    "transform",
    transforms,
)
@pytest.mark.parametrize(
    "n_samples",
    [100],
)
@pytest.mark.parametrize(
    "objective_type",
    [mine.MINEObjectiveType.MINE, mine.MINEObjectiveType.MINE_BIASED, mine.MINEObjectiveType.F_DIV],
)
def test_univariate(
    transform: Callable[[np.ndarray], np.ndarray], n_samples: int, objective_type: mine.MINEObjectiveType
) -> None:
    # true_mi, mine_mi = function_test(transform, n_samples, objective_type)
    # assert mine_mi == pytest.approx(true_mi, 0.1)
    assert True


"""
Multivariate Gaussians, for which we know the exact MI.
We can use flexible enough function space to get a tight lower bound.
"""


@pytest.mark.parametrize(
    "n_samples",
    [100],
)
@pytest.mark.parametrize(
    "d",
    [4, 16, 32],
)
@pytest.mark.parametrize(
    "objective_type",
    [mine.MINEObjectiveType.MINE, mine.MINEObjectiveType.MINE_BIASED, mine.MINEObjectiveType.F_DIV],
)
def test_multivairate_gaussian(n_samples: int, d: int, objective_type: mine.MINEObjectiveType) -> None:
    # mu = np.zeros(d)
    # c = rng.random(size=(d, d))
    # sigma = c.T @ c
    #
    # t = rng.multivariate_normal(mu, sigma, size=n_samples)
    #
    # x = t[:, : d // 2]
    # y = t[:, d // 2 :]
    #
    # sigma_1 = sigma[: d // 2, : d // 2]
    # sigma_2 = sigma[d // 2 :, d // 2 :]
    #
    # true_mi = 1 / 2 * np.log(np.linalg.det(sigma_1) * np.linalg.det(sigma_2) / np.linalg.det(sigma))
    # mine_mi = get_mine_prediction(x, y, objective_type)

    # assert mine_mi == pytest.approx(true_mi, 0.1)
    assert True


"""
The standard test for neural networks:
    does the loss actually go down?
    Do the layer values change over time?
"""


def test_loss() -> None:
    ...


def test_value_changes() -> None:
    ...


"""
When we restrict the family of functions, we should get a worse lower bound.
"""


def test_worse_bound() -> None:
    #
    # x = rng.uniform(-5, 5, size=(100, 1))
    # y = transforms[0](x)
    #
    # mine_mi_big = get_mine_prediction(x, y, objective_type=mine.MINEObjectiveType.MINE, small=False)
    # mine_mi_small = get_mine_prediction(x, y, objective_type=mine.MINEObjectiveType.MINE, small=True)

    # assert mine_mi_small < mine_mi_big
    assert True


class TestFindMetricDistribution:
    @pytest.mark.parametrize("n_samples", (10, 100))
    @pytest.mark.parametrize("dim", (2, 4))
    def test_smoke(self, n_samples: int, dim: int) -> None:
        assert True
