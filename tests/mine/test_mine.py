from typing import Callable, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
import numpy as np
import pytest
import sklearn.feature_selection as skfs

from latte.mine import mine, callbacks as cbs
from latte.models import datamodules as dm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


rng = np.random.default_rng(42)
torch.manual_seed(42)

transforms = [
    lambda x: x + rng.normal(0, 0.05, size=(x.shape[0], 1)),
    lambda x: np.sin(x) + rng.normal(0, 0.05, size=(x.shape[0], 1)),
    lambda x: x**3 + rng.normal(0, 0.05, size=(x.shape[0], 1)),
    lambda x: 1 / (np.abs(x) + 1) + rng.normal(0, 0.05, size=(x.shape[0], 1)),
    lambda x: np.exp(x) + rng.normal(0, 0.05, size=(x.shape[0], 1)),
]


def get_mine_prediction(
    x: np.ndarray,
    y: np.ndarray,
    objective_type: mine.MINEObjectiveType,
    small: bool = False,
    epochs: int = 200,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
) -> float:
    """Calculates the MINE estimate of mutual information between x and y by training the network."""
    data = dm.GenericMINEDataModule(
        X=torch.from_numpy(x), Z=torch.from_numpy(y), p_train=0.5, p_val=0.1, batch_size=batch_size
    )

    # construct the MINE estimator
    if small:
        S = mine.StatisticsNetwork(
            S=nn.Sequential(nn.Linear(x.shape[-1] + y.shape[-1], 16), nn.Linear(16, 8)),
            out_dim=8,
        )
    else:
        S = mine.StatisticsNetwork(
            S=nn.Sequential(
                nn.Linear(x.shape[-1] + y.shape[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            ),
            out_dim=128,
        )

    model = mine.MINE(T=S, kind=objective_type, alpha=0.01, learning_rate=learning_rate)
    trainer = Trainer(max_epochs=epochs, enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(model, datamodule=data)
    test_result = trainer.test(ckpt_path="best", dataloaders=data, verbose=False)

    # get the test set mutual information prediction
    mine_mi = test_result[0]["test_mutual_information_epoch"]

    return mine_mi


def function_test(
    transform: Callable, n_samples: int, n_epochs: int, objective_type: mine.MINEObjectiveType
) -> Tuple[float, float]:
    """Generates a random sample, transforms it with noise, and estimates the "true" and MINE mutual information"""
    x = rng.uniform(-1, 1, size=(n_samples, 1))
    y = transform(x)

    true_mi = skfs.mutual_info_regression(x.reshape((-1, 1)), y, n_neighbors=5)
    mine_mi = get_mine_prediction(x, y, objective_type, epochs=n_epochs, batch_size=512)

    return true_mi, mine_mi


# DONE
@pytest.mark.skip("Too long to run")
@pytest.mark.parametrize(
    "transform",
    transforms,
)
@pytest.mark.parametrize(
    "n_samples",
    [16000],
)
@pytest.mark.parametrize(
    "n_epochs",
    [1000],
)
def test_univariate(transform: Callable[[np.ndarray], np.ndarray], n_samples: int, n_epochs: int) -> None:
    true_mi, mine_mi = function_test(transform, n_samples, n_epochs, mine.MINEObjectiveType.MINE)
    assert mine_mi == pytest.approx(true_mi, 0.1), f"{mine_mi} != {true_mi}"


# DONE
@pytest.mark.skip("Too long to run")
@pytest.mark.parametrize(
    "n_samples",
    [16000],
)
@pytest.mark.parametrize(
    "d",
    [2, 5, 10],
)
@pytest.mark.parametrize(
    "n_epochs",
    [1000],
)
def test_multivairate_gaussian(n_samples: int, d: int, n_epochs: int) -> None:
    mu = np.zeros(d)
    c = rng.random(size=(d, d))
    sigma = c.T @ c

    t = rng.multivariate_normal(mu, sigma, size=n_samples)

    x = t[:, : d // 2]
    y = t[:, d // 2 :]

    sigma_1 = sigma[: d // 2, : d // 2]
    sigma_2 = sigma[d // 2 :, d // 2 :]

    true_mi = 1 / 2 * np.log(np.linalg.det(sigma_1) * np.linalg.det(sigma_2) / np.linalg.det(sigma))
    mine_mi = get_mine_prediction(
        x, y, mine.MINEObjectiveType.MINE, epochs=n_epochs, learning_rate=1e-3, batch_size=512
    )

    assert mine_mi == pytest.approx(true_mi, 0.1)


# DONE
@pytest.mark.skip("Too long to run")
@pytest.mark.parametrize(
    "n_samples",
    [1000],
)
@pytest.mark.parametrize(
    "d",
    [4, 16, 32],
)
@pytest.mark.parametrize(
    "objective_type",
    [mine.MINEObjectiveType.MINE, mine.MINEObjectiveType.MINE_BIASED, mine.MINEObjectiveType.F_DIV],
)
def test_loss_decrease(n_samples: int, d: int, objective_type: mine.MINEObjectiveType) -> None:

    mu = np.zeros(d)
    c = rng.random(size=(d, d))
    sigma = c.T @ c

    x = rng.multivariate_normal(mu, sigma, size=n_samples)
    y = x + rng.normal(0, 1, size=x.shape)

    data = dm.GenericMINEDataModule(X=torch.from_numpy(x), Z=torch.from_numpy(y), p_train=0.5, p_val=0.5, batch_size=32)

    # construct the MINE estimator
    S = mine.StatisticsNetwork(
        S=nn.Sequential(
            nn.Linear(x.shape[-1] + y.shape[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ),
        out_dim=64,
    )

    model = mine.MINE(T=S, kind=objective_type, alpha=0.01, learning_rate=1e-4)

    cb = cbs.MetricTracker()
    trainer = Trainer(max_epochs=50, callbacks=[cb], enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(model, datamodule=data)

    for ii in [16, 32, 48]:
        # assert that the mutual information estimate has increased on average compared to earlier epochs
        assert np.mean(
            [cb.epoch_outputs[ii - d]["validation_mutual_information_epoch"] for d in range(-2, 3)]
        ) > np.mean([cb.epoch_outputs[ii - 12 - d]["validation_mutual_information_epoch"] for d in range(-2, 3)])


# DONE
# @pytest.mark.skip("Too long to run")
def test_value_changes() -> None:

    x = rng.uniform(-5, 5, size=(1000, 1))
    y = transforms[0](x)

    data = dm.GenericMINEDataModule(
        X=torch.from_numpy(x), Z=torch.from_numpy(y), p_train=0.95, p_val=0.025, batch_size=32
    )

    # construct the MINE estimator
    S = mine.StatisticsNetwork(
        S=nn.Sequential(
            nn.Linear(x.shape[-1] + y.shape[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        ),
        out_dim=32,
    )

    model = mine.MINE(T=S, kind=mine.MINEObjectiveType.MINE, alpha=0.01)

    cb = cbs.WeightsTracker()
    trainer = Trainer(max_epochs=16, callbacks=[cb], enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(model, datamodule=data)

    # cb.epoch_weights will contain the weight values for each layer at each epoch
    # We assert that the values for each layer were changed at each epoch
    for ii, entry in cb.epoch_weights.items():
        for jj in range(1, len(entry)):
            assert (
                np.linalg.norm(entry[jj] - entry[jj - 1]) > 0
            )  # since the layer weight are stored as numpy arrays, use numpy linalg


# DONE
@pytest.mark.skip("Too long to run")
@pytest.mark.parametrize(
    "objective_type",
    (mine.MINEObjectiveType.MINE, mine.MINEObjectiveType.MINE_BIASED, mine.MINEObjectiveType.F_DIV),
)
def test_lower_mi(objective_type: mine.MINEObjectiveType) -> None:

    x = rng.uniform(-5, 5, size=(1000, 1))
    y_1 = transforms[0](x)
    y_2 = y_1 + rng.normal(0, 1, size=y_1.shape)

    mine_mi_1 = get_mine_prediction(x, y_1, objective_type=objective_type, epochs=100, small=False)
    mine_mi_2 = get_mine_prediction(x, y_2, objective_type=objective_type, epochs=100, small=False)

    assert mine_mi_1 > mine_mi_2


# DONE
@pytest.mark.skip("Too long to run")
@pytest.mark.parametrize(
    "objective_type",
    (mine.MINEObjectiveType.MINE, mine.MINEObjectiveType.MINE_BIASED, mine.MINEObjectiveType.F_DIV),
)
def test_worse_bound(objective_type: mine.MINEObjectiveType) -> None:

    x = rng.uniform(-5, 5, size=(1000, 1))
    y = transforms[0](x)

    mine_mi_big = get_mine_prediction(x, y, objective_type=objective_type, epochs=100, small=False)
    mine_mi_small = get_mine_prediction(x, y, objective_type=objective_type, epochs=100, small=True)

    assert mine_mi_small < mine_mi_big


# DONE
# @pytest.mark.skip("Too long to run")
class TestMINE:
    @pytest.mark.parametrize("n_samples", (64,))
    @pytest.mark.parametrize("d", (1, 2))
    @pytest.mark.parametrize(
        "objective_type",
        (mine.MINEObjectiveType.MINE, mine.MINEObjectiveType.MINE_BIASED, mine.MINEObjectiveType.F_DIV),
    )
    @pytest.mark.parametrize("n_epochs", (32,))
    def test_smoke(self, n_samples: int, d: int, objective_type: mine.MINEObjectiveType, n_epochs: int) -> None:

        x = rng.uniform(-1, 1, size=(n_samples, 1))
        y = transforms[0](x)

        data = dm.GenericMINEDataModule(
            X=torch.from_numpy(x), Z=torch.from_numpy(y), p_train=0.5, p_val=0.5, batch_size=16
        )

        # construct the MINE estimator
        S = mine.StatisticsNetwork(
            S=nn.Sequential(
                nn.Linear(x.shape[-1] + y.shape[-1], 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
            ),
            out_dim=16,
        )

        model = mine.MINE(T=S, kind=objective_type, alpha=0.01, learning_rate=1e-2)

        initial_layer_weights = []
        for layer in model.T.S:
            if hasattr(layer, "weight"):
                initial_layer_weights.append(layer.weight.clone())

        cb = cbs.MetricTracker()
        trainer = Trainer(max_epochs=n_epochs, callbacks=[cb], enable_model_summary=False, enable_progress_bar=False)
        trainer.fit(model, datamodule=data)

        # Assert that the loss decreases over time (the MI estimate increases)
        assert (
            cb.epoch_outputs[2]["validation_mutual_information_epoch"]
            < cb.epoch_outputs[n_epochs]["validation_mutual_information_epoch"]
        )
        assert (
            cb.epoch_outputs[2]["train_mutual_information_epoch"]
            < cb.epoch_outputs[n_epochs]["train_mutual_information_epoch"]
        )

        # Assert that the weights of all layers have changed
        ii = 0
        for layer in model.T.S:
            if hasattr(layer, "weight"):
                assert torch.linalg.norm(initial_layer_weights[ii] - layer.weight) > 0
                ii += 1

        # Assert that the model call returns a *negative* value
        # (the negative of the MI estimate, which should be *positive*)
        with torch.no_grad():
            for batch in data.test_dataloader():
                assert model(batch) < 0
