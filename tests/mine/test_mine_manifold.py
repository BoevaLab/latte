import torch
import torch.nn as nn
from pytorch_lightning import Trainer
import numpy as np
import pytest

from latte.models import datamodules as dm
from latte.mine import mine, callbacks as cbs
from latte.dataset import synthetic

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


rng = np.random.default_rng(42)
torch.manual_seed(42)


@pytest.mark.skip("Too long to run")
@pytest.mark.parametrize(
    "n_samples",
    [1024],
)
@pytest.mark.parametrize(
    "d",
    [3, 4],
)
@pytest.mark.parametrize(
    "k",
    [1, 2],
)
def test_close_solution(n_samples: int, d: int, k: int) -> None:

    if k == 2:
        fs = [lambda x: np.arctan2(x[:, 0], x[:, 1])]
    else:
        fs = [lambda x: x[:, 0] ** 3]

    # Right now, we only test 2-dimensional subspaces
    dataset = synthetic.synthetic_dataset(n_samples, d=d, k=k, fs=fs)

    data = dm.GenericMINEDataModule(
        torch.from_numpy(dataset.X),
        torch.from_numpy(dataset.Z),
        p_train=0.9,
        p_val=0.05,
        batch_size=128,
        test_batch_size=1024,
    )

    # construct the MINE estimator
    S = mine.ManifoldStatisticsNetwork(
        S=nn.Sequential(
            nn.Linear(k + len(fs), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        ),
        d=d,
        k=k,
        out_dim=128,
    )

    model = mine.ManifoldMINE(T=S, kind=mine.MINEObjectiveType.MINE, alpha=0.01)

    trainer = Trainer(max_epochs=256, enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(model, datamodule=data)

    # Assert that the return k-frame is column orthonormal
    A_hat = model.T.projection_layer.A.data.detach().cpu().numpy()
    assert np.linalg.norm(np.eye(k) - A_hat.T @ A_hat) == pytest.approx(
        0, abs=1e-3
    ), f"A_hat = {A_hat} is not column orthogonal; A_hat.T @ A_hat = {A_hat.T @ A_hat}"

    # Assert that the returned k-frame is close to the solution
    assert np.linalg.norm(dataset.A @ dataset.A.T - A_hat @ A_hat.T) == pytest.approx(0, abs=1e-1), (
        f"A_hat is not close to the solution; A_hat = {A_hat}, A_hat @ A_hat.T = {A_hat @ A_hat.T}, "
        f"A = {dataset.A}, A @ A.T = {dataset.A @ dataset.A.T}"
    )


@pytest.mark.skip("Too long to run")
@pytest.mark.parametrize(
    "n_samples",
    [1024],
)
@pytest.mark.parametrize(
    "d",
    [4, 8],
)
def test_loss_decrease(n_samples: int, d: int) -> None:

    fs = [lambda x: np.arctan2(x[:, 0], x[:, 1])]

    # Right now, we only test 2-dimensional subspaces
    k = 2
    dataset = synthetic.synthetic_dataset(n_samples, d=d, k=k, fs=fs)

    data = dm.GenericMINEDataModule(
        torch.from_numpy(dataset.X),
        torch.from_numpy(dataset.Z),
        p_train=0.5,
        p_val=0.4,
        batch_size=128,
        test_batch_size=1024,
    )

    # construct the MINE estimator
    S = mine.ManifoldStatisticsNetwork(
        S=nn.Sequential(
            nn.Linear(k + len(fs), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ),
        d=d,
        k=k,
        out_dim=64,
    )

    model = mine.ManifoldMINE(T=S, kind=mine.MINEObjectiveType.MINE, alpha=0.01)

    cb = cbs.MetricTracker()
    trainer = Trainer(max_epochs=50, callbacks=[cb], enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(model, datamodule=data)

    for ii in [16, 32, 48]:
        # assert that the mutual information estimate has increased on average compared to earlier epochs
        assert np.mean(
            [cb.epoch_outputs[ii - d]["validation_mutual_information_epoch"] for d in range(-2, 3)]
        ) > np.mean([cb.epoch_outputs[ii - 12 - d]["validation_mutual_information_epoch"] for d in range(-2, 3)])


# DONE
@pytest.mark.skip("Too long to run")
def test_projection_layer_changes() -> None:

    fs = [lambda x: np.arctan2(x[:, 0], x[:, 1])]

    d, k = 3, 2

    dataset = synthetic.synthetic_dataset(1024, d=d, k=k, fs=fs)

    data = dm.GenericMINEDataModule(
        torch.from_numpy(dataset.X),
        torch.from_numpy(dataset.Z),
        p_train=0.95,
        p_val=0.025,
        batch_size=128,
        test_batch_size=1024,
    )

    # construct the MINE estimator
    S = mine.ManifoldStatisticsNetwork(
        S=nn.Sequential(
            nn.Linear(k + len(fs), 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        ),
        d=d,
        k=k,
        out_dim=32,
    )

    model = mine.ManifoldMINE(T=S, kind=mine.MINEObjectiveType.MINE, alpha=0.01)

    cb = cbs.WeightsTracker()
    trainer = Trainer(max_epochs=50, callbacks=[cb], enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(model, datamodule=data)

    # cb.epoch_weights will contain the weight values for each layer at each epoch
    # We assert that the values for the projection layer were changed at each epoch
    entry = cb.epoch_weights[len(model.T.S)]
    for jj in range(1, len(entry)):
        assert np.linalg.norm(entry[jj] - entry[jj - 1]) > 0


class TestMINEManifold:
    @pytest.mark.parametrize("n_samples", (512,))
    @pytest.mark.parametrize("d", (2, 3))
    @pytest.mark.parametrize("k", (1, 2))
    @pytest.mark.parametrize("n_epochs", (64,))
    def test_smoke(self, n_samples: int, d: int, k: int, n_epochs: int) -> None:

        if k == 1:
            fs = [lambda x: x[:, 0] ** 3]
        else:
            fs = [lambda x: np.arctan2(x[:, 0], x[:, 1])]

        dataset = synthetic.synthetic_dataset(n_samples, d, k=k, fs=fs)

        data = dm.GenericMINEDataModule(
            torch.from_numpy(dataset.X),
            torch.from_numpy(dataset.Z),
            p_train=0.7,
            p_val=0.1,
            batch_size=128,
            test_batch_size=1024,
        )

        # construct the MINE estimator
        S = mine.ManifoldStatisticsNetwork(
            S=nn.Sequential(
                nn.Linear(k + len(fs), 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
            ),
            d=d,
            k=k,
            out_dim=16,
        )

        model = mine.ManifoldMINE(
            T=S, kind=mine.MINEObjectiveType.MINE, alpha=0.01, learning_rate=1e-2, manifold_learning_rate=1e-2
        )

        initial_layer_weights = []
        for layer in model.T.S:
            if hasattr(layer, "weight"):
                initial_layer_weights.append(layer.weight.clone())

        initial_layer_weights.append(model.T.projection_layer.A.data.clone())

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

        assert torch.linalg.norm(model.T.projection_layer.A.data.clone() - initial_layer_weights[-1]) > 0

        # Assert that the return k-frame is column orthonormal
        A_hat = model.T.projection_layer.A.data.detach().cpu().numpy()
        assert torch.linalg.norm(torch.eye(k) - A_hat.T @ A_hat) == pytest.approx(0, abs=1e-3)
