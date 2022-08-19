import pytest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pytorch_lightning import Trainer
import numpy as np
import torch

from latte.models.probabilistic_pca import probabilistic_pca, orientator, orientation
from latte.manifolds import utils as mutils
from latte.utils import evaluation
from latte.modules import callbacks as cbs
from latte.modules.data.datamodules import GenericDataModule


# @pytest.mark.skip("Too long to run")
@pytest.mark.parametrize("n", [6, 8, 10])
@pytest.mark.parametrize("d", [2, 3])
def test_loss_decrease(n: int, d: int):
    dataset = probabilistic_pca.generate(N=4096, n=n, d=d, d_measured=d, sigma=0.1, rng=1)

    X_fit, X_eval, Z_fit, Z_eval = train_test_split(dataset.X, dataset.Z, test_size=0.25, random_state=1)

    # We do not need `Z_train`, since we only use the "train" split for pPCA
    X_train, X_orient, _, Z_orient = train_test_split(X_fit, Z_fit, test_size=0.5, random_state=1)

    # Run probabilistic PCA to get the principal components of the observed data
    pca = PCA(n_components=d, random_state=1)
    pca.fit(X_train)

    # Construct the original/raw pPCA estimate of the mixing matrix. See (Tipping, Bishop, 1999)
    A_hat = probabilistic_pca.construct_mixing_matrix_estimate(pca)

    Z_pca = probabilistic_pca.get_latent_representations(dataset.X, A_hat, pca.mean_, pca.noise_variance_**0.5)

    # Prepare the data
    # Learning a rotation matrix from the learned Z to the true Z.
    orientation_X = torch.from_numpy(Z_pca)
    orientation_Y = torch.from_numpy(dataset.Z)

    orientation_data = GenericDataModule(X=orientation_X, Y=orientation_Y)

    r = {1: True, -1: True}
    for sign in [-1, 1]:
        # Initialise the model with the appropriate initialisation of the matrix
        orientation_model = orientator.Orientator(
            n=d,
            d=d,
            loss=torch.nn.MSELoss(),
            init_sign=sign,
            learning_rate=1e-2,
            manifold_optimisation=True,
        )

        cb = cbs.MetricTracker()
        # Train the model
        trainer = Trainer(
            callbacks=[cb],
            max_epochs=32,
            enable_progress_bar=False,
            enable_model_summary=False,
            gpus=0,
        )

        # Train the model to find the optimal projection matrix
        trainer.fit(orientation_model, datamodule=orientation_data)

        for ii in [16, 24]:
            # assert that the mutual information estimate has increased on average compared to earlier epochs
            r[sign] &= sum(
                [cb.epoch_outputs[ii - d]["validation_loss_epoch"].detach().cpu().numpy() for d in range(-4, 5)]
            ) < sum(
                [cb.epoch_outputs[ii - 12 - d]["validation_loss_epoch"].detach().cpu().numpy() for d in range(-4, 5)]
            )

    assert r[-1] or r[1]


# @pytest.mark.skip("Too long to run")
@pytest.mark.parametrize("n", [6, 8, 10])
@pytest.mark.parametrize("d", [2, 3])
def test_orientation(n: int, d: int) -> None:
    dataset = probabilistic_pca.generate(N=10000, n=n, d=d, d_measured=d, sigma=0.1, rng=1)

    X_fit, X_eval, Z_fit, Z_eval = train_test_split(dataset.X, dataset.Z, test_size=0.25, random_state=1)

    # We do not need `Z_train`, since we only use the "train" split for pPCA
    X_train, X_orient, _, Z_orient = train_test_split(X_fit, Z_fit, test_size=0.5, random_state=1)

    # Run probabilistic PCA to get the principal components of the observed data
    pca = PCA(n_components=d, random_state=1)
    pca.fit(X_train)

    # Construct the original/raw pPCA estimate of the mixing matrix. See (Tipping, Bishop, 1999)
    A_hat = probabilistic_pca.construct_mixing_matrix_estimate(pca)

    # Orient the computed mixing matrix to be able to compute the true factors of variation
    orientation_result = orientation.orient(
        X=X_orient,
        Z=Z_orient[:, dataset.measured_factors],
        A_hat=A_hat,
        mu_hat=pca.mean_,
        sigma_hat=np.sqrt(pca.noise_variance_),
        p_train=0.8,
        p_val=0.1,
        max_epochs=256,
        verbose=False,
        gpus=0,
    )

    R_hat = orientation_result.R
    assert mutils.is_orthonormal(torch.from_numpy(R_hat))

    A_hat_oriented = A_hat @ R_hat
    stopped_epoch = orientation_result.stopped_epoch

    original_error = evaluation.mixing_matrix_fit(dataset.A, A_hat)["Value"]["Difference between matrices"]
    oriented_error = evaluation.mixing_matrix_fit(dataset.A, A_hat_oriented)["Value"]["Difference between matrices"]

    assert stopped_epoch < 255
    assert oriented_error < original_error
    assert oriented_error < 1e-1
