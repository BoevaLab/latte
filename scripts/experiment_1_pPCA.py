"""
This is the script for generating the results for "Experiment 1" for the thesis.

It evaluates the solutions found by simple probabilistic PCA on data generated using the probabilistic PCA model.
It runs pPCA on the data and then rotates the found mixing matrix to align with the true one based on a labeled
subsample of the data.
"""
import dataclasses

import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
import geoopt

from sklearn.decomposition import PCA

from latte.dataset import synthetic
from latte.utils import visualisation, evaluation
from latte.models.subspace_orientation import Orientator
from latte.models.datamodules import GenericDataModule
import latte.hydra_utils as hy
import logging
import coloredlogs

coloredlogs.install()


_DEFAULT_TARGET = pathlib.Path("experiment_1")

seed = 2
rng = np.random.default_rng(seed)


@hy.config
@dataclasses.dataclass
class Experiment1Config:
    """
    The configuration of the synthetic data with subspace estimation experiment.

    Attributes:
        n: The number of points to generate.
        d: the dimensionality of the observed space.
        subspace_size: The dimensionality of the subspace ("k" above)
        assumed_subspace_size: The size of the subspace to be fit.
                               Since we do not know this in practice, it is interesting
                               to consider what happens when the model is misspecified
                               (i.e., assumed_subspace_size != subspace_size)
        orientation_ratio: The proportion of the data to use for orientation of the mixing matrix
        orientation_n: The number of samples to use for orientation of the mixing matrix (overrides `orientation_ratio`)
        learning_rate: The learning rate for the Orientator network
        noise: The scale of the gaussian noise to add to the observations
        max_epochs: The number of epochs to train the MINE model for
        seed: The seed for the random generators
    """

    n: int = 1024
    d: int = 3
    subspace_size: int = 2
    assumed_subspace_size: int = 2
    orientation_ratio: float = 0.1
    orientation_n: int = -1
    noise: float = 0.1
    loss: str = "mse"
    learning_rate: float = 1e-3
    max_epochs: int = 128
    seed: int = 42
    gpus: int = 1
    num_workers: int = 12


@hy.main
def main(cfg: Experiment1Config):
    assert (
        cfg.orientation_ratio != -1 or cfg.orientation_n != -1
    ), "At least one of the orientation_ratio or orientation_n has to be specified."
    assert cfg.loss in ["mse"], "The selected loss function is not supported."

    filepath = pathlib.Path(_DEFAULT_TARGET)
    filepath.mkdir(exist_ok=True, parents=True)

    # Generate the synthetic data based on the pPCA model
    dataset = synthetic.gaussian_data(cfg.n, cfg.d, cfg.subspace_size, sigma=cfg.noise, rng=cfg.seed)

    pca = PCA(n_components=cfg.assumed_subspace_size, random_state=cfg.seed)
    pca.fit(dataset.X)

    U_hat = pca.components_.T
    mu_hat = pca.mean_
    sigma_hat = np.sqrt(pca.noise_variance_)
    Lambda_hat = np.diag(pca.singular_values_**2) / cfg.n

    # Construct the original/raw pPCA estimate of the mixing matrix. See (Tipping, Bishop, 1999)
    A_hat = U_hat @ np.sqrt(Lambda_hat - sigma_hat**2 * np.eye(cfg.assumed_subspace_size))

    if cfg.d == 3 and cfg.subspace_size == 2:
        file_name = filepath / "original_mixing_estimate.png"
        visualisation.visualise_generative_model_in_3d(
            Z=dataset.Z,
            A=dataset.A,
            mu=dataset.mu,
            A_hat=A_hat,
            mu_hat=pca.mean_,
            file_name=file_name,
            title="Original pPCA mixing matrix estimate",
        )

    evaluation_result = evaluation.evaluate_found_subspace(U_hat, dataset.Q)
    logging.info("--- Quality of the fit. ---")
    print(evaluation_result)

    # We take the representations to be the means of the posteriors
    M_hat = A_hat.T @ A_hat + sigma_hat**2 * np.eye(cfg.assumed_subspace_size)
    Z_pca = (dataset.X - mu_hat) @ (np.linalg.inv(M_hat) @ A_hat.T).T  # The posterior mean in pPCA

    # Prepare the data to orient the mixing matrix
    orientation_n = int(cfg.orientation_ratio * cfg.n) if cfg.orientation_n == -1 else cfg.orientation_n
    orientation_ixs = rng.choice(cfg.n, orientation_n, replace=False)
    logging.info(f"Will orient the projection matrix based on {orientation_n} samples.")
    orientation_data = GenericDataModule(
        X=torch.from_numpy(Z_pca[orientation_ixs]),
        Y=torch.from_numpy(dataset.Z[orientation_ixs]),
        num_workers=cfg.num_workers,
    )

    loss = None
    if cfg.loss == "mse":
        loss = nn.MSELoss()
    orientation_model = Orientator(d=cfg.assumed_subspace_size, loss=loss, learning_rate=cfg.learning_rate)

    trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
    trainer.fit(orientation_model, datamodule=orientation_data)

    # Take out the found rotation matrix and construct the rotation-corrected mixing matrix
    R_hat = orientation_model.orientator.A.detach().cpu().numpy()
    A_hat_oriented = A_hat @ R_hat
    Z_oriented = Z_pca @ R_hat

    # Calculate some measures of fit (before and after rotation)
    original_loss = (np.linalg.norm(dataset.Z - Z_pca, axis=1) ** 2).mean()
    oriented_loss = (np.linalg.norm(dataset.Z - Z_oriented, axis=1) ** 2).mean()

    original_matrix_evaluation = evaluation.evaluate_found_mixing_matrix(dataset.A, A_hat)
    oriented_matrix_evaluation = evaluation.evaluate_found_mixing_matrix(dataset.A, A_hat_oriented)

    original_ll = synthetic.gaussian_log_likelihood(dataset.X, A_hat, pca.mean_, sigma_hat)
    oriented_ll = synthetic.gaussian_log_likelihood(dataset.X, A_hat_oriented, pca.mean_, sigma_hat)

    rotated_ll_mean = np.mean(
        [
            synthetic.gaussian_log_likelihood(
                dataset.X,
                A_hat
                @ geoopt.ManifoldParameter(
                    geoopt.Stiefel().random(cfg.assumed_subspace_size, cfg.assumed_subspace_size)
                )
                .detach()
                .numpy(),
                pca.mean_,
                np.sqrt(pca.noise_variance_),
            )
            for _ in range(100)
        ]
    )
    random_ll_mean = np.mean(
        [
            synthetic.gaussian_log_likelihood(
                dataset.X, rng.normal(size=(cfg.d, cfg.assumed_subspace_size)), pca.mean_, np.sqrt(pca.noise_variance_)
            )
            for _ in range(100)
        ]
    )

    results_dict = {
        "Value": {
            "Original MSE": original_loss,
            "MSE after orientation": oriented_loss,
            "Original log-likelihood": original_ll,
            "Log-likelihood after orientation": oriented_ll,
            "Average log-likelihood of a rotated mixing matrix": rotated_ll_mean,
            "Average log-likelihood of a random mixing matrix": random_ll_mean,
        }
    }
    results_dict["Value"].update(
        {key + " (original)": value for key, value in original_matrix_evaluation["Value"].items()}
    )
    results_dict["Value"].update(
        {key + " (oriented)": value for key, value in oriented_matrix_evaluation["Value"].items()}
    )

    results_df = pd.DataFrame(results_dict)

    logging.info("--- Results of the fit. ---")
    print(results_df)

    results_df.to_csv(pathlib.Path(_DEFAULT_TARGET) / "results_df.csv")

    if cfg.d == 3 and cfg.subspace_size == 2:
        file_name = filepath / "oriented_mixing_estimate.png"
        visualisation.visualise_generative_model_in_3d(
            Z=dataset.Z,
            A=dataset.A,
            mu=dataset.mu,
            A_hat=A_hat_oriented,
            mu_hat=pca.mean_,
            file_name=file_name,
            title="Oriented pPCA mixing matrix estimate",
        )


if __name__ == "__main__":
    main()
