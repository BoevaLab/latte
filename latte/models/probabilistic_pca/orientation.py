"""
Convenience functions for orientating the original estimate of the probabilistic PCA mixing matrix towards the true one
using the Orientator class.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from latte.modules.data.datamodules import GenericDataModule
from latte.manifolds import utils as mutils
from latte.models.probabilistic_pca.orientator import Orientator
from latte.models.probabilistic_pca import probabilistic_pca


@dataclass
class OrientationResult:
    """A dataclass for holding the result of matrix rotation when estimating the true mixing matrix in pPCA.

    Members:
        R: The estimated rotation matrix.
        loss: The value of the loss with the returned matrix.
        stopped_epoch: The number of epochs performed when orienting.
                       Used to observe the effect of explicit manifold optimisation.
    """

    R: np.ndarray
    loss: float
    train_loss: float
    stopped_epoch: int


def _prepare_data(
    X: np.ndarray,
    Z: np.ndarray,
    A_hat: np.ndarray,
    mu_hat: np.ndarray,
    sigma_hat: float,
    p_train: float,
    p_val: float,
    batch_size: int,
    num_workers: int,
) -> GenericDataModule:
    """A utility function which prepares the data for orienting the mixing matrix found by pPCA.

    Args:
        X (np.ndarray): The observable data matrix of shape `N x n`.
        Z (np.ndarray): The matrix of values of the measured latent factors of shape `N x d`.
        A_hat (np.ndarray): The mixing matrix as estimated by pPCA (with columns corresponding to the principal axes).
        mu_hat (np.ndarray): The observable data mean as estimated by pPCA.
        sigma_hat (float): The observable data noise standard deviation as estimated by pPCA.
        p_train (float): The proportion of the data to use for training.
        p_val (float): The proportion of the data to use for validation.
        batch_size (int): Batch size to be used.
        num_workers (int): Number of workers for loading the data.

    Returns:
        GenericDataModule: A Pytorch Lightning DataModule.
    """

    # Get the latent representations as the means of the posteriors defined by `A_hat`
    Z_pca = probabilistic_pca.get_latent_representations(X, A_hat, mu_hat, sigma_hat)

    # Prepare the data
    # Learning a rotation matrix from the learned Z to the true Z.
    orientation_X = torch.from_numpy(Z_pca)
    orientation_Y = torch.from_numpy(Z)

    return GenericDataModule(
        X=orientation_X, Y=orientation_Y, num_workers=num_workers, batch_size=batch_size, p_train=p_train, p_val=p_val
    )


def orient(
    X: np.ndarray,
    Z: np.ndarray,
    A_hat: np.ndarray,
    mu_hat: np.ndarray,
    sigma_hat: float,
    p_train: float,
    p_val: float,
    batch_size: int = 128,
    learning_rate: float = 1e-2,
    min_delta: float = 1e-2,
    patience: int = 16,
    max_epochs: int = 256,
    num_workers: int = 1,
    loss: str = "mse",
    manifold_optimisation: bool = True,
    verbose: bool = False,
    gpus: int = 1,
) -> OrientationResult:
    """Unsupervised pPCA is only identifiable up to a rotation in the latent space.
    To identify the mixing matrix with the correct orientation in the latent space, a number of examples with measured
    values of the latent factors is needed.
    This function takes the estimated mixing matrix `A_hat` (which was fit by pPCA) and orients it by pre-multiplying it
    by the best rotation matrix found based on a number of observations of the latent factors.
    The optimisation is done over the Stiefel manifold of d-frames.
    The estimated mixing matrix `A_hat` can optionally project from a latent space which is *larger* then there are
    latent factors for which we have observations.
    In that case, a column-orthonormal matrix is found which optimally projects the latent factors inferred by `A_hat`
    to the observed ones.

    Args:
        X (np.ndarray): The observable data matrix of shape `N x n`.
        Z (np.ndarray): The matrix of values of the measured latent factors of shape `N x d`.
        A_hat (np.ndarray): The mixing matrix as estimated by pPCA (with columns corresponding to the principal axes).
        mu_hat (np.ndarray): The observable data mean as estimated by pPCA.
        sigma_hat (float): The observable data noise standard deviation as estimated by pPCA.
        p_train (float): The proportion of the data to use for training.
        p_val (float): The proportion of the data to use for validation.
        batch_size (int): Batch size to be used.
        learning_rate (float): Learning rate for optimisation over the Stiefel manifold.
        min_delta (float): `min_delta` for the learning rate scheduler.
        patience (int): `patience` for the learning rate scheduler.
        max_epochs (int): Maximal number of epochs to optimiser for.
        num_workers (int): Number of workers for loading the data.
        verbose (bool): Verbosity of the `pytorch lightning` Trainer
        loss (str, optional): The loss to use to quantify the fit between the inferred latent factors
                              and the true values. Defaults to "mse".
        manifold_optimisation: Whether to orient the estimate constrained to the Stiefel manifold or not.
                               This should be left True for almost all use cases, it is mostly here to enable
                               investigating the effect of the constrained optimisation versus the unconstrained one.
        gpus (int, optional_description_): Number of GPUs to use. Defaults to 1.

    Returns:
        OrientationResult: _description_
    """
    assert loss in ["mse"], "The selected loss function is not supported."
    if loss == "mse":
        loss_fn = nn.MSELoss()

    # We will be mapping from the latent space found by PCA
    # (which has at least as many factors as there are captured in Z) to the space of the observed factors in Z
    n_captured_directions, n_relevant_factors = A_hat.shape[1], Z.shape[1]
    orientation_data = _prepare_data(X, Z, A_hat, mu_hat, sigma_hat, p_train, p_val, batch_size, num_workers)

    # In case `n_captured_directions`` == `n_relevant_factors`, we are optimising over orthonormal matrices, which
    # form a disconnected manifold (those with determinant 1 and -1), so we have to train the model twice, with
    # different initialisations of the matrix; once with the identity matrix and once with the identity matrix with the
    # first element being -1
    signs = (-1, 1) if n_captured_directions == n_relevant_factors else (1,)
    R_hats, losses, train_losses, stopped_epochs = dict(), dict(), dict(), dict()
    for sign in signs:

        # Initialise the model with the appropriate initialisation of the matrix
        orientation_model = Orientator(
            n=n_captured_directions,
            d=n_relevant_factors,
            loss=loss_fn,
            init_sign=sign,
            learning_rate=learning_rate,
            manifold_optimisation=manifold_optimisation,
        )

        early_stopping_callback = EarlyStopping(
            monitor="validation_loss", min_delta=min_delta, patience=patience, verbose=False, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="validation_loss", mode="min")

        # Train the model
        trainer = Trainer(
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=max_epochs,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
            gpus=gpus,
        )

        # Train the model to find the optimal projection matrix
        trainer.fit(orientation_model, datamodule=orientation_data)

        train_losses[sign] = trainer.callback_metrics["train_loss_epoch"].item()

        # Evaluate the fit
        test_loss = trainer.test(ckpt_path="best", dataloaders=orientation_data, verbose=False)[0]["test_loss_epoch"]

        # Load the best model based on the validation split
        best_model = Orientator.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
            n=n_captured_directions,
            d=n_relevant_factors,
            loss=loss_fn,
            init_sign=sign,
            learning_rate=learning_rate,
        )

        # Assert that the found projection matrix is orthonormal
        assert not manifold_optimisation or mutils.is_orthonormal(best_model.orientator.A.detach().cpu(), atol=1e-2)
        # print(f"THE MATRIX ORTHOGONAL: {mutils.is_orthonormal(best_model.orientator.A.detach().cpu(), atol=1e-1)}.")

        # Extract the projection matrix of the best model
        R_hat = best_model.orientator.A.detach().cpu().numpy()

        R_hats[sign] = R_hat
        losses[sign] = test_loss
        stopped_epochs[sign] = early_stopping_callback.stopped_epoch

    # If n_captured_directions == n_relevant_factors, we did not need to run orientation twice
    # with different starting points
    return OrientationResult(
        R=R_hats[1] if n_captured_directions != n_relevant_factors or losses[1] < losses[-1] else R_hats[-1],
        loss=min(losses.values()),
        train_loss=train_losses[1]
        if n_captured_directions != n_relevant_factors or losses[1] < losses[-1]
        else train_losses[-1],
        stopped_epoch=stopped_epochs[1]
        if n_captured_directions != n_relevant_factors or losses[1] < losses[-1]
        else stopped_epochs[-1],
    )
