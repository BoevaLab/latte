"""
This module contains main wrapper functions around the `MIST` model for finding the linear subspace of a representation
space capturing the most mutual information about a set of known factors of variation.
"""

from typing import Optional, List

import torch
from sklearn import preprocessing

from latte.models.mist import estimation as mist_estimation


def find_subspace(
    Z: torch.Tensor,
    Y: torch.Tensor,
    subspace_size: int,
    standardise: bool = True,
    fitting_indices: Optional[List[int]] = None,
    max_epochs: int = 256,
    mine_network_width: int = 200,
    mine_learning_rate: float = 1e-4,
    manifold_learning_rate: float = 1e-2,
    lr_scheduler_patience: int = 6,
    lr_scheduler_min_delta: float = 5e-3,
    early_stopping_patience: int = 12,
    early_stopping_min_delta: float = 5e-3,
    batch_size: int = 128,
    test_batch_size: int = 8192,
    num_workers: int = 1,
    gpus: int = 1,
) -> mist_estimation.MISTResult:
    """
    Finds the subspaces of the representation space with samples captured in `Z` which contain the most information
    about the true factors of variation with samples captured in `Y` among all linear subspaces of the
    specified dimensionality.
    It uses `MIST` to find the subspace.

    Args:
        Z: Dataset of samples from the distribution of the representations.
        Y: Dataset of matching values of the true factors of variation.
        subspace_size: The dimensionality subspace of `Z` to find.
        standardise: Whether to standardise the data before estimation.
        fitting_indices: Optional sequence of indices of the samples to use to perform estimation
                         on a subset of the data.
        max_epochs: The maximum number of epochs to train the MINE model for.
        mine_network_width: Size of the hidden layers of the standard `MINE` network.
        mine_learning_rate: Learning rate for the parameters of the `MINE` model.
        manifold_learning_rate: Learning rate for the projection matrix.
        lr_scheduler_patience: Patience for the learning rate schedulers.
        lr_scheduler_min_delta: Min delta for the learning rate schedulers.
        early_stopping_patience: The patience (number of epoch without a significant reduction of loss) for the `MIST`
                                 subspace estimation.
        early_stopping_min_delta: The minimal loss reduction between epochs for the `MIST` subspace estimation.
        batch_size: Batch size for the `MIST` optimisation procedure.
        test_batch_size: The test batch size for the `MIST` optimisation procedure.
        num_workers: Number of workers to use for `MIST` training.
        gpus: Number of `gpus` to use for `MIST` training.

    Returns:
        The projection matrix onto the common subspaces and their complements (their orthogonal bases)
        of the two representations spaces, and the mutual information estimated using `MINE`.
        Returned in the sequence
            (
                projection matrix onto the common subspace of representation space 1,
                projection matrix onto the complement of the common subspace of representation space 1,
                projection matrix onto the common subspace of representation space 2,
                projection matrix onto the complement of the common subspace of representation space 2,
                mutual information estimate
            )
    """

    assert Z.shape[0] == Y.shape[0]

    if standardise:
        # Standardise the data
        Z = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z.numpy())).float()
        Y = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Y.numpy())).float()

    if fitting_indices is None:
        # If no subset is provided, train on the entire dataset.
        fitting_indices = list(range(Z.shape[0]))

    mi_estimation_result = mist_estimation.fit(
        X=Z[fitting_indices],
        Z_max=Y[fitting_indices],
        datamodule_kwargs=dict(
            num_workers=num_workers, batch_size=batch_size, test_batch_size=test_batch_size, p_train=0.4, p_val=0.2
        ),
        model_kwargs=dict(
            x_size=Z.shape[1],
            z_max_size=Y.shape[1],
            subspace_size=subspace_size,
            mine_network_width=mine_network_width,
            mine_learning_rate=mine_learning_rate,
            manifold_learning_rate=manifold_learning_rate,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_delta=lr_scheduler_min_delta,
            verbose=False,
        ),
        trainer_kwargs=dict(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            gpus=gpus,
        ),
        early_stopping_kwargs={
            "min_delta": early_stopping_min_delta,
            "patience": early_stopping_patience,
            "verbose": False,
        },
    )

    return mi_estimation_result
