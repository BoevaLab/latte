"""
This module contains main wrapper functions around the `MIST` and `CCA` models for finding the commons subspaces of
two representations spaces based either on mutual information or correlation.
"""

from typing import List, Tuple, Optional, Union

import torch
import numpy as np
from sklearn import preprocessing

from latte.models.mist import estimation as mist_estimation
from latte.models.cca import estimation as cca_estimation


def find_common_subspaces_with_mutual_information(
    Z_1: Union[np.ndarray, torch.Tensor],
    Z_2: Union[np.ndarray, torch.Tensor],
    subspace_sizes: Tuple[int, int] = (1, 1),
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Finds the subspaces of the representation space `Z_1` and `Z_2` which contain the most mutual information among all
    linear subspaces of the specified dimensionality.
    It uses `MIST` to find the two subspaces.

    Args:
        Z_1: Dataset of samples from the first distribution.
        Z_2: Dataset of samples from the second distribution matched to the ones from the first one.
        subspace_sizes: The 2-tuple of subspace dimensionalities to find.
                        The first element specifies the dimensionality of the subspace of the first representation space
                        to look for and the second element the dimensionality of the subspace of the second
                        representation space.
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

    assert Z_1.shape[0] == Z_2.shape[0]

    if standardise:
        # Standardise the data
        Z_1 = Z_1.numpy() if isinstance(Z_1, torch.Tensor) else Z_1
        Z_2 = Z_2.numpy() if isinstance(Z_2, torch.Tensor) else Z_2
        Z_1 = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z_1)).float()
        Z_2 = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z_2)).float()
    else:
        Z_1 = torch.from_numpy(Z_1) if isinstance(Z_1, np.ndarray) else Z_1
        Z_2 = torch.from_numpy(Z_2) if isinstance(Z_2, np.ndarray) else Z_2

    if fitting_indices is None:
        # If no subset is provided, train on the entire dataset.
        fitting_indices = list(range(Z_1.shape[0]))

    # Use `MIST` estimation to find the subspaces with the highest mutual information and the mutual information itself.
    mist_result = mist_estimation.fit(
        X=Z_1[fitting_indices],
        Z_max=Z_2[fitting_indices],
        datamodule_kwargs=dict(
            num_workers=num_workers, p_train=0.4, p_val=0.2, batch_size=batch_size, test_batch_size=test_batch_size
        ),
        model_kwargs=dict(
            x_size=Z_1.shape[1],
            z_max_size=Z_2.shape[1],
            subspace_size=subspace_sizes,
            mine_network_width=mine_network_width,
            mine_learning_rate=mine_learning_rate,
            manifold_learning_rate=manifold_learning_rate,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_delta=lr_scheduler_min_delta,
            verbose=False,
        ),
        trainer_kwargs=dict(max_epochs=max_epochs, enable_model_summary=False, enable_progress_bar=False, gpus=gpus),
        early_stopping_kwargs={
            "min_delta": early_stopping_min_delta,
            "patience": early_stopping_patience,
            "verbose": False,
        },
    )

    return mist_result.A_1, mist_result.E_1, mist_result.A_2, mist_result.E_2, mist_result.mutual_information


def find_common_subspaces_with_correlation(
    Z_1: torch.Tensor,
    Z_2: torch.Tensor,
    subspace_size: int = 1,
    standardise: bool = True,
    fitting_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds the subspaces of the representation space `Z_1` and `Z_2` which are most correlated among all linear
    subspaces of the specified dimensionality.
    The subspaces are found using `CCA`.

    Args:
        Z_1: Dataset of samples from the first distribution.
        Z_2: Dataset of samples from the second distribution matched to the ones from the first one.
        subspace_size: The dimensionalities of the subspaces to find.
                       The dimensionalities of both subspaces have to be the same in CCA.
        standardise: Whether to standardise the data before estimation.
        fitting_indices: Optional sequence of indices of the samples to use to perform estimation
                         on a subset of the data.

    Returns:
        The projection matrix onto the common subspaces and their complements (their orthogonal bases)
        of the two representations spaces.
        Returned in the sequence
            (
                projection matrix onto the common subspace of representation space 1,
                projection matrix onto the complement of the common subspace of representation space 1,
                projection matrix onto the common subspace of representation space 2,
                projection matrix onto the complement of the common subspace of representation space 2,
            )

    """

    assert Z_1.shape[0] == Z_2.shape[0]

    if standardise:
        # Standardise the data
        Z_1 = preprocessing.StandardScaler().fit_transform(Z_1.numpy())
        Z_2 = preprocessing.StandardScaler().fit_transform(Z_2.numpy())

    if fitting_indices is None:
        # If no subset is provided, train on the entire dataset.
        fitting_indices = list(range(Z_1.shape[0]))

    # Use `MIST` estimation to find the subspaces with the highest correlation.
    cca_result = cca_estimation.fit(Z_1[fitting_indices], Z_2[fitting_indices], subspace_size)

    return cca_result.A_1, cca_result.E_1, cca_result.A_2, cca_result.E_2
