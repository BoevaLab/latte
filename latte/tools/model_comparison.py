# TODO (Anej): Documentation

from typing import List, Tuple

import torch
from sklearn import preprocessing

from latte.models.mist import estimation as mist_estimation
from latte.models.cca import estimation as cca_estimation


# TODO (Anej): Paramterise
def find_subspaces_with_mutual_information(
    Z_1: torch.Tensor,
    Z_2: torch.Tensor,
    ixs: List[int],
    subspace_size: Tuple[int, int] = (1, 1),
    standardise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:

    if standardise:
        Z_1 = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z_1.numpy())).float()
        Z_2 = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z_2.numpy())).float()

    mist_result = mist_estimation.find_subspace(
        X=Z_1[ixs],
        Z_max=Z_2[ixs],
        datamodule_kwargs=dict(num_workers=1, p_train=0.4, p_val=0.2, batch_size=128, test_batch_size=8192),
        model_kwargs=dict(
            x_size=Z_1.shape[1],
            z_max_size=Z_2.shape[1],
            subspace_size=subspace_size,
            mine_network_width=200,
            mine_learning_rate=1e-4,
            manifold_learning_rate=1e-2,
            lr_scheduler_patience=6,
            lr_scheduler_min_delta=5e-3,
            alpha=1e-2,
            verbose=True,
        ),
        trainer_kwargs=dict(max_epochs=128, enable_model_summary=False, enable_progress_bar=False, gpus=1),
        early_stopping_kwargs={"min_delta": 5e-3, "patience": 12, "verbose": False},
    )

    print(f"The estimated mutual information is {mist_result.mutual_information: .3f}.")

    return mist_result.A_1, mist_result.E_1, mist_result.A_2, mist_result.E_2, mist_result.mutual_information


def find_correlated_subspaces(
    Z_1: torch.Tensor,
    Z_2: torch.Tensor,
    ixs: List[int],
    subspace_size: int = 1,
    standardise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if standardise:
        Z_1 = preprocessing.StandardScaler().fit_transform(Z_1.numpy())
        Z_2 = preprocessing.StandardScaler().fit_transform(Z_2.numpy())

    cca_result = cca_estimation.find_subspace(Z_1[ixs], Z_2[ixs], subspace_size)

    return cca_result.A_1, cca_result.E_1, cca_result.A_2, cca_result.E_2
