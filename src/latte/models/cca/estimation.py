import dataclasses

import numpy as np
import torch
from sklearn.cross_decomposition import CCA

from latte.tools.ksg import naive_ksg
from latte.dataset import utils as dataset_utils
from latte.utils import subspaces


@dataclasses.dataclass
class CCAResult:
    """A dataclass for holding the result of mutual information estimation using CCA

    Members:
        mutual_information (float): The estimate of the mutual information in the optimal as calculated on the
                                    test dataset
        mutual_information (float): The estimate of the mutual information in the complement of the optimal subspace
                                    as calculated on the test dataset
        A_1: In case of also estimating the linear subspace capturing the most information, this holds the
           d-frame defining the estimated subspace of the first distribution
        A_2: In case of also estimating the linear subspace capturing the most information, this holds the
           d-frame defining the estimated subspace of the second distributions
        E_1: In case of also estimating the linear subspace capturing the most information, this holds the
           `n x n` projection matrix defining the complement of the estimated subspace of the first distribution
        E_2: In case of also estimating the linear subspace capturing the most information, this holds the
           `n x n` projection matrix defining the complement of the estimated subspace of the second distributions
    """

    A_1: torch.Tensor
    A_2: torch.Tensor
    E_1: torch.Tensor
    E_2: torch.Tensor
    mutual_information: float
    complement_mutual_information: float


def fit(X: np.ndarray, Z: np.ndarray, subspace_size: int, p_train: float = 0.8) -> CCAResult:
    """
    Main function of the module.
    Linear CCA version of the `fit_subspace` function from `MIST` evaluation.

    Args:
        X: Samples of the first distribution.
        Z: Corresponding samples of the distribution in regard to which the mutual information should be maximised.
        subspace_size: The number of components to estimate.
        p_train: Proportion of the training data.

    Returns:
        A_1 CCAResult object containing the results of the estimation.
    """

    (X_train, X_test), (Z_train, Z_test) = dataset_utils.split([X, Z], p_train=p_train, p_val=1 - p_train)

    cca = CCA(n_components=subspace_size)
    cca.fit(X_train, Z_train)

    A_hat_1, A_hat_2 = cca.x_weights_, cca.y_weights_
    (d_1, m_1), (d_2, m_2) = A_hat_1.shape, A_hat_2.shape
    E_hat_1 = subspaces.principal_subspace_basis(np.eye(d_1) - A_hat_1 @ A_hat_1.T, d_1 - m_1)
    E_hat_2 = subspaces.principal_subspace_basis(np.eye(d_2) - A_hat_2 @ A_hat_2.T, d_2 - m_2)

    X_test_projected_A_1, X_test_projected_E_1 = X_test @ A_hat_1, X_test @ E_hat_1
    Z_test_projected_A_2, Z_test_projected_E_2 = Z_test @ A_hat_2, Z_test @ E_hat_2

    # Construct the result
    mi_estimate = CCAResult(
        mutual_information=naive_ksg.mutual_information([X_test_projected_A_1, Z_test_projected_A_2], k=3),
        complement_mutual_information=naive_ksg.mutual_information([X_test_projected_E_1, Z_test_projected_E_2], k=3),
        A_1=torch.from_numpy(A_hat_1).float(),
        E_1=torch.from_numpy(E_hat_1).float(),
        A_2=torch.from_numpy(A_hat_2).float(),
        E_2=torch.from_numpy(E_hat_2).float(),
    )

    return mi_estimate
