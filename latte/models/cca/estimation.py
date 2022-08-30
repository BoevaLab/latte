import dataclasses

import numpy as np
import torch
from sklearn.cross_decomposition import CCA

from latte.utils import ksg_estimator
from latte.dataset import utils as dataset_utils


@dataclasses.dataclass
class CCAResult:
    A_1: torch.Tensor
    A_2: torch.Tensor
    E_1: torch.Tensor
    E_2: torch.Tensor
    mutual_information: float
    complement_mutual_information: float


def find_subspace(X: np.ndarray, Z: np.ndarray, subspace_size: int, p_train: float = 0.8) -> CCAResult:
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
    E_hat_1, E_hat_2 = np.eye(A_hat_1.shape[0]) - A_hat_1 @ A_hat_1.T, np.eye(A_hat_2.shape[0]) - A_hat_2 @ A_hat_2.T

    X_test_projected_A_1, X_test_projected_E_1 = X_test @ A_hat_1, X_test @ E_hat_1
    Z_test_projected_A_2, Z_test_projected_E_2 = Z_test @ A_hat_2, Z_test @ E_hat_2

    # Construct the result
    mi_estimate = CCAResult(
        mutual_information=ksg_estimator.mutual_information([X_test_projected_A_1, Z_test_projected_A_2], k=3),
        complement_mutual_information=ksg_estimator.mutual_information(
            [X_test_projected_E_1, Z_test_projected_E_2], k=3
        ),
        A_1=torch.from_numpy(A_hat_1).float(),
        E_1=torch.from_numpy(E_hat_1).float(),
        A_2=torch.from_numpy(A_hat_2).float(),
        E_2=torch.from_numpy(E_hat_2).float(),
    )

    return mi_estimate
