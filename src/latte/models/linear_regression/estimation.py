import dataclasses

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from latte.tools.ksg import naive_ksg
from latte.dataset import utils as dataset_utils


@dataclasses.dataclass
class LinearRegressionResult:
    """
    A dataclass for holding the result of mutual information estimation using linear regression.
    A_1: In case of also estimating the linear subspace capturing the most information, this holds the
       d-frame defining the estimated subspace of the first distribution
    E_1: In case of also estimating the linear subspace capturing the most information, this holds the
       `n x n` projection matrix defining the complement of the estimated subspace of the first distribution
    """

    A_1: torch.Tensor
    E_1: torch.Tensor
    mutual_information: float
    complement_mutual_information: float


def fit(X: np.ndarray, Z: np.ndarray, p_train: float = 0.8) -> LinearRegressionResult:
    """
    Main function of the module.
    Linear regression version of the `fit_subspace` function from `MIST` evaluation.

    Args:
        X: Samples of the first distribution.
        Z: Corresponding samples of the distribution in regard to which the mutual information should be maximised.
        p_train: Proportion of the training data.

    Returns:
        A LinearRegressionResult object containing the results of the estimation.
    """
    assert len(Z.shape) == 1

    (X_train, X_test), (Z_train, Z_test) = dataset_utils.split([X, Z], p_train=p_train, p_val=1 - p_train)

    lr = LinearRegression()
    lr.fit(X, Z)

    A = (lr.coef_.reshape(-1, 1) / np.linalg.norm(lr.coef_)).astype(np.float32)  # Make unit length
    d, m = A.shape
    E = np.linalg.svd(np.eye(A.shape[0], dtype=np.float32) - A @ A.T, full_matrices=False)[0][:, : d - m]

    X_test_projected_A, X_test_projected_E = X_test @ A, X_test @ E

    # Construct the result
    mi_estimate = LinearRegressionResult(
        mutual_information=naive_ksg.mutual_information([X_test_projected_A, Z_test.reshape(-1, 1)], k=3),
        complement_mutual_information=naive_ksg.mutual_information([X_test_projected_E, Z_test.reshape(-1, 1)], k=3),
        A_1=torch.from_numpy(A).float(),
        E_1=torch.from_numpy(E).float(),
    )

    return mi_estimate
