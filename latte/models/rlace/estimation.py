from typing import Dict, Any
import dataclasses

import numpy as np
import torch

from latte.models.rlace import rlace
from latte.dataset import utils as dataset_utils
from latte.manifolds import utils as mutils
from latte.tools.ksg import naive_ksg


@dataclasses.dataclass
class RLACEResult:
    A_1: torch.Tensor
    E_1: torch.Tensor
    mutual_information: float
    complement_mutual_information: float


def fit(X: np.ndarray, Z: np.ndarray, rlace_params: Dict[str, Any], p_train: float = 0.8) -> RLACEResult:
    """
    Main function of the module.
    `rLACE` version of the `fit_subspace` function from `MIST` evaluation.

    Args:
        X: Samples of the first distribution.
        Z: Corresponding samples of the distribution in regard to which the mutual information should be maximised.
        rlace_params: Parameters for training the `rLACE` model.
        p_train: Proportion of the training data.

    Returns:
        A RLACEResult object containing the results of the estimation.
    """

    (X_train, X_test), (Z_train, Z_test) = dataset_utils.split([X, Z], p_train=p_train, p_val=1 - p_train)
    rlace_result = rlace.solve_adv_game(X_train, Z_train, **rlace_params)

    # Extract the found projection matrix
    E_hat_x = torch.from_numpy(rlace_result["P"]).float()
    P_hat_x = torch.eye(E_hat_x.shape[0]) - E_hat_x
    U, _, _ = torch.linalg.svd(P_hat_x)
    A_hat_x = U[:, : rlace_params.get("rank", 1)].float()

    # Assert we get a valid orthogonal matrix
    # We relax the condition for larger matrices since they are harder to keep close to orthogonal
    assert mutils.is_orthonormal(A_hat_x, atol=5e-3 + 1e-4 * A_hat_x.shape[1]), (
        f"A_hat_x.T @ A_hat_x = {A_hat_x.T @ A_hat_x}, "
        f"distance from orthogonal = {torch.linalg.norm(A_hat_x.T @ A_hat_x - torch.eye(A_hat_x.shape[1]))}"
    )

    X_test_projected_A, X_test_projected_E = X_test @ A_hat_x, X_test @ E_hat_x

    # Construct the result
    mi_estimate = RLACEResult(
        mutual_information=naive_ksg.mutual_information([X_test_projected_A.numpy(), Z_test.reshape(-1, 1)], k=3),
        complement_mutual_information=naive_ksg.mutual_information(
            [X_test_projected_E.numpy(), Z_test.reshape(-1, 1)], k=3
        ),
        A_1=A_hat_x,
        E_1=E_hat_x,
    )

    return mi_estimate
