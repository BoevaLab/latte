from typing import Dict, Any
import dataclasses

import numpy as np
import torch

from latte.models.rlace import rlace
from latte.dataset import utils as dataset_utils
from latte.manifolds import utils as mutils
from latte.tools.ksg import ksg
from latte.dataset.utils import RandomGenerator


@dataclasses.dataclass
class RLACEResult:
    """
    A dataclass for holding the result of mutual information estimation using rLACE.
    A_1: In case of also estimating the linear subspace capturing the most information, this holds the
       d-frame defining the estimated subspace of the first distribution
    E_1: In case of also estimating the linear subspace capturing the most information, this holds the
       `n x n` projection matrix defining the complement of the estimated subspace of the first distribution
    """

    A_1: torch.Tensor
    E_1: torch.Tensor
    mutual_information: float
    complement_mutual_information: float


def fit(
    X: np.ndarray, Z: np.ndarray, rlace_params: Dict[str, Any], p_train: float = 0.8, rng: RandomGenerator = 1
) -> RLACEResult:
    """
    Main function of the module.
    `rLACE` version of the `fit_subspace` function from `MIST` evaluation.

    Args:
        X: Samples of the first distribution.
        Z: Corresponding samples of the distribution in regard to which the mutual information should be maximised.
        rlace_params: Parameters for training the `rLACE` model.
        p_train: Proportion of the training data.
        rng: The random generator.

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
    d, m = A_hat_x.shape

    # Assert we get a valid orthogonal matrix
    # We relax the condition for larger matrices since they are harder to keep close to orthogonal
    assert mutils.is_orthonormal(A_hat_x, atol=5e-3 + 1e-4 * A_hat_x.shape[1]), (
        f"A_hat_x.T @ A_hat_x = {A_hat_x.T @ A_hat_x}, "
        f"distance from orthogonal = {torch.linalg.norm(A_hat_x.T @ A_hat_x - torch.eye(A_hat_x.shape[1]))}"
    )

    X_test_projected_A, X_test_projected_E = X_test @ A_hat_x, X_test @ E_hat_x

    # Construct the result
    rng = np.random.default_rng(rng)
    mi_estimation_ix = rng.choice(len(Z_test), min(8000, len(Z_test)), replace=False)
    mi_estimate = RLACEResult(
        mutual_information=ksg.estimate_mi_ksg(
            X_test_projected_A[mi_estimation_ix].numpy(),
            Z_test[mi_estimation_ix].reshape(-1, 1).numpy()
            if isinstance(Z_test, torch.Tensor)
            else Z_test[mi_estimation_ix].reshape(-1, 1),
            neighborhoods=(5,),
        )[5],
        complement_mutual_information=ksg.estimate_mi_ksg(
            X_test_projected_E[mi_estimation_ix].numpy(),
            Z_test[mi_estimation_ix].reshape(-1, 1).numpy()
            if isinstance(Z_test, torch.Tensor)
            else Z_test[mi_estimation_ix].reshape(-1, 1),
            neighborhoods=(5,),
        )[5],
        A_1=A_hat_x,
        E_1=torch.linalg.svd(E_hat_x, full_matrices=False)[0][:, : d - m],
    )

    return mi_estimate
