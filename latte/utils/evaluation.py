"""
Code for evaluating various general results and solutions used in `Latte`.
"""

from typing import List
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch
from scipy import optimize

from sklearn import feature_selection


def subspace_fit(Q: np.ndarray, Q_hat: np.ndarray) -> pd.DataFrame:
    """
    Evaluates the goodness of fit of the found subspace spanned by the estimate of the matrix `A_hat`
    compared to the ground-truth one.
    Args:
        Q: The ground-truth projection matrix
        Q_hat: The orthogonal basis of the estimated subspace

    Returns:
        A pandas dataframe containing two evaluation metrics:
            - the norm of the difference between the projection matrices *in the original space*
            - the same norm normalised by the norm of the ground-truth mixing matrix
    """

    # We look at the distance between the subspace and *its* projection onto the found subspace defined by `Q_hat`
    subspace_distance = np.linalg.norm(Q @ Q.T - Q_hat @ Q_hat.T @ Q @ Q.T)
    normalised_subspace_distance = np.linalg.norm(Q @ Q.T - Q_hat @ Q_hat.T @ Q @ Q.T) / np.linalg.norm(Q @ Q.T)
    return pd.DataFrame(
        {
            "Value": {
                "Difference between subspaces": subspace_distance,
                "Normalised difference between subspaces": normalised_subspace_distance,
            }
        }
    )


def mixing_matrix_fit(A: np.ndarray, A_hat: np.ndarray) -> pd.DataFrame:
    """
    Evaluates the goodness of fit of the found mixing matrix `A_hat` compared to the ground-truth matrix `A`,
    for example in probabilistic PCA.
    Args:
        A: The true mixing matrix
        A_hat: The estimated mixing matrix

    Returns:
        A pandas dataframe containing two evaluation metrics:
            - the norm of the difference between the matrices
            - the same norm normalised by the norm of the true mixing matrix
    """
    subspace_distance = np.linalg.norm(A - A_hat)
    normalised_subspace_distance = np.linalg.norm(A - A_hat) / np.linalg.norm(A)
    return pd.DataFrame(
        {
            "Value": {
                "Difference between matrices": subspace_distance,
                "Normalised difference between matrices": normalised_subspace_distance,
            }
        }
    )


def axis_factor_mutual_information(Z: torch.Tensor, Y: torch.Tensor, factor_names: List[str]) -> pd.DataFrame:
    """
    Calculates the mutual information between all pairs of the provided axes in the data `Z` and the factor `Y`.
    Args:
        Z: A `N x d` matrix containing the data.
        Y: A `N x f` matrix containing ground truth factors of variation for each data point.
        factor_names: Human-understandable factor names for easier interpretation of the produced matrix.

    Returns:
        An `f x d` data frame containing the estimate of the mutual information between every axis in `Z` and every
        factor in `Y`.
    """

    mis = defaultdict(lambda: defaultdict(float))
    for ii, (factor_name, axis) in enumerate(product(factor_names, range(Z.shape[1]))):
        mis[f"Axis {axis}"][factor_name] = feature_selection.mutual_info_regression(
            Y[:, ii // Z.shape[1]].reshape((-1, 1)), Z[:, axis]
        )[0]
        # Since mutual_info_regression returns 1 value per feature (in this,
        # we simulate 1 feature, we extract the first one

    return pd.DataFrame(mis)


def distance_to_permutation_matrix(R: torch.Tensor) -> float:
    """
    Calculates the Frobenius norm of the closest *signed* permutation matrix as measured by the Fribenius norm.
    The closest signed permutation matrix is computed using the linear sum assignment algorithm with the cost
    matrix defined by the contributions of the permutation matrix entries to the Frobenius norm.
    *Importantly*, this assumes that the matrix R is *orthogonal*.
    Implemented with the help of
    https://math.stackexchange.com/questions/2951811/nearest-signed-permutation-matrix-to-a-given-matrix-a.

    Args:
        R (np.ndarray): The projection matrix to evaluate.

    Returns:
        float: The Frobenius norm to the closest signed permutation matrix.
    """
    # assert mutils.is_orthonormal(R, atol=5e-2)

    # The cost matrix for the Hungarian algorithm
    # In the general case, the first term would include the *squared L2 row norms*, however, in the orthogonal case,
    # they are zero
    # See the link above to see what the elements of this matrix corresponds to
    # C = 1 - R**2 + (torch.abs(R) - 1) ** 2

    # The general case
    C = torch.linalg.norm(R, dim=1).reshape(-1, 1) ** 2 - R**2 + (R.abs() - 1) ** 2

    M = optimize.linear_sum_assignment(C)  # M includes the rows and columns of where ones should go

    P = torch.zeros(R.shape)
    P[M] = 1  # Put ones into the entries where they should be
    P *= torch.sign(R)  # Multiply with the sign to take into account the absolute values from the calculation of `C`

    return torch.linalg.norm(R - P).item()
