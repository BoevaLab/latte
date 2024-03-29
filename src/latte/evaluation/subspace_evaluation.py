"""
Code for evaluating various general results and solutions used in `Latte`.
"""

from typing import List
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch

from sklearn import feature_selection


def subspace_fit(U: np.ndarray, U_hat: np.ndarray) -> pd.DataFrame:
    """
    Evaluates the goodness of fit of the found subspace spanned by the estimate of the matrix `U_hat`
    compared to the ground-truth one.
    Args:
        U: The ground-truth projection matrix
        U_hat: The orthogonal basis of the estimated subspace

    Returns:
        A pandas dataframe containing two evaluation metrics:
            - the norm of the difference between the projection matrices *in the original space*
            - the same norm normalised by the norm of the ground-truth mixing matrix
    """

    # We look at the distance between the subspace and *its* projection onto the found subspace defined by `U_hat`
    subspace_distance = np.linalg.norm(U @ U.T - U_hat @ U_hat.T @ U @ U.T)
    normalised_subspace_distance = np.linalg.norm(U @ U.T - U_hat @ U_hat.T @ U @ U.T) / np.linalg.norm(U @ U.T)
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
