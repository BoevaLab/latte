"""
This module implements various representation similarity metrics (most of them based on CCA) as described in Chapter 32
of Murphy. 2023. Probabilistic Machine Learning: Advanced Topics.
"""

import numpy as np


def _get_rho(Z_1: np.ndarray, Z_2: np.ndarray) -> np.ndarray:
    return np.asarray([np.corrcoef(Z_1[:, ii], Z_2[:, ii])[0, 1] for ii in range(Z_1.shape[1])])


def mean_squared_canonical_correlation(Z_1: np.ndarray, Z_2: np.ndarray) -> float:
    """
    Computes the mean squared canonical correlation (MSCC) score between the projections onto the two most-correlated
    subspaces `Z_1` and `Z_2`.
    Based on Murphy. 2023., Chapter 32, Eq. 32.4.
    Args:
        Z_1: The *projections* onto the subspace of the first representation space with the highest correlation.
        Z_2: The *projections* onto the subspace of the second representation space with the highest correlation.

    Returns:
        The MSCC score.
    """
    assert Z_1.shape[1] == Z_2.shape[1]

    # Compute the axis-wise correlations.
    rhos = _get_rho(Z_1, Z_2)

    # Compute MSCC.
    return np.linalg.norm(rhos, 2) ** 2 / len(rhos)


def mean_canonical_correlation(Z_1: np.ndarray, Z_2: np.ndarray) -> float:
    """
    Computes the mean canonical correlation (MCC) score between the projections onto the two most-correlated
    subspaces `Z_1` and `Z_2`.
    Based on Murphy. 2023., Chapter 32, Eq. 32.5
    Args:
        Z_1: The *projections* onto the subspace of the first representation space with the highest correlation.
        Z_2: The *projections* onto the subspace of the second representation space with the highest correlation.

    Returns:
        The MCC score.
    """
    assert Z_1.shape[1] == Z_2.shape[1]

    # Compute the axis-wise correlations.
    rhos = _get_rho(Z_1, Z_2)

    # Compute MCC.
    # TODO: Absolute values of rhos?
    return np.sum(rhos) / len(rhos)


def projection_weighted_canonical_correlation_analysis(
    Z_1: np.ndarray,
    Z_2: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
) -> float:
    """
    Computes the projection weighted canonical correlation analysis (PWCCA) score between the projections onto the two
    most-correlated subspaces `Z_1` and `Z_2`.
    Based on Murphy. 2023., Chapter 32, Eq. 32.7
    Args:
        Z_1: The *projections* onto the subspace of the first representation space with the highest correlation.
        Z_2: The *projections* onto the subspace of the second representation space with the highest correlation.
        X: The matching *original* representations of *one* of the two representation spaces.
           Any of the two representation spaces can be used, however, the result is not identical.
        W: The matrix whose columns are the orthogonal vectors spanning the CCA subspace in X.

    Returns:
        The PWCCA score.
    """
    assert Z_1.shape[1] == Z_2.shape[1]

    # Compute the axis-wise correlations.
    rhos = _get_rho(Z_1, Z_2)

    # Compute the per-axis weights
    alphas = np.sum(np.abs((X @ W).T @ X), axis=1)

    # Compute PWCCA
    return float(np.sum(alphas * rhos / alphas.sum()))


def projection_weighted_linear_regression(
    Z_1: np.ndarray,
    Z_2: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
) -> float:
    """
    Computes the projection weighted linear regression (PWLR) score between the projections onto the two
    most-correlated subspaces `Z_1` and `Z_2`.
    Similar to projection_weighted_canonical_correlation_analysis, however, the way the weights are computed and
    used is different.
    Based on Murphy. 2023., Chapter 32, Eq. 32.8.
    Args:
        Z_1: The *projections* onto the subspace of the first representation space with the highest correlation.
        Z_2: The *projections* onto the subspace of the second representation space with the highest correlation.
        X: The matching *original* representations of *one* of the two representation spaces.
           Any of the two representation spaces can be used, however, the result is not identical.
        W: The matrix whose columns are the orthogonal vectors spanning the CCA subspace in X.

    Returns:
        The PWCCA score.
    """
    assert Z_1.shape[1] == Z_2.shape[1]

    # Compute the axis-wise correlations.
    rhos = _get_rho(Z_1, Z_2)

    # Compute the axis-wise weights
    alphas = np.linalg.norm((X @ W).T @ X, 2, axis=1) ** 2

    # Compute PWLR
    return float(np.sum(alphas * rhos**2 / alphas.sum()))


def projection_weighted_linear_centered_kernel_alignment(
    X_1: np.ndarray,
    X_2: np.ndarray,
) -> float:
    """
    Computes the projection weighted linear centered kernel alignment (PWLCKA) score between original representations
    `X_1` and `X_2`.
    Based on Murphy. 2023., Chapter 32, Eq. 32.12.
    Args:
        X_1: The *original* representation from the first representation space with the highest correlation.
        X_2: The *original* representation from the second representation space with the highest correlation.

    Returns:
        The PWLCKA score.
    """
    # Factorise the original representations
    U_1, Sigma_1, _ = np.linalg.svd(X_1, full_matrices=False)
    U_2, Sigma_2, _ = np.linalg.svd(X_2, full_matrices=False)

    # Compute the individual terms in the PWLCKA sum
    lambdas = np.outer(Sigma_1, Sigma_2)
    Us = U_1.T @ U_2

    # Compute the PWLCKA sum
    return np.sum(lambdas * Us**2) / (np.linalg.norm(Sigma_1) * np.linalg.norm(Sigma_2))
