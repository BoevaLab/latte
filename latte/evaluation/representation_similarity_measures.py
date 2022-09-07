import numpy as np


def _get_rho(Z_1: np.ndarray, Z_2: np.ndarray) -> np.ndarray:
    return np.asarray([np.corrcoef(Z_1[:, ii], Z_2[:, ii])[0, 1] for ii in range(Z_1.shape[1])])


def mean_squared_canonical_correlation(Z_1: np.ndarray, Z_2: np.ndarray) -> float:
    """
    Murphy. 2023., Chapter 32, Eq. 32.4
    Args:
        Z_1:
        Z_2:

    Returns:

    """
    assert Z_1.shape[1] == Z_2.shape[1]
    rhos = _get_rho(Z_1, Z_2)

    return np.linalg.norm(rhos, 2) ** 2 / len(rhos)


def mean_canonical_correlation(Z_1: np.ndarray, Z_2: np.ndarray) -> float:
    """
    Murphy. 2023., Chapter 32, Eq. 32.5
    Args:
        Z_1:
        Z_2:

    Returns:

    """
    assert Z_1.shape[1] == Z_2.shape[1]
    rhos = _get_rho(Z_1, Z_2)

    # TODO: Absolute values of rhos?
    return np.sum(rhos) / len(rhos)


def projection_weighted_canonical_correlation_analysis(
    Z_1: np.ndarray,
    Z_2: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
) -> float:
    """
    Murphy. 2023., Chapter 32, Eq. 32.7
    Args:
        Z_1:
        Z_2:
        X:
        W:

    Returns:

    """
    assert Z_1.shape[1] == Z_2.shape[1]
    rhos = _get_rho(Z_1, Z_2)

    alphas = np.sum(np.abs((X @ W).T @ X), axis=1)

    return float(np.sum(alphas * rhos / alphas.sum()))


def projection_weighted_linear_regression(
    Z_1: np.ndarray,
    Z_2: np.ndarray,
    X_1: np.ndarray,
    W_1: np.ndarray,
) -> float:
    """
    Murphy. 2023., Chapter 32, Eq. 32.8
    Args:
        Z_1:
        Z_2:
        X_1:
        W_1:

    Returns:

    """
    assert Z_1.shape[1] == Z_2.shape[1]
    rhos = _get_rho(Z_1, Z_2)

    alphas = np.linalg.norm((X_1 @ W_1).T @ X_1, 2, axis=1) ** 2

    return float(np.sum(alphas * rhos**2 / alphas.sum()))


def projection_weighted_linear_centered_kernel_alignment(
    X_1: np.ndarray,
    X_2: np.ndarray,
) -> float:
    """
    Murphy. 2023., Chapter 32, Eq. 32.7
    Args:
        X_1:
        X_2:

    Returns:

    """
    U_1, Sigma_1, _ = np.linalg.svd(X_1, full_matrices=False)
    U_2, Sigma_2, _ = np.linalg.svd(X_2, full_matrices=False)

    lambdas = np.outer(Sigma_1, Sigma_2)
    Us = U_1.T @ U_2

    return np.sum(lambdas * Us**2) / (np.linalg.norm(Sigma_1) * np.linalg.norm(Sigma_2))
