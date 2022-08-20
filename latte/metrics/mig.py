import numpy as np

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from scipy.stats import differential_entropy


def mig(Z: np.ndarray, Y: np.ndarray, discrete: bool = False) -> float:
    """
    Calculates the MIG score of the learned representations `Z` against the true factors of variation `Y`.
    Mutual information is calculated with `sklearn`'s `mutual_info_regression` function.

    Args:
        Z: An `N x k` matrix of learned representations for `N` samples.
        Y: An `N x d` matrix of true factors of variation for `N` samples.
        discrete: Whether the distributions of the factors/learned representations are discrete.

    Returns:
        The MIG score.
    """

    k, d = Z.shape[1], Y.shape[1]

    # Compute the pairwise mutual information matrix
    M = np.zeros((k, d))
    for i in range(k):
        M[i, :] = mutual_info_regression(Y, Z[:, i])

    # Compute the factor of variation entropies
    H = np.zeros(k)
    for j in range(k):
        # TODO (Anej): Which one to use?
        # `differential_entropy` results in much bigger difference between the oriented and non-oriented pPCA solutions,
        # but results in an entropy estimate lower than the MI estimate given by `mutual_info_regression` (resulting in
        # MIG > 1)
        H[j] = differential_entropy(Y[:, j]) if not discrete else mutual_info_score(Y[:, j], Y[:, j])
        # H[j] = mutual_info_regression(Y[:, [j]], Y[:, j])[0]

    # Compute the MIG score
    s = np.argsort(-M, axis=0)[:2]
    mig_score = np.mean((M[s[0, :], np.arange(d)] - M[s[1, :], np.arange(d)]) / H)

    return float(mig_score)
