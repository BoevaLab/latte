from scipy import optimize
import numpy as np
import torch

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
