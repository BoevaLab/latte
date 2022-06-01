from typing import Callable, List
import dataclasses


import numpy as np
import geoopt


rng = np.random.default_rng(42)


@dataclasses.dataclass
class SyntheticDataset:
    """
    A synthetic dataset holding data generated by the `synthetic_dataset` function.
    See function definition for more detailed documentation of the generated data.

    Members:
        X: A `n x d` matrix of the (noisy) observables
        A: The `d x k` projection matrix into the k-dimensional linear subspace
        Y: The `n x k` matrix of the projections of the observables into the k-dimensional linear subspace
        Z: The `n x m` matrix of the manifold coordinates of the observables
    """

    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    Z: np.ndarray


@dataclasses.dataclass
class GaussianGenerativeDataset:
    """
    A synthetic dataset holding the data generated by the `gaussian_data` function

    Members:
        X: A `n x d` matrix of the observables
        Z: The `n x k` matrix of the generative latent factors
        A: The `d x k` mixing/loading matrix
    """

    X: np.ndarray
    Z: np.ndarray
    A: np.ndarray


def synthetic_dataset(
    n: int,
    d: int,
    k: int,
    fs: List[Callable[[np.ndarray], np.ndarray]],
    spread: float = 1,
    sigma: float = 0,
) -> SyntheticDataset:
    """
    Generate a synthetic dataset of n d-dimensional points (observations) X sampled from an isotropic Gaussian.
    The data actually live in a lower-dimensional nonlinear subspace/manifold M.
    M itself depends only on a (random) k-dimensional linear subspace (characterised by a `d x k` projection matrix)
    and is defined by the functions `fs` which map the k-dimensional subspace into the true subspace,
    e.g., a 1D curve or a 2D surface which depends on the k dimensions of the subspace.
    In general, this true space will have dimension m := len(fs).

    We assume we only have access to the d-dimensional observations, while the k-dimensional subspace and the
    true m-dimensional space (and the projections of the observations on them) are unknown.
    A small subset of the true projection values can be used to try to find these subspaces, and then to evaluate the
    quality of the fit.

    Args:
        n: Number of samples to generate.
        d: The dimension of the observable space.
        k: The dimension of the linear subspace on which the manifold M depends.
        fs: The list of functions mapping from the k-dimensional linear subspace into M.
        spread: The standard deviation of the Gaussian used to generate the observations.
        sigma: The standard deviation of the noise added to the observations
               to make the discovery of the subspace harder.
    Returns:
        A SyntheticDataset object
    """

    X = rng.normal(scale=spread, size=(n, d))  # The true observations

    A = geoopt.Stiefel().random(d, k).numpy()  # The k-frame (projection matrix) of the linear subspace
    Y = X @ A  # The coordinates in the linear subspace

    Z = np.asarray([f(Y) for f in fs]).T  # The true value of the factors of variation in the manifold M

    X_noisy = X + rng.normal(0, sigma, size=(n, d))  # Noisy observations

    return SyntheticDataset(X=X_noisy, A=A, Y=Y, Z=Z)


def gaussian_data(n: int, d: int, k: int, sigma: float = 0.1, spread: float = 1.0) -> GaussianGenerativeDataset:
    """
    Function generating a dataset according to the probabilistic PCA generative model.

    We assume we only have access to the d-dimensional observations, while the k-dimensional subspace
    (and the projections of the observations on it) are unknown.
    A small subset of the true projection values can be used to try to find these subspace, and then to evaluate the
    quality of the fit.
    """
    Z = rng.normal(scale=spread, size=(n, k))  # The generative factors

    A = rng.normal(size=(d, k))  # The mixing/loading matrix

    X = Z @ A.T + rng.normal(scale=sigma, size=(n, d))  # The observables

    return GaussianGenerativeDataset(X=X, Z=Z, A=A)