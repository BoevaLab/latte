"""
Code for generation of data from the probabilistic PCA generative model as well as some utility functions for
estimation and computation.
"""
import dataclasses
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from latte.dataset.utils import RandomGenerator


@dataclasses.dataclass
class ProbabilisticPCADataset:
    """
    A synthetic dataset holding the data generated by the `gaussian_data` function

    Members:
        X: A `N x n` matrix of the observables
        Z: The `N x n` matrix of the generative latent factors
        A: The `n x d` mixing/loading matrix
        mu: The shift added to the transformed generative factors (mean of the observables)
        sigma: The noise standard deviation of the observation model
        n: The dimensionality of the observational space
        d: The true dimensionality of the latent space
        d_measured: The number of "measured factors"
        measured_factors: The indices of the factors we are interested in / have measurements of.
    """

    X: np.ndarray
    Z: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    sigma: float
    n: int
    d: int
    d_measured: int
    measured_factors: np.ndarray


def generate(
    N: int,
    n: int,
    d: int,
    d_measured: Optional[int] = None,
    sigma: float = 0.1,
    spread: float = 1.0,
    rng: RandomGenerator = 42,
) -> ProbabilisticPCADataset:
    """
    Function generating a dataset according to the probabilistic PCA generative model.

    We assume we only have access to the `n`-dimensional observations, while the `d`-dimensional subspace
    (and the projections of the observations on it) are unknown.
    Further, we can extend this to the case where a subset of those `d` (`d_measured`) factors are observed.
    A small subset of the true projection values can be used to try to find the true mapping, and to evaluate the
    quality of the fit subspace.

    Args:
        N: Number of samples to generate.
        n: The dimension of the observable space.
        d: The dimension of the linear subspace which the data is projected on.
        d_measured: The number of generative factors we are interested in, or we assume have measurements of.
        spread: The standard deviation of the Gaussian used to generate the observations.
        sigma: The standard deviation of the noise added to the observations
               to make the discovery of the subspace harder.
        rng: The random number generator for the dataset.

    Returns:
        Populated GaussianGenerativeDataset
    """
    if d_measured is None:
        d_measured = d

    rng = np.random.default_rng(rng)

    Z = rng.normal(scale=spread, size=(N, d))  # The generative factors

    A = rng.normal(size=(n, d))  # The mixing/loading matrix; it can be any `d x d` matrix
    mu = rng.normal(size=(n,))
    epsilon = rng.normal(scale=sigma, size=(N, n))

    X = Z @ A.T + mu + epsilon  # The observables

    # Only return the values of the factors we are interested in, not all used in the generation
    measured_factors = np.sort(rng.choice(d, size=d_measured, replace=False))

    return ProbabilisticPCADataset(
        X=X.astype(np.float32),
        Z=Z.astype(np.float32),
        A=A.astype(np.float32),
        mu=mu.astype(np.float32),
        sigma=sigma,
        n=n,
        d=d,
        d_measured=d_measured,
        measured_factors=measured_factors,
    )


def gaussian_log_likelihood(X: np.ndarray, A_hat: np.ndarray, mu_hat: np.ndarray, sigma_hat: float) -> float:
    """
    Calculates the log-likelihood of the parameters `mu_hat1`, `sigma_hat`, and `A_hat` as the parameters of a
    probabilistic PCA model given the data `X`.
    Args:
        X: The observed data
        A_hat: The mixing matrix
        mu_hat: The mean of the data
        sigma_hat: The standard deviation of the observation model

    Returns:
        The log-likelihood
    """
    cov_hat = A_hat @ A_hat.T + sigma_hat**2 * np.eye(A_hat.shape[0])
    return stats.multivariate_normal(mean=mu_hat, cov=cov_hat).logpdf(X).sum()


def generative_model_mutual_information(A: np.ndarray, sigma: float) -> float:
    """
    Analytically computes the mutual information between the latent and the observable vectors from the probabilistic
    PCA generative model with the mixing matrix A.
    Args:
        A: The model mixing matrix
        sigma: The variance of the observable model

    Returns:
        The true mutual information between the subvectors.
    """

    n, d = A.shape
    sigma = np.vstack((np.hstack((np.eye(d), A.T)), np.hstack((A, A @ A.T + sigma**2 * np.eye(n)))))

    return 1 / 2 * np.log(np.linalg.det(sigma[:d, :d]) * np.linalg.det(sigma[d:, d:]) / np.linalg.det(sigma))


def gaussian_mutual_information(sigma: np.ndarray, d: int) -> float:
    """
    Analytically computes the mutual information between two subvectors of a joint-Gaussian random vector.
    Args:
        sigma: The covariance of the joint Gaussian distribution
        d: The dimensionality of the first subvector (the second one has dimensionality sigma.shape[0] - d)

    Returns:
        The true mutual information between the subvectors.
    """
    return 1 / 2 * np.log(np.linalg.det(sigma[:d, :d]) * np.linalg.det(sigma[d:, d:]) / np.linalg.det(sigma))


def construct_mixing_matrix_estimate(pca: PCA) -> np.ndarray:
    """
    Constructs the mixing matrix estimate based on the values returned by `sklearn`'s PCA model.
    See Tipping, M. E., & Bishop, C. M. (1999). Mixtures of Probabilistic Principal Component Analysers.

    Args:
        pca (PCA): The `sklearn` PCA model fit to the observable data

    Returns:
        np.ndarray: The maximum likelihood estimate of the mixing matrix with the columns corresponding to the
                    (scaled) principal directions.
    """

    U_hat = pca.components_.T
    sigma_hat = np.sqrt(pca.noise_variance_)
    Lambda_hat = np.diag(pca.singular_values_**2) / pca.n_samples_
    A_hat = U_hat @ np.sqrt(Lambda_hat - sigma_hat**2 * np.eye(Lambda_hat.shape[0]))
    return A_hat.astype(np.float32)


def get_latent_representations(X: np.ndarray, A: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Constructs the estimates of the latent factors based on the mixing matrix (estimate) `A` corresponding to
       the observables in `X`.
       The latent representation are taken to be the means of the posterior distributions of the latents given the
       observations.
       See Tipping, M. E., & Bishop, C. M. (1999). Mixtures of Probabilistic Principal Component Analysers.

    Args:
        X (np.ndarray): The `N x n` matrix of observable data.
        A (np.ndarray): The (estimate) of the mixing matrix.
        mu (np.ndarray): The (estimate) of the observable data mean.
        sigma (float): The (estimate) of the observable data noise standard deviation.

    Returns:
        np.ndarray: The (estimated) latent values.
    """
    # We represent the observables as the **means of their posteriors**
    M = A.T @ A + sigma**2 * np.eye(A.shape[1])
    P = np.linalg.inv(M) @ A.T

    return ((X - mu) @ P.T).astype(np.float32)


def get_observations(Z: np.ndarray, A: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Constructs the estimate of the observation means given the (estimates) of the pPCA parameters.

    Args:
        Z (np.ndarray): The latent values to produce the observations for.
        A (np.ndarray): The (estimate) of the mixing matrix.
        mu (np.ndarray): The (estimate) of the observable data mean.

    Returns:
        np.ndarray: The (estimated) observable data means.
    """
    return (Z @ A.T + mu).astype(np.float32)
