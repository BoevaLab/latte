"""Implementation of the KSG Mutual Information estimator,
proposed in
    A. Kraskov, H. Stoegbauer, P. Grassberger, Estimating Mutual Information, 2008
      https://arxiv.org/abs/cond-mat/0305641
"""
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from scipy import special

_DIGAMMA = special.digamma


def _find_kth_neighbour(distances: np.ndarray, k: int) -> int:
    """Finds the index of the `k`th neighbour.

    Args:
        distances: the `i`th entry represents the distances from the given point to the `i`th point
            in the dataset. One entry is expected to be 0, being the distance from the point to itself
        k: the distance to the `k`th neighbour (not including the point itself)

    Returns:
        the index of the `k`th neighbour
    """
    return sorted(range(len(distances)), key=lambda i: distances[i])[k]


def _prepare_matrices(distance_x: ArrayLike, distance_y: ArrayLike, normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Auxiliary function, validating the inputs.

    Args:
        distance_x: (i,j)-th entry should be ||xi-xj|| according to some sensible norm. Shape (N, N)
        distance_y: (i,j)-th entry should be ||yi-yj|| according to some sensible norm on the Y space. Shape (N, N)
        normalize: boolean flag whether to normalize the distance matrices (to values ranging in [0, 1]).
            If the distances are not appropriately normalized (and the scales are much different),
              the maximum distance on `X\\times Y` may effectively ignore either `X` or `Y`.

    Returns:
        distance_x, shape (N, N)
        distance_y, shape (N, N)
    """
    x, y = np.asarray(distance_x), np.asarray(distance_y)

    if x.shape != y.shape:
        raise ValueError(f"Distance matrices have different shape: {x.shape} != {y.shape}.")
    if x.shape != (len(x), len(x)):
        raise ValueError("The distance matrices must be square.")

    if normalize:
        return x / np.max(x), y / np.max(y)
    else:
        return x, y


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"Parameter k should be a positive integer (was {k}).")


def estimate_mi_ksg1(distance_x: ArrayLike, distance_y: ArrayLike, k: int = 10, normalize: bool = True) -> float:
    """Assume you have `N` samples (xi, yi) from probability distribution P(x, y),
    defined on the space `X \\times Y`, and you want to estimate mutual information I(X; Y).

    If you have meaningful norm on these spaces (and can calculate distances between points),
    you can give these distance matrices to this function.

    Args:
        distance_x: (i,j)-th entry should be ||xi-xj|| according to some sensible norm. Shape (N, N)
        distance_y: (i,j)-th entry should be ||yi-yj|| according to some sensible norm on the Y space. Shape (N, N)
        k: number of neighbors. Trades bias for variance.
        normalize: boolean flag whether to normalize the distance matrices (to values ranging in [0, 1]).
            If the distances are not appropriately normalized (and the scales are much different),
              the maximum distance on `X\\times Y` may effectively ignore either `X` or `Y`.

    Returns:
        estimate of the mutual information
    """
    _validate_k(k)
    x, y = _prepare_matrices(distance_x, distance_y, normalize=normalize)
    n_points = len(x)

    # This list holds digamma(n_x + 1) + digamma(n_y + 1) for all points
    digammas = []

    for i in range(n_points):
        row = np.maximum(x[i], y[i])
        kth_neighbour = _find_kth_neighbour(row, k)

        distance = row[kth_neighbour]

        # Don't include the `i`th point itself in n_x and n_y
        n_x = (x[i, :] < distance).sum() - 1
        n_y = (y[i, :] < distance).sum() - 1

        digammas.append(_DIGAMMA(n_x + 1) + _DIGAMMA(n_y + 1))

    mi_estimate = _DIGAMMA(k) - np.mean(digammas) + _DIGAMMA(n_points)
    return max(mi_estimate, 0.0)


def estimate_mi_ksg2(distance_x: ArrayLike, distance_y: ArrayLike, k: int = 10, normalize: bool = True) -> float:
    """Hopefully an alternative to `estimate_mi_ksg1`.

    TODO(Pawel):
        Currently it gives wrong estimates. Debug.
    """
    _validate_k(k)
    x, y = _prepare_matrices(distance_x, distance_y, normalize=normalize)
    n_points = len(x)

    # This list holds digamma(n_x) + digamma(n_y)
    digammas = []

    for i in range(n_points):
        row = np.maximum(x[i], y[i])
        kth_neighbour = _find_kth_neighbour(row, k)

        dist_x = x[i, kth_neighbour]
        dist_y = y[i, kth_neighbour]

        # Don't include the `i`th point itself in n_x and n_y
        n_x = (x[i, :] <= dist_x).sum() - 1
        n_y = (y[i, :] <= dist_y).sum() - 1

        digammas.append(_DIGAMMA(n_x) + _DIGAMMA(n_y))

    mi_estimate = _DIGAMMA(k) - 1 / k - np.mean(digammas) + _DIGAMMA(n_points)
    return max(mi_estimate, 0.0)


class Multinormal:
    """Auxiliary object for representing the multivariate normal distribution."""

    def __init__(self, mean: ArrayLike, covariance: ArrayLike, rng=10) -> None:
        """

        Args:
            mean: mean vector of the distribution, shape (dim,)
            covariance: covariance matrix of the distribution, shape (dim, dim)
            rng: random number generator, used to keep internal random state
        """
        # Mean and the covariance
        self._mean = np.asarray(mean)
        self._covariance = np.asarray(covariance)

        # The determinant of the covariance, used to calculate
        # the entropy
        self._det_covariance: float = np.linalg.det(self._covariance)

        # Internal random state
        self._rng = np.random.default_rng(rng)

        # Dimensionality of the space
        self._dim = self._mean.shape[0]

        # Validate the shape
        if self._covariance.shape != (self._dim, self._dim):
            raise ValueError(f"Covariance has shape {self._covariance.shape} rather than {(self._dim, self._dim)}.")
        # TODO(Pawel): Validate whether covariance is positive definite

    def sample(self, n_samples: int, rng=None) -> np.ndarray:
        """Sample from the distribution.

        Args:
            n_samples: number of samples to generate
            rng: provide a custom random number generator. By default the internal state will be used

        Returns:
            samples, shape (n_samples, dim)
        """
        my_rng = self._rng if rng is None else np.random.default_rng(rng)
        return my_rng.multivariate_normal(mean=self._mean, cov=self._covariance, size=n_samples)

    @property
    def dim(self) -> int:
        """The dimensionality."""
        return self._dim

    def entropy(self) -> float:
        """Entropy in nats."""
        return 0.5 * (np.log(self._det_covariance) + self.dim * (1 + np.log(2 * np.pi)))


class SplitMultinormal:
    """Represents a joint distribution P(X, Y) which is multivariate normal
    of variables X and Y which also have multivariate normal distributions.
    """

    def __init__(self, mean: ArrayLike, covariance: ArrayLike, split: int, rng=123) -> None:
        """

        Args:
            mean: mean of the joint distribution P(X, Y).
                Components 0, 1, ..., `split`-1 represent the mean vector of P(X)
                Components `split`, `split`+1, ... represent the mean vector of P(Y)
            covariance: covariance matrix of P(X, Y)
                Upper left `split` x `split` block represents the covariance of P(X)
                Bottom right `n-split` x `n-split` block represents the covariance of P(Y)
            split: divides the distribution P(X, Y) into P(X) and P(Y)
            rng: internal random state
        """
        mean = np.asarray(mean)
        covariance = np.asarray(covariance)

        # Define the random state
        self._default_rng = np.random.default_rng(rng)

        self._joint_distribution = Multinormal(mean=mean, covariance=covariance, rng=self._default_rng)
        self._x_distribution = Multinormal(
            mean=mean[:split], covariance=covariance[:split, :split], rng=self._default_rng
        )
        self._y_distribution = Multinormal(
            mean=mean[split:], covariance=covariance[split:, split:], rng=self._default_rng
        )

        self._split = split

    def sample_joint(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        samples = self._joint_distribution.sample(n)
        return samples[:, : self._split], samples[:, self._split :]

    def sample_x(self, n: int) -> np.ndarray:
        return self._x_distribution.sample(n)

    def sample_y(self, n: int) -> np.ndarray:
        return self._y_distribution.sample(n)

    @property
    def dim_x(self) -> int:
        return self._x_distribution.dim

    @property
    def dim_y(self) -> int:
        return self._y_distribution.dim

    def mutual_information(self) -> float:
        """Calculates the mutual information I(X; Y) using an exact formula.

        Returns:
            mutual information, in nats

        The mutual information is given by
            0.5 * log( det(covariance_x) * det(covariance_y) / det(full covariance) )
        what follows from the
            I(X; Y) = H(X) + H(Y) - H(X, Y)
        formula and the entropy of the multinormal distribution.
        """
        h_x = self._x_distribution.entropy()
        h_y = self._y_distribution.entropy()
        h_xy = self._joint_distribution.entropy()
        mi = h_x + h_y - h_xy  # Mutual information estimate
        return max(0.0, mi)  # Mutual information is always non-negative
