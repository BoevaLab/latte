from typing import Protocol

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


class SimilarityMetric(Protocol):
    """Interface for measuring the similarity between projections onto 1D vectors
    and the true generative factor values."""

    def score(self, projections: np.ndarray, scores: np.ndarray) -> float:
        """

        Args:
            projections: latent space coordinates projected onto a 1D subspace, shape (n_points,)
            scores: true generative factors values, shape (n_points,)

        Returns:
            some measure of similarity between these two vectors, the larger the better

        Note:
            This function does not need to be symmetric.
        """
        ...


def _check_shape(projections: np.ndarray, scores: np.ndarray) -> None:
    """Checks whether `projections` and `scores` are both of shape (n,).

    Raises:
        ValueError, if the shapes are wrong
    """
    if not (projections.shape == scores.shape == (len(projections),)):
        raise ValueError(f"Shape mismatch: projections {projections.shape} != scores {scores.shape}.")


class SpearmanCorrelation(SimilarityMetric):
    """Calculates the Spearman correlation.

    It is a thin wrapper around SciPy's Spearman correlation function,
    which matches the interface.
    """

    def score(self, projections: np.ndarray, scores: np.ndarray) -> float:
        _check_shape(projections=projections, scores=scores)
        return stats.spearmanr(projections, scores)[0]


class KendallTau(SimilarityMetric):
    """Calculates Kendall's tau.

    It is a thin wraper around the respective SciPy's function.
    """

    def score(self, projections: np.ndarray, scores: np.ndarray) -> float:
        _check_shape(projections=projections, scores=scores)
        return stats.kendalltau(projections, scores)[0]


class MutualInformation(SimilarityMetric):
    """Calculates the 1D mutual information.

    This is a thin wrapper around SciKit Learn's mutual information function,
    which matches the interface.
    """

    def __init__(self, n_neighbors: int = 5, random_state: int = 123) -> None:
        self._n_neighbors = n_neighbors
        self._random_state = random_state

    def score(self, projections: np.ndarray, scores: np.ndarray) -> float:
        _check_shape(projections=projections, scores=scores)

        # mutual_info expects 2D features and returns an array of values, 1 for each feature (1 in our case)
        return mutual_info_regression(
            projections.reshape((-1, 1)),
            scores,
            n_neighbors=self._n_neighbors,
            discrete_features=False,
            random_state=self._random_state,
        )[0]


class AxisMetric(Protocol):
    """Interface for a method of finding the optimal 1D axis describing a given factor of variation."""

    def score(self, axis: np.ndarray, points: np.ndarray, scores: np.ndarray) -> float:
        """Calculates the score of a given axis. The larger the better.

        Args:
            axis: axis of variation, shape (n_dim,)
            points: coordinates, shape (n_points, n_dim)
            scores: generative factor of interest for each observation, shape (n_points,)

        Returns:
            measure how good the axis to describe the change of `scores`
             attributed to the `points` (the larger the better)
        """
        ...


class ProjectionAxisMetric(AxisMetric):
    """Implementation of `AxisMetric` interface, which projects
    the scores onto a 1D subspace and calculates the score.
    """

    def __init__(self, projection_metric: SimilarityMetric) -> None:
        """

        Args:
            projection_metric: projection metric to be used to calculate
             the similarity between the projections and true generative factors
        """
        self.projection_metric = projection_metric

    def score(self, axis: np.ndarray, points: np.ndarray, scores: np.ndarray) -> float:
        projections = project(points=points, vector=axis)
        return self.projection_metric.score(projections=projections, scores=scores)


class SpearmanCorrelationMetric(ProjectionAxisMetric):
    """Spearman correlation metric between projections onto specified axis
    and true generative factor scores."""

    def __init__(self) -> None:
        super().__init__(projection_metric=SpearmanCorrelation())


class MutualInformationMetric(ProjectionAxisMetric):
    """Mutual information metric between projections onto specified axis
    and true generative factor scores.

    See Also:
        MutualInformation
    """

    def __init__(self, n_neighbors: int = 5, random_state: int = 128) -> None:
        super().__init__(projection_metric=MutualInformation(n_neighbors=n_neighbors, random_state=random_state))


class KendallTauMetric(ProjectionAxisMetric):
    """Kendall's tau coefficient between projections onto specified axis
    and true generative factors."""

    def __init__(self) -> None:
        super().__init__(projection_metric=KendallTau())


def project(points: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Finds the component in the `directions` of `vector`.

    Args:
        points: points coordinates, shape (n_points, n_dim)
        vector: direction at which the points will be projected, shape (n_dim,)

    Returns:
        projections, shape (n_points,)
    """
    unit = unit_vector(vector)
    return np.einsum("ij,j->i", points, unit)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns a unit vector parallel to the `vector`."""
    return vector / np.linalg.norm(vector)
