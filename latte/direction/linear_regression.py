import dataclasses
from typing import Sequence

import numpy as np
from sklearn.linear_model import LinearRegression


@dataclasses.dataclass
class OptimizationResult:
    value: float
    direction: np.ndarray
    intercept: float


def find_best_direction(points: np.ndarray, scores: Sequence[float]) -> OptimizationResult:
    """Finds the optimal direction by minimising the sum
    of squares of the differences between projections and scores.

    Args:
        points: shape (n_points, n_dim)
        scores: length n_points

    Returns:
        R squared of the fit
        direction: shape (n_dim,)
        intercept
    """
    lr = LinearRegression()
    lr.fit(points, scores)
    return OptimizationResult(value=lr.score(points, scores), direction=lr.coef_, intercept=lr.intercept_)
