"""Gradient-free optimization of the optimal axis."""
import nevergrad as ng
import numpy as np

import latte.direction.common as common
from latte.direction.random import OptimizationResult


def optimize(
    points: np.ndarray,
    scores: np.ndarray,
    metric: common.AxisMetric,
    budget: int = 200,
    seed: int = 321,
) -> OptimizationResult:
    """Finds the direction maximizing `metric` calculated using
    `points` and `scores`.

    Args:
        points: point coordinates, shape (n_points, n_dim)
        scores: scores of the points, shape (n_points,)
        metric: the function that returns the measure of similarity between projections
            onto a given direction and the scores
        budget: optimization budget. Increases the chance to find a good axis,
            but makes the optimization longer
        seed: random seed, used for reproducibility

    Returns:
        the unit vector pointing in the direction of the highest ascent of `scores`
        the metric value for that direction

    Todo:
        TODO(Pawel): Add callbacks to the optimizer. Both custom ones
            as well as the optimization history.
    """
    n_dim = points.shape[-1]
    optimizer = ng.optimizers.NGOpt(parametrization=n_dim, budget=budget)
    # Set the random seed for reproducibility
    optimizer.parametrization.random_state.seed(seed)

    def objective(vector: np.ndarray) -> float:
        """As nevergrad minimizes the objective and we want to maximize,
        we need an additional minus sign."""
        direction = common.unit_vector(vector)
        return -metric.score(axis=direction, points=points, scores=scores)

    # The optimal vector
    axis = common.unit_vector(optimizer.minimize(objective).value)

    # Metric value at the optimal point
    value = metric.score(axis=axis, points=points, scores=scores)

    return OptimizationResult(value=value, direction=axis)
