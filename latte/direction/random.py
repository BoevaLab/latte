import dataclasses
from typing import NamedTuple, Sequence, Union

import numpy as np
import numpy.typing as ntypes

import latte.direction.common as common


# Seed (or generator) we can use to initialize a random number generator
_Generator = Union[int, np.random.Generator]


def get_random_directions(
    n_directions: int,
    dim: int,
    random_generator: _Generator = 1,
    _scale: float = 10.0,
) -> np.ndarray:
    """Generates `n_directions` random directions in `dim`-dimensional space.

    Args:
        n_directions: number of vectors to generate
        dim: dimension of the feature space
        random_generator: NumPy's random generator, used for reproducibility
        _scale: parameter used in the sampling process. If the zero vector is sampled,
            increasing this parameter may help (it should not be modified without a good reason)

    Returns:
        sampled axes, shape (n_directions, dim)
    """
    random_generator = np.random.default_rng(random_generator)
    vectors = random_generator.normal(loc=0.0, scale=_scale, size=(n_directions, dim))

    # TODO(Pawel): Make sure that we didn't sample the zero vector.
    return vectors


class _MetricsAndVectors(NamedTuple):
    """Auxiliary class used to bind metric values and vectors.

    Attrs:
        metrics: metric values, shape (n,)
        vectors: vector representing the direction, shape (n, dim)
    """

    metrics: np.ndarray
    vectors: np.ndarray


def find_metric_distribution(
    points: ntypes.ArrayLike,
    scores: Sequence[float],
    metric: common.AxisMetric,
    budget: int = 200,
    random_generator: _Generator = 1,
) -> _MetricsAndVectors:
    """Uniformly samples over all possible directions and returns
    the (sample from the) distribution of scores and directions.

    Args:
        points: shape (n_points, n_dim)
        scores: length n_points
        metric: metric used to score the axes
        budget: how many random directions will be sampled
        random_generator: used for reproducibility

    Returns:
        scores, shape (budget,)
        vectors, shape (budget, n_dim)
    """
    points, scores = np.asarray(points), np.asarray(scores)
    vectors = get_random_directions(n_directions=budget, dim=points.shape[-1], random_generator=random_generator)
    metric_vals = [metric.score(axis=vector, points=points, scores=scores) for vector in vectors]

    return _MetricsAndVectors(metrics=np.asarray(metric_vals), vectors=vectors)


@dataclasses.dataclass
class OptimizationResult:
    value: float
    direction: np.ndarray


def select_best_direction(data: _MetricsAndVectors) -> OptimizationResult:
    """Finds the direction with the best score.

    Args:
        data: output of `find_metric_distribution`

    Returns:
        best metric
        direction, shape (n_dim,)
    """
    i = np.argmax(data.metrics)
    return OptimizationResult(
        value=data.metrics[i],
        direction=common.unit_vector(data.vectors[i]),
    )


def find_best_direction(
    points: np.ndarray,
    scores: Sequence[float],
    metric: common.AxisMetric,
    budget: int = 200,
    random_generator: Union[int, np.random.Generator] = 1,
) -> OptimizationResult:
    data = find_metric_distribution(
        points=points, scores=scores, metric=metric, budget=budget, random_generator=random_generator
    )
    return select_best_direction(data)
