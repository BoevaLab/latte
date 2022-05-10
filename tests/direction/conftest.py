from typing import Sequence

import pytest
import numpy as np


class Auxiliary:
    """This is a (bit hacky) auxiliary class
    with functions shared between different test modules.

    This trick is taken from https://stackoverflow.com/a/42156088
    """

    @staticmethod
    def score_according_to_gradient(direction: Sequence[float], points: np.ndarray) -> np.ndarray:
        """Calculates the score by taking a scalar product with a given direction vector.

        Args:
            direction: shape (n_dim,)
            points: shape (n_points, n_dim)

        Returns:
            scores, shape (n_points,)
        """
        direction_vector = np.asarray(direction)
        return np.einsum("ij,j -> i", points, direction_vector)


@pytest.fixture
def auxiliary() -> Auxiliary:
    return Auxiliary()
