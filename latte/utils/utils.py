from typing import List
from collections import defaultdict
from itertools import product

import pandas as pd
import torch

from sklearn import feature_selection


def axis_factor_mutual_information(Z: torch.Tensor, F: torch.Tensor, factor_names: List[str]) -> pd.DataFrame:
    """
    Calculates the mutual information between all pairs of the provided axes in the data `Z` and the factor `F`.
    Args:
        Z: A `n x k` matrix containing the data.
        F: A `n x f` matrix containing ground truth factors of variation for each data point.
        factor_names: Human-understandable factor names for easier interpretation of the produced matrix.

    Returns:
        An `f x k` data frame containing the estimate of the mutual information between every axis in `Z` and every
        factor in `F`.
    """

    mis = defaultdict(lambda: defaultdict(float))
    for ii, (factor_name, axis) in enumerate(product(factor_names, range(Z.shape[1]))):
        mis[f"Axis {axis}"][factor_name] = feature_selection.mutual_info_regression(
            F[:, ii // Z.shape[1]].reshape((-1, 1)), Z[:, axis]
        )[0]
        # Since mutual_info_regression returns 1 value per feature (in this,
        # we simulate 1 feature, we extract the first one

    return pd.DataFrame(mis)
