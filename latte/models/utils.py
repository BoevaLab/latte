from typing import Tuple, Union

import numpy as np


def split(
    X: np.ndarray, y: np.ndarray, p_train: float = 0.7, p_val: float = 0.1, seed: Union[np.random.Generator, int] = 1
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split the data into train, validation, and test sets."""
    n = len(X)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    end_train, end_val = int(p_train * n), int((p_train + p_val) * n)
    X_train, X_val, X_test = (
        X[permutation][:end_train],
        X[permutation][end_train:end_val],
        X[permutation][end_val:],
    )
    y_train, y_val, y_test = (
        y[permutation][:end_train],
        y[permutation][end_train:end_val],
        y[permutation][end_val:],
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
