from typing import Union

import numpy as np
import torch

TensorOrNDArray = Union[torch.Tensor, np.ndarray]


def principal_subspace_basis(A: TensorOrNDArray, d: int):
    """
    Returns an orthonormal basis of the d-dimensional principal subspace spanned by the image of A.
    Args:
        A: A projection matrix.
        d: The dimensionality of the principal subspace.
    """
    # Extracts the first d columns of the `U` matrix in A = USV^T.
    return np.linalg.svd(A, full_matrices=False)[0][:, :d]
