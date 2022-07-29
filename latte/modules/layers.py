"""
Implementations of some custom layers used throughout `Latte`.
"""

from typing import Optional

import torch
import torch.nn as nn
import geoopt


class ManifoldProjectionLayer(nn.Module):
    """A custom layer to project a tensor onto a linear subspace spanned by a d-frame from a Stiefel manifold."""

    def __init__(self, n: int, d: int, init: Optional[torch.Tensor] = None):
        """
        Args:
            n: Dimension of the observable space (the one we want to project *from*).
            d: Dimension of the linear subspace (the one we want to project *onto*).
            init: An optional initial setting of the manifold parameter
        """
        super().__init__()
        if init is None:
            self.A = geoopt.ManifoldParameter(geoopt.Stiefel().random(n, d))
        else:
            self.A = geoopt.ManifoldParameter(init, manifold=geoopt.Stiefel())
            # self.A = geoopt.ManifoldParameter(init, manifold=geoopt.Euclidean())
        # self.A is a `n x d` matrix
        # Notice that due to the shape of the matrix, we have to multiply with its transpose to project a single
        # data point onto the subspace.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A


class ConcatLayer(nn.Module):
    """A custom layer to concatenate two vectors."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, y), self.dim)


class MultiInputSequential(nn.Sequential):
    """A custom Module which augments the normal Sequential Module
    to be able to take multiple inputs at the same time."""

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        for module in self._modules.values():
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x


class View(nn.Module):
    """A utility layer to enable convenient construction of a Sequential model with tensor reshaping."""

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
