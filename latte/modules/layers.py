"""
Implementations of some custom layers used throughout `Latte`.
"""

from typing import Optional

import torch
import torch.nn as nn
import geoopt


class ManifoldProjectionLayer(nn.Module):
    """A custom layer to project a tensor onto a linear subspace spanned by a d-frame from a Stiefel manifold."""

    def __init__(self, n: int, d: int, init: Optional[torch.Tensor] = None, stiefel_manifold: bool = True):
        """
        Args:
            n: Dimension of the observable space (the one we want to project *from*).
            d: Dimension of the linear subspace (the one we want to project *onto*).
            init: An optional initial setting of the manifold parameter
            stiefel_manifold: Whether the projection layer should be on the Stiefel manifold.
                              This should be left True for almost all use cases, it is mostly here to enable
                              investigating the effect of the constrained optimisation versus the unconstrained one.
        """
        super().__init__()
        if init is None:
            if stiefel_manifold:
                self.A = geoopt.ManifoldParameter(geoopt.Stiefel().random(n, d))
            else:
                self.A = geoopt.ManifoldParameter(geoopt.Euclidean().random(n, d))
        else:
            if stiefel_manifold:
                self.A = geoopt.ManifoldParameter(init, manifold=geoopt.Stiefel())
            else:
                self.A = geoopt.ManifoldParameter(init, manifold=geoopt.Euclidean())
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
