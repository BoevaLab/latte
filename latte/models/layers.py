from typing import Optional

import torch
import torch.nn as nn
import geoopt


class ManifoldProjectionLayer(nn.Module):
    """A custom layer to project a tensor onto a linear subspace spanned by a k-frame from a Stiefel manifold."""

    def __init__(self, d: int, k: int, init: Optional[torch.Tensor] = None):
        """
        Args:
            d: Dimension of the observable space
            k: Dimension of the subspace to project to
            init: An optional initial setting of the manifold parameter
        """
        super().__init__()
        self.A = geoopt.ManifoldParameter(geoopt.Stiefel().random(d, k) if init is None else init)

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
