"""
This file implements a linear encoder-decoder pair mainly intended to be used when the projection matrix is learned
on the *raw* data.
This for example allows us to use the `pythae` samplers to sample directly from the representational space defined
by the projection matrix.
"""

import torch

from pythae.models.base.base_model import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
import geoopt


class LinearEncoder(BaseEncoder):
    """
    An implementation of a linear encoder which takes a *learned*/fit projection matrix `A` and uses it as the
    encoding matrix.
    This is mostly a convenience implementation to be able to use the projection matrices learned on the raw
    representations into the `pythae` workflow.
    As the variance is not meant to be learned, it uses a constant unit variance for all dimensions.
    """

    def __init__(self, A: geoopt.ManifoldTensor, device: str = "cuda") -> None:
        """
        Args:
            A (geoopt.ManifoldTensor): The projection matrix
            device: The device to use.
        """
        super().__init__()
        self.A = A
        self.device = device

    def forward(self, x):
        z = x @ self.A
        return ModelOutput(embedding=z, log_covariance=torch.zeros_like(z).to(self.device))


class LinearDecoder(BaseDecoder):
    """
    An implementation of a linear encoder which takes a *learned*/fit projection matrix `A` and uses it as the
    decoding matrix.
    This is mostly a convenience implementation to be able to use the projection matrices learned on the raw
    representations into the `pythae` workflow.
    """

    def __init__(self, A: geoopt.ManifoldTensor) -> None:
        """
        Args:
            A (geoopt.ManifoldTensor): The projection d-frame
        """
        super().__init__()
        self.A = A

    def forward(self, x):
        return ModelOutput(reconstruction=x @ self.A.T)
