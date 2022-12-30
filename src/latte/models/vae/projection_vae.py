"""
This file implements wrappers around the BaseEncoder and BaseDecoder of the `pythae` package to enable the use of
the (learned) projection matrices in these VAE components.
The models basically take a trained encoder/decoder and project the latent representations with the provided matrix,
defining a new latent space of the model.
This for example allows us to use the `pythae` samplers to sample directly from the linear subspace defined by the
projection matrix.
"""

import torch

from pythae.models.base.base_model import BaseEncoder, BaseDecoder
import geoopt


class ProjectionBaseEncoder(BaseEncoder):
    """
    A wrapper around the base encoder to project onto a linear subspace with a projection matrix.
    The latent representations produced by this model are the representations of the base encoder projected with
    the projection matrix.
    """

    def __init__(self, encoder: BaseEncoder, A: geoopt.ManifoldTensor) -> None:
        """
        Args:
            encoder (BaseEncoder): The encoder to whose latent space the projection will be applied
            A (geoopt.ManifoldTensor): The projection matrix
        """
        super().__init__()
        self.encoder = encoder
        self.A = A

    def forward(self, x):
        y = self.encoder(x)
        y.embedding = y.embedding @ self.A  # Project the entire batch with the projection matrix
        y.log_covariance = torch.diagonal(self.A.T @ torch.diag_embed(y.log_covariance) @ self.A, dim1=1, dim2=2)
        return y


class ProjectionBaseDecoder(BaseDecoder):
    """
    A wrapper around the base decoder to project from a linear subspace of the original encoder to the subspace
    from which the original decoder generates.
    The decoder works with latent representations from a base encoder projected onto a subspace
    with a projection matrix.
    The projection decoder projects these representations back into the original decoder representation space
    with the *same* projection matrix and then decodes with the original decoder.
    """

    def __init__(self, decoder: BaseDecoder, A: geoopt.ManifoldTensor) -> None:
        """
        Args:
            decoder (BaseDecoder): The decoder to whose latent space the projection will be applied
            A (geoopt.ManifoldTensor): The projection matrix
        """
        super().__init__()
        self.decoder = decoder
        self.A = A

    def forward(self, x):
        # Project the entire batch back into the base decoder representation space and decode it
        return self.decoder(x @ self.A.T)
