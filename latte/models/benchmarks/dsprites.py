"""
Implementation of two simple encoder/decoder `pythae`-compatible pairs for the `dSprites` dataset.
One of the pairs is a simple fully-connected network while the other is based on (transposed) convolutions.
"""

from torch import nn
import torch

from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

from latte.modules.layers import View


class DSpritesConvEncoder(BaseEncoder):
    def __init__(self, latent_size: int) -> None:
        BaseEncoder.__init__(self)

        self.latent_size = latent_size

        # The convolutional encoder; same as in the original beta-VAE paper by Higgins et al. (2017)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(),
            View((-1, 256)),  # B, 256
            nn.Linear(256, self.latent_size * 2),  # B, latent_size*2
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:

        y = self.encoder(x)

        # The same network produces both the mean and the variance parameters
        mu = y[:, : self.latent_size]
        log_var = y[:, self.latent_size :]

        output = ModelOutput(embedding=mu, log_covariance=log_var)

        return output


class DSpritesConvDecoder(BaseDecoder):
    def __init__(self, latent_size: int) -> None:
        BaseDecoder.__init__(self)

        # The convolutional decoder; same as in the original beta-VAE paper by Higgins et al. (2017)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # B, 1, 64, 64
            nn.Sigmoid(),  # To make the reconstruction values in [0, 1]
        )

    def forward(self, z: torch.Tensor):

        x_hat = self.decoder(z)
        output = ModelOutput(reconstruction=x_hat)

        return output


class DSpritesFCEncoder(BaseEncoder):
    def __init__(self, latent_size: int) -> None:
        BaseEncoder.__init__(self)

        self.latent_size = latent_size

        # A simple baseline fully-connected encoder
        self.encoder = nn.Sequential(
            View((-1, 64 * 64)),
            nn.Linear(64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_size * 2),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:

        y = self.encoder(x)

        # The same network produces both the mean and the variance parameters
        mu = y[:, : self.latent_size]
        log_var = y[:, self.latent_size :]

        output = ModelOutput(embedding=mu, log_covariance=log_var)

        return output


class DSpritesFCDecoder(BaseDecoder):
    def __init__(self, latent_size: int) -> None:
        BaseDecoder.__init__(self)

        # A simple baseline fully-connected decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 64),
            View((-1, 1, 64, 64)),
            nn.Sigmoid(),  # To make the reconstruction values in [0, 1]
        )

    def forward(self, z: torch.Tensor):

        x_hat = self.decoder(z)
        output = ModelOutput(reconstruction=x_hat)

        return output
