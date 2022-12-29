"""
Simple implementations of baseline VAE components compatible with `pythae` for the *black-and-white* version of
the CelebA dataset.
Based on the similar implementations for the `dSprites` dataset.
"""

from torch import nn
import torch

from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

from latte.modules.layers import View


class CelebAConvEncoderBW(BaseEncoder):
    """A base architecture of an encoder on the black and white CelebA dataset resized to 64 x 64."""

    def __init__(self, latent_size: int, n_channels: int = 128) -> None:
        BaseEncoder.__init__(self)

        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, n_channels, 5, 2, 2),  # B,  n_channels, 32, 32
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, 2 * n_channels, 5, 2, 2),  # B,  2 * n_channels, 16, 16
            nn.BatchNorm2d(2 * n_channels),
            nn.ReLU(),
            nn.Conv2d(2 * n_channels, 4 * n_channels, 5, 2, 2),  # B,  4 * n_channels,  8,  8
            nn.BatchNorm2d(4 * n_channels),
            nn.ReLU(),
            nn.Conv2d(4 * n_channels, 8 * n_channels, 5, 2, 2),  # B,  8 * n_channels,  4,  4
            nn.BatchNorm2d(8 * n_channels),
            nn.ReLU(),
            View((-1, 8 * n_channels * 4 * 4)),  # B, 512
            nn.Linear(8 * n_channels * 4 * 4, self.latent_size * 2),  # B, latent_size*2
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:

        y = self.encoder(x)

        # The same network produces both the mean and the variance parameters
        mu = y[:, : self.latent_size]
        log_var = y[:, self.latent_size :]

        output = ModelOutput(embedding=mu, log_covariance=log_var)

        return output


class CelebAConvDecoderBW(BaseDecoder):
    """A base architecture of an decoder on the black and white CelebA dataset resized to 64 x 64."""

    def __init__(self, latent_size: int, n_channels: int = 128) -> None:
        BaseDecoder.__init__(self)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 8 * n_channels * 4 * 4),  # B, 8 * n_channels * 4 * 4
            View((-1, 8 * n_channels, 4, 4)),  # B, 8 * n_channels,  4,  4
            nn.BatchNorm2d(8 * n_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                8 * n_channels, 4 * n_channels, 5, 2, padding=2, output_padding=1
            ),  # B,  4 * n_channels,  8,  8
            nn.BatchNorm2d(4 * n_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4 * n_channels, 2 * n_channels, 5, 2, padding=2, output_padding=1
            ),  # B,  2 * n_channels, 16, 16
            nn.BatchNorm2d(2 * n_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * n_channels, n_channels, 5, 2, padding=2, output_padding=1),  # B,  n_channels, 32, 32
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels, 1, 5, 2, padding=2, output_padding=1),  # B, 1, 64, 64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):

        x_hat = self.decoder(z)
        output = ModelOutput(reconstruction=x_hat)

        return output
