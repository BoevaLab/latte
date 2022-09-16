"""
Simple implementations of baseline CCN image VAE components compatible with `pythae`.
Based on the similar implementations from https://arxiv.org/pdf/1802.05983.pdf.
These architectures work on the 3-layer 64x64 images from e.g. resized `CelebA` or `Shapes3D` datasets.
"""

from torch import nn
import torch

from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.trainers import BaseTrainerConfig

from latte.modules.layers import View


class ConvEncoder(BaseEncoder):
    """A base architecture of an encoder."""

    def __init__(self, model_config: BaseTrainerConfig) -> None:
        BaseEncoder.__init__(self)

        self.model_config = model_config

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # B, 32, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),  # B, 32, 16, 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # B, 64, 8, 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),  # B, 4, 4, 4
            nn.ReLU(),
            View((-1, 64 * 4 * 4)),  # B, 64 * 4 * 4
            nn.Linear(64 * 4 * 4, 256),  # B, 256
            nn.ReLU(),
            nn.Linear(256, self.model_config.latent_dim * 2),  # B, model_config.latent_dim * 2
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:

        y = self.encoder(x)

        # The same network produces both the mean and the variance parameters
        mu = y[:, : self.model_config.latent_dim]
        log_var = y[:, self.model_config.latent_dim :]

        output = ModelOutput(embedding=mu, log_covariance=log_var)

        return output


class ConvDecoder(BaseDecoder):
    """A base architecture of a decoder."""

    def __init__(self, model_config: BaseTrainerConfig) -> None:
        BaseDecoder.__init__(self)

        self.decoder = nn.Sequential(
            nn.Linear(model_config.latent_dim, 256),  # B, 256
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4),  # B, 64 * 4 * 4
            nn.ReLU(),
            View((-1, 64, 4, 4)),  # B, 64,  4,  4
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B, 64, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B, 32, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B, 32, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # B, 3, 64, 64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):

        x_hat = self.decoder(z)
        output = ModelOutput(reconstruction=x_hat)

        return output
