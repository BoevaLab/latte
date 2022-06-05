from abc import abstractmethod, ABCMeta
from typing import Tuple
from enum import Enum

import torch
import torch.nn as nn
import pytorch_lightning as pl


class VAEReconstructionLossType(Enum):
    bernoulli = 1
    gaussian = 2


class View(nn.Module):
    """A utility layer to enable the construction of a Sequential model with tensor reshaping."""

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def kl_divergence(mu, log_var):

    kl = -0.5 * (1 + log_var - torch.pow(mu, 2) - torch.exp(log_var))
    total_kl = torch.mean(torch.sum(kl, dim=1), dim=0)
    dimension_wise_kl = torch.mean(kl, dim=0)

    return total_kl, dimension_wise_kl


class VAE(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        latent_size: int,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_type: VAEReconstructionLossType = VAEReconstructionLossType.bernoulli,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = encoder
        self.decoder = decoder
        self.loss_type = loss_type
        self.learning_rate = learning_rate

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        gamma = self.encoder(x)

        mu = gamma[:, : self.latent_size]
        log_var = gamma[:, self.latent_size :]

        z = mu + torch.exp(log_var / 2) * torch.randn_like(mu)

        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:

        if self.loss_type == VAEReconstructionLossType.bernoulli:
            loss = nn.BCEWithLogitsLoss()(x_hat, x)
        elif self.loss_type == VAEReconstructionLossType.gaussian:
            x_hat = torch.sigmoid(x_hat)
            loss = nn.MSELoss()(x_hat, x)

        return loss

    @abstractmethod
    def full_loss(
        self,
        reconstruction_loss: torch.Tensor,
        total_kl: torch.Tensor,
        dimension_wise_kl: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        x, z = batch

        x_hat, mu, log_var = self(x)
        reconstruction_loss = self.reconstruction_loss(x, x_hat)
        total_kl, dimension_wise_kl = kl_divergence(mu, log_var)

        full_loss = self.full_loss(reconstruction_loss, total_kl, dimension_wise_kl)

        self.log("train_full_loss", full_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train_reconstruction_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_total_kl", total_kl, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return full_loss

    def validation_step(self, batch, batch_idx) -> None:
        with torch.no_grad():
            full_loss = self.training_step(batch, batch_idx)
            self.log("validation_full_loss", full_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx) -> None:
        with torch.no_grad():
            full_loss = self.training_step(batch, batch_idx)
            self.log("test_full_loss", full_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class BetaVAE(VAE):
    def __init__(self, beta: float, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def full_loss(
        self,
        reconstruction_loss: torch.Tensor,
        total_kl: torch.Tensor,
        dimension_wise_kl: torch.Tensor,
    ) -> torch.Tensor:
        return reconstruction_loss + self.beta * total_kl
