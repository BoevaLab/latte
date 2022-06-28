from typing import Union, Dict, Callable

import torch
import pytorch_lightning as pl
import geoopt

from latte.models.layers import ManifoldProjectionLayer


class Orientator(pl.LightningModule):
    def __init__(
        self,
        d: int,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        learning_rate: float = 1e-4,
        random_init: bool = False,
    ):
        """
        Args:
            d: The dimensionality of the space
            loss: The loss to use to assess the fit between the rotated and the true values
            learning_rate: The learning rate for the rotation matrix
            random_init: Whether to initialise the rotation matrix randomly. If not, it is initialised to the identity.
        """
        super().__init__()

        self.orientator = ManifoldProjectionLayer(d, d, init=torch.eye(d) if not random_init else None)
        self.loss = loss
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orientator(x)

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        with torch.no_grad():
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx) -> None:
        with torch.no_grad():
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return geoopt.optim.RiemannianAdam([self.orientator.A], lr=self.learning_rate)
