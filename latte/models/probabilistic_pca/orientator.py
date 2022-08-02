"""
Implementation of a `Pytorch Lightning` module designed to take an original estimate of the mixing matrix from the
probabilistic PCA model (which corresponds to the true one up to a rotation in the latent space if the estimate is good)
and orient it towards the true one using a set of *labelled* data points with known latent values.
It works by orienting the matrix by a pre-multiplication of the estimate with a column-orthogonal matrix from the
Stiefel manifold.
"""

# TODO(Pawel): When pytype starts supporting Literal, remove the comment
from typing import Literal, Union, Dict, Callable, Any  # pytype: disable=not-supported-yet

import torch
import pytorch_lightning as pl
import geoopt

from latte.modules.layers import ManifoldProjectionLayer


class Orientator(pl.LightningModule):
    """This model can be used to orient a latent space to align it to a (subset of) the true latent factors."""

    def __init__(
        self,
        n: int,
        d: int,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        learning_rate: float = 1e-4,
        init_sign: Literal[-1, 1] = 1,
        lr_scheduler_patience: int = 8,
        lr_scheduler_min_delta: float = 1e-2,
        manifold_optimisation: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            n: The dimensionality of the space to project from
            d: The dimensionality of the space to project to
            loss: The loss to use to assess the fit between the rotated and the true values
            learning_rate: The learning rate for the rotation matrix
            init_sign: Determines the initialization projection matrix.
                       If init_sign = 1, the projection is initialized as the identity matrix, and if it is -1, it is
                       initialized as the identity with the sign of the first element flipped (-1).
            lr_scheduler_patience: The patience for the ReduceOnPlateu learning rate scheduler
            lr_scheduler_min_delta: The minimum improvement for the ReduceOnPlateu learning rate scheduler
            manifold_optimisation: Whether to orient the estimate constrained to the Stiefel manifold or not.
                                   This should be left True for almost all use cases, it is mostly here to enable
                                   investigating the effect of the constrained optimisation versus unconstrained.
            verbose: Controls the verbosity of callbacks and learning rate schedulers
        """
        super().__init__()

        assert init_sign in [-1, 1], "init_sign must be in {-1, 1}"
        init = torch.eye(d) if n == d else None
        # If n = d, the Stiefel manifold is *disconnected* and we have to initialize the projection manually
        # This enables optimisation over the disconnected Stiefel manifold
        if init is not None:
            init[0, 0] = init_sign
        self.orientator = ManifoldProjectionLayer(n, d, init=init, stiefel_manifold=manifold_optimisation)
        self.loss = loss
        self.learning_rate = learning_rate
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_min_delta = lr_scheduler_min_delta
        self.verbose = verbose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the rotation to the tensor
        return self.orientator(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
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

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, Dict[str, Any]]]:
        # The `geoopt` optimiser over the Stiefel manifold
        optimizer = geoopt.optim.RiemannianAdam([self.orientator.A], lr=self.learning_rate)

        # A learning rate scheduler for the optimizer over the Stiefel manifold
        network_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_min_delta,
            threshold_mode="abs",
            verbose=self.verbose,
            min_lr=1e-6,
        )

        lr_scheduler_config = {
            "scheduler": network_lr_scheduler,
            "interval": "epoch",
            "monitor": "validation_loss",
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
