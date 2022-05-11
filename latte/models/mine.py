"""
Implementation of the modular MINE model for estimating the mutual information
between two arbitrary random variables.
Based on:
Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y., Courville, A., & Hjelm, R. D. (2018).
Mutual information neural estimation. 35th International Conference on Machine Learning, ICML 2018, 2, 864–873.

The code is inspired by and based on https://github.com/gtegner/mine-pytorch.
"""
from enum import Enum
from typing import Any, Union, Dict, Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor


EPSILON = 1e-6


class MINELossType(Enum):
    MINE = 1
    F_DIV = 2
    MINE_BIASED = 3


class MINEBiasCorrection(torch.autograd.Function):
    """
    The custom running-mean-based correction factor to account for the bias of the
    minibatch-based gradient estimate of the MINE loss function.
    """

    @staticmethod
    def forward(ctx: Any, t_marginal: torch.Tensor, running_mean: torch.Tensor) -> Any:
        ctx.save_for_backward(t_marginal, running_mean)
        y = -torch.logsumexp(t_marginal, 0) + torch.log(torch.tensor(t_marginal.shape[0]))
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Union[float, Any], None]:
        x, running_mean = ctx.saved_tensors
        m_grad = grad_output * x.exp().detach() / (running_mean + EPSILON) / x.shape[0]
        return -m_grad, None


def mine_loss(
    t: torch.Tensor, t_marginal: torch.Tensor, running_mean: torch.Tensor, alpha: float
) -> Union[torch.Tensor, torch.Tensor]:
    """The partially bias-corrected loss function which keeps a running mean
    of the denominator of the second term from expression (12) in the paper."""
    t_marginal_exp = torch.exp(-torch.logsumexp(t_marginal, 0) + torch.log(torch.tensor(t_marginal.shape[0]))).detach()
    running_mean = t_marginal_exp if running_mean == 0 else (1 - alpha) * running_mean + alpha * t_marginal_exp
    t_new = MINEBiasCorrection.apply(t_marginal, running_mean)

    return torch.mean(t) + t_new, running_mean


def f_div_loss(t: torch.Tensor, t_marginal: torch.Tensor) -> torch.Tensor:
    """The f-divergence upper-bound-based loss."""
    return torch.mean(t) - torch.mean(torch.exp(t_marginal - 1))


def mine_loss_biased(t: torch.Tensor, t_marginal: torch.Tensor) -> torch.Tensor:
    """The simple version of the DV-formulation-based MINE loss function
    which does not correct for the bias in any way."""
    return torch.mean(t) - torch.logsumexp(t_marginal, 0) + torch.log(torch.tensor(t_marginal.shape[0]))


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


class StatisticsNetwork(nn.Module):
    """The general statistics network to combine the two random variables X and Z.
    It works by taking in the two variables, concatenating them, and then
    passing them to any intermediate network S constructed before turning the
    output of that network into the estimate of mutual information.
    """

    def __init__(self, S: nn.Module, out_dim: int):
        """
        Args:
        S: The arbitrary network connecting X and Z
        out_dim: The size of the output of S, to be able to stick a final single-output linear layer on top
        """
        super().__init__()

        if isinstance(S, nn.Sequential):
            self.S = MultiInputSequential(ConcatLayer(), *S, nn.Linear(out_dim, 1))
        else:
            self.S = MultiInputSequential(ConcatLayer(), S, nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.S(x, z)


class MINE(pl.LightningModule):
    def __init__(
        self,
        T: StatisticsNetwork,
        kind: MINELossType = MINELossType.MINE,
        learning_rate: float = 1e-3,
        alpha: Optional[float] = 0.01,
    ):
        """
        Args:
        T: The statistics network which will be optimised to estimate the mutual information
        kind: The type of loss to use
        lr: The learning rate for the running mean to correct for the bias in the gradient estimates
        """
        super().__init__()

        self.running_mean = torch.zeros((1,))
        self.T = T
        self.learning_rate = learning_rate
        self.alpha = alpha

        self.kind = kind

    def forward(self, x: torch.Tensor, z: torch.Tensor, z_marginal: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
        x: The sample of X from the joint distribution
        z: The sample of Z from the joint distribution
        z_marginal: (optional) A separate sample of the marginal distribution of Z

        """

        # If the marginal z is not provided explicitly, "sample" it by permuting the joint sample batch
        if z_marginal is None:
            z_marginal = z[torch.randperm(x.shape[0])]
        # "Sample" x by permuting the joint sample
        x_marginal = x[torch.randperm(x.shape[0])]

        # Pass it to the general function T
        t = self.T(x, z)
        t_marginal = self.T(x_marginal, z_marginal)

        # Calculate the loss / output of the model
        if self.kind == MINELossType.MINE:
            mi, self.running_mean = mine_loss(t, t_marginal, self.running_mean, self.alpha)
        elif self.kind == MINELossType.F_DIV:
            mi = f_div_loss(t, t_marginal)
        elif self.kind == MINELossType.MINE_BIASED:
            mi = mine_loss_biased(t, t_marginal)

        return mi

    def mi(self, x: torch.Tensor, z: torch.Tensor, z_marginal: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            mi = self.forward(x, z, z_marginal)

        return mi

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x, z = batch
        loss = -self(x, z)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x, z = batch
        loss = -self(x, z)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
