"""
Implementation of the modular MINE model for estimating the mutual information between two arbitrary random variables.
Based on:
Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y., Courville, A., & Hjelm, R. D. (2018).
Mutual information neural estimation. 35th International Conference on Machine Learning, ICML 2018, 2, 864–873.

The code is inspired by and based on https://github.com/gtegner/mine-pytorch.

It contains a standalone Pytorch Lightning module implementation which can be used for estimation and a pure
pytorch version which is easier to plug into other models, e.g., MIST.
"""
from enum import Enum
from typing import Any, Union, Dict, Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from latte.modules.layers import ManifoldProjectionLayer, ConcatLayer, MultiInputSequential


EPSILON = 1e-6


class MINEObjectiveType(Enum):
    """The different types of objectives used by MINE.
    `Latte` mainly uses the bias-corrected MINE objective."""

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
        y = torch.logsumexp(t_marginal, 0) - torch.log(torch.tensor(t_marginal.shape[0]))
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Union[float, Any], None]:
        t_marginal, running_mean = ctx.saved_tensors
        corrected_grad = grad_output * t_marginal.exp().detach() / (running_mean + EPSILON) / t_marginal.shape[0]
        return corrected_grad, None


def mine_objective(
    t: torch.Tensor, t_marginal: torch.Tensor, running_mean: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The partially bias-corrected loss function which keeps a running mean
    of the denominator of the second term from expression (12) in the paper."""

    # detached such that the running mean computation does not affect the gradient, which are corrected manually
    exp_t_marginal = torch.exp(torch.logsumexp(t_marginal, 0) - torch.log(torch.tensor(t_marginal.shape[0]))).detach()

    running_mean = exp_t_marginal if running_mean == 0 else (1 - alpha) * running_mean + alpha * exp_t_marginal

    y = MINEBiasCorrection.apply(t_marginal, running_mean)

    return torch.mean(t) - y, running_mean


def f_div_objective(t: torch.Tensor, t_marginal: torch.Tensor) -> torch.Tensor:
    """The f-divergence upper-bound-based loss."""
    return torch.mean(t) - torch.mean(torch.exp(t_marginal - 1))


def mine_objective_biased(t: torch.Tensor, t_marginal: torch.Tensor) -> torch.Tensor:
    """The simple version of the DV-formulation-based MINE loss function
    which does not correct for the bias in any way."""
    return torch.mean(t) - (torch.logsumexp(t_marginal, 0) - torch.log(torch.tensor(t_marginal.shape[0])))


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


class ManifoldStatisticsNetwork(StatisticsNetwork):
    """The specialised statistics network which projects the random variable X
    onto the linear subspace spanned by the d-frame V before combining it with Z.
    """

    def __init__(self, S: nn.Module, n: int, d: int, out_dim: int, init: Optional[torch.Tensor] = None):
        """
        Args:
            S: The arbitrary network connecting X and Z
            n: Dimension of the observable space
            d: Dimension of the subspace to project to
            out_dim: The size of the output of S, to be able to stick a final single-output linear layer on top
            init: An optional initial setting of the manifold parameter
        """
        super().__init__(S, out_dim)

        self.projection_layer = ManifoldProjectionLayer(n, d, init=init)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.S(self.projection_layer(x), z)


class StandaloneMINE(pl.LightningModule):
    """
    Pytorch Lightning module for a network estimating the mutual information between two random variables based
    on their empirical samples.
    This is done by maximising a lower bound on the mutual information.
    Note that the forward call to the model computes the *negative* of the estimate to frame the optimisation
    as a more standard minimisation.
    The estimate of the mutual information can be obtained by calling model.mi().

    This is a standalone implementation of the model which can be used to approximate the mutual information
    between two distributions.
    The `MINE` class below is used as a module in other models, e.g., MIST.
    """

    def __init__(
        self,
        T: StatisticsNetwork,
        kind: MINEObjectiveType = MINEObjectiveType.MINE,
        learning_rate: float = 1e-4,
        alpha: float = 0.01,
        lr_scheduler_patience: int = 8,
        lr_scheduler_min_delta: float = 0.0,
        verbose: bool = False,
    ):
        """
        Args:
            T: The statistics network which will be optimised to estimate the mutual information
            kind: The type of objective to use
            learning_rate: The learning rate for the statistics network parameters
            alpha: The update step size of the bias correction
            lr_scheduler_patience: The patience for the ReduceOnPlateu learning rate scheduler
            lr_scheduler_min_delta: The minimum improvement for the ReduceOnPlateu learning rate scheduler
            verbose: Controls the verbosity of callbacks and learning rate schedulers
        """
        super().__init__()

        self.running_mean = torch.zeros((1,))
        self.T = T
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_min_delta = lr_scheduler_min_delta
        self.verbose = verbose

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
        # x_marginal = x

        # Pass it to the general function T
        t = self.T(x, z)
        t_marginal = self.T(x_marginal, z_marginal)

        # Calculate the output of the model
        if self.kind == MINEObjectiveType.MINE:
            mi, self.running_mean = mine_objective(t, t_marginal, self.running_mean, self.alpha)
        elif self.kind == MINEObjectiveType.F_DIV:
            mi = f_div_objective(t, t_marginal)
        elif self.kind == MINEObjectiveType.MINE_BIASED:
            mi = mine_objective_biased(t, t_marginal)

        # Since we are *maximising* the estimate of the mutual information, we return the negative of the estimate
        # as the value of the loss to frame this as a more standard *minimisation* problem
        return -mi

    def mi(self, x: torch.Tensor, z: torch.Tensor, z_marginal: torch.Tensor = None) -> torch.Tensor:
        """
        A utility function that returns the estimate of the mutual information from a trained model.
        """
        with torch.no_grad():
            mi = -self(x, z, z_marginal)

        return mi

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        x, z = batch
        loss = self(x, z)
        self.log("train_mutual_information", -loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "train_mutual_information": -loss}

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        x, z = batch
        mi = self.mi(x, z)
        self.log("validation_mutual_information", mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"validation_mutual_information": mi}

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        x, z = batch
        mi = self.mi(x, z)
        self.log("test_mutual_information", mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"test_mutual_information": mi}

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, Dict[str, Any]]]:

        # Optimizer of the statistics network parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        network_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
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
            "monitor": "validation_mutual_information",
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class MINE(nn.Module):
    """
    A normal Pytorch module for a network estimating the mutual information between two random variables based
    on their empirical samples.
    This is done by maximising a lower bound on the mutual information.
    Note that the forward call to the model computes the *negative* of the estimate to frame the optimisation
    as a more standard minimisation.
    The estimate of the mutual information can be obtained by calling model.mi().

    This is a separate implementation which can be plugged into other estimators (e.g., MIST).
    """

    def __init__(
        self,
        T: StatisticsNetwork,
        kind: MINEObjectiveType = MINEObjectiveType.MINE,
        alpha: float = 0.01,
    ):
        """
        Args:
            T: The statistics network which will be optimised to estimate the mutual information
            kind: The type of objective to use
            alpha: The update step size of the bias correction
        """
        super().__init__()

        self.running_mean = torch.zeros((1,))
        self.T = T
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
        # x_marginal = x

        # Pass it to the general function T
        t = self.T(x, z)
        t_marginal = self.T(x_marginal, z_marginal)

        # Calculate the output of the model
        if self.kind == MINEObjectiveType.MINE:
            mi, self.running_mean = mine_objective(t, t_marginal, self.running_mean, self.alpha)
        elif self.kind == MINEObjectiveType.F_DIV:
            mi = f_div_objective(t, t_marginal)
        elif self.kind == MINEObjectiveType.MINE_BIASED:
            mi = mine_objective_biased(t, t_marginal)

        # Since we are *maximising* the estimate of the mutual information, we return the negative of the estimate
        # as the value of the loss to frame this as a more standard *minimisation* problem
        return -mi

    def mi(self, x: torch.Tensor, z: torch.Tensor, z_marginal: torch.Tensor = None) -> torch.Tensor:
        """
        A utility function that returns the estimate of the mutual information from a trained model.
        """
        with torch.no_grad():
            mi = -self(x, z, z_marginal)

        return mi
