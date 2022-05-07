from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class MINELossType(Enum):
    MINE = 1
    F_DIV = 2
    MINE_BIASED = 3


class MINELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass


def mine_loss(t, t_marginal, running_mean, lr):
    running_mean = (1 - lr) * running_mean + lr * torch.mean(t_marginal)
    return torch.mean(t) - torch.zeros()


def f_div_loss(t, t_marginal):
    return torch.mean(t) - torch.mean(torch.exp(t_marginal - 1))


def mine_loss_biased(t, t_marginal):
    return torch.mean(t) - torch.logsumexp(t_marginal, 0) + torch.log(torch.tensor(t_marginal.shape[0]))


class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)


class CustomSequential(nn.Sequential):
    def forward(self, *x):
        for module in self._modules.values():
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x


class StatisticsNetwork(nn.Module):
    def __init__(self, S, out_dim):
        super().__init__()

        if isinstance(S, nn.Sequential):
            self.S = CustomSequential(ConcatLayer(), *S, nn.Linear(out_dim, 1))
        else:
            self.S = CustomSequential(ConcatLayer(), S, nn.Linear(out_dim, 1))

    def forward(self, x, z):
        return self.S(x, z)


class MINE(nn.Module):
    def __init__(self, T: StatisticsNetwork, kind: MINELossType = MINELossType.MINE, lr: float = None):
        super().__init__()

        self.running_mean = 0
        self.T = T
        self.lr = lr

        self.kind = kind

    def forward(self, x: torch.Tensor, z: torch.Tensor, z_marginal: torch.Tensor = None) -> torch.Tensor:
        if z_marginal is None:
            z_marginal = z[torch.randperm(x.shape[0])]
        x_marginal = x[torch.randperm(x.shape[0])]

        t = self.T(x, z)
        t_marginal = self.T(x_marginal, z_marginal)

        if self.kind == MINELossType.MINE:
            mi, self.running_mean = mine_loss(t, t_marginal, self.running_mean, self.lr)
        elif self.kind == MINELossType.F_DIV:
            mi = f_div_loss(t, t_marginal)
        elif self.kind == MINELossType.MINE_BIASED:
            mi = mine_loss_biased(t, t_marginal)

        return -mi

    def mi(self, x: torch.Tensor, z: torch.Tensor, z_marginal: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            mi = -self.forward(x, z, z_marginal)

        return mi
