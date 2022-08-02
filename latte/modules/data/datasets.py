"""
Utility implementations of various Dataset classes used throughout `Latte`.
They only differ in the number of distributions (number of different variables) they serve.
"""

from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    A general dataset class to store samples of observations only (*one* distribution).
    """

    def __init__(self, X: torch.Tensor):
        self.X = X

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


class GenericDataset(Dataset):
    """
    A general dataset class to store samples of observations and targets (*two* distributions).
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        assert len(X) == len(Y), f"The two datasets are not of the same size; {X.shape}, {Y.shape}"

        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.X[idx]) if self.transform is not None else self.X[idx], self.Y[idx]


class MISTDataset(Dataset):
    """
    A general dataset class to store samples of *three* distributions: the original representations, the factors with
    which mutual information should be maximised, and the factors with which mutual information should be minimised.
    """

    def __init__(
        self,
        X: torch.Tensor,
        Z_max: torch.Tensor,
        Z_min: Optional[torch.Tensor] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        assert len(X) == len(Z_max) and (Z_min is None or len(X) == len(Z_min)), (
            f"The datasets are not of the same size; "
            f"X.shape = {X.shape}, Z_max.shape = {Z_max.shape}" + f", Z_min.shape = {Z_min.shape}."
            if Z_min is not None
            else ""
        )

        self.X = X
        self.Z_max = Z_max
        self.Z_min = Z_min

        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.transform(self.X[idx]) if self.transform is not None else self.X[idx],
            self.Z_max[idx],
            self.Z_min[idx] if self.Z_min is not None else torch.tensor(0.0),
        )
