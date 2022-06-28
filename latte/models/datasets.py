from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    """A general dataset class to store samples of observations and targets."""

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        assert len(X) == len(Y), f"The two datasets are not of the same size; {X.shape}, {Y.shape}"

        self.X: torch.Tensor = X
        self.Y: torch.Tensor = Y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.X[idx]) if self.transform is not None else self.X[idx], self.Y[idx]


class MINEDataset(GenericDataset):
    """A general dataset class to store samples of variables we would like to
    calculate the mutual information between."""

    def __init__(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(X=X, Y=Z, transform=transform)


class PretrainDataset(Dataset):
    """A general dataset class to store samples of variables we would like to
    calculate the mutual information between."""

    def __init__(self, X: torch.Tensor):
        self.X: torch.Tensor = X

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]
