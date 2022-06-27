from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset


class MINEDataset(Dataset):
    """A general dataset class to store samples of variables we would like to
    calculate the mutual information between."""

    def __init__(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        assert len(X) == len(Z), f"The two datasets are not of the same size; {X.shape}, {Z.shape}"

        self.X: torch.Tensor = X
        self.Z: torch.Tensor = Z
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.X[idx]) if self.transform is not None else self.X[idx], self.Z[idx]


class PretrainDataset(Dataset):
    """A general dataset class to store samples of variables we would like to
    calculate the mutual information between."""

    def __init__(self, X: torch.Tensor):
        self.X: torch.Tensor = X

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]
