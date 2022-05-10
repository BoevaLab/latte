import pathlib
from typing import Optional, Union, List

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

from latte.dataset.dsprites import DEFAULT_TARGET, load_dsprites_preprocessed, DSpritesFactor, split
from latte.models.datasets import MINEDataset


class DSpritesPCADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, pathlib.Path] = DEFAULT_TARGET,
        batch_size: int = 32,
        target_factors: Optional[List[DSpritesFactor]] = None,
        ignored_factors: Optional[List[DSpritesFactor]] = None,
        n_components: int = 12,
        p_train: float = 0.7,
        p_val: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_factors = target_factors
        self.ignored_factors = ignored_factors
        self.n_components = n_components
        self.p_train = p_train
        self.p_val = p_val

    def setup(self, stage: Optional[str] = None):

        # TODO (Anej): make use of stage

        dataset = load_dsprites_preprocessed(
            method="PCA",
            n_components=self.n_components,
            factors=self.target_factors,
            ignored_factors=self.ignored_factors,
            return_ratios=False,
        )

        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = split(
            dataset.X.astype(np.float32), dataset.y.astype(np.float32), p_train=self.p_train, p_val=self.p_val
        )
        self.mine_train = MINEDataset(X=torch.from_numpy(self.X_train), Z=torch.from_numpy(self.y_train))
        self.mine_val = MINEDataset(X=torch.from_numpy(self.X_val), Z=torch.from_numpy(self.y_val))
        self.mine_test = MINEDataset(X=torch.from_numpy(self.X_test), Z=torch.from_numpy(self.y_test))

    def train_dataloader(self):
        return DataLoader(self.mine_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mine_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mine_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mine_test, batch_size=self.batch_size)
