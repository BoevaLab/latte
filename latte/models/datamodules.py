import pathlib
from typing import Optional, Union, List

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

from latte.dataset import dsprites
from latte.models import datasets as ds
from latte.models import utils


class GenericMINEDataModule(pl.LightningDataModule):
    """A generic DataModule for serving data from two distributions to MINE."""

    def __init__(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        p_train: float = 0.7,
        p_val: float = 0.1,
        batch_size: int = 32,
        test_batch_size=512,
        num_workers: int = 1,
    ):
        """

        Args:
            X: data from the first distribution
            Z: data from the second distribution
            p_train: The portion of the data for training
            p_val: The portion of the data for validation
            batch_size: Batch size for training
            batch_size: Batch size for validation and test
            num_workers: The number of workers for the DataLoaders
        """

        super().__init__()
        self.X = X
        self.Z = Z
        self.p_train = p_train
        self.p_val = p_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    # TODO (Anej): this is duplicated...
    def setup(self, stage: Optional[str] = None):

        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = utils.split(
            self.X.astype(np.float32), self.Z.astype(np.float32), p_train=self.p_train, p_val=self.p_val
        )
        self.mine_train = ds.MINEDataset(X=torch.from_numpy(self.X_train), Z=torch.from_numpy(self.y_train))
        self.mine_val = ds.MINEDataset(X=torch.from_numpy(self.X_val), Z=torch.from_numpy(self.y_val))
        self.mine_test = ds.MINEDataset(X=torch.from_numpy(self.X_test), Z=torch.from_numpy(self.y_test))

    def train_dataloader(self):
        return DataLoader(self.mine_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mine_val, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mine_test, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mine_test, batch_size=self.test_batch_size, num_workers=self.num_workers)


class DSpritesPCADataModule(pl.LightningDataModule):
    """A DataModule for the dSprites dataset.
    It keeps preprocessed (PCA or FactorAnalysis decomposed) dSprites data."""

    def __init__(
        self,
        data_dir: Union[str, pathlib.Path] = dsprites.DEFAULT_TARGET,
        target_factors: Optional[List[dsprites.DSpritesFactor]] = None,
        ignored_factors: Optional[List[dsprites.DSpritesFactor]] = None,
        n_components: int = 12,
        p_train: float = 0.7,
        p_val: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        """

        Args:
            data_dir: The directory of the dSprites data
            target_factors: The factors to keep in the data, i.e., the ones which we want to predict/compute with
            ignored_factors: The factors in the data to ignore. The dataset will be filtered such that these factors
                             have a constant value
            n_components: Number of components to project to in PCA or FactorAnalysis
            p_train: The portion of the data for training
            p_val: The portion of the data for validation
            batch_size: Batch size for training
            num_workers: The number of workers for the DataLoaders
        """
        super().__init__()
        self.data_dir = data_dir
        self.target_factors = target_factors
        self.ignored_factors = ignored_factors
        self.n_components = n_components
        self.p_train = p_train
        self.p_val = p_val
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):

        # TODO (Anej): (optional) make use of stage [fit, test, predict]

        dataset = dsprites.load_dsprites_preprocessed(
            method="PCA",
            n_components=self.n_components,
            factors=self.target_factors,
            ignored_factors=self.ignored_factors,
        )

        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = utils.split(
            dataset.X.astype(np.float32), dataset.y.astype(np.float32), p_train=self.p_train, p_val=self.p_val
        )
        self.mine_train = ds.MINEDataset(X=torch.from_numpy(self.X_train), Z=torch.from_numpy(self.y_train))
        self.mine_val = ds.MINEDataset(X=torch.from_numpy(self.X_val), Z=torch.from_numpy(self.y_val))
        self.mine_test = ds.MINEDataset(X=torch.from_numpy(self.X_test), Z=torch.from_numpy(self.y_test))

    def train_dataloader(self):
        return DataLoader(self.mine_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mine_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mine_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mine_test, batch_size=self.batch_size, num_workers=self.num_workers)
