import pathlib

# TODO(Pawel): When pytype starts supporting Literal, remove the comment
from typing import Optional, Union, List, Dict  # pytype: disable=not-supported-yet

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from latte.dataset import dsprites
from latte.models import datasets as ds
from latte.dataset import utils as dataset_utils


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        p_train: float = 0.7,
        p_val: float = 0.1,
        batch_size: int = 32,
        test_batch_size: int = 512,
        num_workers: int = 1,
    ):
        """

        Args:
            p_train: The portion of the data for training
            p_val: The portion of the data for validation
            batch_size: Batch size for training
            batch_size: Batch size for validation and test
            num_workers: The number of workers for the DataLoaders
        """
        super().__init__()

        self.p_train = p_train
        self.p_val = p_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_batch_size = test_batch_size

        self.train_data, self.val_data, self.test_data = None, None, None

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.val_data,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=shuffle,
        )

    def test_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=shuffle,
        )

    def predict_dataloader(self, shuffle: bool = False):
        # TODO: What is the role of this method?
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=shuffle,
        )


class GenericMINEDataModule(BaseDataModule):
    """A generic DataModule for serving data from two distributions to MINE."""

    def __init__(self, X: torch.Tensor, Z: torch.Tensor, **kwargs):
        """

        Args:
            X: data from the first distribution
            Z: data from the second distribution
            kwargs: kwargs to be passed to BaseDataModule
        """

        super().__init__(**kwargs)

        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = dataset_utils.split(
            X.type(torch.FloatTensor), Z.type(torch.FloatTensor), p_train=self.p_train, p_val=self.p_val
        )

    def setup(self, stage: Optional[str] = None):

        self.train_data = ds.MINEDataset(X=self.X_train, Z=self.y_train)
        self.val_data = ds.MINEDataset(X=self.X_val, Z=self.y_val)
        self.test_data = ds.MINEDataset(X=self.X_test, Z=self.y_test)


class SplitMINEDataModule(GenericMINEDataModule):
    """A generic DataModule for serving data from two distributions to MINE.
    Already split into train, val, and test splits."""

    def __init__(self, X: Dict[str, torch.Tensor], Z: Dict[str, torch.Tensor], **kwargs):
        """

        Args:
            X: A dictionary containing split data from the first distribution.
               Should contain "train", "val", and "test" splits
            Z: A dictionary containing split data from the second distribution.
               Should contain "train", "val", and "test" splits
            kwargs: kwargs to be passed to BaseDataModule
        """

        super(GenericMINEDataModule, self).__init__(**kwargs)
        self.X_train = X["train"]
        self.y_train = Z["train"]
        self.X_val = X["val"]
        self.y_val = Z["val"]
        self.X_test = X["test"]
        self.y_test = Z["test"]


class DSpritesDataModule(BaseDataModule):
    """A DataModule for the dSprites dataset.
    It keeps optionally preprocessed (PCA or FactorAnalysis decomposed) dSprites data."""

    def __init__(
        self,
        data_dir: Union[str, pathlib.Path] = dsprites.DEFAULT_TARGET,
        target_factors: Optional[List[dsprites.DSpritesFactor]] = None,
        ignored_factors: Optional[List[dsprites.DSpritesFactor]] = None,
        raw: bool = False,
        n_components: int = 12,
        noise_amount: float = 0.0,
        rng: dataset_utils.RandomGenerator = 42,
        **kwargs,
    ):
        """

        Args:
            data_dir: The directory of the dSprites data
            target_factors: The factors to keep in the data, i.e., the ones which we want to predict/compute with
            ignored_factors: The factors in the data to ignore. The dataset will be filtered such that these factors
                             have a constant value
            raw: Whether to keep the raw image data (and not decompose it using the PCA).
            n_components: Number of components to project to in PCA or FactorAnalysis.
            noise_amount: The amount of salt and pepper noise to inject into images.
            rng: The random number generator for the noise in case noise is added to the images.
            kwargs: kwargs to be passed to BaseDataModule.
        """
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.target_factors = target_factors
        self.ignored_factors = ignored_factors
        self.raw = raw
        self.n_components = n_components
        self.noise_amount = noise_amount
        self.rng = np.random.default_rng(rng)

    def setup(self, stage: Optional[str] = None):

        # The raw dataset with the original images and principal components
        self.dataset = dsprites.load_dsprites_preprocessed(
            return_raw=self.raw,
            method="PCA",
            n_components=self.n_components,
            factors=self.target_factors,
            ignored_factors=self.ignored_factors,
        )

        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = dataset_utils.split(
            self.dataset.X if self.noise_amount > 0 else torch.from_numpy(self.dataset.X).type(torch.FloatTensor),
            torch.from_numpy(self.dataset.y).type(torch.FloatTensor),
            p_train=self.p_train,
            p_val=self.p_val,
        )

        def _transform(img):
            pepper = self.rng.binomial(1, p=self.noise_amount, size=img.shape)
            salt = self.rng.binomial(1, p=self.noise_amount, size=img.shape)
            return torch.from_numpy(np.clip(img + salt - pepper, 0, 1)).type(torch.FloatTensor)

        self.train_data = ds.MINEDataset(
            X=self.X_train, Z=self.y_train, transform=_transform if self.noise_amount > 0 else None
        )
        self.val_data = ds.MINEDataset(
            X=self.X_val, Z=self.y_val, transform=_transform if self.noise_amount > 0 else None
        )
        self.test_data = ds.MINEDataset(
            X=self.X_test, Z=self.y_test, transform=_transform if self.noise_amount > 0 else None
        )
