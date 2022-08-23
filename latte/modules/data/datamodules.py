"""
Code implementing various specific and general `Pytorch Lightning` data modules used throughout `Latte`.
The most important general classes implemented here are GenericDataModule, serving data from two distributions,
MISTDataModule, serving data from three distributions useful for the supervised version of MIST.
Both also have version where the data has been split previously and the split should be preserved.
BaseDataModule implements all common processing steps.
"""

import pathlib

from typing import Optional, Union, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from latte.dataset import dsprites
from latte.modules.data import datasets as ds
from latte.dataset import utils as dataset_utils


class BaseDataModule(pl.LightningDataModule):
    """
    A utility base data module for preforming tasks common to all specific data modules used in `Latte`.
    """

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
        # TODO (Anej): What is the purpose of this method?
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=shuffle,
        )


class GenericDataModule(BaseDataModule):
    """
    A generic DataModule for serving observations and targets (data from *two* distributions).
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, **kwargs):
        """
        Args:
            X: observations
            Y: targets
            kwargs: kwargs to be passed to BaseDataModule
        """

        super().__init__(**kwargs)

        (self.X_train, self.X_val, self.X_test), (self.y_train, self.y_val, self.y_test) = dataset_utils.split(
            [X, Y], p_train=self.p_train, p_val=self.p_val
        )

    def setup(self, stage: Optional[str] = None):

        self.train_data = ds.GenericDataset(X=self.X_train, Y=self.y_train)
        self.val_data = ds.GenericDataset(X=self.X_val, Y=self.y_val)
        self.test_data = ds.GenericDataset(X=self.X_test, Y=self.y_test)


class SplitGenericDataModule(GenericDataModule):
    """
    A generic DataModule for serving data from two distributions to MINE.
    Already split into train, val, and test splits.
    This is useful for preventing data leakage in subsequent steps of a pipeline where some data was already used for
    training.
    """

    def __init__(self, X: Dict[str, torch.Tensor], Y: Dict[str, torch.Tensor], **kwargs):
        """
        Args:
            X: A dictionary containing split data from the first distribution.
               Should contain "train", "val", and "test" splits
            Z: A dictionary containing split data from the second distribution.
               Should contain "train", "val", and "test" splits
            kwargs: kwargs to be passed to GenericDataModule
        """

        super(GenericDataModule, self).__init__(**kwargs)
        self.X_train = X["train"]
        self.X_val = X["val"]
        self.X_test = X["test"]
        self.y_train = Y["train"]
        self.y_val = Y["val"]
        self.y_test = Y["test"]


class MISTDataModule(BaseDataModule):
    def __init__(self, X: torch.Tensor, Z_max: torch.Tensor, Z_min: Optional[torch.Tensor] = None, **kwargs):
        """
        This data module implements the basis data module used by the MIST model.
        It thus contains samples from at least two or possibly three distributions.
        The first distribution (`X`) is the "observed" one - usually the learned representations of a VAE model.
        The second distribution (`Z_max`) contains the values of the true factors such that we want to *maximise*
        the mutual information between them and the first distribution.
        The third, optional, distribution (`Z_min`) contains the values of the true factors such that we want to
        *minimise* the mutual information between them and the first distribution.
        We can choose to only maximise the mutual information with some factors and not minimise with any.
        In that case, `Z_min` can be left out (defaults to `None`).
        Args:
            X: data from the first distribution
            Z_max: data from the distribution to maximise the mutual information with
            Z_min: data from the distribution to maximise the mutual information with
            kwargs: kwargs to be passed to BaseDataModule
        """

        super().__init__(**kwargs)
        # Initialise the minimisation datasets as None, since they are passed to the underlying Dataset
        self.Z_min_train, self.Z_min_val, self.Z_min_test = None, None, None

        # If minimising, split all three datasets, otherwise just the two relevant ones.
        if Z_min is not None:
            (
                (self.X_train, self.X_val, self.X_test),
                (self.Z_max_train, self.Z_max_val, self.Z_max_test),
                (self.Z_min_train, self.Z_min_val, self.Z_min_test),
            ) = dataset_utils.split([X, Z_max, Z_min], p_train=self.p_train, p_val=self.p_val)
        else:
            (
                (self.X_train, self.X_val, self.X_test),
                (self.Z_max_train, self.Z_max_val, self.Z_max_test),
            ) = dataset_utils.split([X, Z_max], p_train=self.p_train, p_val=self.p_val)

    def setup(self, stage: Optional[str] = None):

        self.train_data = ds.MISTDataset(X=self.X_train, Z_max=self.Z_max_train, Z_min=self.Z_min_train)
        self.val_data = ds.MISTDataset(X=self.X_val, Z_max=self.Z_max_val, Z_min=self.Z_min_val)
        self.test_data = ds.MISTDataset(X=self.X_test, Z_max=self.Z_max_test, Z_min=self.Z_min_test)


class SplitMISTDataModule(MISTDataModule):
    """
    Same as `MISTDataModule`, but initialised with data with is already split into train/validation/test splits.
    This is useful for preventing data leakage in subsequent steps of a pipeline where some data was already used for
    training.
    """

    def __init__(
        self,
        X: Dict[str, torch.Tensor],
        Z_max: Dict[str, torch.Tensor],
        Z_min: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        """

        Args:
            X: A dictionary containing split data from the first distribution.
               Should contain "train", "val", and "test" splits
            Z_max: A dictionary containing split data from the distribution to maximise the mutual information with.
                   Should contain "train", "val", and "test" splits
            Z_min: An optional dictionary containing split data from the distribution to minimise the
                   mutual information with.
                   Should contain "train", "val", and "test" splits
            kwargs: kwargs to be passed to BaseDataModule
        """

        super(BaseDataModule, self).__init__(**kwargs)
        self.X_train = X["train"]
        self.X_val = X["val"]
        self.X_test = X["test"]
        self.Z_max_train = Z_max["train"]
        self.Z_max_val = Z_max["val"]
        self.Z_max_test = Z_max["test"]
        self.Z_min_train = Z_min["train"] if Z_min is not None else None
        self.Z_min_val = Z_min["val"] if Z_min is not None else None
        self.Z_min_test = Z_min["test"] if Z_min is not None else None


class DSpritesDataModule(BaseDataModule):
    """
    A specific DataModule for the dSprites dataset.
    It keeps optionally preprocessed (PCA or FactorAnalysis decomposed) dSprites data.
    """

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
        self.dataset = None

    def setup(self, stage: Optional[str] = None):

        # The raw dataset with the original images and principal components
        self.dataset = dsprites.load_dsprites_preprocessed(
            return_raw=self.raw,
            method="PCA",
            n_components=self.n_components,
            factors=self.target_factors,
            ignored_factors=self.ignored_factors,
        )

        # If noise is added later, keep the images as numpy arrays for compatibility with the noise transformation,
        # otherwise transform them to a tensor
        (self.X_train, self.X_val, self.X_test), (self.y_train, self.y_val, self.y_test) = dataset_utils.split(
            [
                self.dataset.X if self.noise_amount > 0 else torch.from_numpy(self.dataset.X),
                torch.from_numpy(self.dataset.y),
            ],
            p_train=self.p_train,
            p_val=self.p_val,
        )

        def _transform(img):
            pepper = self.rng.binomial(1, p=self.noise_amount, size=img.shape).astype(np.float32)
            salt = self.rng.binomial(1, p=self.noise_amount, size=img.shape).astype(np.float32)
            return torch.from_numpy(np.clip(img + salt - pepper, 0, 1))

        self.train_data = ds.GenericDataset(
            X=self.X_train, Y=self.y_train, transform=_transform if self.noise_amount > 0 else None
        )
        self.val_data = ds.GenericDataset(
            X=self.X_val, Y=self.y_val, transform=_transform if self.noise_amount > 0 else None
        )
        self.test_data = ds.GenericDataset(
            X=self.X_test, Y=self.y_test, transform=_transform if self.noise_amount > 0 else None
        )
