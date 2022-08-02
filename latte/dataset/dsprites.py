""" Script for downloading and loading the `dSprites` dataset (https://github.com/deepmind/dsprites-dataset). """

import dataclasses
import pathlib
from enum import IntEnum
from typing import Tuple, Union, List, Dict, Optional, Sequence

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis

import latte.dataset.utils as dsutils


DEFAULT_TARGET = "dataset/raw/dsprites.npz"
DSPRITES_URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
MD5_CHECKSUM = "7da33b31b13a06f4b04a70402ce90c2e"


# Factors of variation in the dSprites dataset and their dimensions
# in the latents data structure
class DSpritesFactor(IntEnum):
    COLOR = 0
    SHAPE = 1
    SCALE = 2
    ORIENTATION = 3
    X = 4
    Y = 5


# Maps the dSprites factors of variation to their descriptive names
factor_names = {
    DSpritesFactor.SHAPE: "Shape",
    DSpritesFactor.SCALE: "Scale",
    DSpritesFactor.ORIENTATION: "Orientation",
    DSpritesFactor.X: "Position x",
    DSpritesFactor.Y: "Position y",
}


# Maps the dSprites factors of variation to a value in their range
# so that we can effectively remove the factor of variation by filtering
# the images which do not have this specified value.
# This removes the variation in that factor (keeps it constant) and downsizes the dataset.
fixed_values = {
    DSpritesFactor.SHAPE: 1,
    DSpritesFactor.SCALE: 1,
    DSpritesFactor.ORIENTATION: 0,
    DSpritesFactor.X: 15 / 31,
    DSpritesFactor.Y: 15 / 31,
}


@dataclasses.dataclass
class DSpritesMetadata:
    """
    Provided `dSprites` metadata.
    """

    date: str
    description: str
    version: int
    latents_names: Tuple[str, ...]


@dataclasses.dataclass
class DSpritesDataset:
    """
    The original `dSprites` dataset as downloaded from the source.
    """

    metadata: DSpritesMetadata
    imgs: np.ndarray
    latents_values: np.ndarray
    latents_classes: np.ndarray


@dataclasses.dataclass
class PreprocessedDSpritesDataset:
    """
    Preprocessed `dSprites` dataset, where `X` and `y` can either be the principal components of the images or the
    raw images themselves.
    See `load_dsprites_preprocessed` for more details.
    """

    X: torch.Tensor
    y: torch.Tensor
    factor2idx: Dict[DSpritesFactor, int]
    pca_ratios: Optional[np.ndarray]


@dataclasses.dataclass
class DecompositionResult:
    X: np.ndarray
    pca_ratios: Optional[np.ndarray]


def _parse_metadata(data) -> DSpritesMetadata:
    """Extracts metadata from the raw dSprites dataset.

    Args:
        data: raw dSprites dataset

    Returns:
        metadata object
    """
    metadata = data["metadata"].tolist()

    def _extract(key: str):
        return metadata[key.encode("UTF-8")]

    return DSpritesMetadata(
        date=_extract("date").decode("UTF-8"),
        description=_extract("description").decode("UTF-8"),
        version=_extract("version"),
        latents_names=tuple([x.decode("UTF-8") for x in _extract("latents_names")]),
    )


def load_dsprites(filepath: Union[str, pathlib.Path] = DEFAULT_TARGET, download_ok: bool = True) -> DSpritesDataset:
    """Function loading the dSprites dataset.

    Args:
        filepath: where the dataset should be loaded from
        download_ok: if the dataset should be downloaded (if it doesn't exist)

    Returns:
        DSpritesDataset object
    """
    filepath = pathlib.Path(filepath)

    # If the file doesn't exist and we have permission to download,
    # we will download it
    if (not filepath.exists()) and download_ok:
        dsutils.download(target=filepath, _source=DSPRITES_URL, _checksum=MD5_CHECKSUM)

    # Load the dataset to the memory
    data = np.load(filepath, allow_pickle=True, encoding="bytes")

    metadata: DSpritesMetadata = _parse_metadata(data)

    # Now we can open the dataset and reshape it
    return DSpritesDataset(
        metadata=metadata,
        latents_values=data["latents_values"],
        latents_classes=data["latents_classes"],
        imgs=data["imgs"],
    )


def prepare_data(
    data: DSpritesDataset,
    factors: List[DSpritesFactor],
    ignored_factors: Optional[List[DSpritesFactor]],
    return_raw: bool = False,
    keep_all_dimensions: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[DSpritesFactor, int]]:
    """Converts the raw dSprites data into Numpy arrays which will be
    processed by PCA.
    Args:
        data: DSpritesDataset object
        factors: Which factors to keep the information about in the returned data
        ignored_factors: Which factors should be held fixed (all the variation in them will be removed)
        return_raw: Whether to return the raw image data (and not decompose it using the specified method).
        keep_all_dimensions: If true, do not remove dimensions with zero variance (useful for later reconstruction)

    Returns:
        X: flattened images with constant-valued pixels removed
        y: the values of the selected factors of variation for each image
        factor2idx: a dictionary mapping each dSprites factor (DSpritesFactor)
                    to the dimension of y in which it is captured
    """

    X = data.imgs

    # Select the images of interest by ignoring the variation of factors
    # we are not interested in
    sample_mask = np.ones(data.latents_values.shape[0], dtype=bool)
    if ignored_factors is not None:
        for factor in ignored_factors:
            sample_mask *= data.latents_values[:, factor] == fixed_values[DSpritesFactor(factor)]

    X = X[sample_mask]
    y = data.latents_values[sample_mask, :]

    if not return_raw and not keep_all_dimensions:
        # Flatten images
        X = data.imgs.reshape((len(data.imgs), -1))

        # Filter out constant pixels
        X = X[:, X.max(axis=0) == 1]

    # Keep only the factors that we want
    y = y[:, factors]

    # The dictionary which maps each of the factors of interest
    # into the dimension of the returned factors y
    factor2idx = {factor: ii for ii, factor in enumerate(factors)}

    return X, y, factor2idx


def decompose(X: np.ndarray, method: str, n_components: int) -> DecompositionResult:
    """Decomposes the flattened images either PCA or FactorAnalysis.

    Args:
        X: the array of flattened images
        method: The way to preprocess the data, {PCA, FA}
        n_components: Number of component to produce in the representation.

    Returns:
        A DecompositionResult object with the representations of the images
        in the decomposed space and, in the case of PCA, the explained variance ratios
    """
    assert method in [
        "PCA",
        "FA",
        "FactorAnalysis",
    ], f"Method {method} is not supported, only PCA and FactorAnalysis supported."

    if method == "PCA":
        trans = PCA(n_components=n_components)
    else:
        trans = FactorAnalysis(n_components=n_components)

    X = StandardScaler().fit_transform(X)
    X = trans.fit_transform(X)

    return DecompositionResult(X=X, pca_ratios=trans.explained_variance_ratio_ if method == "PCA" else None)


def load_dsprites_preprocessed(
    filepath: Union[str, pathlib.Path] = DEFAULT_TARGET,
    download_ok: bool = True,
    return_raw: bool = False,
    method: Optional[str] = "PCA",
    n_components: Optional[int] = 20,
    factors: Optional[Sequence[DSpritesFactor]] = None,
    ignored_factors: Optional[Sequence[DSpritesFactor]] = None,
    keep_all_dimensions: bool = False,
) -> PreprocessedDSpritesDataset:
    """Load the dSprites dataset and optionally preprocess it using PCA or FactorAnalysis.
    Args:
        filepath: The location of the dsprites data.
        download_ok: If the data can be downloaded.
        return_raw: Whether to return the raw image data (and not decompose it using the specified method).
        method: The way to preprocess the data, {PCA, FA}
        n_components: Number of component to produce in the representation.
        factors: Which factors to return.
        ignored_factors: Which factors should be kept constant in the dataset.
            The data is filtered and reduced such that these factors have a constant value.
        keep_all_dimensions: If true, do not remove dimensions with zero variance (useful for later reconstruction)

    Returns:
        A preprocessedDSpritesDataset holding the filtered and optionally transformed data
        (coordinates in PCA or FA component space), the filtered latent factors,
        the PCA variance explained proportions in case method="PCA", and a dictionary
        mapping each of the returned factors to the dimension in `y` to be able to select
        appropriate factors from the returned ones.
    """

    if factors is None:
        # By default, return all factors
        factors = list(factor_names.keys())

    # Load raw data
    data = load_dsprites(filepath, download_ok)

    # Flatten and filter the data
    X, y, factor2idx = prepare_data(
        data,
        factors=factors,
        ignored_factors=ignored_factors,
        keep_all_dimensions=keep_all_dimensions,
        return_raw=return_raw,
    )

    if not return_raw:
        # Reduce the dimensionality
        decomposition_result = decompose(X, method, n_components=n_components)
        X = decomposition_result.X
        pca_ratios = decomposition_result.pca_ratios
    else:
        pca_ratios = None

    if return_raw:
        # Expand dimensions to be of the correct form for pytorch processing...
        X = np.expand_dims(X, axis=1)

    return PreprocessedDSpritesDataset(
        torch.tensor(X.astype(np.float32)), torch.tensor(y.astype(np.float32)), factor2idx, pca_ratios=pca_ratios
    )


if __name__ == "__main__":
    metadata = load_dsprites().metadata
    print(metadata)
