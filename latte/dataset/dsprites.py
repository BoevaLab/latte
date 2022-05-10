import dataclasses
import hashlib
import pathlib
from enum import IntEnum

import requests

# TODO(Pawel): When pytype starts supporting Literal, remove the comment
from typing import Literal, Tuple, Union, List, Dict, Optional  # pytype: disable=not-supported-yet

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis


DEFAULT_TARGET = "dataset/raw/dsprites.npz"
DSPRITES_URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
MD5_CHECKSUM = "7da33b31b13a06f4b04a70402ce90c2e"


class DSpritesFactor(IntEnum):
    COLOR = 0
    SHAPE = 1
    SCALE = 2
    ORIENTATION = 3
    X = 4
    Y = 5


factor_names = {
    DSpritesFactor.SHAPE: "Shape",
    DSpritesFactor.SCALE: "Scale",
    DSpritesFactor.ORIENTATION: "Orientation",
    DSpritesFactor.X: "Position x",
    DSpritesFactor.Y: "Position y",
}


fixed_values = {
    DSpritesFactor.SHAPE: 1,
    DSpritesFactor.SCALE: 1,
    DSpritesFactor.ORIENTATION: 0,
    DSpritesFactor.X: 0.5,
    DSpritesFactor.Y: 0.5,
}


def _check_file_ready(file: pathlib.Path, md5_checksum: str, open_mode: Literal["r", "rb"]) -> bool:
    """Checks if the file is ready to use (exists and has the right MD5 checksum)

    Args:
        file: path to the file one wants to check
        md5_checksum: expected MD5 checksum
        open_mode: how the file should be opened

    Returns:
        true, if the file exists and has the right checksum
    """
    if not file.exists():
        return False

    with open(file, open_mode) as f:
        checksum = hashlib.md5(f.read()).hexdigest()

    return checksum == md5_checksum


def download(target: Union[str, pathlib.Path], _source: str = DSPRITES_URL, _checksum: str = MD5_CHECKSUM) -> None:
    target = pathlib.Path(target)

    # If the file already exists, we do nothing
    if _check_file_ready(file=target, md5_checksum=_checksum, open_mode="rb"):
        return

    # Otherwise, download the file
    req = requests.get(DSPRITES_URL)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(req.content)

    # Check if the file is ready now. Otherwise, raise an exception
    if not _check_file_ready(target, md5_checksum=_checksum, open_mode="rb"):
        raise ValueError(
            "The downloaded file is corrupted. Download it again. "
            "If it doesn't help, check whether the MD5 checksum are right. "
            "Note that opening a corrupted file may be a threat to your computer, due to pickle."
        )


@dataclasses.dataclass
class DSpritesMetadata:
    date: str
    description: str
    version: int
    latents_names: Tuple[str, ...]


@dataclasses.dataclass
class DSpritesDataset:
    metadata: DSpritesMetadata
    imgs: np.ndarray
    latents_values: np.ndarray
    latents_classes: np.ndarray


@dataclasses.dataclass
class PreprocessedDSpritesDataset:
    X: np.ndarray
    y: np.ndarray
    factor2idx: Dict[DSpritesFactor, int]
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
        download(target=filepath, _source=DSPRITES_URL, _checksum=MD5_CHECKSUM)

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
    data: DSpritesDataset, factors: List[DSpritesFactor], ignored_factors: List[DSpritesFactor]
) -> Tuple[np.ndarray, np.ndarray, Dict[DSpritesFactor, int]]:
    """Converts the raw dSprites data into Numpy arrays which will be
    processed by PCA.
    Args:
        data: DSpritesDataset object
        factors: Which factors to keep the information about in the returned data
        ignored_factors: Which factors should be held fixed (all the variation in them will be removed)

    Returns:
        Flattened filtered images and factors

    """
    # Flatten images
    X = data.imgs.reshape((len(data.imgs), -1))

    # Select the images of interest by ignoring the variation of factors
    # we are not interested in
    sample_mask = np.ones(data.latents_values.shape[0], dtype=bool)
    for factor in ignored_factors:
        sample_mask *= data.latents_values[:, factor] == fixed_values[DSpritesFactor(factor)]

    X = X[sample_mask, :]
    y = data.latents_values[sample_mask, :]

    # Filter out constant pixels
    X = X[:, X.max(axis=0) == 1]

    # Keep only the factors that we want
    y = y[:, factors]

    # The dictionary which maps each of the factors of interest
    # into the dimension of the returned factors y
    factor2idx = {factor: ii for ii, factor in enumerate(factors)}

    return X, y, factor2idx


def load_dsprites_preprocessed(
    filepath: Union[str, pathlib.Path] = DEFAULT_TARGET,
    download_ok: bool = True,
    method: str = "PCA",
    n_components: int = 20,
    factors: Optional[List[DSpritesFactor]] = None,
    ignored_factors: Optional[List[DSpritesFactor]] = None,
    return_ratios: bool = False,
) -> PreprocessedDSpritesDataset:
    """Load the dSprites dataset and preprocess it using PCA or FactorAnalysis.
    Args:
        filepath: The location of the dsprites data.
        download_ok: If the data can be downloaded.
        method: The way to preprocess the data, {PCA, FA]
        n_components: Number of component to produce in the representation.
        factors: Which factors to return.
        ignored_factors: Which factors should be kept constant in the dataset.
            The data is filtered and reduced such that these factors have a constant value.
        return_ratios: In case of PCA, whether to also return the ratios of variance explained.

    Returns:
        The filtered transformed data (coordinates in PCA or FA component space),
        the filtered latent factors, optionally PCA variance explained proportions.
        To be able to select appropriate factors from the returned ones, a dictionary
        mapping each of the returned factors to the dimension in `y` is also returned.
    """

    assert method in [
        "PCA",
        "FA",
        "FactorAnalysis",
    ], f"Method {method} is not supported, only PCA and FactorAnalysis supported."

    assert not return_ratios or method in ["PCA"], "Ratios can only be returned for PCA."

    if factors is None:
        # By default, return all factors
        factors = [
            DSpritesFactor.SHAPE,
            DSpritesFactor.SCALE,
            DSpritesFactor.ORIENTATION,
            DSpritesFactor.X,
            DSpritesFactor.Y,
        ]

    # Load raw data
    data = load_dsprites(filepath, download_ok)

    # Flatten and filter the data
    X, y, factor2idx = prepare_data(
        data, factors=factors, ignored_factors=ignored_factors if ignored_factors is not None else []
    )

    # Reduce the dimensionality
    if method == "PCA":
        trans = PCA(n_components=n_components)
    else:
        trans = FactorAnalysis(n_components=n_components)

    X = StandardScaler().fit_transform(X)
    X = trans.fit_transform(X)

    return PreprocessedDSpritesDataset(
        X, y, factor2idx, pca_ratios=trans.explained_variance_ratio_ if return_ratios else None
    )


# TODO (Anej)
def split(
    X: np.ndarray, y: np.ndarray, p_train: float = 0.7, p_val: float = 0.1
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split the data into train, validation, and train sets.
    Could be made deterministic for this specific dataset later.
    """
    n = len(X)
    permutation = np.random.permutation(n)
    end_train, end_val = int(p_train * n), int((p_train + p_val) * n)
    X_train, X_val, X_test = (
        X[permutation, :][:end_train],
        X[permutation, :][end_train:end_val],
        X[permutation, :][end_val:],
    )
    y_train, y_val, y_test = (
        y[permutation, :][:end_train],
        y[permutation, :][end_train:end_val],
        y[permutation, :][end_val:],
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    metadata = load_dsprites().metadata
    print(metadata)
