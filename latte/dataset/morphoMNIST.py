""" Script for downloading and loading the `Cars3D` dataset.
    Some parts based on the morpho-MNIST package (https://github.com/dccastro/Morpho-MNIST). """

import dataclasses
import os
import pathlib
import zipfile
import tempfile

# TODO(Pawel): When pytype starts supporting Literal, remove the comment
from typing import Union, Optional, Literal  # pytype: disable=not-supported-yet

import numpy as np
import pandas as pd
import h5py

import latte.dataset.utils as dsutils


DEFAULT_TARGET = "dataset/raw/"

TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TRAIN_LATENTS = "train-morpho.csv"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"
TEST_LATENTS = "t10k-morpho.csv"
TRAIN_PERTURBATIONS = "train-pert-idx1-ubyte.gz"
TEST_PERTURBATIONS = "t10k-pert-idx1-ubyte.gz"

MORPHOMNIST_URL = {
    "plain": "https://drive.google.com/uc?export=download&id=1-E3sbKtzN8NGNefUdky2NVniW1fAa5ZG",
    "local": "https://drive.google.com/uc?export=download&id=1ECYmtpPvGH0AkK0JfrGfA2FpOCZK1VX2",
    "global": "https://drive.google.com/uc?export=download&id=1fFGJW0IHoBmLuD6CEKCB8jz3Y5LJ5Duk",
}
MD5_CHECKSUM = {
    "plain": "5864359c29f8019ec2191714fde5a257",
    "local": "cce1064561c19bc8b05f6c2d9a1f962e",
    "global": "0584bd1e0e359b96c88d1b719a5959db",
}


@dataclasses.dataclass
class MorphoMNISTDataset:
    """Stores the MorphoMNIST data from https://github.com/dccastro/Morpho-MNIST.
    The dataset contains 28x28 black and white images from the MNIST dataset together with their true
    latent factors: the area, length, thickness, slant, width, height.
    The images can also be perturbed (in a global or local way); the labels of the perturbations are:
    0: plain; 1: thinned; 2: thickened; 3: swollen; 4: fractured
    Members:
        train_imgs: ndarray of shape (60000, 28, 28)
        train_labels: ndarray of shape (60000, )
        train_latents: ndarray of shape (60000, 7) with the first column storing the index
        train_perturbations: optional ndarray of shape (60000, )
        test_imgs: ndarray of shape (10000, 28, 28)
        test_labels: ndarray of shape (10000, )
        test_latents: ndarray of shape (10000, 7) with the first column storing the index
        test_perturbations: optional ndarray of shape (10000, )
    """

    train_imgs: np.ndarray
    train_latents_values: np.ndarray
    train_labels: np.ndarray
    train_perturbations: Optional[np.ndarray]

    test_imgs: np.ndarray
    test_latents_values: np.ndarray
    test_labels: np.ndarray
    test_perturbations: Optional[np.ndarray]


def save_to_file(
    version: Literal["plain", "local", "global"],
    split: Literal["train", "test"],
    filepath: pathlib.Path,
    images: np.ndarray,
    labels: np.ndarray,
    latents: np.ndarray,
    perturbations: Optional[np.ndarray],
) -> None:
    """A convenience function to save the data into a h5 file.

    Params:
        version: the version of the data (plain or local/global perturbations)
        split: split of the dataset to save train or test
        filepath: directory to save to
        images: the dataset images
        labels: the dataset labels (digits)
        latents: the dataset latents
        perturbations: the perturbations applied when version != plain
    """

    data = h5py.File(filepath / f"morphoMNIST_{version}_{split}.h5", "w")
    data.create_dataset("images", data=images)
    data.create_dataset("labels", data=labels)
    data.create_dataset("latents", data=latents)
    if version in ["local", "global"]:
        data.create_dataset("perturbations", data=perturbations)

    data.close()


def prepare_files(
    filepath: pathlib.Path, version: Literal["plain", "local", "global"], _source: str, _md5_checksum: str
) -> None:
    """Function extracting and preparing the MorphoMNIST data.

    Args:
        filepath: directory where the dataset should be stored
        version: the version of the data to load (plain or local/global perturbations)
        _source: the URL of the data to be downloaded
        _md5_checksum: the MD5 checksum of the file to be downloaded

    Returns:
        MorphoMNISTDataset object
            In case that the version is not "plain", it also includes the information about the perturbations performed,
            which is not there for the plain (non-perturbed) data.
    """

    archive_file = tempfile.NamedTemporaryFile()
    extract_dir = tempfile.TemporaryDirectory()

    dsutils.download(target=archive_file.name, _source=_source, _checksum=_md5_checksum)

    with zipfile.ZipFile(archive_file, "r") as zf:
        zf.extractall(extract_dir.name)

    def right_path(_version: str, filename: str) -> str:
        return os.path.join(extract_dir.name, _version, filename)

    train_images = dsutils.load_idx(right_path(version, TRAIN_IMAGES))
    train_labels = dsutils.load_idx(right_path(version, TRAIN_LABELS))
    train_latents = pd.read_csv(right_path(version, TRAIN_LATENTS))

    test_images = dsutils.load_idx(right_path(version, TEST_IMAGES))
    test_labels = dsutils.load_idx(right_path(version, TEST_LABELS))
    test_latents = pd.read_csv(right_path(version, TEST_LATENTS))

    train_perturbations, test_perturbations = None, None
    # if the data is perturbed (it is not "plain"), we also collect the perturbation information,
    # which does not exist for the plain data
    if version in ["local", "global"]:
        train_perturbations = dsutils.load_idx(right_path(version, TRAIN_PERTURBATIONS))
        test_perturbations = dsutils.load_idx(right_path(version, TEST_PERTURBATIONS))

    save_to_file(version, "train", filepath, train_images, train_labels, train_latents, train_perturbations)
    save_to_file(version, "test", filepath, test_images, test_labels, test_latents, test_perturbations)


def load_morphoMNIST(
    version: Literal["plain", "local", "global"] = "plain",
    filepath: Union[str, pathlib.Path] = DEFAULT_TARGET,
    download_ok: bool = True,
) -> MorphoMNISTDataset:
    """Function loading the MorphoMNIST dataset.

    Args:
        version: the version of the data to load (plain or local/global perturbations)
        filepath: where the initial archive should be stored
        download_ok: if the dataset should be downloaded (if it doesn't exist)

    Returns:
        MorphoMNISTDataset object
    """

    filepath = pathlib.Path(filepath)
    final_filepath_train, final_filepath_test = (
        filepath / f"morphoMNIST_{version}_train.h5",
        filepath / f"morphoMNIST_{version}_test.h5",
    )

    # If the file doesn't exist and we have permission to download,
    # we will download it
    if (not final_filepath_train.exists() or not final_filepath_test.exists()) and download_ok:
        prepare_files(
            filepath,
            _source=MORPHOMNIST_URL[version],
            _md5_checksum=MD5_CHECKSUM[version],
            version=version,
        )

    # Load the dataset to the memory
    train_data = h5py.File(filepath / f"morphoMNIST_{version}_train.h5", "r")
    test_data = h5py.File(filepath / f"morphoMNIST_{version}_test.h5", "r")

    # Now we can open the dataset and reshape it
    return MorphoMNISTDataset(
        train_imgs=np.expand_dims(train_data["images"][:], axis=1),  # Expand the dimensions to have a channel dimension
        train_labels=train_data["labels"][:],
        train_latents_values=train_data["latents"][:],
        train_perturbations=train_data["perturbations"][:] if version in ["local", "global"] else None,
        test_imgs=np.expand_dims(test_data["images"][:], axis=1),
        test_labels=test_data["labels"][:],
        test_latents_values=test_data["latents"][:],
        test_perturbations=test_data["perturbations"][:] if version in ["local", "global"] else None,
    )


if __name__ == "__main__":
    for dataset_version in ["plain", "local", "global"]:
        load_morphoMNIST(version=dataset_version)
    print("DONE")
