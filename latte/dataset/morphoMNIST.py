""" Script for downloading and loading the Cars3D dataset.
    Some parts based on the morpho-MNIST package (https://github.com/dccastro/Morpho-MNIST). """

import dataclasses
import os
import pathlib
import shutil
import zipfile

# TODO(Pawel): When pytype starts supporting Literal, remove the comment
from typing import Union, Optional, Literal  # pytype: disable=not-supported-yet

import numpy as np
import pandas as pd
import h5py

import latte.dataset.utils as dsutils


DEFAULT_TARGET = "dataset/raw/"

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
    """
    archive_filepath = filepath / f"{version}_archive.zip"
    extract_filepath = filepath

    dsutils.download(target=archive_filepath, _source=_source, _checksum=_md5_checksum)

    extract_filepath.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_filepath, "r") as zf:
        zf.extractall(extract_filepath)

    os.remove(archive_filepath)

    train_images = dsutils.load_idx(str(extract_filepath / os.path.join(version, "train-images-idx3-ubyte.gz")))
    train_labels = dsutils.load_idx(str(extract_filepath / os.path.join(version, "train-labels-idx1-ubyte.gz")))
    train_latents = pd.read_csv(str(extract_filepath / os.path.join(version, "train-morpho.csv")))

    test_images = dsutils.load_idx(str(extract_filepath / os.path.join(version, "t10k-images-idx3-ubyte.gz")))
    test_labels = dsutils.load_idx(str(extract_filepath / os.path.join(version, "t10k-labels-idx1-ubyte.gz")))
    test_latents = pd.read_csv(str(extract_filepath / os.path.join(version, "t10k-morpho.csv")))

    train_perturbations, test_perturbations = None, None
    if version in ["local", "global"]:
        train_perturbations = dsutils.load_idx(
            str(extract_filepath / os.path.join(version, "train-pert-idx1-ubyte.gz"))
        )
        test_perturbations = dsutils.load_idx(str(extract_filepath / os.path.join(version, "t10k-pert-idx1-ubyte.gz")))

    shutil.rmtree(extract_filepath / version)

    save_to_file(version, "train", filepath, train_images, train_labels, train_latents, train_perturbations)
    save_to_file(version, "test", filepath, test_images, test_labels, test_latents, test_perturbations)


def load_morphomnist(
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
        train_imgs=train_data["images"][:],
        train_labels=train_data["labels"][:],
        train_latents_values=train_data["latents"][:],
        train_perturbations=train_data["perturbations"][:] if version in ["local", "global"] else None,
        test_imgs=test_data["images"][:],
        test_labels=test_data["labels"][:],
        test_latents_values=test_data["latents"][:],
        test_perturbations=test_data["perturbations"][:] if version in ["local", "global"] else None,
    )


if __name__ == "__main__":
    for version in ["plain", "local", "global"]:
        load_morphomnist(version=version)
    print("DONE")
