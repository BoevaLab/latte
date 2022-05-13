""" Script for downloading and loading the Cars3D dataset.
    Some parts based on disentanglement_lib (https://github.com/google-research/disentanglement_lib). """

import dataclasses
import pathlib
import os
import tarfile
import shutil

from typing import Union

from PIL import Image
import h5py
import numpy as np
import scipy.io as sio

from latte.dataset import utils as dsutils


DEFAULT_TARGET = "dataset/raw"
CARS3D_URL = "http://www.scottreed.info/files/nips2015-analogy-data.tar.gz"
MD5_CHECKSUM = "4e866a7919c1beedf53964e6f7a23686"


@dataclasses.dataclass
class Cars3DDataset:
    imgs: np.ndarray
    latents_values: np.ndarray


def prepare_files(filepath: pathlib.Path) -> None:
    """Function extracting and preparing the Cars3D data.

    Args:
        filepath: directory for storing the temporary data and the dataset

    Returns:
        Cars3DDataset object
    """

    archive_filepath = filepath / "cars3d.tar.gz"
    extract_filepath = filepath / "cars3d_data"
    final_filepath = filepath / "cars3d.h5"

    dsutils.download(target=archive_filepath, _source=CARS3D_URL, _checksum=MD5_CHECKSUM)

    extract_filepath.parent.mkdir(parents=True, exist_ok=True)
    final_filepath.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_filepath) as af:
        subdir_and_files = [tarinfo for tarinfo in af.getmembers() if tarinfo.name.startswith("data/cars/")]
        af.extractall(members=subdir_and_files, path=extract_filepath)

    # remove the original archive
    os.remove(archive_filepath)

    images = np.zeros((184 * 4 * 24, 64, 64, 3))
    latents = np.zeros((184 * 4 * 24, 3))

    elevation, azimuth = np.linspace(0, 45, 4) / 45, np.linspace(0, 345, 24) / 345

    for car_idx, car_name in enumerate(os.listdir(extract_filepath / os.path.join("data", "cars"))):

        if (
            not os.path.isfile(extract_filepath / (os.path.join("data", "cars", str(car_name))))
            or not car_name[-4:] == ".mat"
        ):
            continue
        car_images = _load_mesh(extract_filepath / (os.path.join("data", "cars", str(car_name))))
        all_factors = np.transpose(
            [
                np.tile(elevation, 24),  # vertical rotation
                np.repeat(azimuth, 4),  # horizontal rotation
                np.tile(car_idx, 4 * 24),  # car color/model
            ]
        )

        latents[car_idx * 4 * 24 : (car_idx + 1) * 4 * 24, :] = all_factors
        images[car_idx * 4 * 24 : (car_idx + 1) * 4 * 24, :, :, :] = car_images

    # remove the extracted files
    shutil.rmtree(extract_filepath)

    car3d_data = h5py.File(final_filepath, "w")
    car3d_data.create_dataset("images", data=images)
    car3d_data.create_dataset("latents", data=latents)

    car3d_data.close()


def load_cars3d(
    filepath: Union[str, pathlib.Path] = DEFAULT_TARGET,
    download_ok: bool = True,
) -> Cars3DDataset:
    """Function loading the Cars3D dataset.

    Args:
        filepath: directory for storing the temporary data and the dataset
        download_ok: if the dataset should be downloaded (if it doesn't exist)

    Returns:
        Cars3DDataset object
    """
    filepath = pathlib.Path(filepath)
    final_filepath = filepath / "cars3d.h5"

    # If the file doesn't exist and we have permission to download,
    # we will download it
    if (not final_filepath.exists()) and download_ok:
        prepare_files(filepath)

    data = h5py.File(final_filepath, "r")

    # Now we can open the dataset and reshape it
    return Cars3DDataset(imgs=data["images"][:], latents_values=data["latents"][:])


def _load_mesh(filename):
    """Parses a single source file and rescales contained images.
    From disentanglement_lib; https://github.com/google-research/disentanglement_lib/
    """
    with open(filename, "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):

        image = Image.fromarray(flattened_mesh[i, :, :, :])
        image.thumbnail((64, 64), Image.ANTIALIAS)
        rescaled_mesh[i, :, :, :] = np.array(image)

    return rescaled_mesh / 255


if __name__ == "__main__":
    load_cars3d()
    print("DONE")
