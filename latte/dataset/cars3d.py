""" Script for downloading and loading the Cars3D dataset from http://www.scottreed.info/.
    Some parts based on disentanglement_lib (https://github.com/google-research/disentanglement_lib). """

import dataclasses
import pathlib
import os
import tarfile
import tempfile

from typing import Union

from PIL import Image
import h5py
import numpy as np
import scipy.io as sio

from latte.dataset import utils as dsutils


DEFAULT_TARGET = "dataset/raw/cars3d.h5"
CARS3D_URL = "http://www.scottreed.info/files/nips2015-analogy-data.tar.gz"
MD5_CHECKSUM = "4e866a7919c1beedf53964e6f7a23686"


@dataclasses.dataclass
class Cars3DDataset:
    """The dataclass for the Cars3D data.
    It stores the images of the cars and their true factors of variation: the elevation, azimuth, and the model
    Members:
        imgs: ndarray of shape (17664, 64, 64, 3). The original images are of shape (128, 128, 3),
              but they are reshaped for memory efficiency.
        latent values: ndarray of shape (17664, 3)
    """

    imgs: np.ndarray
    latents_values: np.ndarray


def prepare_files(filepath: pathlib.Path) -> None:
    """Function extracting and preparing the Cars3D data.

    Args:
        filepath: the filepath of where to store the dataset

    Returns:
        Cars3DDataset object
    """

    archive_file = tempfile.NamedTemporaryFile()
    extract_dir = tempfile.TemporaryDirectory()

    dsutils.download(target=archive_file.name, _source=CARS3D_URL, _checksum=MD5_CHECKSUM)

    with tarfile.open(archive_file.name) as af:
        subdir_and_files = [tarinfo for tarinfo in af.getmembers() if tarinfo.name.startswith("data/cars/")]
        af.extractall(members=subdir_and_files, path=extract_dir.name)

    images = np.zeros((184 * 4 * 24, 64, 64, 3))
    latents = np.zeros((184 * 4 * 24, 3))

    elevation, azimuth = np.linspace(0, 45, 4) / 45, np.linspace(0, 345, 24) / 345

    for car_idx, car_name in enumerate(os.listdir(os.path.join(extract_dir.name, "data", "cars"))):

        if (
            not os.path.isfile((os.path.join(extract_dir.name, "data", "cars", str(car_name))))
            or not car_name[-4:] == ".mat"
        ):
            continue
        car_images = _load_mesh((os.path.join(extract_dir.name, "data", "cars", str(car_name))))
        all_factors = np.transpose(
            [
                np.tile(elevation, 24),  # vertical rotation
                np.repeat(azimuth, 4),  # horizontal rotation
                np.tile(car_idx, 4 * 24),  # car color/model
            ]
        )

        latents[car_idx * 4 * 24 : (car_idx + 1) * 4 * 24, :] = all_factors
        images[car_idx * 4 * 24 : (car_idx + 1) * 4 * 24, :, :, :] = car_images

    filepath.parent.mkdir(parents=True, exist_ok=True)

    car3d_data = h5py.File(filepath, "w")
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

    # If the file doesn't exist and we have permission to download,
    # we will download it
    if (not filepath.exists()) and download_ok:
        prepare_files(filepath)

    data = h5py.File(filepath, "r")

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
