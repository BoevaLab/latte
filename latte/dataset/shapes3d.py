import dataclasses
import pathlib

from typing import Union

import numpy as np
import h5py

from latte.dataset import utils as dsutils


DEFAULT_TARGET = "dataset/raw/shapes3d.h5"
SHAPES3D_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
MD5_CHECKSUM = "099a2078d58cec4daad0702c55d06868"


@dataclasses.dataclass
class Shapes3DDataset:
    imgs: np.ndarray
    latents_values: np.ndarray


def load_shapes3d(filepath: Union[str, pathlib.Path] = DEFAULT_TARGET, download_ok: bool = True) -> Shapes3DDataset:
    """Function loading the Shapes3D dataset.

    Args:
        filepath: where the dataset should be loaded from
        download_ok: if the dataset should be downloaded (if it doesn't exist)

    Returns:
        Shapes3DDataset object
    """
    filepath = pathlib.Path(filepath)

    # If the file doesn't exist and we have permission to download,
    # we will download it
    if (not filepath.exists()) and download_ok:
        dsutils.download(target=filepath, _source=SHAPES3D_URL, _checksum=MD5_CHECKSUM)

    # Load the dataset to the memory
    data = h5py.File(filepath, "r")

    # Now we can open the dataset and reshape it
    return Shapes3DDataset(
        imgs=data["images"],
        latents_values=data["labels"],
    )


if __name__ == "__main__":
    load_shapes3d()
    print("DONE")
