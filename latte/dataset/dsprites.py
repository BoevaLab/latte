import dataclasses
import pathlib

from typing import Tuple, Union

import numpy as np

import latte.dataset.utils as dsutils


DEFAULT_TARGET = "dataset/raw/dsprites.npz"
DSPRITES_URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
MD5_CHECKSUM = "7da33b31b13a06f4b04a70402ce90c2e"


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


if __name__ == "__main__":
    metadata = load_dsprites().metadata
    print(metadata)
