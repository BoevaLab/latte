import dataclasses
import hashlib
import pathlib
import requests

# TODO(Pawel): When pytype starts supporting Literal, remove the comment
from typing import Literal, Tuple, Union  # pytype: disable=not-supported-yet

import numpy as np


DEFAULT_TARGET = "dataset/raw/dsprites.npz"
DSPRITES_URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
MD5_CHECKSUM = "7da33b31b13a06f4b04a70402ce90c2e"


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


if __name__ == "__main__":
    metadata = load_dsprites().metadata
    print(metadata)
