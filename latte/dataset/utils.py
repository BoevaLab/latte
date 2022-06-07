import pathlib
import hashlib
import requests
import gzip
import struct

import numpy as np

# TODO(Pawel): When pytype starts supporting Literal, remove the comment
from typing import Literal, Union  # pytype: disable=not-supported-yet


# Seed (or generator) we can use to initialize a random number generator
RandomGenerator = Union[int, np.random.Generator]


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.

    Params:
        path: Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns:
        Output array of dtype ``uint8``.
    """
    open_fcn = gzip.open if path.endswith(".gz") else open
    with open_fcn(path, "rb") as f:
        return _load_uint8(f)


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


def download(target: Union[str, pathlib.Path], _source: str, _checksum: str) -> None:
    """Downloads a dataset.

    Params:
        target: the target location of the dataset
        _source: the URL to download the dataset from
        _checksum: the MD5 checksum of the file to check for correctness
    """

    target = pathlib.Path(target)

    # If the file already exists, we do nothing
    if _check_file_ready(file=target, md5_checksum=_checksum, open_mode="rb"):
        return

    # Otherwise, download the file
    req = requests.get(_source)
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
