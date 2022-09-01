import pathlib
import hashlib
import requests
import gzip
import struct

import torch
import numpy as np

# TODO(Pawel): When pytype starts supporting Literal, remove the comment
from typing import List, Literal, Union, Tuple, Any  # pytype: disable=not-supported-yet


# Seed (or generator) we can use to initialize a random number generator
RandomGenerator = Union[int, np.random.Generator]
TensorOrNDArray = Union[torch.Tensor, np.ndarray]


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
    print(f"Downloading the file from {_source} into {target}.")
    req = requests.get(_source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(req.content)
    print("Download complete.")

    # Check if the file is ready now. Otherwise, raise an exception
    if not _check_file_ready(target, md5_checksum=_checksum, open_mode="rb"):
        raise ValueError(
            "The downloaded file is corrupted. Download it again. "
            "If it doesn't help, check whether the MD5 checksum are right. "
            "Note that opening a corrupted file may be a threat to your computer, due to pickle."
        )


def split(
    D: List[TensorOrNDArray],
    p_train: float = 0.7,
    p_val: float = 0.1,
    seed: RandomGenerator = 1,
) -> List[Union[Tuple[TensorOrNDArray, TensorOrNDArray, TensorOrNDArray], Tuple[TensorOrNDArray, TensorOrNDArray]]]:
    """
    Split the list of datasets into train, validation, and test sets.
    `D` should contain a list of datasets of the same length to be split.
    Args:
        D: A list of datasets to split.
           This can for example be a list of two elements [X, Y], the observations and the targets, but it can also
           contain more datasets.
        p_train: The fraction of the samples to be in the training split.
        p_val: The fraction of the samples to be in the validation or train split.
               If p_train + p_val < 1, the datasets will be split into three splits; the train, validation, and the test
               splits, with the last containing 1 - p_train - p_val fraction of the dataset.
               Otherwise, if p_train + p_val = 1, the datasets are split into two splits; the train and the test one.
        seed: The randomness generator.

    Returns:
        A list of tuples.
        The list is of the same length as `D`, and each element is itself a tuple of two or three datasets
    """
    assert len(np.unique([len(x) for x in D])) == 1
    assert p_train + p_val <= 1

    rng = np.random.default_rng(seed)
    N = len(D[0])
    permutation = rng.permutation(N)
    end_train, end_val = int(p_train * N), int((p_train + p_val) * N)

    if end_val < N:
        D_split = [(x[permutation][:end_train], x[permutation][end_train:end_val], x[permutation][end_val:]) for x in D]
    else:
        D_split = [(x[permutation][:end_train], x[permutation][end_train:]) for x in D]

    return D_split


def load_split_data(
    file_paths_x: Tuple[str, str, str],
    file_paths_y: Tuple[str, str, str],
    dataset: str,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load the data of the specified dataset split into an existing train-validation-test split.
    Args:
        file_paths_x: Tuple of the paths to the train, validation, and test splits of the observational files in this
                      order.
        file_paths_y: Tuple of the paths to the train, validation, and test splits of the target files in this order.
        dataset: The dataset to load.
                 Currently, "shapes3d" and "celeba" are supported.

    Returns:
        Two tuples; the first one is the observation data split into the train, validation, and test splits, and the
        second one is the latent-factor data split into the same splits.
    """
    print("Loading the data.")

    X_train = torch.from_numpy(np.load(file_paths_x[0])).float()
    Y_train = torch.from_numpy(np.load(file_paths_y[0])).float()
    X_val = torch.from_numpy(np.load(file_paths_x[1])).float()
    Y_val = torch.from_numpy(np.load(file_paths_y[1])).float()
    X_test = torch.from_numpy(np.load(file_paths_x[2])).float()
    Y_test = torch.from_numpy(np.load(file_paths_y[2])).float()

    if dataset in ["celeba", "shapes3d"]:
        X_train /= 255
        X_val /= 255
        X_test /= 255

    print("Data loaded.")
    return (X_train, X_val, X_test), (Y_train, Y_val, Y_test)


def get_dataset_module(dataset: str) -> Any:
    from latte.dataset import shapes3d, celeba

    """Returns the module to use for various utilities according to the dataset."""
    if dataset == "celeba":
        return celeba
    elif dataset == "shapes3d":
        return shapes3d
