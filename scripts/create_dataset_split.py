from typing import Tuple
import dataclasses
from os import path

import numpy as np

from latte.dataset import shapes3d, dsprites, morphoMNIST, utils as dsutils
from latte import hydra_utils as hy


@hy.config
@dataclasses.dataclass
class DatasetSplitConfig:
    dataset: str
    dataset_path: str
    save_dir: str
    seed: int = 1
    p_train: float = 0.7
    p_val: float = 0.1


def _preprocess_morphomnist_data(data: morphoMNIST.MorphoMNISTDataset) -> Tuple[np.ndarray, np.ndarray]:

    X = np.vstack((data.train_imgs, data.test_imgs))
    Y_labels = np.hstack((data.train_labels, data.test_labels))
    Y_perturbations = (
        np.hstack((data.train_perturbations, data.test_perturbations)) if data.train_perturbations is not None else None
    )
    Y_latents_values = np.vstack((data.train_latents_values, data.test_latents_values))
    if Y_perturbations is not None:
        Y = np.hstack((Y_labels.reshape(-1, 1), Y_latents_values, Y_perturbations.reshape(-1, 1)))
    else:
        Y = np.hstack((Y_labels.reshape(-1, 1), Y_latents_values))

    return X, Y


def get_dataset(dataset: str, path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the data (the observation only) for the specified dataset.
    Args:
        dataset: The dataset to load.
        path: The path to the data file.
    Returns:
        The data in the form of a Tensor.
    """
    print("Loading the dataset.")
    if dataset == "dsprites":
        data = dsprites.load_dsprites(filepath=path)
        X = data.imgs[:]
        Y = data.latents_values[:]
    elif dataset == "shapes3d":
        data = shapes3d.load_shapes3d(filepath=path)
        X = data.imgs[:]
        Y = data.latents_values[:]
    elif dataset == "morphomnist":
        data = morphoMNIST.load_morphoMNIST(version="global", filepath=path)
        X, Y = _preprocess_morphomnist_data(data)
    else:
        raise NotImplementedError

    del data

    if dataset == "dsprites":
        X = np.expand_dims(X, 1)
    elif dataset == "shapes3d":
        X = np.transpose(X, (0, 3, 1, 2))

    print("Dataset loaded.")
    return X, Y


@hy.main
def main(cfg: DatasetSplitConfig):
    X, Y = get_dataset(cfg.dataset, cfg.dataset_path)
    print("Splitting dataset.")
    (X_train, X_val, X_test), (Y_train, Y_val, Y_test) = dsutils.split(
        D=[X, Y], p_train=cfg.p_train, p_val=cfg.p_val, seed=cfg.seed
    )
    print("Dataset split.")
    del X  # Save memory
    print("X deleted.")

    print("Saving...")
    for X, Y, split in zip([X_train, X_val, X_test], [Y_train, Y_val, Y_test], ["train", "val", "test"]):
        np.save(path.join(cfg.save_dir, f"{cfg.dataset}_images_{split}_split"), X)
        np.save(path.join(cfg.save_dir, f"{cfg.dataset}_attributes_{split}_split"), Y)

    print("DONE.")


if __name__ == "__main__":
    main()
