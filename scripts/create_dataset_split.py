import dataclasses
from os import path

import numpy as np

from latte.dataset import shapes3d, utils as dsutils
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


def get_dataset(dataset: str, path: str) -> np.ndarray:
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
        ...
    elif dataset == "shapes3d":
        dataset = shapes3d.load_shapes3d(filepath=path)
        X = dataset.imgs[:]
        del dataset
        X = np.transpose(X, (0, 3, 1, 2))

    print("Dataset loaded.")
    return X


@hy.main
def main(cfg: DatasetSplitConfig):
    X = get_dataset(cfg.dataset, cfg.dataset_path)
    print("Splitting dataset.")
    X_train, X_val, X_test = dsutils.split(D=[X], p_train=cfg.p_train, p_val=cfg.p_val, seed=cfg.seed)[0]
    print("Dataset split.")
    del X  # Save memory
    print("X deleted.")

    print("Saving...")
    for X, split in zip([X_train, X_val, X_test], ["train", "val", "test"]):
        np.save(path.join(cfg.save_dir, f"{cfg.dataset}_{split}_split"), X)

    print("DONE.")


if __name__ == "__main__":
    main()
