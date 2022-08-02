"""
A simple script to convert the entire `CelebA` dataset to a numpy array for easier handling by the `pythae` library.
"""

import pathlib

import torchvision
import numpy as np

from latte.dataset import celeba


def main():
    _ROOT = pathlib.Path("/cluster/work/boeva/asvete/dataset/raw")

    for split in ["train", "valid", "test"]:
        dataset = torchvision.datasets.CelebA(_ROOT, split=split, download=False)
        X = celeba.to_numpy(dataset=dataset, N=len(dataset), black_white=False, w=64, h=64)

        with open(_ROOT / f"celeba_images_{split}_color.npy", "wb") as np_file:
            np.save(np_file, X)

        with open(_ROOT / f"celeba_attrs_{split}_color.npy", "wb") as np_file:
            np.save(np_file, dataset.attr.numpy())


if __name__ == "__main__":
    main()
