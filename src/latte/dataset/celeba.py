"""
Utility functions and attributes for working with the `CelebA` dataset
(https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
"""

import torchvision
import numpy as np
from skimage.color import rgb2gray
from tqdm import tqdm

from latte.dataset.utils import RandomGenerator


# Original dimensions of the images
_w, _h = 218, 178


attribute_names = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


attribute2idx = {name: jj for jj, name in enumerate(attribute_names)}


def to_numpy(
    dataset: torchvision.datasets.CelebA,
    N: int,
    black_white: bool = False,
    w: int = _w,
    h: int = _h,
    rng: RandomGenerator = 42,
) -> np.ndarray:
    """Converts a portion of the images in `dataset` to a numpy array needed for benchmark_VAE.

    Args:
        dataset (torchvision.Dataset): The data loaded by `torchvision`.
        N (int): The number of images to return.
        black_white (bool, optional): Whether to tranform to black and white. Defaults to False.
        w (int): Optional target width. Defaults to 218.
        h (int): Optional target height. Defaults to 178.
        rng (RandomGenerator): A random generator

    Returns:
        np.ndarray: `N` images transformed into numpy arrays, either in RGB or black and white.
    """

    rng = np.random.default_rng(rng)
    ixs = rng.choice(len(dataset), size=N, replace=False) if N < len(dataset) else range(N)
    if black_white:
        images = np.zeros(shape=(N, h, w), dtype=np.float32)
    else:
        images = np.zeros(shape=(N, h, w, 3), dtype=np.float32)

    for i, ix in enumerate(tqdm(ixs)):
        images[i, ...] = np.asarray(
            rgb2gray(dataset[ix][0].resize((w, h))) if black_white else dataset[ix][0].resize((w, h))
        )

    if black_white:
        return np.expand_dims(images, axis=1)
    else:
        return np.transpose(images, axes=(0, 3, 1, 2))
