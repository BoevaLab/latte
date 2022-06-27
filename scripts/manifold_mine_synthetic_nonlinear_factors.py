"""
This script generates a random dataset of n points in a d-dimensional space, projects them onto a random k-dimensional
linear subspace, and produces a set of factors which depend only on the k dimensions of the subspace.

Since the factors depend only on the k dimensions, they capture all the information about the factors.
The goal is to find this linear subspace by searching for the k-dimensional linear subspace which captures all
(or, the most) information about the factors.

The script trains a ManifoldMINE model to find the subspace which the factors depend on by optimising over the model
parameters as well as the projection matrix onto the linear subspace.

The data can optionally be noisy.

It produces the evaluation of the fit, and if the data is 2- or 3-dimensional,
it also produces plots of the fit.
"""
import dataclasses

from scipy import special

import torch
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from latte.dataset import synthetic
from latte.mine import estimation
from latte.utils import evaluation, visualisation
from latte.manifolds import utils as mutils
import latte.hydra_utils as hy

import logging
import coloredlogs

coloredlogs.install()


_DEFAULT_TARGET = "figures/manifold_mine_nonlinear_factors"


def _generate_data(n: int, d: int, k: int, noise: float, seed: int) -> synthetic.SyntheticDataset:

    fs = [lambda x: special.logsumexp([x[:, j] for j in range(x.shape[1])], axis=0)]

    # Produce the synthetic dataset
    dataset = synthetic.synthetic_dataset(n, d, k=k, sigma=noise, fs=fs, rng=seed)  # len(fs) = m

    # Scale the data
    dataset.X = StandardScaler().fit_transform(dataset.X)
    dataset.Z = StandardScaler().fit_transform(dataset.Z)
    dataset.Y = StandardScaler().fit_transform(dataset.Y)

    return dataset


@hy.config
@dataclasses.dataclass
class SyntheticDataManifoldConfig:
    """
    The configuration of the synthetic data with subspace estimation experiment.

    Attributes:
        n: The number of points to generate.
        d: the dimensionality of the observed space.
        subspace_size: The dimensionality of the subspace ("k" above)
        assumed_subspace_size: The size of the subspace to be fit.
                               Since we do not know this in practice, it is interesting
                               to consider what happens when the model is misspecified
                               (i.e., assumed_subspace_size != subspace_size)
        noise: The scale of the gaussian noise to add to the observations
        network_width: The width of the MINE statistics network
        max_epochs: The number of epochs to train the MINE model for
        seed: The seed for the random generators
    """

    n: int = 128
    d: int = 3
    subspace_size: int = 2
    assumed_subspace_size: int = 2
    noise: float = 0.1
    network_width: int = 128
    max_epochs: int = 128
    seed: int = 42
    gpus: int = 1
    num_workers: int = 12


@hy.main
def main(cfg: SyntheticDataManifoldConfig):

    pl.seed_everything(cfg.seed)

    assert cfg.subspace_size <= cfg.d, "The subspace size should be at most the size of the full space."

    dataset = _generate_data(cfg.n, cfg.d, cfg.subspace_size, cfg.noise, cfg.seed)

    number_of_factors = dataset.Z.shape[1]

    if dataset.Z.shape[-1] == 1:
        h = mutual_info_regression(dataset.Z, dataset.Z.flatten())[0]
        logging.info(f"The entropy the factor of variation is {h: .3f}.")

    mi_estimation_result = estimation.estimate_mutual_information(
        subspace_estimation=True,
        X=torch.from_numpy(dataset.X),
        Z=torch.from_numpy(dataset.Z),
        data_presplit=False,
        datamodule_kwargs=dict(num_workers=12),
        model_kwargs=dict(
            x_size=cfg.d,
            z_size=number_of_factors,
            network_width=cfg.network_width,
            subspace_size=cfg.assumed_subspace_size,
        ),
        trainer_kwargs=dict(
            max_epochs=cfg.max_epochs, gpus=cfg.gpus, enable_progress_bar=False, enable_model_summary=False
        ),
    )

    subspace_mi = mi_estimation_result.mutual_information
    logging.info(
        f"The captured mutual information between the subspace and the factor of variation is {subspace_mi: .3f}."
    )

    A_hat = mi_estimation_result.subspace
    assert mutils.is_orthonormal(A_hat)
    A_hat = A_hat.numpy()

    evaluation_result = evaluation.evaluate_found_subspace(dataset.A, A_hat)
    logging.info("--- Quality of the fit. ---")
    print(evaluation_result)

    # Plot the fit
    if cfg.d in [2, 3]:
        file_name = f"nonlinear_synthetic_dataset_{cfg.d}d_{cfg.subspace_size}d.png"
        visualisation.visualise_synthetic_data(dataset.X, dataset.A, A_hat, file_name=file_name)

        # If the factor of variation is 1-dimensional, also visualise it by the color of the observable points
        if dataset.Z.shape[-1] == 1:

            file_name = f"nonlinear_synthetic_dataset_{cfg.d}d_{cfg.subspace_size}d_factors.png"
            visualisation.visualise_synthetic_data(
                dataset.X,
                dataset.A,
                A_hat,
                f=dataset.Z,
                file_name=file_name,
                alphas={"observable": 0.5, "true_projections": 0.2, "estimated_projections": 0.3},
            )


if __name__ == "__main__":
    main()
