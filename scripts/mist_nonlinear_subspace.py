"""
This script generates a random dataset of N points in an n-dimensional space, projects them onto a random d-dimensional
linear subspace, and produces a set of factors which depend only on the d dimensions of the subspace.

Since the factors depend only on the d dimensions, these d dimensions capture all the information about the factors.
The goal is to find this linear subspace by searching for the d-dimensional linear subspace which captures all
(or, the most) information about the factors.

The script trains a MIST model to find the subspace which the factors depend on by optimising over the model
parameters as well as the projection matrix onto the linear subspace.

The data can optionally be noisy.

It produces the evaluation of the fit, and if the data is 2- or 3-dimensional, it also produces plots of the fit.
"""
import dataclasses

from scipy import special

import torch
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from latte.dataset import synthetic
from latte.models.mist import estimation
from latte.evaluation import subspace_evaluation
from latte.utils.visualisation import datasets as dataset_visualisation
import latte.hydra_utils as hy

import logging
import coloredlogs

coloredlogs.install()


_DEFAULT_TARGET = "figures/manifold_mine_nonlinear_factors"


@hy.config
@dataclasses.dataclass
class SyntheticDataManifoldConfig:
    """
    The configuration of the synthetic data with subspace estimation experiment.

    Members:
        N: The number of points to generate.
        n: the dimensionality of the observed space.
        subspace_size: The dimensionality of the subspace ("d" above)
        assumed_subspace_size: The size of the subspace to be fit.
                               Since we do not know this in practice, it is interesting
                               to consider what happens when the model is misspecified
                               (i.e., assumed_subspace_size != subspace_size)
        noise: The scale of the gaussian noise to add to the observations
        network_width: The width of the MINE statistics network
        max_epochs: The maximum number of epochs to train the MINE model for
        learning_rate: The learning rate for the MINE statistics network
        manifold_learning_rate: The learning rate for the projection matrix
        batch_size: The batch size for the training
        num_workers: The number of workers for the data module
        gpus: The gpus parameter for the Pytorch Lightning Trainer
        seed: The seed for the random generators
    """

    N: int = 128
    n: int = 3
    subspace_size: int = 2
    assumed_subspace_size: int = 2
    noise: float = 0.1
    network_width: int = 128
    max_epochs: int = 128
    learning_rate: float = 1e-3
    manifold_learning_rate: float = 1e-2
    batch_size: int = 128
    num_workers: int = 12
    gpus: int = 1
    seed: int = 42


def _generate_data(N: int, n: int, d: int, noise: float, seed: int) -> synthetic.NonlinearManifoldDataset:
    """
    Generates the data from a nonlinear manifold with the `synthetic.generate` function based on the functional
    relationships provided by the `logsumexp` function.
    Returns:
        The generated dataset.
    """

    fs = [lambda x: special.logsumexp([x[:, j] for j in range(x.shape[1])], axis=0)]

    # Produce the synthetic dataset
    dataset = synthetic.generate(N, n=n, d=d, sigma=noise, fs=fs, rng=seed)  # len(fs) = m

    # Scale the data
    dataset.X = StandardScaler().fit_transform(dataset.X)
    dataset.Z = StandardScaler().fit_transform(dataset.Z)
    dataset.Y = StandardScaler().fit_transform(dataset.Y)

    return dataset


@hy.main
def main(cfg: SyntheticDataManifoldConfig):

    pl.seed_everything(cfg.seed)

    assert cfg.subspace_size <= cfg.n, "The subspace size should be at most the size of the full space."

    # Generate the nonlinear manifold data
    dataset = _generate_data(cfg.N, cfg.n, cfg.subspace_size, cfg.noise, cfg.seed)

    # The number of factors is determined by the function passed to the data generation function
    number_of_factors = dataset.Z.shape[1]

    if dataset.Z.shape[-1] == 1:
        # For reference, calculate the entropy of the sample if the data is 1-dimensional
        h = mutual_info_regression(dataset.Z, dataset.Z.flatten())[0]
        logging.info(f"The entropy the factor of variation is {h: .3f}.")

    # Find the subspace with the SupervisedMIST model
    mi_estimation_result = estimation.fit(
        X=torch.from_numpy(dataset.X),
        Z_max=torch.from_numpy(dataset.Z),
        Z_min=None,
        datamodule_kwargs=dict(num_workers=cfg.num_workers, batch_size=cfg.batch_size),
        model_kwargs=dict(
            x_size=cfg.n,
            z_max_size=number_of_factors,
            subspace_size=cfg.assumed_subspace_size,
            z_min_size=None,  # No minimisation with regard to any factors
            mine_learning_rate=cfg.learning_rate,
            mine_network_width=cfg.network_width,
            manifold_learning_rate=cfg.manifold_learning_rate,
            club_network_width=None,
            gamma=1.0,
        ),
        trainer_kwargs=dict(
            max_epochs=cfg.max_epochs, gpus=cfg.gpus, enable_progress_bar=True, enable_model_summary=False
        ),
    )

    subspace_mi = mi_estimation_result.mutual_information
    logging.info(
        f"The captured mutual information between the subspace and the factor of variation is {subspace_mi: .3f}."
    )

    # Extract the estimate of the projection matrix
    A_hat = mi_estimation_result.A_1.detach().cpu().numpy()

    # Evaluate the fit
    evaluation_result = subspace_evaluation.subspace_fit(dataset.A, A_hat)
    logging.info("--- Quality of the fit. ---")
    print(evaluation_result)

    # Plot the fit if the dimensionality is suitable
    if cfg.n in [2, 3]:
        file_name = f"nonlinear_synthetic_dataset_{cfg.n}d_{cfg.subspace_size}d.png"
        dataset_visualisation.synthetic_data(dataset.X, dataset.A, A_hat, file_name=file_name)

        # If the factor of variation is 1-dimensional, also visualise it by the color of the observable points
        if dataset.Z.shape[-1] == 1:

            file_name = f"nonlinear_synthetic_dataset_{cfg.n}d_{cfg.subspace_size}d_factors.png"
            dataset_visualisation.synthetic_data(
                dataset.X,
                dataset.A,
                A_hat,
                f=dataset.Z,
                file_name=file_name,
                alphas={"observable": 0.5, "true_projections": 0.2, "estimated_projections": 0.3},
            )


if __name__ == "__main__":
    main()
