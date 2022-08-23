"""
The Pytorch Lightning implementation of the MIST model.
It can be used for two purposes:
1. To estimate the mutual information between two distributions in the form of a lower bound estimator `MINE` as
   described in "Belghazi et al. (2018). Mutual information neural estimation.".
   Primarily, this is intended to be used to compare the mutual information between two representation functions and
   thus the consistency of a representation learning algorithm.
2. To find a linear subspace of the support of some distribution (e.g., over the latent representations) which at
   the same time maximises the mutual information with regard to a second distribution (in the form of a set of observed
   factors) and minimises it with regard to a third distribution (in the form of a set of observed factors).
   In the main application of this model, the first distribution would represent the learned representations of some
   data, while the second and third distributions would correspond to some observed (generative) factors we would like
   to investigate (e.g., see where they are (not) captured in the full representation space.
   **Note**: the minimisation aspect does not perform well yet.
"""

from typing import Any, Union, Dict, Tuple, List, Optional, Sequence

import geoopt

import torch
import pytorch_lightning as pl

from latte.modules.layers import ManifoldProjectionLayer
from latte.models.club.club import CLUB
from latte.models.mine.mine import MINE
from latte.utils import ksg_estimator


class MIST(pl.LightningModule):
    def __init__(
        self,
        n: Tuple[int, int],
        mine_args: Dict[str, Any],
        subspace_fit: bool = True,
        d: Optional[Tuple[int, int]] = None,
        gamma: float = 1.0,
        club_args: Optional[Dict[str, Any]] = None,
        n_density_updates: int = 0,
        mine_learning_rate: float = 1e-3,
        club_learning_rate: float = 1e-4,
        manifold_learning_rate: float = 1e-2,
        lr_scheduler_patience: int = 8,
        lr_scheduler_min_delta: float = 0.0,
        ksg_num_neighbours: int = 3,
        verbose: bool = False,
    ):
        """
        Implementation of the supervised MIST (Mutual Information optimisation over the STiefel manifold) model.
        The model is designed to:
        (i) estimate the mutual information between two distributions based on samples from them with `MINE` or
        (2) find the linear subspace of the data space of a random vector `X` which captures the
            most information about some random variables `Z_max` and as little information as possible about some other
            random variable `Z_min` among all possible linear subspace of the same dimensionality.
            This is done by projecting the samples of `X` onto the linear subspace with a column-orthogonal matrix
            (a k-frame), and finding the matrix which projects onto the subspace with the desired properties.
            The optimisation over the k-frames is done by optimising over the Stiefel manifold of k-frames, while the
            mutual information between random vectors is lower bounded with MINE (for maximisation) and upper-bounded by
            CLUB (for minimisation).
            This is the *supervised* version of the MIST model, since it works with observations of the latent factors
            we are trying to capture.
            **WARNING**: Minimisation currently does not seem to work to satisfaction.
            Please keep gamma = 1 for good performance.

        The model also continually assesses the mutual information between the distributions with an implementation of
        the non-parametric KSG estimator.
        Args:
            n: The dimensionality of the original samples of the first and second distributions.
            subspace_fit: Whether to find the optimal `d`-dimensional subspace or just estimate the mutual information
                          on the entire space.
            d: The optional dimensionality of the subspace to project onto for the first and second distributions.
               Can be None if just estimating the mutual information between two distributions.
            mine_args: The args for MINE.
            club_args: The args for CLUB.
            gamma: The coefficient determining the contribution of the MINE and CLUB losses.
                   Final loss is computed as `gamma * mine_loss + (1 - gamma) * club_loss`, meaning that larger values
                   of gamma put more emphasis on maximising the mutual information w.r.t. the factors of interest while
                   smaller values of gamma put more emphasis on minimising the mutual information w.r.t. the factors
                   which should be ignored/erased.
                   Should be between 0 and 1.
            n_density_updates: Number of updates to the density estimator for CLUB to make before each mutual
                               information estimation step.
            lr_scheduler_patience: The patience for the ReduceOnPlateu learning rate scheduler
            lr_scheduler_min_delta: The minimum improvement for the ReduceOnPlateu learning rate scheduler
            mine_learning_rate: The learning rate for the MINE parameters.
            club_learning_rate: The learning rate for the CLUB parameters.
            manifold_learning_rate: The learning rate for the projection matrix.
            ksg_num_neighbours: Number of neighbours used in the estimation of mutual information by the KSG estimator.
        """
        assert 0 <= gamma <= 1.0, "gamma should be between 0 and 1."
        assert gamma == 1.0 or club_args is not None
        # If mutual information is not minimised, the CLUB density estimator will not be used and should not be updated
        assert gamma < 1.0 or n_density_updates == 0
        assert not subspace_fit or d is not None
        assert subspace_fit or gamma == 1.0  # When estimating the MI between two distribution, we only use `MINE`
        super().__init__()

        # If subspace fitting is not selected, project onto the same space by default
        # (corresponds to a "rotation"/orthogonal transformation of the space)
        if not subspace_fit:
            d = n

        # Sets the flag of whether to do manifold optimisation (to find the optimal subspace)
        self.gamma = gamma
        self.minimise_mi = self.gamma < 1.0  # Whether to also minimise MI w.r.t. some factors (depends on gamma)

        self.mine = MINE(**mine_args) if self.gamma > 0 else None
        self.club = CLUB(**club_args) if self.minimise_mi else None
        self.projection_layer_x = ManifoldProjectionLayer(n[0], d[0])
        self.projection_layer_z = ManifoldProjectionLayer(n[1], d[1])

        self.n_density_updates = n_density_updates
        self.mine_learning_rate = mine_learning_rate
        self.club_learning_rate = club_learning_rate
        self.manifold_learning_rate = manifold_learning_rate
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_min_delta = lr_scheduler_min_delta
        self.verbose = verbose
        self.ksg_num_neighbours = ksg_num_neighbours

        # We are using multiple optimisers and manual optimisation
        self.automatic_optimization = False

    def forward(
        self, x: torch.Tensor, z_max: torch.Tensor, z_min: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward call of the model in which it directly calculates the objective of the model - the negative
        of the mutual information estimate by the MINE model (the lower bound) and the mutual information estimate by
        the CLUB model (the upper bound).
        These two terms are weighted by the gamma hyperparameter and combined into the objective.
        Args:
            x: The sample of the observed random vector.
            z_max: The sample of the factors in regard to which we want to maximise the mutual information.
            z_min: The sample of the factors in regard to which we want to minimise the mutual information.

        Returns:
            The value of the objetive of the model.
        """

        x = self.projection_layer_x(x)
        z_max = self.projection_layer_z(z_max)

        # Negative of the estimate of the mutual information by MINE (the lower bound)
        mine_loss = self.mine(x, z_max) if self.mine is not None else 0
        # Estimate of the mutual information by CLUB (the upper bound)
        club_loss = self.club(x, z_min) if self.club is not None else 0

        return self.gamma * mine_loss + (1 - self.gamma) * club_loss, -mine_loss, club_loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        """
        The main training step of the supervised MIST model.
        It updates the parameters of the estimators of the mutual information as well as the projection matrix.
        """

        # Depending on whether mutual information is also being minimised or not, load 2 or 3 optimizers of the model
        if self.minimise_mi:
            mine_optimizer, club_density_optimizer, manifold_optimizer = self.optimizers()
        else:
            mine_optimizer, manifold_optimizer = self.optimizers()

        x, z_max, z_min = batch  # Samples from the first, second, and the third distribution

        # We first update the parameters of the CLUB estimator of mutual information by updating its density estimator
        # by maximising the log-likelihood of the data
        # This density estimator is used to estimate the mutual information
        for _ in range(self.n_density_updates):
            density_loss = -self.club.log_likelihood(self.projection_layer_x(x), z_min)
            club_density_optimizer.zero_grad()
            self.manual_backward(density_loss)
            club_density_optimizer.step()

        loss, mine_mi, club_mi = self(x, z_max, z_min)

        # We update the parameters of the MINE model and the projection matrix at the same time since they optimise the
        # same objective - the mutual information directly
        mine_optimizer.zero_grad()
        manifold_optimizer.zero_grad()
        self.manual_backward(loss)
        mine_optimizer.step()
        manifold_optimizer.step()

        # Log various aspects of the estimate separately
        self.log("train_mutual_information_mine", mine_mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.minimise_mi:
            self.log(
                "train_mutual_information_club",
                club_mi,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def log_on_evaluation_step(
        self,
        step: str,
        loss: torch.Tensor,
        mine_mi: torch.Tensor,
        club_mi: torch.Tensor,
        x_projected: torch.Tensor,
        z_max: torch.Tensor,
        z_min: torch.Tensor,
    ) -> None:
        """
        Utility function to log various quantities recorded at the non-training steps of the model.

        Args:
            step: "test" or "validation" to correctly label the log
            loss: The values of the loss
            mine_mi: The lower bound computed by MINE of the mutual information between the projected sample of the
                     first distribution and a sample from the second one
            club_mi: The upper bound computed by CLUB of the mutual information between the projected sample of the
                     first distribution and a sample from the third one
            x_projected: The projected sample from the first distribution
            z_max: The sample from the second distribution
            z_min: The sample from the third distribution

        Returns:

        """

        # Log various aspects of the estimate separately
        # The full loss
        self.log(f"{step}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # The estimate of the mutual information between the projected sample from the first distribution and the
        # sample from the second one
        self.log(
            f"{step}_maximised_mutual_information",
            ksg_estimator.mutual_information(
                [x_projected.detach().cpu().numpy(), z_max.detach().cpu().numpy()], k=self.ksg_num_neighbours
            ),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # The lower bound estimated by MINE
        self.log(f"{step}_mutual_information_mine", mine_mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.minimise_mi:
            # The estimate of the mutual information between the projected sample from the first distribution and the
            # sample from the third one
            self.log(
                f"{step}_minimised_mutual_information",
                ksg_estimator.mutual_information(
                    [x_projected.detach().cpu().numpy(), z_min.detach().cpu().numpy()], k=self.ksg_num_neighbours
                ),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            # The upper bound estimated by CLUB
            self.log(
                f"{step}_mutual_information_club",
                club_mi,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def predict_without_gradients(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], step: str
    ) -> Dict[str, torch.Tensor]:
        """
        Helper function to perform a validation or test step without logging gradients.
        Args:
            batch: The batch of data
            step: "validation" or "test"

        Returns:
            The dictionary that `*_step` `pytorch_lightning` methods should return.
        """

        x, z_max, z_min = batch  # Samples from the first, second, and the third distribution

        with torch.no_grad():
            x_projected = self.projection_layer_x(x)
            loss, mine_mi, club_mi = self(x, z_max, z_min)

        self.log_on_evaluation_step(step, loss, mine_mi, club_mi, x_projected, z_max, z_min)

        return {f"{step}_loss": loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.predict_without_gradients(batch, "validation")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.predict_without_gradients(batch, "test")

    def validation_epoch_end(self, outputs: Sequence[Dict[str, torch.Tensor]]) -> None:
        """
        Log metrics to the learning-rate schedulers at the end of a validation epoch.
        """

        validation_loss = torch.stack([o["validation_loss"] for o in outputs]).mean()

        if self.minimise_mi:
            mine_scheduler, club_density_scheduler, manifold_scheduler = self.lr_schedulers()
            club_density_scheduler.step(validation_loss)
        else:
            mine_scheduler, manifold_scheduler = self.lr_schedulers()
            manifold_scheduler.step(validation_loss)

        mine_scheduler.step(validation_loss)

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.ReduceLROnPlateau]]:
        """
        Configure all optimisers and schedulers used by the full model.
        There are three sets of parameters over which the model is optimising: the projection matrix, the parameters of
        the MINE model, and the parameters of the CLUB model, so each of those gets its own optimiser and scheduler.
        """

        # Separate optimisers for all parameters in the model
        mine_optimizer = torch.optim.Adam(list(self.mine.parameters()), lr=self.mine_learning_rate)
        # Separate learning rate schedulers for all parameters in the model
        mine_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            mine_optimizer,
            mode="min",
            factor=0.1,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_min_delta,
            threshold_mode="abs",
            verbose=self.verbose,
            min_lr=1e-8,
        )

        # A geoopt optimizer for optimization over the Stiefel manifold
        # It is instantiated only if search of the optimal subspace should be performed
        manifold_optimizer = geoopt.optim.RiemannianAdam(
            params=[self.projection_layer_x.A, self.projection_layer_z.A], lr=self.manifold_learning_rate
        )
        manifold_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            manifold_optimizer,
            mode="min",
            factor=0.2,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_min_delta,
            threshold_mode="abs",
            verbose=self.verbose,
            min_lr=1e-6,
        )

        if self.minimise_mi:
            club_density_optimizer = torch.optim.Adam(list(self.club.parameters()), lr=self.club_learning_rate)
            club_density_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                club_density_optimizer,
                mode="min",
                factor=0.1,
                patience=self.lr_scheduler_patience,
                threshold=self.lr_scheduler_min_delta,
                threshold_mode="abs",
                verbose=self.verbose,
                min_lr=1e-6,
            )

            return [mine_optimizer, club_density_optimizer, manifold_optimizer], [
                mine_scheduler,
                club_density_scheduler,
                manifold_scheduler,
            ]
        else:

            return [mine_optimizer, manifold_optimizer], [mine_scheduler, manifold_scheduler]
