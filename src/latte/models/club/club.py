"""
Implementation of the basic CLUB model for estimating and minimising the mutual information between two random vectors
using the CLUB upper bound.
Implementation based on
Cheng, Pengyu et al. “CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information.” ICML (2020). and the
corresponding code from https://github.com/Linear95/CLUB.
"""

import torch
from torch import nn


class CLUB(nn.Module):
    """
    A pytorch implementation of the `CLUB` model which can be used to estimate the mutual information between two
    distributions based on their samples or plugged into other modules for minimisation of the mutual information.
    """

    def __init__(
        self,
        mean_estimator: nn.Module,
        log_variance_estimator: nn.Module,
    ):
        """
        Args:
            mean_estimator: A `pytorch` module estimating the mean of the conditional distribution of a given sample `x`
            log_variance_estimator: A `pytorch` module estimating the logarithm of the diagonal covariance of the
                                    conditional distribution of a given sample `x`
        """
        super(CLUB, self).__init__()
        # Networks implementing the two components of the Gaussian approximation of the conditional density
        self.mean_estimator = mean_estimator
        self.log_variance_estimator = log_variance_estimator

    def log_likelihood(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Returns the likelihood of `z` under the variational Gaussian distribution for
        the conditional distribution of `z` given `x`.
        Args:
            x: A sample from the first distribution.
            z: A sample from the first distribution.

        Returns:
            The likelihood given the variational approximation.
        """

        # Get the parameters of the variational conditional distribution
        mean, log_variance = self.mean_estimator(x), self.log_variance_estimator(x)
        return (-((mean - z) ** 2) / log_variance.exp() - log_variance).sum(dim=1).mean(dim=0)

    def forward(self, x, z):
        """
        Forward function of the model, which uses the variational approximation of the conditional distribution given
        by the two models to compute an upper bound on the mutual information between the distributions generating the
        samples.

        Args:
            x: A sample from the first distribution
            z: The corresponding sample from the second distribution

        Returns: Estimate of the upper bound of the mutual information between the distributions of `x` and `z` based on
        this batch of data.
        """

        # Get the parameters of the variational conditional distribution
        mean, log_variance = self.mean_estimator(x), self.log_variance_estimator(x)

        # Perform a 1-sample negative sampling
        random_index = torch.randperm(len(x)).long()

        positive = -((mean - z) ** 2) / log_variance.exp()
        negative = -((mean - z[random_index]) ** 2) / log_variance.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

        return upper_bound / 2
