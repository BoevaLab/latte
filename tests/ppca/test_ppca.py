import pytest
from sklearn.decomposition import PCA

from latte.models.probabilistic_pca import probabilistic_pca, evaluation as ppca_evaluation


@pytest.mark.parametrize("n", [8, 16, 32])
@pytest.mark.parametrize("d", [2, 4])
def test_dataset(n: int, d: int) -> None:
    dataset = probabilistic_pca.generate(N=1024, n=n, d=d, d_measured=d, sigma=0.3, rng=1)

    assert dataset.X.shape == (1024, n)
    assert dataset.Z.shape == (1024, d)
    assert dataset.A.shape == (n, d)
    assert len(dataset.measured_factors) == d


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("d", [2, 3])
def test_ppca(n: int, d: int) -> None:
    dataset = probabilistic_pca.generate(N=10000, n=n, d=d, d_measured=d, sigma=0.1, rng=1)

    # Run probabilistic PCA to get the principal components of the observed data
    pca = PCA(n_components=d, random_state=1)
    pca.fit(dataset.X)

    assert pytest.approx(dataset.sigma**2, abs=1e-3) == pca.noise_variance_, dataset.sigma**2 - pca.noise_variance_
    assert pytest.approx(dataset.mu, abs=1e-1) == pca.mean_, dataset.mu - pca.mean_
    assert ppca_evaluation.subspace_fit(dataset, pca.components_.T)["Value"]["Difference between subspaces"] < 1e-2
