import numpy as np

from qrc.ipc import capacities, legendre_targets
from qrc.model import Params
from qrc.reservoir import features, simulate
from qrc.sampling import noisy_features, outcome_distribution, readout_capacity


def test_outcome_distribution_normalised_and_mean():
    n, lam = 3, 0.6
    pop = np.random.default_rng(0).dirichlet(np.ones(8), size=5)
    P = outcome_distribution(pop, n, lam)
    assert np.allclose(P.sum(axis=1), 1.0) and P.min() >= 0
    s = 1 - 2 * ((np.arange(8)[:, None] >> (n - 1 - np.arange(n))) & 1)   # (2**n, n) outcome signs
    assert np.allclose(P @ s, features(pop, n, lam), atol=1e-12)


def test_noisy_features_mean_and_variance():
    pop = np.zeros((20000, 2)); pop[:, 0] = 0.8; pop[:, 1] = 0.2      # one spin, <Z> = 0.6
    x_hat = noisy_features(pop, 1, 1.0, S=10, rng=np.random.default_rng(1))
    assert abs(x_hat.mean() - 0.6) < 0.01
    assert abs(x_hat.var() - (1 - 0.36) / 10) < 0.003


def test_sampled_matches_analytic():
    run = simulate(Params(n=4, eps=0.5), lam=0.8, p=0.3, M=40000, T_wash=300)
    Y, table = legendre_targets(run.u, run.T_wash, delays=(2,))
    S = 50
    C_analytic = capacities(run.x, run.cov, Y, S)
    x_hat = noisy_features(run.pop, 4, 0.8, S, np.random.default_rng(2))
    C_sampled = readout_capacity(x_hat, Y)
    assert C_analytic[0] > 0.05
    assert np.all(np.abs(C_sampled - C_analytic) < 0.03)
