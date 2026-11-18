import numpy as np
import pytest

from qrc.model import Params
from qrc.reservoir import features, shot_covariance, simulate, trace_distance_curve

SMALL = Params(n=3, eps=0.5)


def test_lambda0_features_zero():
    run = simulate(SMALL, lam=0.0, p=0.3, M=500, T_wash=300)
    assert np.all(run.x == 0.0)


def test_p0_features_decay():
    run = simulate(SMALL, lam=1.0, p=0.0, M=500, T_wash=500)
    assert np.abs(run.x).max() < 1e-6


def test_covariance_diag_and_psd():
    run = simulate(SMALL, lam=0.8, p=0.3, M=500, T_wash=300)
    diag = np.diagonal(run.cov, axis1=1, axis2=2)
    assert np.allclose(diag, 1.0 - run.x**2)
    assert np.linalg.eigvalsh(run.cov).min() > -1e-12


def test_features_and_covariance_from_populations():
    pop = np.zeros((1, 8)); pop[0, 0] = 1.0           # |000>
    assert np.allclose(features(pop, 3, 0.5), 0.5)
    assert np.allclose(shot_covariance(pop, 3, 1.0), 0.0)


def test_populations_normalised():
    run = simulate(SMALL, lam=0.8, p=0.3, M=500, T_wash=300)
    assert np.allclose(run.pop.sum(axis=1), 1.0)
    assert run.pop.min() > -1e-14
    assert run.u.shape == (800,) and run.T_wash == 300


def test_washout_warning():
    with pytest.warns(UserWarning, match="trace distance"):
        simulate(SMALL, lam=1.0, p=0.01, M=50, T_wash=5)


def test_trace_distance_curve_decreases_to_zero():
    u = np.random.default_rng(0).uniform(-1, 1, 400)
    curve = trace_distance_curve(SMALL, 1.0, 0.5, u)
    assert curve.shape == (400,)
    assert curve[-1] < 1e-8 < curve[0]
