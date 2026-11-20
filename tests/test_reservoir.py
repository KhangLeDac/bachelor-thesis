import numpy as np
import pytest

from qrc.model import Params
from qrc.reservoir import features, reset_sequence, shot_covariance, simulate, trace_distance_curve

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


def test_constant_function_reproduces_constant_p():
    a = simulate(SMALL, lam=0.8, p=0.7, M=300, T_wash=300)
    b = simulate(SMALL, lam=0.8, p=lambda u: np.full_like(u, 0.7), M=300, T_wash=300)
    assert np.array_equal(a.pop, b.pop) and np.all(b.p_seq == 0.7)


def test_input_dependent_reset_sequence_and_effect():
    f = lambda u: 0.5 + 0.3 * u
    run = simulate(SMALL, lam=0.8, p=f, M=300, T_wash=300)
    assert np.allclose(run.p_seq, f(run.u))           # p_t = f(u_t), aligned with u
    const = simulate(SMALL, lam=0.8, p=0.5, M=300, T_wash=300)
    assert np.array_equal(run.u, const.u)             # same inputs: paired comparison
    assert not np.allclose(run.pop, const.pop)


def test_reset_function_out_of_range():
    with pytest.raises(ValueError, match="p must be in"):
        simulate(SMALL, lam=0.8, p=lambda u: 0.5 + u, M=10, T_wash=10)
    with pytest.raises(ValueError, match="shape"):
        reset_sequence(lambda u: 0.5, np.zeros(5))


def test_trace_distance_curve_with_function():
    u = np.random.default_rng(0).uniform(-1, 1, 200)
    assert np.allclose(trace_distance_curve(SMALL, 1.0, lambda v: np.full_like(v, 0.5), u),
                       trace_distance_curve(SMALL, 1.0, 0.5, u))
