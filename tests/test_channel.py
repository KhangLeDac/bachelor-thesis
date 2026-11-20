import numpy as np
import pytest

from qrc.channel import check_strengths, kraus_measure, kraus_reset, measure_reset, measure_reset_spin, step
from qrc.model import Params, single_site, unitary


def random_density_matrix(d, rng):
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def test_spin_map_matches_kraus():
    n, rng = 3, np.random.default_rng(1)
    rho = random_density_matrix(2**n, rng)
    for lam, p in [(0.3, 0.2), (0.8, 0.7), (1.0, 1.0), (0.0, 0.0)]:
        for spin in range(n):
            ref = np.zeros_like(rho)
            for E in kraus_reset(p):
                for Om in kraus_measure(lam):
                    K = single_site(E @ Om, spin, n)
                    ref += K @ rho @ K.conj().T
            assert np.allclose(measure_reset_spin(rho, spin, n, lam, p), ref, atol=1e-12)


def test_step_trace_and_hermiticity():
    params, rng = Params(n=3), np.random.default_rng(2)
    rho = step(random_density_matrix(8, rng), unitary(params, 0.3), 3, 0.6, 0.4)
    assert np.isclose(np.trace(rho).real, 1.0)
    assert np.allclose(rho, rho.conj().T)


def test_unital_at_p0():
    params = Params(n=3)
    mixed = np.eye(8) / 8
    assert np.allclose(step(mixed, unitary(params, -0.5), 3, 0.7, 0.0), mixed, atol=1e-12)


def test_full_reset():
    rng = np.random.default_rng(3)
    target = np.zeros((8, 8)); target[0, 0] = 1.0
    assert np.allclose(measure_reset(random_density_matrix(8, rng), 3, 1.0, 1.0), target, atol=1e-12)


@pytest.mark.parametrize("lam,p", [(-0.1, 0.5), (1.1, 0.5), (np.nan, 0.5), (0.5, -0.1), (0.5, 1.1), (0.5, np.nan)])
def test_invalid_strengths(lam, p):
    with pytest.raises(ValueError):
        check_strengths(lam, p)
    with pytest.raises(ValueError):
        step(np.eye(2) / 2, np.eye(2), 1, lam, p)


@pytest.mark.parametrize("lam,p", [(0.5, -0.1), (0.5, 1.1), (1.2, 0.5)])
def test_measure_reset_rejects_invalid(lam, p):
    with pytest.raises(ValueError):
        measure_reset(np.eye(2) / 2, 1, lam, p)


def test_full_reset_for_any_lambda():
    rng = np.random.default_rng(4)
    target = np.zeros((8, 8)); target[0, 0] = 1.0
    for lam in (0.0, 0.5):
        assert np.allclose(measure_reset(random_density_matrix(8, rng), 3, lam, 1.0), target, atol=1e-12)


def test_check_strengths_arrays():
    check_strengths(0.5, np.array([0.0, 0.3, 1.0]))
    with pytest.raises(ValueError, match="p must be in"):
        check_strengths(0.5, np.array([0.2, 1.1]))
    with pytest.raises(ValueError, match="p must be in"):
        check_strengths(0.5, np.array([0.2, np.nan]))
