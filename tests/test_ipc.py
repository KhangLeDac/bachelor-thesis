import numpy as np
import pytest

from qrc.ipc import capacities, ipc, legendre_targets, summarize
from qrc.model import Params
from qrc.reservoir import simulate


def test_target_normalisation_and_table():
    u = np.random.default_rng(0).uniform(-1, 1, 20100)
    Y, table = legendre_targets(u, T_wash=100, delays=(2, 1, 1))
    assert Y.shape == (20000, 10)
    assert table[:3] == [(1, (0,)), (1, (1,)), (1, (2,))]
    assert (3, (0, 1, 1)) in table
    assert np.all(np.abs(Y.mean(axis=0)) < 0.05)
    assert np.all(np.abs(Y.var(axis=0) - 1.0) < 0.05)


def test_targets_aligned_with_delays():
    u = np.random.default_rng(1).uniform(-1, 1, 60)
    Y, table = legendre_targets(u, T_wash=10, delays=(3,))
    k = table.index((1, (2,)))
    assert np.allclose(Y[:, k], np.sqrt(3) * u[8:58])


def test_feature_equal_to_target():
    u = np.random.default_rng(2).uniform(-1, 1, 20100)
    Y, _ = legendre_targets(u, T_wash=100, delays=(3,))
    x = Y[:, [0]]
    C = capacities(x, np.zeros((len(x), 1, 1)), Y, S=np.inf)
    assert abs(C[0] - 1.0) < 0.02
    assert np.all(np.abs(C[1:]) < 0.03)


def test_total_bounded_by_n():
    run = simulate(Params(n=4, eps=0.5), lam=1.0, p=0.3, M=4000, T_wash=300)
    C, table = ipc(run, S=np.inf, delays=(20, 10, 7))
    assert summarize(C, table)["total"] <= 4 + 0.05


def test_no_memory_at_full_reset():
    run = simulate(Params(n=3, eps=0.5), lam=1.0, p=1.0, M=4000, T_wash=50)
    C, table = ipc(run, S=np.inf, delays=(10, 5, 3))
    delayed = np.array([max(delays) >= 1 for _, delays in table])
    assert np.all(np.abs(C[delayed]) < 0.02)
    assert C[~delayed].sum() > 0.1


def test_p0_ipc_zero():
    run = simulate(Params(n=3, eps=0.5), lam=1.0, p=0.0, M=3000, T_wash=500)
    C, table = ipc(run, S=1e4, delays=(10, 5, 3))
    assert abs(summarize(C, table)["total"]) < 0.05


def test_lambda0_no_nan():
    run = simulate(Params(n=3, eps=0.5), lam=0.0, p=0.3, M=2000, T_wash=300)
    C, _ = ipc(run, S=100, delays=(5, 3, 2))
    assert np.all(np.isfinite(C)) and np.all(np.abs(C) < 0.02)


def test_S_inf_matches_large_S():
    run = simulate(Params(n=3, eps=0.5), lam=0.9, p=0.3, M=2000, T_wash=300)
    C_inf, _ = ipc(run, S=np.inf, delays=(5, 3, 2))
    C_big, _ = ipc(run, S=1e12, delays=(5, 3, 2))
    assert np.allclose(C_inf, C_big, atol=1e-6)


def test_summarize_keys_and_memory():
    table = [(1, (0,)), (1, (1,)), (2, (0, 0)), (3, (0, 0, 1))]
    s = summarize(np.array([0.5, 0.25, 0.1, 0.05]), table)
    assert np.isclose(s["total"], 0.9) and np.isclose(s["degree1"], 0.75)
    assert np.isclose(s["degree2"], 0.1) and np.isclose(s["degree3"], 0.05)
    assert np.allclose(s["memory"], [0.5, 0.25])


def test_delay_too_long_raises():
    with pytest.raises(ValueError):
        legendre_targets(np.zeros(3000), T_wash=1000, delays=(2000,))


def test_ipc_warns_when_memory_not_decayed():
    run = simulate(Params(n=3, eps=0.5), lam=0.6, p=0.05, M=3000, T_wash=1500)
    with pytest.warns(UserWarning, match="delays"):
        ipc(run, S=np.inf, delays=(2, 1, 1))


def test_ipc_quiet_when_memory_decayed():
    import warnings
    run = simulate(Params(n=3, eps=0.5), lam=1.0, p=1.0, M=2000, T_wash=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ipc(run, S=np.inf, delays=(5, 3, 2))


def test_summarize_higher_degrees():
    table = [(1, (0,)), (2, (0, 0)), (4, (0, 0, 0, 0))]
    s = summarize(np.array([0.5, 0.2, 0.1]), table)
    assert np.isclose(s["degree4"], 0.1) and np.isclose(s["degree3"], 0.0)
    assert np.isclose(s["total"], 0.8)
