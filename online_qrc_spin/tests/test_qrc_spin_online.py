"""Tests for OnlineSpinQRC. Run from online_qrc_spin/:  python -m pytest tests -q"""

import os
import sys

import numpy as np
import pytest
import qutip as qt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.qrc_spin_online import ConstantPolicy, OnlineSpinQRC, trace_distance  # noqa: E402

try:
    from utils.qrc_spin import SpinQRC
except ImportError:  # SpinQRC needs utils/shots.py
    SpinQRC = None


def dims(n):
    return [[2] * n, [2] * n]


def random_density_matrix(n, seed=0):
    rng = np.random.default_rng(seed)
    dim = 2 ** n
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = A @ A.conj().T
    return qt.Qobj(rho / np.trace(rho), dims=dims(n))


def z_expectations(res, rho):
    """<Z_i> for every spin."""
    return np.array([qt.expect(op, rho) for op in res.sz_ops])


def reference_hamiltonian(res, s):
    """H(s) written out spin by spin, independent of H_static / sx_ops / sz_ops."""
    n = res.n

    def site(op, i):
        return qt.tensor([op if k == i else qt.qeye(2) for k in range(n)])

    field = res.field_strengths(s)
    H = 0
    for i in range(n):
        if res.model == "TiltedTFIM":
            H += res.constant_field * site(qt.sigmaz(), i) + field[i] * site(qt.sigmax(), i)
        else:
            H += field[i] * site(qt.sigmaz(), i)
        for j in range(i + 1, n):
            H += res.J[i, j] * site(qt.sigmax(), i) * site(qt.sigmax(), j)
    return H


def make(**kw):
    defaults = dict(n=4, encoding_strength=0.5, coupling_strength=1.0, gamma=0.3, theta=np.pi / 2, dt=1.0, probes=2)
    defaults.update(kw)
    return OnlineSpinQRC(**defaults)


# -------------------------------------------------------------- construction


def test_operators_and_state_share_dims():
    res = make(n=3, probes=[0, 2])
    ops = [*res.sx_ops, *res.sz_ops, *res.probe_P0_ops, *res.probe_P1_ops, *res.probe_E01_ops,
           *res.lowering_ops, res.H_static, res.initial_state]
    for op in ops:
        assert op.dims == dims(3)
    assert res.initial_state.tr() == pytest.approx(1.0)
    assert qt.expect(res.sz_ops[0], res.initial_state) == pytest.approx(1.0)  # |0...0>


def test_probes_int_and_list():
    assert make(n=4, probes=3).probes == [0, 1, 2]
    assert make(n=4, probes=[3, 1]).probes == [1, 3]


@pytest.mark.parametrize("kw", [
    dict(probes=[0, 4]),
    dict(probes=[-1]),
    dict(model="XY"),
    dict(encoding_method="sparse"),
    dict(damping_on="some"),
    dict(decay_to=2),
    dict(n_shots=0.5),
])
def test_invalid_arguments_raise(kw):
    with pytest.raises(ValueError):
        make(**kw)


@pytest.mark.parametrize("damping_on, expected", [("all", 4), ("probes", 2)])
def test_damping_on_selects_spins(damping_on, expected):
    assert len(make(n=4, probes=2, damping_on=damping_on).lowering_ops) == expected


# --------------------------------------------------------------- propagation


@pytest.mark.parametrize("model", ["TiltedTFIM", "TFIM"])
def test_evolve_without_damping_matches_matrix_exponential(model):
    """gamma = 0: evolve is rho -> U rho U^+ with U = exp(-i H dt) for the hand-built H(s)."""
    res = make(model=model)
    rho0 = random_density_matrix(res.n)
    s = np.array([0.7])

    U = (-1j * reference_hamiltonian(res, s) * res.dt).expm()
    ref = U * rho0 * U.dag()

    np.testing.assert_allclose(res.evolve(rho0, s, 0.0).full(), ref.full(), atol=1e-5)


@pytest.mark.parametrize("model", ["TiltedTFIM", "TFIM"])
def test_evolve_keeps_a_valid_state(model):
    res = make(model=model)
    out = res.evolve(random_density_matrix(res.n, seed=2), np.array([0.3]), 0.4)
    assert out.tr().real == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(out.full(), out.dag().full(), atol=1e-8)
    assert np.linalg.eigvalsh(0.5 * (out + out.dag()).full()).min() > -1e-6


@pytest.mark.parametrize("decay_to", [0, 1])
def test_damping_drives_spins_to_the_fixed_point(decay_to):
    """Strong damping from the opposite state: Z -> +1 for decay_to=0 (|0>), -1 for decay_to=1 (|1>)."""
    res = make(decay_to=decay_to, damping_on="all")
    start = qt.tensor([qt.fock_dm(2, 1 - decay_to)] * res.n)
    z = z_expectations(res, res.evolve(start, np.array([0.5]), 50.0))
    sign = 1 if decay_to == 0 else -1
    assert np.all(sign * z > 0.5)


@pytest.mark.skipif(SpinQRC is None, reason="utils/shots.py is missing, SpinQRC cannot be imported")
@pytest.mark.parametrize("encoding_method", ["one_to_one", "fill", "dense"])
def test_regression_against_spinqrc(encoding_method):
    """Without back-action and with decay_to=1, the evolution reproduces SpinQRC exactly."""
    n, steps, seed = 4, 15, 7
    data = np.random.default_rng(1).uniform(0, 1, (steps, 1))
    kw = dict(n=n, encoding_strength=0.6, coupling_strength=1.2, gamma=0.2, dt=1.0,
              model="TiltedTFIM", encoding_method=encoding_method, seed=seed)

    original = SpinQRC(observables="z_zz", **kw)
    ref = original.run(data)[:, :n]

    res = OnlineSpinQRC(theta=0.0, probes=1, damping_on="all", decay_to=1, **kw)
    np.testing.assert_allclose(res.J, original.J)
    rho = res.initial_state
    ours = []
    for s in data:
        rho = res.evolve(rho, s, res.policy.gamma)  # no measure_and_reset
        ours.append(z_expectations(res, rho))
    np.testing.assert_allclose(np.array(ours), ref, atol=1e-6)


# ------------------------------------------------------------ measurement channel


@pytest.mark.parametrize("theta", [0.0, 0.8, np.pi / 2, np.pi])
def test_channel_is_cptp(theta):
    res = make(n=4, probes=[0, 2])
    out = res.measure_and_reset(random_density_matrix(res.n, seed=3), theta)
    assert out.tr().real == pytest.approx(1.0)
    np.testing.assert_allclose(out.full(), out.dag().full(), atol=1e-12)
    assert np.linalg.eigvalsh(out.full()).min() > -1e-12


def test_theta_zero_is_pure_dephasing_of_probes():
    """theta = 0: Z populations unchanged, coherences between different probe outcomes removed."""
    res = make(n=4, probes=[1, 3])
    rho = random_density_matrix(res.n, seed=4)
    out = res.measure_and_reset(rho, 0.0).full()
    rho = rho.full()
    np.testing.assert_allclose(np.diag(out), np.diag(rho), atol=1e-12)

    idx = np.arange(res.dim)
    probe_bits = lambda i: tuple((i >> (res.n - 1 - q)) & 1 for q in res.probes)  # noqa: E731
    same = np.array([[probe_bits(a) == probe_bits(b) for b in idx] for a in idx])
    np.testing.assert_allclose(out[~same], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[same], rho[same], atol=1e-12)


def test_theta_pi_resets_probes_to_zero():
    res = make(n=4, probes=[0, 1, 2])
    out = res.measure_and_reset(random_density_matrix(res.n, seed=5), np.pi)
    p = res.outcome_probabilities(out)
    assert np.isclose(p[0], 1.0)  # all probes in |0>
    np.testing.assert_allclose(res.features(out), 1.0, atol=1e-12)


def test_partial_reset_population():
    """Outcome 1 ends in |0> with probability sin^2(theta/2)."""
    res = make(n=2, probes=[0])
    rho = qt.tensor(qt.fock_dm(2, 1), qt.fock_dm(2, 0))  # |10>: probe 0 in |1>
    theta = 1.1
    out = res.measure_and_reset(rho, theta)
    assert np.isclose(res.outcome_probabilities(out)[0], np.sin(theta / 2) ** 2)


# ------------------------------------------------------------------ protocol


def test_feature_layout():
    res = make(n=4, probes=3)
    assert res.n_features == 7
    assert res.feature_labels == ["Z0", "Z1", "Z2", "Z0Z1", "Z0Z2", "Z1Z2", "Z0Z1Z2"]
    out = res.run(np.random.default_rng(0).uniform(0, 1, 5))
    assert out.shape == (5, 7)
    assert np.all(np.abs(out) <= 1 + 1e-12)


def test_run_returns_states():
    res = make(n=3, probes=2)
    data = np.random.default_rng(0).uniform(0, 1, 6)
    features, states = res.run(data, return_states=True)
    assert features.shape == (6, res.n_features)
    assert len(states) == 6
    for rho in states:
        assert rho.dims == dims(3)
        assert rho.tr().real == pytest.approx(1.0, abs=1e-6)


def test_features_are_read_before_reset():
    """With theta = pi the post-reset probes are always |0>, so pre-reset features must differ from 1."""
    res = make(theta=np.pi)
    out = res.run(np.random.default_rng(0).uniform(0, 1, 10))
    assert not np.allclose(out, 1.0)


def test_policy_argument_overrides_default():
    res = make(gamma=0.3, theta=np.pi / 2)
    data = np.random.default_rng(2).uniform(0, 1, 8)
    a = res.run(data)
    b = res.run(data, policy=ConstantPolicy(0.3, np.pi / 2))
    c = res.run(data, policy=ConstantPolicy(0.9, np.pi))
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_policy_sees_previous_features():
    """prev_features is None at k = 0 and the features of step k - 1 afterwards."""
    seen = []

    def policy(k, s_k, prev):
        seen.append(None if prev is None else prev.copy())
        return 0.3, np.pi / 2

    res = make()
    out = res.run(np.random.default_rng(0).uniform(0, 1, 4), policy=policy)
    assert seen[0] is None
    for k in range(1, 4):
        np.testing.assert_allclose(seen[k], out[k - 1])


def test_finite_shots_close_to_exact():
    data = np.random.default_rng(0).uniform(0, 1, 20)
    exact = make().run(data)
    sampled = make(n_shots=200_000, shot_seed=1).run(data)
    np.testing.assert_allclose(sampled, exact, atol=0.02)


def test_esp_two_initial_states_converge():
    """Same input, different initial states: trace distance shrinks to ~0 (fading memory)."""
    res = make(n=4, probes=2, gamma=0.5, theta=np.pi)
    data = np.random.default_rng(0).uniform(0, 1, 40)
    mixed = qt.tensor([qt.qeye(2)] * res.n) / res.dim
    _, s1 = res.run(data, return_states=True)
    _, s2 = res.run(data, rho0=mixed, return_states=True)
    d = [trace_distance(a, b) for a, b in zip(s1, s2)]
    assert d[-1] < 1e-3 * d[0]
