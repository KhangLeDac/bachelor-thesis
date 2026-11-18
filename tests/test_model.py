import numpy as np

from qrc.model import Params, draw_couplings, hamiltonian, unitary, z_signs


def test_couplings_reproducible_and_normalised():
    for n in (3, 6):
        a, b, c = draw_couplings(n, 1.3, 7), draw_couplings(n, 1.3, 7), draw_couplings(n, 1.3, 8)
        assert np.array_equal(a, b)
        assert not np.allclose(a, c)
        assert np.array_equal(a, a.T)
        assert np.all(np.diag(a) == 0)
        assert np.isclose((a**2).sum() / n, 1.3**2)


def test_z_signs_convention():
    assert np.array_equal(z_signs(2), [[1, 1, -1, -1], [1, -1, 1, -1]])


def test_hamiltonian_hermitian_and_unitary_is_unitary():
    params = Params(n=3)
    for u in (-0.7, 0.2):
        H = hamiltonian(params, u)
        assert np.allclose(H, H.T)
        U = unitary(params, u)
        assert np.allclose(U @ U.conj().T, np.eye(8), atol=1e-12)


def test_hamiltonian_single_spin():
    params = Params(n=1, h_x=0.5, h_z=1.0, eps=0.3)
    X, Z = np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0])
    u = 0.4
    assert np.allclose(hamiltonian(params, u), 1.0 * Z + (0.5 + 0.3 * u) * X)
