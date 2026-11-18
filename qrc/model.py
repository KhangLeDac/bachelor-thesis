"""Spin model: random couplings, Hamiltonian H(u) and step unitary U(u).

    H(u) = h_z sum_i Z_i + sum_{i<j} J_ij X_i X_j + (h_x + eps * u) sum_i X_i

Basis convention: spin 0 is the most significant bit of the basis index, and
|0> is the Z = +1 eigenstate.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_Z = np.diag([1.0, -1.0])


@dataclass(frozen=True)
class Params:
    """Model parameters. ``seed`` fixes the couplings (one seed = one device)."""

    n: int = 6
    J: float = 1.0
    h_x: float = 0.5
    h_z: float = 1.0
    eps: float = 0.1
    tau: float = 2.0
    seed: int = 0

    @property
    def dim(self) -> int:
        return 2**self.n


def draw_couplings(n: int, J: float, seed: int) -> np.ndarray:
    """Symmetric (n, n) couplings with zero diagonal, J_ij ~ U(-1, 1), rescaled by one
    factor so that (1/n) sum_ij J_ij^2 = J^2 (rms local coupling J)."""
    if n == 1:
        return np.zeros((1, 1))
    A = np.triu(np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, n)), 1)
    A = A + A.T
    return A * (J * np.sqrt(n / np.sum(A**2)))


def z_signs(n: int) -> np.ndarray:
    """(n, 2**n) array: entry [i, k] is the Z eigenvalue (+1 or -1) of spin i in basis state k."""
    k = np.arange(2**n)
    bits = (k[None, :] >> (n - 1 - np.arange(n))[:, None]) & 1
    return 1 - 2 * bits # -1 for |1>, +1 for |0>


def single_site(op: np.ndarray, i: int, n: int) -> np.ndarray:
    """Dense (2**n, 2**n) operator acting as ``op`` on spin i and as identity elsewhere."""
    return np.kron(np.kron(np.eye(2**i), op), np.eye(2 ** (n - i - 1)))


@lru_cache(maxsize=32)
def hamiltonian_parts(params: Params) -> tuple[np.ndarray, np.ndarray]:
    """(H0, H1) with H(u) = H0 + u * H1; H1 = eps * sum_i X_i. Cached per parameter set."""
    n = params.n
    Jm = draw_couplings(n, params.J, params.seed)
    X = [single_site(_X, i, n) for i in range(n)]
    H0 = sum(params.h_z * single_site(_Z, i, n) + params.h_x * X[i] for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            H0 = H0 + Jm[i, j] * X[i] @ X[j]
    H1 = params.eps * sum(X)
    H0.flags.writeable = False
    H1.flags.writeable = False
    return H0, H1


def hamiltonian(params: Params, u: float) -> np.ndarray:
    """Dense real H(u)."""
    H0, H1 = hamiltonian_parts(params)
    return H0 + float(u) * H1


def unitary(params: Params, u: float) -> np.ndarray:
    """U(u) = exp(-i H(u) tau), via the eigendecomposition of H(u)."""
    w, v = np.linalg.eigh(hamiltonian(params, u))
    return (v * np.exp(-1j * w * params.tau)) @ v.conj().T
