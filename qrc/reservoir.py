"""Drive the reservoir with an input sequence and collect what the readout sees.

At every step t the state sigma_t = U(u_t) rho_{t-1} U(u_t)^dagger is measured. Its
populations pop_t = diag(sigma_t) determine
    features          x_i(t) = lam <Z_i>                     (mean of one shot s_i)
    shot covariance   Sigma_ij(t) = E[s_i s_j] - x_i x_j,    E[s_i s_j] = lam^2 <Z_i Z_j> (i != j), s_i^2 = 1.
With S shots per step the estimated features have noise covariance Sigma(t) / S.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .channel import check_strengths, measure_reset, step
from .model import Params, unitary, z_signs


@dataclass
class Run:
    """Result of ``simulate``: inputs (washout included) and per-sample populations, features, covariances."""

    u: np.ndarray          # (T_wash + M,) inputs; sample t used u[T_wash + t]
    pop: np.ndarray        # (M, 2**n) populations of the measured state
    x: np.ndarray          # (M, n) features lam <Z_i>
    cov: np.ndarray        # (M, n, n) single-shot covariance
    washout_distance: float
    T_wash: int
    params: Params
    lam: float
    p: float


def features(pop: np.ndarray, n: int, lam: float) -> np.ndarray:
    """(M, n) features lam <Z_i> from (M, 2**n) populations (pop)."""
    return lam * pop @ z_signs(n).T


def shot_covariance(pop: np.ndarray, n: int, lam: float) -> np.ndarray:
    """(M, n, n) covariance of one shot s in {+1, -1}^n."""
    zs = z_signs(n).astype(float)
    second = lam**2 * np.einsum("ik,tk,jk->tij", zs, pop, zs)
    idx = np.arange(n)
    second[:, idx, idx] = 1.0
    x = features(pop, n, lam)
    return second - x[:, :, None] * x[:, None, :]


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Trace distance 0.5 * ||a - b||_1 of two density matrices."""
    return 0.5 * float(np.abs(np.linalg.eigvalsh(a - b)).sum())


def _ground(d: int) -> np.ndarray:
    """ |0...0><0...0|."""
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def trace_distance_curve(params: Params, lam: float, p: float, u: np.ndarray, rho_a=None, rho_b=None) -> np.ndarray:
    """Trace distance after every step between two initial states driven by the same inputs u
    (defaults |0...0> and the maximally mixed state). Its decay measures how fast the reservoir forgets."""
    d, n = params.dim, params.n
    a = _ground(d) if rho_a is None else np.asarray(rho_a, dtype=complex)
    b = np.eye(d, dtype=complex) / d if rho_b is None else np.asarray(rho_b, dtype=complex)
    out = np.empty(len(u))
    for t, ut in enumerate(u):
        U = unitary(params, ut)
        a, b = step(a, U, n, lam, p), step(b, U, n, lam, p)
        out[t] = trace_distance(a, b)
    return out


def simulate(params: Params, lam: float, p: float, M: int = 10000, T_wash: int = 1000,
             input_seed: int = 0, washout_tol: float = 1e-6) -> Run:
    """Run T_wash + M steps with inputs u_t ~ U(-1, 1) from ``input_seed``, starting in |0...0>.

    During the washout the maximally mixed state is propagated alongside; if the two still
    differ by more than ``washout_tol`` in trace distance, a UserWarning is issued.
    => Done to ensure that the reservoir has forgotten its initial state before the actual run."""

    check_strengths(lam, p)
    n, d = params.n, params.dim

    # inputs randomly drawn from uniform distribution 
    u = np.random.default_rng(input_seed).uniform(-1.0, 1.0, T_wash + M)

    """ Washout phase """
    # two initial states stacked in rho: rho[0] = |0...0><0...0|, rho[1] = I / d. 
    rho = np.stack([_ground(d), np.eye(d, dtype=complex) / d])
    # washout
    for t in range(T_wash):
        rho = step(rho, unitary(params, u[t]), n, lam, p)
    dist = trace_distance(rho[0], rho[1])
    # compare trace distance of the two states
    if dist > washout_tol:
        warnings.warn(f"washout too short: trace distance {dist:.2e} after {T_wash} steps "
                      f"(tolerance {washout_tol:.0e}); increase T_wash", UserWarning, stacklevel=2)
    
    """ Actual run phase """
    rho = rho[0]
    pop = np.empty((M, d))
    for t in range(M):
        # state propagation: rho_t = sigma  = U(u_t) rho_{t-1} U(u_t)^dagger = U @ rho @ U^dagger
        U = unitary(params, u[T_wash + t])
        sigma = U @ rho @ U.conj().T
        # pop[t,z] = <z|sigma_t|z> = sigma[z,z] = P_t(z): Probability of measuring basis state z at step t 
        pop[t] = np.diagonal(sigma).real
        # weak measurement and amplitude damping
        rho = measure_reset(sigma, n, lam, p)

    # bundels the parameters and results together 
    return Run(u=u, pop=pop, x=features(pop, n, lam), cov=shot_covariance(pop, n, lam),
               washout_distance=dist, T_wash=T_wash, params=params, lam=lam, p=p)
