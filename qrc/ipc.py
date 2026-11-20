"""Information processing capacity (IPC) of the reservoir with S shots per step.

Targets are products of normalised Legendre polynomials sqrt(2n+1) P_n of delayed inputs
u_{t-k} (zero mean and unit variance for u ~ U(-1, 1)). With h(t) = (1, x_1(t), ..., x_N(t)),
    G = <h h^T>,   V = <Sigma(t)> (zero row and column for the constant),   R = <h y>,
a linear readout trained on features estimated from S shots reaches the capacity
    C_y(S) = R^T (G + V/S)^+ R          (S = inf: V/S = 0).
R is estimated from data, which biases this estimate upward by about (N + 1)/M per target.
``capacities`` removes that bias by cross-fitting: C = R_A^T (G + V/S)^+ R_B with R_A and R_B
from the first and second half of the samples (a gap between them). Targets the features
cannot see then give C close to 0, possibly slightly negative.
With N features the total capacity at S = inf is at most N; the sum over a finite target set is a
lower bound (see ``ipc``, which warns when the memory reaches the largest delay).
"""

import warnings
from itertools import combinations_with_replacement

import numpy as np
from numpy.polynomial import legendre

from .reservoir import Run


def _legendre(u: np.ndarray, order: int) -> np.ndarray:
    """
    inputs:
        u: array of length M with zero mean and unit variance
        order: degree of the Legendre polynomial
    returns:
        array of length M with the Legendre polynomial evaluated at u: [P_order(u[0]), ..., P_order(u[M-1])]
    """
    coef = np.zeros(order + 1)
    coef[order] = 1.0
    return np.sqrt(2 * order + 1) * legendre.legval(u, coef)


def legendre_targets(u: np.ndarray, T_wash: int, delays=(20, 10, 7)):
    """Targets for the samples after the washout: Y (M, L) with row t built from u[T_wash + t - k].

    Degree d = 1 .. len(delays) uses all delay multisets of size d with delays <= delays[d - 1].
    Returns Y and the table [(degree, delays), ...] describing the columns."""
    u = np.asarray(u, dtype=float)
    if max(delays) >= T_wash:
        raise ValueError(f"largest delay {max(delays)} must be smaller than T_wash = {T_wash}")
    M = len(u) - T_wash

    # table of delayed inputs: lagged[t, k] = u[T_wash + t - k] with dim = (M, max(delays) + 1)
    lagged = np.stack([u[T_wash - k: T_wash - k + M] for k in range(max(delays) + 1)], axis=1)


    cols, table = [], []
    for degree, k_max in enumerate(delays, start=1):
        for combo in combinations_with_replacement(range(k_max + 1), degree):
            """
            degree 1: (0,), (1,), ..., (delays[0],)
            degree 2: (0, 0), (0, 1), ..., (delays[1], delays[1])
            degree 3: (0, 0, 0), (0, 0, 1), ..., (delays[2], delays[2], delays[2])
            """
            # turn combination into time series
            y = np.ones(M) # vector of ones the size M
            # goes through the 
            for k in set(combo): # set(combo) removes duplicates
                y = y * _legendre(lagged[:, k], combo.count(k)) 
                # count(k) gives the multiplicity of k in the combination
                # component-wise multiplication 
            cols.append(y)
            table.append((degree, combo))
    return np.column_stack(cols), table


def _pinv_psd(A: np.ndarray, rcut: float = 1e-12) -> np.ndarray:
    """Pseudo-inverse of a symmetric positive semi-definite matrix A with eigenvalues < rcut * max(eig)."""
    w, v = np.linalg.eigh((A + A.T) / 2)
    keep = w > rcut * max(w.max(), 0.0)
    return (v[:, keep] / w[keep]) @ v[:, keep].T


def capacities(x: np.ndarray, cov: np.ndarray, Y: np.ndarray, S: float, gap: int = 50) -> np.ndarray:
    """
    Cross-fitted capacity per target (column of Y) with S shots per step.
    returns: C = array of length L (number of targets)
    """
    M, n = x.shape
    h = np.column_stack([np.ones(M), x])
    G = h.T @ h / M 
    # G = <h h^T>
 
    V = np.zeros((n + 1, n + 1))
    V[1:, 1:] = cov.mean(axis=0)
    # V = ([0,  0        ]
    #      [0, <Sigma(t)>])

    # Pseudo-inverse of the sum of G and V/S
    Minv = _pinv_psd(G if np.isinf(S) else G + V / S) 

    # Cross-fitting: R_A from the first half of the samples, R_B from the second half
    half = (M - gap) // 2
    # R_A [a,l] = <h_a y_l> for the first half of the samples, R_B for the second half
    # -> aproximation of the features onto the legendre targets: form a subspace of the feature space
    R_A = h[:half].T @ Y[:half] / half
    R_B = h[M - half:].T @ Y[M - half:] / half
    # summation over k: i-th entry is the capacity of the i-th target 
    return np.einsum("kl,kl->l", R_A, Minv @ R_B)


def summarize(C: np.ndarray, table) -> dict:
    """
    Total capacity, totals per degree 1..3, and the linear memory function (degree-1 capacity per delay).
    returns: out = {total, degree1, degree2, degree3, memory: array of length K}
    """
    degree = np.array([d for d, _ in table])
    out = {"total": float(C.sum())}
    for d in (1, 2, 3):
        out[f"degree{d}"] = float(C[degree == d].sum())
    lin = sorted((delays[0], c) for (d, delays), c in zip(table, C) if d == 1)
    out["memory"] = np.array([c for _, c in lin])
    return out


def ipc(run: Run, S: float, delays=(20, 10, 7), gap: int = 50, memory_tol: float = 0.01):
    """Capacities of all targets for a simulated run. Returns (C, table).

    Warns if the degree-1 capacity at the largest delay exceeds ``memory_tol``: the reservoir then
    remembers beyond the target set, and the summed capacity underestimates the true total."""
    Y, table = legendre_targets(run.u, run.T_wash, delays)
    C = capacities(run.x, run.cov, Y, S, gap)
    last = summarize(C, table)["memory"][-1]
    if last > memory_tol:
        warnings.warn(f"degree-1 capacity at the largest delay {delays[0]} is {last:.3f} > {memory_tol}: "
                      "the memory extends beyond the targets; increase delays", UserWarning, stacklevel=2)
    return C, table
