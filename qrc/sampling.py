"""Finite-shot simulation: draw S shots per step and train a linear readout on the noisy features.

This is a cross-check of the formula C(S) in ``qrc.ipc``: with many samples both agree.
"""

import numpy as np

from .model import z_signs


def outcome_distribution(pop: np.ndarray, n: int, lam: float) -> np.ndarray:
    """Joint distribution of the n outcomes of one shot, shape (..., 2**n), from populations (..., 2**n).

    P(s) = sum_z pop(z) prod_i (1 + lam s_i z_i) / 2; outcome index bit i = 0 means s_i = +1
    (same ordering as the basis states)."""
    flip = np.array([[1 + lam, 1 - lam], [1 - lam, 1 + lam]]) / 2
    lead = pop.shape[:-1]
    P = pop.reshape(lead + (2,) * n)
    for i in range(n):
        P = np.moveaxis(np.tensordot(P, flip, axes=([len(lead) + i], [1])), -1, len(lead) + i)
    return P.reshape(pop.shape)


def noisy_features(pop: np.ndarray, n: int, lam: float, S: int, rng: np.random.Generator) -> np.ndarray:
    """(M, n) features estimated from S shots per step: the mean of the sampled outcomes s."""
    P = np.clip(outcome_distribution(pop, n, lam), 0.0, None)
    counts = rng.multinomial(int(S), P / P.sum(axis=1, keepdims=True))
    return counts @ z_signs(n).T / S


def readout_capacity(x_hat: np.ndarray, Y: np.ndarray, train_frac: float = 0.5, gap: int = 50) -> np.ndarray:
    """Capacity 1 - NMSE per target of a least-squares readout (with constant) trained on the first
    ``train_frac`` of the samples and tested on the rest (``gap`` samples skipped in between)."""
    M = len(x_hat)
    h = np.column_stack([np.ones(M), x_hat])
    n_train = int(train_frac * M)
    W, *_ = np.linalg.lstsq(h[:n_train], Y[:n_train], rcond=None)
    test = slice(n_train + gap, M)
    err = Y[test] - h[test] @ W
    return 1.0 - (err**2).mean(axis=0) / Y[test].var(axis=0)
