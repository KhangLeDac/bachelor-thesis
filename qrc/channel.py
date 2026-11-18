"""One reservoir step: unitary evolution, weak measurement of every spin, reset of every spin.

Weak (dephasing) measurement of strength lam = cos(theta), outcome s = +1 or -1:
    Omega_+ = cos(theta/2) |0><0| + sin(theta/2) |1><1|
    Omega_- = sin(theta/2) |0><0| + cos(theta/2) |1><1|
lam = 1 is a projective Z measurement, lam = 0 reveals nothing. One shot gives <s> = lam <Z>.

Reset of strength p, outcome not observed:
    E_0 = |0><0| + sqrt(1 - p) |1><1|,   E_1 = sqrt(p) |0><1|
p = 1 resets the spin to |0>, p = 0 does nothing.

Averaged over the measurement outcomes, the measurement followed by the reset maps the
2x2 block of one spin as
    r00 -> r00 + p r11,   r11 -> (1 - p) r11,   r01, r10 -> sqrt(1 - p) sqrt(1 - lam^2) r01, r10.
"""

import numpy as np


def check_strengths(lam: float, p: float) -> None:
    """Raise ValueError unless lam and p are finite numbers in [0, 1]."""
    for name, value in (("lam", lam), ("p", p)):
        if not (np.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def kraus_measure(lam: float) -> list[np.ndarray]:
    """Kraus operators (Omega_+, Omega_-) of the weak measurement of one spin."""
    theta = np.arccos(lam)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return [np.diag([c, s]), np.diag([s, c])]


def kraus_reset(p: float) -> list[np.ndarray]:
    """Kraus operators (E_0, E_1) of the reset of one spin."""
    return [np.diag([1.0, np.sqrt(1.0 - p)]), np.array([[0.0, np.sqrt(p)], [0.0, 0.0]])]


def measure_reset_spin(rho: np.ndarray, spin: int, n: int, lam: float, p: float) -> np.ndarray:
    """Averaged weak measurement, then reset, of one spin. ``rho`` has shape (..., 2**n, 2**n)."""
    a, b = 2**spin, 2 ** (n - spin - 1)
    r = np.asarray(rho).reshape(rho.shape[:-2] + (a, 2, b, a, 2, b))
    out = np.empty_like(r)
    coh = np.sqrt(1.0 - p) * np.sqrt(1.0 - lam**2)
    r11 = r[..., :, 1, :, :, 1, :]
    out[..., :, 0, :, :, 0, :] = r[..., :, 0, :, :, 0, :] + p * r11
    out[..., :, 1, :, :, 1, :] = (1.0 - p) * r11
    out[..., :, 0, :, :, 1, :] = coh * r[..., :, 0, :, :, 1, :]
    out[..., :, 1, :, :, 0, :] = coh * r[..., :, 1, :, :, 0, :]
    return out.reshape(rho.shape)


def measure_reset(rho: np.ndarray, n: int, lam: float, p: float) -> np.ndarray:
    """Averaged weak measurement, then reset, of every spin. Raises ValueError unless lam, p are in [0, 1]."""
    check_strengths(lam, p)
    for spin in range(n):
        rho = measure_reset_spin(rho, spin, n, lam, p)
    return rho


def step(rho: np.ndarray, U: np.ndarray, n: int, lam: float, p: float) -> np.ndarray:
    """One full step averaged over the measurement outcomes: U rho U^dagger, then measure and reset."""
    return measure_reset(U @ rho @ U.conj().T, n, lam, p)
