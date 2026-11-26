"""Input-dependent reset strength p_t = f(u_t) (RQ2).

Every response function has the form

    f(u) = p0 + a * g(u),        <g> = 0,  <g^2> = 1   for u ~ U(-1, 1),

where g is a fixed *shape* and a the *strength*. Because g has zero mean, the average reset strength is
<p_t> = p0 for every a (fair comparison with the constant baseline p0), and because g has unit variance,
a is the standard deviation of p_t, so strengths of different shapes are directly comparable.
For legendre, only the direction of `coefs` (c_1, c_2, ...) matters; the absolute coefficients of
p - p0 in the basis e_n are b_n = a c_n / |c|.

Shapes (raw shape h, centred and normalised numerically to g):
    linear      h = u                                   odd: reset depends on the sign of u
    magnitude   h = |u|                                 even: reset depends on the size of u
    quadratic   h = u^2                                 even, smooth
    sigmoid     h = 1 / (1 + exp(-k (|u| - u0)))        even, switch at |u| = u0 (parameters k, u0)
    legendre    h = sum_n c_n e_n(u)                    general, e_n = sqrt(2n+1) P_n (direction c_1, c_2, ...)

f must stay in [0, 1] on [-1, 1]. ``a_max`` gives the largest allowed |a| for each sign, and
``Reset.from_fraction`` picks a = frac * a_max, so frac in [-1, 1] is always valid.
``Reset`` objects are plain Python objects and can be sent to worker processes (unlike lambdas).
"""

import numpy as np
from numpy.polynomial import legendre

SHAPES = ("constant", "linear", "magnitude", "quadratic", "sigmoid", "legendre")

# Gauss-Legendre nodes on [-1, 0] and [0, 1] separately: exact for polynomials and |u| (kink at 0)
_X, _W = legendre.leggauss(200)
_NODES = np.concatenate([(_X - 1) / 2, (_X + 1) / 2])
_WEIGHTS = np.concatenate([_W, _W]) / 4          # /2 for the half intervals, /2 for the density of U(-1, 1)
_GRID = np.linspace(-1.0, 1.0, 20001)             # for min / max of g


def _raw_shape(kind: str, u: np.ndarray, shape: dict) -> np.ndarray:
    """Raw shape h(u) before centring and normalising."""
    u = np.asarray(u, dtype=float)
    if kind == "linear":
        return u
    if kind == "magnitude":
        return np.abs(u)
    if kind == "quadratic":
        return u**2
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-shape["k"] * (np.abs(u) - shape["u0"])))
    if kind == "legendre":
        c = np.asarray(shape["coefs"], dtype=float)                  # direction c_1, c_2, ... (no constant term)
        coef = np.concatenate([[0.0], c * np.sqrt(2 * np.arange(1, len(c) + 1) + 1)])
        return legendre.legval(u, coef)
    raise ValueError(f"unknown shape {kind!r}; known: {SHAPES}")


_SHAPE_KEYS = {"constant": set(), "linear": set(), "magnitude": set(), "quadratic": set(),
               "sigmoid": {"k", "u0"}, "legendre": {"coefs"}}


class Reset:
    """Response function f(u) = p0 + a g(u) with <g> = 0 and <g^2> = 1 for u ~ U(-1, 1)."""

    def __init__(self, kind: str, p0: float, a: float = 0.0, **shape):
        if kind not in _SHAPE_KEYS:
            raise ValueError(f"unknown shape {kind!r}; known: {SHAPES}")
        if set(shape) != _SHAPE_KEYS[kind]:
            raise ValueError(f"shape {kind!r} needs parameters {sorted(_SHAPE_KEYS[kind])}, got {sorted(shape)}")
        self.kind, self.p0, self.a, self.shape = kind, float(p0), float(a), dict(shape)
        if kind == "constant":
            self.mean, self.std = 0.0, 1.0
        else:
            h = _raw_shape(kind, _NODES, self.shape)
            self.mean = float(_WEIGHTS @ h)
            self.std = float(np.sqrt(_WEIGHTS @ (h - self.mean) ** 2))
            if self.std < 1e-12:
                raise ValueError(f"shape {kind!r} with {self.shape} is constant on [-1, 1]")
        lo, hi = self.p_range()
        if lo < -1e-12 or hi > 1 + 1e-12:
            raise ValueError(f"{self.describe()} leaves [0, 1]: p in [{lo:.4f}, {hi:.4f}]; "
                             f"allowed a in [-{self.a_max(-1):.4f}, {self.a_max(+1):.4f}]")

    def g(self, u) -> np.ndarray:
        """Normalised shape g(u): zero mean and unit variance for u ~ U(-1, 1)."""
        if self.kind == "constant":
            return np.zeros_like(np.asarray(u, dtype=float))
        return (_raw_shape(self.kind, u, self.shape) - self.mean) / self.std

    def __call__(self, u) -> np.ndarray:
        """p = f(u) = p0 + a g(u), clipped against rounding errors only (the range is checked in __init__)."""
        return np.clip(self.p0 + self.a * self.g(u), 0.0, 1.0)

    def a_max(self, sign: int = +1) -> float:
        """Largest |a| with the given sign of a such that p0 + a g(u) stays in [0, 1] on [-1, 1]."""
        if self.kind == "constant":
            return 0.0
        g = self.g(_GRID)
        up, down = (g.max(), -g.min()) if sign > 0 else (-g.min(), g.max())   # how far a g goes up / down
        return float(min((1 - self.p0) / up, self.p0 / down))

    @classmethod
    def from_fraction(cls, kind: str, p0: float, frac: float, **shape) -> "Reset":
        """a = frac * a_max(sign of frac): frac = +-1 uses the full allowed range, frac = 0 is the constant p0."""
        if not -1.0 <= frac <= 1.0:
            raise ValueError(f"frac must be in [-1, 1], got {frac}")
        probe = cls(kind, p0, 0.0, **shape)
        a = frac * probe.a_max(+1 if frac >= 0 else -1)
        return cls(kind, p0, a * (1 - 1e-12), **shape)          # stay inside [0, 1] despite rounding

    def p_range(self) -> tuple[float, float]:
        """(min, max) of p0 + a g(u) on [-1, 1] (unclipped)."""
        p = self.p0 + self.a * self.g(_GRID)
        return float(p.min()), float(p.max())

    def describe(self) -> str:
        extra = "".join(f", {k}={v}" for k, v in self.shape.items())
        return f"{self.kind}(p0={self.p0:g}, a={self.a:.4g}{extra})"

    def __repr__(self) -> str:
        return f"Reset<{self.describe()}>"
