---
title: Legendre basis for the reset response g(u)
status: adopted
date: 2026-11-25
revised: 2026-11-26
relates: [fair-comparison-mean-reset]
code: [qrc/reset.py::Reset, qrc/reset.py::_raw_shape, rq2/configs/phaseA_legendre.toml, rq2/configs/phaseA_legendre_grid.toml]
diary: [25.11]
---

## Idea
Write the input-dependent reset as $f(u) = p^*_S + a\,g(u)$ and expand the shape $g$ in normalised
Legendre polynomials. The hand-picked families (linear, quadratic, magnitude, sigmoid) become points in
one systematic space, and the same basis is used by the IPC targets.

## Formulation
Orthonormal basis for $u \sim \mathcal U(-1,1)$:
$$e_n(u) = \sqrt{2n+1}\,P_n(u), \qquad \langle e_n e_m\rangle_u = \delta_{nm}.$$
Shape and response:
$$g(u) = \sum_{n\ge1} c_n\, e_n(u), \qquad f(u) = p^*_S + a\, g(u).$$
- $n \ge 1$ only: $\langle e_n\rangle = 0$, so $\langle p_t\rangle = p^*_S$ for every $a$, $c$.
- $\langle g^2\rangle = \sum_n c_n^2$; `Reset` normalises $g$ anyway, so $a$ is the standard deviation
  of $p_t$ and only the *direction* of $c$ matters.
- Odd $n$: response to the sign of $u$; even $n$: response to its size ($\lvert u\rvert - \tfrac12$ is
  mostly $e_2$).

Two-mode scan (phase A): $(c_1, c_2) = (\cos\theta, \sin\theta)$, $\theta\in[0,\pi)$; the half circle
$[\pi, 2\pi)$ is $-g$ and is covered by the sign of $a = \mathrm{frac}\cdot a_{\max}$,
$\mathrm{frac}\in[-1,1]$. $\theta = 0$ linear, $\pi/2$ quadratic, $\pi/4$ equal mixture.
Result: heatmap of $\Delta C(\theta, \mathrm{frac})$ for every $S$.

## Why
- Contains all earlier families; avoids optimising ad-hoc shapes one by one.
- Zero mean is built in, so the fairness condition holds automatically
  ([fair-comparison-mean-reset](fair-comparison-mean-reset.md)).
- Gives a *response spectrum*: $\partial C/\partial a_n$ at $a = 0$ shows which kind of response helps
  before optimising shapes.
- Matches the Legendre product basis of the IPC targets (`qrc/ipc.py`), so response and capacity are
  described in the same language.

## Alternatives considered
- Separate families (linear, magnitude, sigmoid) only — not systematic, overlapping, no notion of
  "direction" in function space.
- Monomials $u^n$ centred by hand — not orthogonal, so coefficients mix and normalisation is awkward.
- Scanning $(c_1, c_2)$ on a square grid — redundant, since only the direction matters after
  normalisation; the circle parametrisation halves the work via the sign of $a$.

## Open questions / next steps
- Extend to $n \ge 3$ (first $P_3$ run exists) once the $(\theta, \mathrm{frac})$ map is understood.
- Use the coefficients $a_n$ as the parameters of an adaptive rule in phase C (gradient on a capacity
  proxy, see `rq2/PLAN.md` §4).
