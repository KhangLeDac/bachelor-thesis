---
title: Legendre basis for the reset response g(u)
status: adopted
date: 2026-11-25
revised: 2026-11-26
relates: [fair-comparison-mean-reset]
code: [qrc/reset.py::Reset, qrc/reset.py::_raw_shape, rq2/configs/phaseA_legendre.toml, rq2/configs/phaseA_legendre_grid.toml, rq2/legendre_map.py::grid_points, rq2/plot_shapes.py::plot_best, rq2/plot_shapes.py::plot_fixed]
diary: [25.11, 26.11]
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

*Coverage (26.11).* The grid $\theta\in[0,\pi)\times\mathrm{frac}\in[-1,1]$ covers every allowed response
in $\mathrm{span}\{e_1,e_2\}$ (up to resolution): the norm of $c$ is absorbed into $a$, $(\theta+\pi, a)
\equiv(\theta,-a)$, and `from_fraction` maps $\mathrm{frac}$ one-to-one onto $[0,\pm a_{\max}(\theta,\pm)]$.
Not covered: $e_{n\ge3}$, and an $e_0$ shift (excluded by the fairness condition). The map wraps with a
flip, $\Delta C(\pi^-,\mathrm{frac})\to\Delta C(0,-\mathrm{frac})$.

*Feasible set in absolute coefficients (26.11).* With $b_n = a\,c_n$, $p(u) = p^*_S + \sum_n b_n e_n(u)$
and $\lVert b\rVert = \operatorname{std}(p_t)$ (orthonormality). The allowed set
$$\mathcal F = \{b : 0 \le p^*_S + b\cdot e(u) \le 1\ \ \forall u\in[-1,1]\}$$
is an intersection of half-planes: convex, but not a disc. The constraint is a sup-norm, the metric an
$L^2$ norm:
- the extremes of $g$ depend on the direction: $e_1\in[-1.73, 1.73]$, $e_2\in[-1.12, 2.24]$;
- $p^*_S \ne \tfrac12$: room $1-p^*$ above, $p^*$ below (e.g. $S=10^4$, $p^*=0.8$, $\theta=90°$:
  $a_{\max}(+)\approx0.089$, $a_{\max}(-)\approx0.179$);
- the boundary is the envelope of the lines from the parabola arc
  $e(u) = (\sqrt3u, \tfrac{\sqrt5}{2}(3u^2-1))$: curved pieces plus edges and corners from $u=\pm1, 0$.

$\mathrm{frac} = \lVert b\rVert/\lVert b\rVert_{\max}(\theta,\pm)$ rescales each ray separately and maps
$\mathcal F$ onto the unit disc. Coverage is complete, but equal frac $\ne$ equal std of $p_t$ across
$\theta$, sign and $S$.

*Grid results (26.11, `phaseA_legendre_grid.toml`, 10 devices, paired $\Delta C$ ± sem).*

| $S$ | $p^*_S$ | best $(b_1, b_2)$ | $\Delta C$ |
|---|---|---|---|
| $10^4$ | 0.8 | $(\pm0.194, -0.112)$, $\theta=\mp30°$ | $+1.20\pm0.13$ |
| $10^5$ | 0.7 | $(\pm0.265, -0.071)$, $\theta=\mp15°$ | $+1.94\pm0.09$ |
| $10^6$ | 0.25 | $(-0.220, +0.059)$, $\theta=165°$ | $+1.12\pm0.09$ |

- Mostly $e_1$ (odd); $\Delta C$ nearly unchanged under $b_1\to-b_1$. The sign of $b_2$ follows $p^*_S$:
  the tilt points to the side where $\mathcal F$ is larger.
- **Optimum on the boundary:** all best points at $\lvert\mathrm{frac}\rvert = 1$ ($S=10^6$: the 0.75
  point equal within errors) → the $[0,1]$ constraint is active. Headroom (Bhatia–Davis): for
  $p_t\in[0,1]$ with mean $p^*$, $\operatorname{Var}(p_t)\le p^*(1-p^*)$, attained by $p_t\in\{0,1\}$;
  $\operatorname{std}\le0.40/0.46/0.43$ vs $0.22/0.27/0.23$ reached → step-like shapes can modulate
  about twice as strongly.
- **$\Delta C$ (essentially) never negative:** 125/136/122 of 144 points have $\Delta C>2\,$sem; the
  4–9 negative points per $S$ lie at $\lvert\mathrm{frac}\rvert\le0.1$, within 1–2 sem of zero.
  Hypotheses:
  - reset as a second, stronger input channel: in $H(u)$ the encoding is weak ($\varepsilon u = \pm0.1$
    vs $h_x = 0.5$), while $\operatorname{std}(p_t)\sim0.2$ writes $u_t$ directly into the state; also
    explains the $e_1$ dominance ($e_1$ keeps the sign of $u$, $e_2$ loses it) and the approximate
    $b_1\to-b_1$ symmetry;
  - shot noise: modulation increases the input-driven variance of $\langle Z_i\rangle$, strong resets
    push towards $\lvert0\rangle$ (less noise); fits the small loss at $S=\infty$ in the linquad run
    ($-0.07\pm0.01$); the grid has no $S\le10^3$ or $S=\infty$;
  - if $\Delta C$ is nearly even in $b_1$, it starts as $\kappa b_1^2$ → no first-order loss for
    either sign;
  - not bias: $\lambda^*_S$ and the missing degrees $\ge4$ work *against* $f$; estimator bias is shared
    by the paired runs; only the capacity threshold is unchecked.

*Best shapes are ramps between the extremes (26.11, `rq2/plot_shapes.py --best / --fixed`).* The best
responses are (nearly) monotone in $u$ and try to span $[0,1]$: one end of the input range gives low,
the other high reset, $u=-1 \mapsto p\approx0$, $u=+1 \mapsto p\approx1$ or the other way round (the sign
of $b_1$ is free, see the mirror symmetry above).

| $S$ | $p^*_S$ | best: $f(-1) \to f(+1)$, range | linear, frac $=\pm1$: $f(-1)\to f(+1)$, range |
|---|---|---|---|
| $10^4$ | 0.8 | $0.21 \to 0.89$ (max $1.0$), $0.79$ | $0.6 \to 1.0$, $0.4$ |
| $10^5$ | 0.7 | $0.08 \to 1.0$, $0.92$ | $0.4 \to 1.0$, $0.6$ |
| $10^6$ | 0.25 | $0.76 \to 0.0$, $0.76$ | $0.5 \to 0.0$, $0.5$ |

- With $\langle p_t\rangle = p^*_S \ne \tfrac12$ a *straight* ramp cannot reach both 0 and 1; it hits the
  nearer boundary first and uses only part of the range (linear: std $0.115/0.173/0.144$,
  $\Delta C = 0.85/1.54/0.72$).
- The $e_2$ part bends the ramp (concave for large $p^*$, convex for small $p^*$) so that it covers much
  more of $[0,1]$ at the same mean: std $0.224/0.274/0.228$, $\Delta C$ higher by $\approx0.4$ at every
  $S$. The $e_2$ tilt is therefore mainly a way to fit a longer ramp into $[0,1]$, not an independent
  "magnitude" response.
- Rule for phase B: *map the input monotonically onto the full reset range, low $u$ → one extreme,
  high $u$ → the other, at mean $p^*_S$.* The limit of this is the step $p_t = \Theta(\pm(u_t - u_0))$
  with $P(\pm(u - u_0) > 0) = p^*_S$, which saturates $\operatorname{Var}(p_t) = p^*(1-p^*)$.

## Why
- Contains all earlier families; avoids optimising ad-hoc shapes one by one.
- Zero mean is built in, so the fairness condition holds automatically
  ([fair-comparison-mean-reset](fair-comparison-mean-reset.md)).
- Gives a *response spectrum*: $\partial\Delta C/\partial b_n$ at $b = 0$ shows which kind of response helps
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
- Compare directions at equal $a$ (slope $\partial\Delta C/\partial a$ from the $\pm0.05, \pm0.1$
  points); the absolute-coordinate heat maps exist (`rq2/legendre_map.py`, left panel).
- Control: $p_t$ with the same distribution as $f(u_t)$ but independent of $u_t$ (shuffled) — gain from
  input dependence or from modulation itself?
- Control: larger $\varepsilon$ — does the gain shrink, as the "second channel" hypothesis predicts?
- Step / two-valued responses up to $\operatorname{std} = \sqrt{p^*(1-p^*)}$, beyond the polynomial
  boundary.
- Full-range monotone ramps at mean $p^*_S$ (e.g. a nonlinear monotone map $u \mapsto p$ from 0 to 1,
  or a clipped linear ramp), interpolating between the bent Legendre ramp and the step: does
  $\Delta C$ keep growing with the covered range / std, or is there an optimum before the step?
- Degree split and memory function of the best points; add $S=10^3$ and $S=\infty$ to the grid.
- Does the capacity threshold favour the modulated reservoir?
- Use the coefficients $b_n$ as the parameters of an adaptive rule in phase C (gradient on a capacity
  proxy, see `rq2/PLAN.md` §4).
