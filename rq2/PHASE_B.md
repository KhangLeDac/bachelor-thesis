# RQ2 phase B: reset with input memory

**Question.** Does the reset strength gain from depending on *past* inputs as well,
$p_t = f(u_t, u_{t-1}, \dots, u_{t-d})$, compared with the memoryless responses $p_t = f(u_t)$ of
phase A? Which combination of current and past inputs is best, and does a rule follow from it?

Plan: implementation → test the baseline shape with the current metrics → general formalism (Legendre
product basis) → derive a rule. Phase A: [PHASE_A.md](PHASE_A.md), overview: [PLAN.md](PLAN.md).

---

## 1. Function space

The inputs are i.i.d. $u_t \sim \mathcal U(-1,1)$, so products of normalised Legendre polynomials of
different delays are orthonormal for the joint input distribution — the same basis as the IPC targets:
$$\phi_{\mathbf n}(u_t,\dots,u_{t-d}) = \prod_{k=0}^{d} e_{n_k}(u_{t-k}), \qquad
e_n(u) = \sqrt{2n+1}\,P_n(u), \qquad \langle \phi_{\mathbf n}\phi_{\mathbf m}\rangle = \delta_{\mathbf n\mathbf m}.$$
The response keeps the phase-A form and notation (strength $a$, unit direction $c$, absolute
coefficients $b = a\,c$):
$$p_t = p^*_S + a\,g(u_t,\dots,u_{t-d}) = p^*_S + \sum_{\mathbf n \ne 0} b_{\mathbf n}\,\phi_{\mathbf n},
\qquad \sum_{\mathbf n} c_{\mathbf n}^2 = 1, \qquad \lVert b\rVert = \lvert a\rvert = \operatorname{std}(p_t).$$
- $\mathbf n \ne 0$: every term has zero mean, so $\langle p_t\rangle = p^*_S$ (same fairness condition as
  phase A, [fair-comparison-mean-reset](../report/ideas/fair-comparison-mean-reset.md)).
- Phase A is the subspace $n_1 = \dots = n_d = 0$; every phase-A shape is a phase-B shape.
- Valid range: $f$ maps $[-1,1]^{d+1} \to [0,1]$; $a = \mathrm{frac}\cdot a_{\max}(c,\pm)$ as in phase A,
  with $a_{\max}$ from min/max of $g$ over the cube.
- Scope for now: degree $\le 2$, delay $d \le 1$ (later possibly $d \le 2$).

**First search space** (as the Legendre grid of phase A): the four single-delay modes
$$\{\,e_1(u_t),\ e_2(u_t),\ e_1(u_{t-1}),\ e_2(u_{t-1})\,\}.$$
The cross term $e_1(u_t)\,e_1(u_{t-1})$ is *not* in this space (see §2, $\lvert\Delta u\rvert$).
How to scan it (grid on planes, response spectrum, optimisation) is decided after §5 step 2.

## 2. Shapes

| shape | $g$ (normalised) | basis content | idea |
|---|---|---|---|
| **$\Delta u$ (baseline)** | $\Delta u/\sqrt{2/3} = \tfrac{1}{\sqrt2}\big(e_1(u_t) - e_1(u_{t-1})\big)$ | degree 1, in the search space | signed change: rising input → strong reset, falling → weak (analogue of *linear*) |
| $\lvert\Delta u\rvert$ (compared alongside) | $(\lvert\Delta u\rvert - \tfrac23)/\sqrt{2/9}$ | 72 % $-e_1(u_t)e_1(u_{t-1})$, 2 × 10 % $e_2$, rest higher degree | any strong change → strong reset (analogue of *magnitude*) |
| linear (phase A, reference shape) | $e_1(u_t)$ | degree 1, delay 0 | memoryless comparison |
| moving average | $\tfrac{1}{\sqrt2}\big(e_1(u_t) + e_1(u_{t-1})\big)$ | degree 1, in the search space | low-pass input; together with $\Delta u$ and linear on one circle |

- $\Delta u$, linear and the moving average all lie in the plane $\mathrm{span}\{e_1(u_t), e_1(u_{t-1})\}$
  ($\theta = -45°, 0°, +45°$): a phase-A-like $\theta$ grid on this plane contains all three.
- The baseline is the *first shape tested*, not the reference. If $\lvert\Delta u\rvert$ (or another shape)
  shows a clear trend, the baseline shape can be switched; the reference stays $p^*_S$.
- Other candidates for later: $u_t u_{t-1}$ (correlation, $= e_1e_1/3$), exponential moving average
  $\sum_k (1-\gamma)\gamma^k u_{t-k}$ (infinite memory, one parameter; bridge to phase C).

**Range penalty of memory shapes.** At frac $=+1$ the strength is
$a_{\max}(+) = \min\big((1-p^*)/\max g,\ p^*/(-\min g)\big) = \operatorname{std}(p_t)$ (and with $\pm g$
swapped for frac $=-1$). Shapes spread over several delays have a larger $\sup\lvert g\rvert$ and so a
smaller allowed std:

| $p^*_S$ | linear $e_1(u_t)$ ($\pm1.73$) | $\Delta u$ ($\pm2.45$) | $\lvert\Delta u\rvert$, $a>0$ / $a<0$ ($g\in[-1.41, 2.83]$) |
|---|---|---|---|
| 0.8 ($S=10^4$) | 0.115 | 0.082 | 0.071 / 0.141 |
| 0.7 ($S=10^5$) | 0.173 | 0.122 | 0.106 / 0.212 |
| 0.25 ($S=10^6$) | 0.144 | 0.102 | 0.177 / 0.088 |

Comparison is at the full allowed range (frac $=\pm1$, decided). Since phase A showed $\Delta C$ growing
with $\operatorname{std}(p_t)$, every result reports $\operatorname{std}(p_t)$ next to $\Delta C$, so that a
loss of $\Delta u$ against linear can be told apart from its smaller range.

## 3. Comparison and metrics

- **Reference:** the constant $p^*_S$ of RQ1, paired (same device, same input sequence), exactly as in
  phase A. Memoryless and memory shapes are thus measured on the same ground; phase-A results stay valid
  and comparable without re-running.
- **Metrics (unchanged):** total IPC, degrees 1–3, `delays = (20, 10, 7)`, $\Delta C$ ± sem over 10
  devices, sign test, per-device spread; plus $\operatorname{std}(p_t)$, degree split and memory function.
- **Possibly new:** capacity split by delay pattern of the targets (does a $u_{t-1}$-dependent reset move
  capacity to targets that contain $u_{t-1}$?). Decide after the first runs.
- **$S$:** $10^4, 10^5, 10^6$ as in the Legendre grid (where $\Delta C$ is significant); add $10^3$ and
  $\infty$ if the baseline behaves differently from phase A.

## 4. Implementation (to do)

- **`qrc/reset.py`:** a response with memory depth $d$ that takes the lag matrix
  $U_t = (u_t, u_{t-1}, \dots, u_{t-d})$, shape $(T, d+1)$:
  - Legendre products from a list of terms $(\mathbf n, c_{\mathbf n})$; mean and variance analytic
    (orthonormal basis, no quadrature), $a_{\max}$ from min/max of $g$ on a grid over $[-1,1]^{d+1}$;
  - special shapes `delta` ($\Delta u$) and `abs_delta` ($\lvert\Delta u\rvert$, normalised with the exact
    mean $2/3$ and variance $2/9$);
  - same interface as `Reset` (`g`, `__call__`, `a_max`, `from_fraction`, `p_range`, picklable); the
    memoryless `Reset` stays unchanged, so phase-A numbers are reproduced exactly.
- **`qrc/reservoir.py`:** `reset_sequence` builds the lag matrix when the response has depth $d > 0$.
  The first $d$ steps lie in the washout; pad $u_{-1},\dots,u_{-d}$ (e.g. with 0) — this only affects the
  washout. Timing as in phase A: $p_t$ acts after the features of step $t$ are read.
- **Job files:** shape kind for lagged Legendre terms, e.g. `terms = [[[1, 0], 0.707], [[0, 1], -0.707]]`
  ($[n_0, n_1]$, $c$), plus `delta` / `abs_delta`.
- **`rq2/scan.py`, `rq2/compare.py`:** should work unchanged if shapes are addressed by name; check.
- **Tests:** orthonormality of the products, $\langle g\rangle = 0$, $\langle g^2\rangle = 1$ for `delta` and
  `abs_delta`, depth-0 terms reproduce `Reset` exactly, $p_t \in [0,1]$.

## 5. Steps

1. Implement §4 with tests.
2. Baseline run: $\Delta u$ and $\lvert\Delta u\rvert$ at frac $\in[-1,1]$ for $S = 10^4, 10^5, 10^6$,
   10 devices; compare with linear (phase A) at the same frac and report std.
3. Decide the search space and strategy from step 2 (e.g. $\theta$ grid on
   $\mathrm{span}\{e_1(u_t), e_1(u_{t-1})\}$, then the full four-mode space, possibly adding
   $e_1(u_t)e_1(u_{t-1})$).
4. Derive the rule (which past inputs, which sign, how strong) and pass it to phase C.

## 6. Open points

- Is degree ≤ 2, delay ≤ 1 enough, or does the rule need $u_{t-2}$?
- Should the cross term $e_1(u_t)e_1(u_{t-1})$ be part of the first search space? It carries 72 % of
  $\lvert\Delta u\rvert$, which the four single-delay modes cannot represent.
- Comparing at full range favours shapes with small $\sup\lvert g\rvert$; if the conclusion depends on it,
  add a comparison at equal $\operatorname{std}(p_t)$.
