# RQ2: input-dependent dissipation

**Question.** Can the IPC at finite shot number $S$ be increased by making the reset strength depend on the
input, $p_t = f(u_t)$, instead of using the best constant $p^*_S$ from RQ1? If yes: what is the optimal
*response* of the dissipation to the input, and can it be turned into a control rule that does not need
a hand-picked $f$?

Plan in three phases:

| phase | question | output |
|---|---|---|
| A | which fixed functions $f(u)$ beat the constant $p^*_S$, and by how much? | $\Delta C$ per function family, $S$, device |
| B | what is the common behaviour of the good functions? | a qualitative rule ("reset more when …") |
| C | can that behaviour be produced by a rule that adapts by itself? | a control rule (ODE / feedback / adaptive) |

---

## 0. Baseline and conventions

- **Baseline:** for every $S$ the RQ1 optimum $(\lambda^*_S, p^*_S)$ (mean over devices, from
  `rq1/analyze.py`). Every result of RQ2 is reported as the gain over that baseline,
  $$\Delta C = C[f] - C[p^*_S],$$
  computed on the **same device and the same input sequence** (paired comparison). Both runs share the
  same randomness, so the noise of $\Delta C$ is much smaller than the noise of $C$ itself.
- **λ:** kept at $\lambda^*_S$ in phase A, so only the dissipation changes. Re-optimising $\lambda$ for
  the best $f$ is a later check.
- **Timing:** step $t$ is $\sigma_t = U(u_t)\rho_{t-1}U(u_t)^\dagger$, then the features are read from
  $\sigma_t$, then measure and reset with $p_t = f(u_t)$. The reset therefore uses the *current* input
  and first acts on the features of step $t+1$.
- **Range:** $f$ must map $[-1, 1] \to [0, 1]$. Parameters that leave this range are excluded (not
  clipped), so every tested function is exactly the formula it claims to be.
- **Fair comparison: same mean dissipation.** Every family is centred so that
  $\langle p_t \rangle_u = p^*_S$ for $u \sim \mathcal{U}(-1, 1)$. A gain then comes from *how* the
  reset is distributed over the inputs, not from more or less reset on average.
- **Targets:** degrees 1–3, `delays = (20, 10, 7)`, as in RQ1 (decided 23.11). The total is then a
  lower bound: an input-dependent reset makes the reservoir more nonlinear, and capacity in degrees ≥ 4
  is not counted (see 19.11: at ε = 0.5, degrees 4–5 held 1.2 of 5.9; at the default ε = 0.1 less).

## 1. Code changes (before phase A)

**Done**, details in [PHASE_A.md](PHASE_A.md):

| file | content | status |
|---|---|---|
| `qrc/channel.py` | `check_strengths` accepts arrays | done |
| `qrc/reservoir.py` | `simulate` / `trace_distance_curve` take a constant `p` or a function $f$; `Run.p_seq` | done |
| `qrc/reset.py` | `Reset`: $f(u) = p_0 + a\,g(u)$ with $\langle g\rangle = 0$, $\langle g^2\rangle = 1$, so $a$ = std of $p_t$ (the TOML cannot hold Python code, and lambdas cannot be sent to worker processes). Shapes `constant`, `linear`, `magnitude`, `quadratic`, `sigmoid` ($k$, $u_0$), `legendre` (`coefs` = direction $c_1, c_2, \ldots$ in the basis $e_n = \sqrt{2n+1}\,P_n$, any scale). `Reset.from_fraction`: $a = \text{frac}\cdot a_{\max}$, $\text{frac} \in [-1, 1]$ | done |
| `rq2/scan.py` | job system as in `rq1/map.py`: one job = one (baseline $S$, shape, frac, device), saved to its own `.npz` (total IPC, degree split, memory function, mean and std of $p_t$, washout distance); frac $= 0$ (constant $p^*_S$) always runs | done |
| `rq2/configs/` | job files `phaseA.toml`, `phaseA_linquad.toml`, `phaseA_legendre.toml`, `phaseA_legendre_grid.toml` (see §2) | done |
| `rq2/compare.py` | paired $\Delta C$ per $S$, shape and frac (sem over devices, devices with $\Delta C > 0$, worst/best device, sign test); figures IPC vs $S$, $\Delta C$ vs frac, $\Delta C$ vs $S$, degree split at the best frac | done |
| `rq2/legendre_grid.py` | writes `phaseA_legendre_grid.toml` (polar grid in the $(e_1, e_2)$ plane) | done |
| `rq2/legendre_map.py` | heat maps of $\Delta C$ over the $(e_1, e_2)$ plane, one per $S$ | done |
| `rq2/plot_shapes.py` | plots of all shapes $g(u)$ and $f(u)$ | done |

## 2. Phase A: fixed response functions

All families are written around the baseline $p^*_S$ as $f(u) = p^*_S + a\,g(u)$ with
$\langle g\rangle = 0$, $\langle g^2\rangle = 1$; $a$ is the std of $p_t$, $a = 0$ is the RQ1 baseline.
The strength is set by $a = \text{frac}\cdot a_{\max}$: $\text{frac} = \pm 1$ is the edge of the
allowed range ($p$ touches 0 or 1).

| family | $g(u)$ | idea | run in |
|---|---|---|---|
| linear (odd) | $\sqrt{3}\,u$ | reset depends on the sign of the input | `phaseA`, `phaseA_linquad` |
| magnitude (even) | $\sqrt{12}\,(\lvert u\rvert - \tfrac12)$ | reset depends on the size of the input | `phaseA` |
| quadratic (even) | $\tfrac{3\sqrt5}{2}\,(u^2 - \tfrac13)$ | smooth version of the magnitude rule | `phaseA_linquad` |
| $P_3$ (odd) | $\sqrt{7}\,P_3(u)$ | odd response of order 3 | `phaseA` |
| sigmoid | $\sigma(k(\lvert u\rvert - u_0))$, centred and normalised | on/off reset above a threshold | not run |
| Legendre | $\sum_{n\ge1} c_n e_n(u)$, $e_n = \sqrt{2n+1}\,P_n$, $\sum_n c_n^2 = 1$ | general: contains the families above | `phaseA_legendre`, `phaseA_legendre_grid` |

The Legendre form is the systematic one: $P_n$ with $n \ge 1$ has zero mean, so every term $b_n e_n$ keeps
$\langle p \rangle = p^*_S$; odd $n$ are "sign" responses, even $n$ "magnitude" responses
($\lvert u\rvert - \frac12$ is mostly $P_2$). It uses the same basis as the IPC targets. With $n \le 2$,
$g$ is normalised, so only the direction of $(c_1, c_2)$ matters:
$(c_1, c_2) = (\cos\theta, \sin\theta)$, $\theta \in [0, \pi)$; the other half circle is the sign of
frac. $\theta = 0$ is linear, $\theta = \pi/2$ quadratic. In absolute coefficients
$p(u) = p^*_S + \sum_n b_n e_n(u)$ with $b_n = a\,c_n$ and $\lVert b\rVert = \lvert a\rvert = \operatorname{std}(p_t)$. See
[legendre-basis-g](../report/ideas/legendre-basis-g.md).

**Common settings:** $\lambda = \lambda^*_S$, devices: seeds 0–9, $M = 50000$, $T_\text{wash} = 1000$,
`delays = (20, 10, 7)`, one input sequence for all jobs. Baselines (RQ1 optimum, `rq1/results/map`):

| $S$ | $10^2$ | $10^3$ | $10^4$ | $10^5$ | $10^6$ | $\infty$ |
|---|---|---|---|---|---|---|
| $(\lambda^*_S, p^*_S)$ | (1.0, 0.95) | (1.0, 0.95) | (1.0, 0.80) | (0.8, 0.70) | (0.4, 0.25) | (0.7, 0.85) |

$S \le 10^3$ sit at the grid edge ($p^*_S = 0.95$): only weak responses fit below $p = 1$.

**Job files:**

| job file | shapes | $S$ | frac | status |
|---|---|---|---|---|
| `phaseA.toml` | linear, magnitude, $P_3$ | all 6 | $\pm 0.25, \pm 0.5, \pm 0.75, \pm 1$ | run |
| `phaseA_linquad.toml` | linear, quadratic | all 6 | $\pm 0.25, \pm 0.5, \pm 0.75, \pm 1$ | run |
| `phaseA_legendre.toml` | Legendre $\theta = \pi/4$: $(c_1, c_2) = (1/\sqrt2, 1/\sqrt2)$ | all 6 | $\pm 0.25, \pm 0.5, \pm 0.75, \pm 1$ | run |
| `phaseA_legendre_grid.toml` | Legendre, $\theta = 0°, 15°, \ldots, 165°$ (12 directions) | $10^4, 10^5, 10^6$ | $\pm 0.05, \pm 0.1, \pm 0.25, \pm 0.5, \pm 0.75, \pm 1$ | run, heat maps made, not yet discussed |

**First results** (linear response, `phaseA_linquad`; details in the diary, entry 25.11,
[implementation.tex](../report/diary/content/implementation.tex), `sec:rq2-phaseA-results`):
- gain $\Delta C > 0$ for every finite $S$, largest at $S = 10^5$: $\Delta C = 1.54 \pm 0.12$ at
  frac $= -1$;
- small loss at $S = \infty$: $\Delta C = -0.07 \pm 0.01$;
- the gain is roughly symmetric in frac and grows with $\lvert\text{frac}\rvert$, so the best frac is at
  the edge $\pm 1$.

**Results to look at:**
- $\Delta C(\text{frac})$ for each family and $S$: is there a gain, for which sign of $a$, and does it
  grow or shrink with $S$? (`rq2/compare.py`; done for `phaseA`, `phaseA_linquad`, `phaseA_legendre`)
- **Response spectrum:** which directions in the $(e_1, e_2)$ plane help ($\partial\Delta C/\partial b_n$ at
  $b = 0$ and $\Delta C(b_1, b_2)$), before optimising shapes
  (`rq2/legendre_map.py` on `phaseA_legendre_grid`; heat maps made, interpretation open).
- Degree split and memory function of the best $f$: does the gain come from more memory (degree 1 at
  larger delays) or more nonlinearity (degrees ≥ 2)? Degree split in `rq2/compare.py`; memory function
  saved but not yet plotted.
- Regret between devices: is the best $a$ the same for every device? (`rq2/compare.py` gives the
  per-device spread of $\Delta C$ and the sign test, not yet the best $a$ per device.)

## 3. Phase B: from functions to a behaviour

Goal: one sentence like *"reset strongly after large inputs, weakly after small ones, and more so at
small $S$"*. Questions to answer from phase A:

1. Odd or even response? (sign of $u$ vs size of $u$)
2. Direction: more reset for large $\lvert u\rvert$ or for small?
3. How does the optimal response depend on $S$, i.e. on the noise level?
4. Is it the same for every device? If not, which device property predicts it (memory time from
   `rq1/memory`, degree split at the baseline)?

Possible mechanisms to test against the data:
- the reset multiplies the stored state by $(1-p_t)$, so an input-dependent $p_t$ multiplies past memory
  by a function of $u_t$ → creates products of the current and past inputs (higher degrees);
- a strong reset after an "informative" input clears old information to make room, a weak reset keeps
  it longer → trade-off between memory and nonlinearity that the reservoir can adapt to.

## 4. Phase C: towards an automatic control rule

$f(u)$ is chosen by hand and probably system dependent. The aim is a rule that produces a good
dissipation from what the system sees, without fixing $f$ beforehand. Candidates, from simple to
ambitious:

1. **Input filter (discrete ODE).** An internal variable follows the input,
   $$z_t = (1-\alpha)\,z_{t-1} + \alpha\, g(u_t), \qquad p_t = h(z_t),$$
   i.e. $\dot z = -\alpha (z - g(u))$. The dissipation then reacts to recent input statistics (e.g. the
   recent size $g(u) = \lvert u\rvert$) and not only to $u_t$. Phase A is the limit $\alpha = 1$.
2. **Measurement feedback.** $p_t$ depends on the measured outcomes or the estimated features
   $\hat x_t$, e.g. reset more when the spins are far from $\lvert 0\rangle$. Natural for a monitored
   reservoir and system independent (uses only observed data). **Caveat:** if $p_t$ depends on the
   *outcome of a single shot*, the outcome-averaged $\rho$ can no longer be propagated, and trajectories
   must be simulated, which is much more expensive. Depending on the outcome *averaged* over $S$ shots
   (i.e. on $\hat x_t$) keeps the current simulation approximately valid.
3. **Adaptive parameters.** Keep a family (e.g. the Legendre coefficients $b_n$) and adapt them online
   with a gradient step on a task-independent proxy of the capacity that can be estimated from the
   features alone, e.g. $\operatorname{Tr}\big((G + V/S)^{-1} G\big)$ or the expressive capacity /
   eigentasks of "Tackling sampling noise …":
   $$b \leftarrow b + \eta\, \nabla_b\, \widehat{\text{proxy}}.$$
   This is the version that does not need a hand-picked $f$ at all.

Phase B decides which of these is worth building: if the good responses depend only on $u_t$, option 1
with $\alpha = 1$ is already enough; if they depend on the noise level or the device, options 2–3.

## 5. Pitfalls

- **Small gains vs estimation noise:** $M = 50000$ and paired comparison; report $\Delta C$ with the
  standard error over devices.
- **Truncated targets:** check that the degree-5 and largest-delay capacities are small for the best
  $f$, otherwise extend `delays`.
- **Washout:** an input-dependent $p_t$ can have phases of weak reset → check the washout distance as in
  RQ1.
- **Mean dissipation:** keep $\langle p\rangle = p^*_S$ in phase A; only later allow the mean to move.

## 6. Open questions (for Tobi)

- Is $\langle p\rangle = p^*_S$ the right fairness condition, or should the mean of $f$ also be free
  (compared against the best constant $p^*_S$ either way)?
- Is measurement feedback (phase C, option 2) in the scope of the thesis, given the trajectory cost?
- Should λ also become input dependent later, or stay fixed?
