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
  `rq1_analyze.py`). Every result of RQ2 is reported as the gain over that baseline,
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

**Done**, details in [IMPLEMENTATION.md](IMPLEMENTATION.md):

| file | content | status |
|---|---|---|
| `qrc/channel.py` | `check_strengths` accepts arrays | done |
| `qrc/reservoir.py` | `simulate` / `trace_distance_curve` take a constant `p` or a function $f$; `Run.p_seq` | done |
| `qrc/reset.py` | `Reset`: $f(u) = p_0 + a\,g(u)$ with $\langle g\rangle = 0$, $\langle g^2\rangle = 1$, so $a$ = std of $p_t$ (the TOML cannot hold Python code, and lambdas cannot be sent to worker processes) | done |
| `rq2/rq2_scan.py` | job system as in `rq1_map.py`: one job = one (baseline $S$, shape, frac, device), saved to its own `.npz` | done |
| `rq2/configs/phaseA.toml` | first job file: linear and magnitude, RQ1 optima as baseline | done |
| `rq2/rq2_analyze.py` | $\Delta C$ vs parameters, degree split and memory function of the best $f$ | after first results |

## 2. Phase A: fixed response functions

All families written around the baseline $p^*_S$ with one strength parameter $a$ (and possibly a shape
parameter); $a = 0$ is the RQ1 baseline.

| family | $f(u)$ | mean over $u$ | idea |
|---|---|---|---|
| linear (odd) | $p^*_S + a\,u$ | $p^*_S$ | reset depends on the sign of the input |
| magnitude (even) | $p^*_S + a\,(\lvert u\rvert - \tfrac12)$ | $p^*_S$ | reset depends on the size of the input |
| quadratic (even) | $p^*_S + a\,(u^2 - \tfrac13)$ | $p^*_S$ | smooth version of the magnitude rule |
| step / sigmoid | $p^*_S + a\,\big(\sigma(k(\lvert u\rvert - u_0)) - c\big)$ | $p^*_S$ ($c$ chosen) | on/off reset above a threshold |
| Legendre expansion | $p^*_S + \sum_{n\ge1} a_n \sqrt{2n+1}\,P_n(u)$ | $p^*_S$ | general: contains the families above |

The Legendre form is the systematic one: $P_n$ with $n \ge 1$ has zero mean, so every coefficient
$a_n$ keeps $\langle p \rangle = p^*_S$; odd $n$ are "sign" responses, even $n$ "magnitude"
responses ($\lvert u\rvert - \frac12$ is mostly $P_2$). It uses the same basis as the IPC targets.

**Scan (first job file):**
- $S \in \{10^2, 10^3, 10^4, 10^5, \infty\}$, $\lambda = \lambda^*_S$, devices: seeds 0–9.
- linear and magnitude: $a$ on a grid over the full allowed range (both signs).
- record: total IPC, degree split, memory function, and the mean and spread of $p_t$.

**Results to look at:**
- $\Delta C(a)$ for each family and $S$: is there a gain, for which sign of $a$, and does it grow or
  shrink with $S$?
- **Response spectrum:** the slope $\partial C / \partial a_n$ at $a = 0$ for $n = 1, 2, 3, \ldots$
  Which kinds of response help at all, before optimising shapes.
- Degree split and memory function of the best $f$: does the gain come from more memory (degree 1 at
  larger delays) or more nonlinearity (degrees ≥ 2)?
- Regret between devices: is the best $a$ the same for every device?

## 3. Phase B: from functions to a behaviour

Goal: one sentence like *"reset strongly after large inputs, weakly after small ones, and more so at
small $S$"*. Questions to answer from phase A:

1. Odd or even response? (sign of $u$ vs size of $u$)
2. Direction: more reset for large $\lvert u\rvert$ or for small?
3. How does the optimal response depend on $S$, i.e. on the noise level?
4. Is it the same for every device? If not, which device property predicts it (memory time from
   `rq1_memory`, degree split at the baseline)?

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
3. **Adaptive parameters.** Keep a family (e.g. the Legendre coefficients $a_n$) and adapt them online
   with a gradient step on a task-independent proxy of the capacity that can be estimated from the
   features alone, e.g. $\operatorname{Tr}\big((G + V/S)^{-1} G\big)$ or the expressive capacity /
   eigentasks of "Tackling sampling noise …":
   $$a \leftarrow a + \eta\, \nabla_a\, \widehat{\text{proxy}}.$$
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
