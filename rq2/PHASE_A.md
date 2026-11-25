# RQ2: code changes for input-dependent reset

This file lists every code change made for RQ2 (input-dependent reset $p_t = f(u_t)$), why it was
needed, and where it enters the computation. The research plan is in [PLAN.md](PLAN.md).

Summary:

| file | change | purpose |
|---|---|---|
| `qrc/channel.py` | `check_strengths` accepts arrays | check a whole sequence $p_t$ at once |
| `qrc/reservoir.py` | `p` may be a function; new `reset_sequence`; `Run.p_seq` | use $p_t = f(u_t)$ in the simulation |
| `qrc/reset.py` (new) | class `Reset`: response functions $f(u) = p_0 + a\,g(u)$ | the families of phase A, centred and normalised |
| `rq2/scan.py` (new) | job system for phase A | scan shapes and strengths around the RQ1 optimum |
| `rq2/configs/phaseA.toml` (new) | first job file | linear and magnitude response, 6 values of $S$, 10 devices |
| `tests/` | 4 + 1 + 23 new tests | constant $f$ reproduces RQ1 exactly, shapes are normalised, range checks |

Unchanged: the physics of one step (`measure_reset_spin`), the features and shot covariance, and
the IPC (`qrc/ipc.py`). Every existing call with a constant `p` behaves exactly as before.

---

## 1. Where $p_t$ enters the computation

One step of the protocol (see diary, 13.11) with an input-dependent reset:

$$
\rho_{t-1} \;\xrightarrow{\;U(u_t)\;}\; \sigma_t = U(u_t)\,\rho_{t-1}\,U(u_t)^\dagger
\;\xrightarrow{\;\text{read}\;}\; x_i(t) = \lambda\langle Z_i\rangle_{\sigma_t}
\;\xrightarrow{\;\mathcal{M}_\lambda,\ \mathcal{R}_{p_t}\;}\; \rho_t,
\qquad p_t = f(u_t).
$$

- The features of step $t$ are read **before** the reset, so $p_t$ first changes the features of step
  $t+1$.
- Per spin, the reset with $p_t$ is the same map as in RQ1, only with a different number in every step:
  $r_{00} \to r_{00} + p_t r_{11}$, $r_{11} \to (1-p_t) r_{11}$, coherences
  $\times \sqrt{1-p_t}\sqrt{1-\lambda^2}$.
- Since $p_t$ depends only on the input (not on measurement outcomes), the outcome-averaged $\rho$ can
  still be propagated exactly; no trajectories are needed.
- Why it can help: the remaining excitation is multiplied by $(1 - p_t)$ every step, so the memory of
  past inputs is multiplied by a function of the current input. This creates products of current and
  past inputs, i.e. new nonlinear (higher-degree) capacity.

## 2. `qrc/channel.py`

**Change:** `check_strengths(lam, p)` accepts arrays as well as numbers. Every entry must be finite and
in $[0, 1]$; the error message shows the range of the offending array.

**Purpose:** `simulate` checks the whole sequence $p_1, \ldots, p_{T_{wash}+M}$ once before the loop,
so an invalid $f$ fails immediately instead of after part of the run.

**Not changed:** `measure_reset_spin`, `measure_reset` and `step` already take a scalar `p` per call.
The loop now simply passes a different scalar each step.

## 3. `qrc/reservoir.py`

**New function `reset_sequence(p, u)`:**
- `p` a number → $p_t = p$ for every step (RQ1).
- `p` a function → $p_t = f(u_t)$, evaluated once for the whole input array (`f` must be vectorised:
  array in, array of the same shape out).
- checks the shape and the range $[0, 1]$ (via `check_strengths`).

**`simulate(params, lam, p, ...)`:**
1. draws the inputs $u$ exactly as before (they do not depend on `p`),
2. `p_seq = reset_sequence(p, u)`,
3. washout loop: `step(rho, U(u_t), n, lam, p_seq[t])`,
4. main loop: features from $\sigma_t$, then `measure_reset(sigma, n, lam, p_seq[T_wash + t])`.

The washout check (ground state vs. maximally mixed state) is unchanged; both states see the same
$p_t$, so it still measures whether the initial state is forgotten.

**Paired comparison:** runs with different `p` but the same `input_seed` (default 0) use the identical
input sequence. Differences $C[f] - C[p^*]$ are therefore not blurred by different inputs.

**`trace_distance_curve(params, lam, p, u)`:** the same change, so the memory time of RQ1
(`rq1/memory.py`) can also be computed for an input-dependent reset.

**`Run`:** new field `p_seq` (shape $(T_{wash}+M,)$, aligned with `u`). `Run.p` holds what was passed
(number or function).

## 4. `qrc/reset.py` (new): the response functions

Every response function has the form

$$
f(u) = p_0 + a\,g(u), \qquad \langle g\rangle = 0,\quad \langle g^2\rangle = 1 \quad (u \sim \mathcal{U}(-1,1)).
$$

- $\langle g\rangle = 0$ → the mean reset $\langle p_t\rangle = p_0$ for every $a$: the comparison with
  the constant baseline $p_0 = p^*_S$ is at equal average dissipation.
- $\langle g^2\rangle = 1$ → $a$ is the standard deviation of $p_t$: strengths of different shapes are
  directly comparable.

**Shapes** (raw shape $h$; $g = (h - \langle h\rangle)/\operatorname{std}(h)$):

| `kind` | raw shape $h(u)$ | $g(u)$ | parameters |
|---|---|---|---|
| `constant` | – | $0$ | – |
| `linear` | $u$ | $\sqrt3\,u$ | – |
| `magnitude` | $\lvert u\rvert$ | $\sqrt{12}\,(\lvert u\rvert - \tfrac12)$ | – |
| `quadratic` | $u^2$ | $\sqrt{45/4}\,(u^2 - \tfrac13)$ | – |
| `sigmoid` | $1/(1+e^{-k(\lvert u\rvert - u_0)})$ | numerically | `k`, `u0` |
| `legendre` | $\sum_n c_n \sqrt{2n+1}\,P_n(u)$ | numerically | `coefs` $= [c_1, c_2, \ldots]$ |

**How mean and std are computed:** Gauss–Legendre quadrature with 200 nodes on $[-1, 0]$ and on
$[0, 1]$ separately (exact for polynomials and for $\lvert u\rvert$, whose kink is at 0).

**Allowed range:** $f$ must stay in $[0, 1]$ on $[-1, 1]$. For $a > 0$:

$$
a_{\max}^{+} = \min\left(\frac{1 - p_0}{\max g},\; \frac{p_0}{-\min g}\right),
$$

and correspondingly for $a < 0$ (`a_max(sign)`, min/max of $g$ on a grid of 20001 points). A `Reset`
outside this range is rejected (not clipped), so every tested function is exactly its formula.

**`Reset.from_fraction(kind, p0, frac)`:** $a = \text{frac}\cdot a_{\max}^{\pm}$ with
$\text{frac} \in [-1, 1]$; $\pm 1$ uses the full allowed range, $0$ is the constant $p_0$. Used by the
scan because $a_{\max}$ depends on $p_0$, which differs between the $S$.

**Why a class and not a lambda:** `ProcessPoolExecutor` sends jobs to worker processes by pickling;
lambdas cannot be pickled, `Reset` objects can.

Example:

```python
from qrc.reset import Reset
from qrc.reservoir import simulate
f = Reset.from_fraction("magnitude", p0=0.8, frac=0.5)   # p = 0.8 + a sqrt(12)(|u| - 1/2)
run = simulate(params, lam=1.0, p=f, M=50000)            # run.p_seq[t] = f(run.u[t])
```

## 5. `rq2/scan.py` and `rq2/configs/phaseA.toml` (new): phase A scan

Same job system as `rq1/map.py` (TOML job file, one file per job, restartable, parallel, see diary
18.11). Differences:

- **Baseline per $S$:** `[[baseline]]` entries give $(S, \lambda^*_S, p^*_S)$. Because $p^*_S$ and
  $\lambda^*_S$ differ between the $S$, every $S$ needs its own simulation (in RQ1 one simulation served
  all $S$).
- **One job** = (baseline $S$, shape, frac, device seed): simulate with $\lambda^*_S$ and
  $f = $ `Reset.from_fraction(shape, p*_S, frac)`, evaluate the capacities at that $S$.
- **frac = 0** (the constant baseline) is always run once per $(S, \text{seed})$; it is the reference for
  $\Delta C$.
- All $f$ are built once before the run starts, so an invalid shape fails before any worker starts.
- **Saved per job:** $S, \lambda, p_0$, shape, frac, $a$, $a_{\max}$, empirical mean and std of $p_t$,
  total IPC, capacity per degree (1–3), memory function, washout distance, warnings, commit, runtime.
  File name: `S1e+04_magnitude_frac+0.50_seed3.npz`.

`phaseA.toml`: $S \in \{10^2, \ldots, 10^6, \infty\}$ with the RQ1 optima as baseline, shapes `linear`
and `magnitude`, frac $\in \{\pm0.25, \pm0.5, \pm0.75, \pm1\}$, 10 devices, $M = 50000$, delays
$(20, 10, 7)$ (degrees 1–3, as in RQ1) → 1020 jobs (about the size of the RQ1 map).

```bash
uv run python rq2/scan.py --config rq2/configs/phaseA.toml --dry-run
nohup uv run python rq2/scan.py --config rq2/configs/phaseA.toml > rq2/logs/phaseA.log 2>&1 &
```

**Caveat:** for $S = 10^2, 10^3$ the RQ1 optimum lies at the edge of the RQ1 grid ($\lambda = 1$,
$p = 0.95$). There $a_{\max}$ is small (the reset cannot go much above 0.95), so phase A can only test
weak responses at small $S$.

Not implemented yet: `rq2/analyze.py` ($\Delta C$ vs. frac per shape and $S$, degree split and memory
function of the best $f$); it is easier to design once the first results exist.

## 6. Tests

| test | checks |
|---|---|
| `test_reservoir.py::test_constant_function_reproduces_constant_p` | $f \equiv 0.7$ gives **bitwise** the same populations as $p = 0.7$ (RQ1 is a special case) |
| `test_reservoir.py::test_input_dependent_reset_sequence_and_effect` | `p_seq` $= f(u)$, same inputs as the constant run, different populations |
| `test_reservoir.py::test_reset_function_out_of_range` | $f$ leaving $[0,1]$ or returning the wrong shape is rejected |
| `test_reservoir.py::test_trace_distance_curve_with_function` | trace distance curve with constant $f$ equals the constant-$p$ curve |
| `test_channel.py::test_check_strengths_arrays` | array check, including NaN |
| `test_reset.py` (23 tests) | every shape has mean 0 and std 1; $\langle p\rangle = p_0$ and $\operatorname{std}(p) = a$; frac $= \pm1$ touches 0 or 1 but stays inside; known closed forms; invalid input rejected; `Reset` is picklable |

All 69 tests pass. A small end-to-end run of `rq2/scan.py` (1 device, $M = 3000$, $S = 10^4$) produced
the expected files with $\langle p_t\rangle = p^*_S$ and $\operatorname{std}(p_t) = a$.
