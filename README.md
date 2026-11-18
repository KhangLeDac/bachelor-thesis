# qrc: a monitored spin quantum reservoir with weak measurement and reset

Simulation code for the bachelor project *Input-dependent reset strength in a monitored quantum reservoir*
(see [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md)). A small transverse-field Ising spin system is driven by an
input sequence. After every input step all spins are weakly measured and partially reset, and the measured
expectation values ⟨Z_i⟩, estimated from S shots, feed a linear readout. Performance is measured with the
information processing capacity (IPC).

## Physics

**Hamiltonian** (N spins, default N = 6):

    H(u) = h_z Σ_i Z_i + Σ_{i<j} J_ij X_i X_j + (h_x + ε u) Σ_i X_i

J_ij ~ U(−1, 1), symmetric, rescaled so that (1/N) Σ_ij J_ij² = J² (rms local coupling J). The integer `seed`
fixes the couplings; one seed is one "device". Inputs u_t are i.i.d. uniform on [−1, 1].

**One step** with input u_t, for density matrix ρ:

1. Unitary evolution: ρ → U ρ U†, U = exp(−i H(u_t) τ).
2. Weak measurement of every spin, strength λ = cos θ ∈ [0, 1], outcome s = ±1:
   Ω₊ = cos(θ/2)|0⟩⟨0| + sin(θ/2)|1⟩⟨1|, Ω₋ = sin(θ/2)|0⟩⟨0| + cos(θ/2)|1⟩⟨1|.
   λ = 1 is a projective Z measurement, λ = 0 reveals nothing.
3. Reset of every spin, strength p ∈ [0, 1], outcome not observed:
   E₀ = |0⟩⟨0| + √(1−p)|1⟩⟨1|, E₁ = √p |0⟩⟨1|.

Averaged over the outcomes, steps 2 and 3 act on the 2×2 block of each spin as
r₀₀ → r₀₀ + p r₁₁, r₁₁ → (1−p) r₁₁, r₀₁ → √(1−p) √(1−λ²) r₀₁. The code propagates this averaged density
matrix (2^N × 2^N; 64 × 64 for N = 6).

Useful limits:
- p = 0: the step is unital, the state relaxes to I/d and carries no input information.
- λ = 0: the measurement reveals nothing.
- p = 1 (any λ): every step resets the system to |0…0⟩, so there is no memory of earlier inputs.

**Features and shot noise.** At step t the state σ_t = U(u_t) ρ_{t−1} U(u_t)† is measured. One shot gives
s ∈ {±1}^N with mean x_i(t) = λ⟨Z_i⟩ and covariance

    Σ_ij(t) = λ² ⟨Z_i Z_j⟩ − x_i x_j   (i ≠ j),     Σ_ii(t) = 1 − x_i².

With S shots per step the readout sees x(t) plus noise of covariance Σ(t)/S. The readout uses only the
current step: h(t) = (1, x_1(t), …, x_N(t)).

**IPC.** Targets y are products of normalised Legendre polynomials √(2n+1) P_n(u_{t−k}) of delayed inputs,
of total degree 1–3. With G = ⟨h hᵀ⟩, V = ⟨Σ(t)⟩ (zero for the constant) and R = ⟨h y⟩, a linear readout
trained on noisy features reaches

    C_y(S) = Rᵀ (G + V/S)⁺ R.

The code cross-fits R: C = R_Aᵀ (G + V/S)⁺ R_B, with R_A and R_B from the two halves of the data (separated by `gap` = 50 samples). This
removes the upward bias of about (N+1)/M per target that comes from estimating R. Targets the reservoir
cannot reproduce then give C ≈ 0, possibly slightly negative.

With N features the total capacity at S = ∞ is at most N, and it approaches N when the target set is large
enough. Two caveats:
- **Cut-off.** `total` sums only the included targets (degree ≤ 3, delays ≤ (20, 10, 7) by default), so it is a
  lower bound. For long memory (small p) it can fall well below N because capacity sits at larger delays. Check
  that `summarize(C, table)["memory"]` has decayed to ≈ 0 at the largest delay; `ipc` warns if it has not.
  Then increase `delays`.
- **Noise.** The estimate fluctuates by about 0.1 at M = 10⁴ and can slightly exceed N.

So compare settings at finite S, or compare how the capacity splits over degrees and delays.

## Installation and running

The project uses [uv](https://docs.astral.sh/uv/):

    uv sync                                      # create .venv and install dependencies
    uv run pytest                                # tests (~15 s)
    uv run python examples/single_point.py       # IPC vs S for one (λ, p)
    uv run python examples/lambda_p_map.py       # total IPC over a (λ, p) grid at fixed S

Figures go to `figures/`. Both examples take `--help`.

**Runtime.** For 64 × 64 matrices, multithreaded BLAS is much slower than a single thread (about 40× on a
busy machine). The examples set `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) before numpy is
imported, and run independent settings in parallel processes. Do the same in your own scripts. One run with
the defaults (1000 washout steps + 10 000 samples, N = 6) takes about 10 s.

## Code map

| module | contents |
|---|---|
| `qrc/model.py` | `Params` (n, J, h_x, h_z, eps, tau, seed), `draw_couplings`, `z_signs`, `single_site`, `hamiltonian`, `unitary` |
| `qrc/channel.py` | `kraus_measure(lam)`, `kraus_reset(p)`, `measure_reset_spin`, `measure_reset`, `step(rho, U, n, lam, p)`, `check_strengths` |
| `qrc/reservoir.py` | `simulate(params, lam, p, M, T_wash, input_seed) -> Run`, `features`, `shot_covariance`, `trace_distance`, `trace_distance_curve` |
| `qrc/ipc.py` | `legendre_targets(u, T_wash, delays)`, `capacities(x, cov, Y, S)`, `summarize(C, table)`, `ipc(run, S)` |
| `qrc/sampling.py` | finite-shot cross-check: `outcome_distribution`, `noisy_features(pop, n, lam, S, rng)`, `readout_capacity` |

A `Run` holds the inputs `u` (washout included), the populations `pop` of the measured state, the features
`x`, the shot covariances `cov`, the washout check `washout_distance`, and the settings. Sample t uses input
`u[T_wash + t]`.

Minimal use:

```python
import numpy as np
from qrc.model import Params
from qrc.reservoir import simulate
from qrc.ipc import ipc, summarize

run = simulate(Params(seed=0), lam=0.8, p=0.5, M=10000, T_wash=1000)
C, table = ipc(run, S=1e4)            # capacity per target, table of (degree, delays)
print(summarize(C, table))            # total, degree1..3, memory (degree-1 capacity per delay)
```

## Conventions

- Spin 0 is the most significant bit of the basis index. |0⟩ is the Z = +1 eigenstate and corresponds to
  outcome s = +1.
- λ and p are scalars in [0, 1], the same for all spins and all steps. Other values raise `ValueError`.
- Defaults: N = 6, J = 1, h_x = 0.5, h_z = 1, ε = 0.1, τ = 2. Target delays: (20, 10, 7) for degrees 1, 2, 3.
- Washout: `simulate` also propagates the maximally mixed state through the washout. It warns if the trace
  distance to the |0…0⟩ run is still above 1e-6 (increase `T_wash`, typically needed for small p).
  `trace_distance_curve` gives the full decay, from which a memory time can be read off.
