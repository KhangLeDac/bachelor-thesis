# `qrc_spin_online.py`: online spin quantum reservoir

Documentation of [utils/qrc_spin_online.py](utils/qrc_spin_online.py). The file simulates a
quantum reservoir of `n` spin-1/2 particles in QuTiP. The reservoir is driven by an input
sequence, a subset of its spins (the *probes*) is measured after every input step, and the
measurement acts back on the state (the *online* protocol).

It follows `SpinQRC` in `qrc_spin.py` (same Hamiltonians, same encodings, same QuTiP
conventions). The protocol is what differs. See [Differences to `SpinQRC`](#differences-to-spinqrc).

## Contents

1. [What the file contains](#what-the-file-contains)
2. [One input step](#one-input-step)
3. [Quick start](#quick-start)
4. [Physical model](#physical-model)
5. [`ConstantPolicy`](#constantpolicy)
6. [`OnlineSpinQRC`](#onlinespinqrc)
7. [`trace_distance`](#trace_distance)
8. [Differences to `SpinQRC`](#differences-to-spinqrc)
9. [Known issues and caveats](#known-issues-and-caveats)

## What the file contains

| Name | Kind | Purpose |
|---|---|---|
| `ConstantPolicy` | class | Supplies the same `(gamma, theta)` at every step. |
| `OnlineSpinQRC` | class | The reservoir: builds the Hamiltonian, evolves the state, reads out and resets the probes. |
| `trace_distance` | function | Trace distance between two density matrices (for fading-memory tests). |

## One input step

For input step `k`, `step` does the following in order:

1. **Evolve.** The state evolves for time `dt` under the Lindblad equation. `H` depends on the
   input `s_k`, and amplitude damping has rate `gamma_k`.
2. **Read out.** The Z-basis outcome distribution of the probes is turned into all
   `2^M - 1` Z-string features (`M` = number of probes). The features are read *before* the reset.
3. **Back-action.** Every probe is measured and partially reset with the Kraus operators
   `K0 = P0 = |0><0|` and `K1 = R_y(theta) P1 = cos(theta/2) |1><1| - sin(theta/2) |0><1|`.
   `theta = 0` is pure dephasing, `theta = pi` is a full reset to `|0>`, anything in between is a partial reset.

The state is carried over to the next step. It is never reset between inputs.

The back-action is applied as the *unconditional* map `rho -> K0 rho K0^+ + K1 rho K1^+`. This is the
average over infinitely many copies of the experiment. With finite `n_shots` only the
features are sampled, and the state still follows the unconditional map.

## Quick start

```python
import numpy as np
from utils.qrc_spin_online import OnlineSpinQRC

qrc = OnlineSpinQRC(
    n=4, encoding_strength=0.5, coupling_strength=1.0,
    gamma=0.3, theta=np.pi, dt=1.0, probes=2,
)
data = np.random.default_rng(0).uniform(0, 1, 100)   # scalar input per step
features = qrc.run(data)                              # shape (100, 3): Z0, Z1, Z0Z1
```

A linear readout (for example ridge regression) is then trained on `features`. That part is not in this file.

## Physical model

All operators act on `n` qubits as full `2^n x 2^n` `Qobj`. Pauli matrices are used directly
(no factor 1/2) and `hbar = 1`. `h = constant_field = 1.0`.

**Static Hamiltonian** (built once):

```
H_static = sum_i h Z_i  +  sum_{i<j} J_ij X_i X_j
```

`J` is an `n x n` matrix drawn uniformly from `[-coupling_strength, coupling_strength]`. Only
the upper triangle (`i < j`) is used.

**Input-dependent part** (built every step). `field_i` is the *total* coefficient on spin `i`, including the constant part:

| `model` | Input drives | Total Hamiltonian |
|---|---|---|
| `"TiltedTFIM"` | X fields | `sum_i h Z_i + sum_i field_i X_i + sum_{i<j} J_ij X_i X_j` |
| `"TFIM"` | Z fields | `sum_i field_i Z_i + sum_{i<j} J_ij X_i X_j` |

For `TFIM` the code adds `(field_i - h) Z_i` on top of the static `h Z_i`, so the total Z coefficient is `field_i`.

**Dissipation.** One jump operator per damped spin, `sqrt(gamma_k) * lowering_i`.

| `decay_to` | Lowering operator | Fixed point |
|---|---|---|
| `0` (default) | `qt.sigmap()` = `|0><1|` | `|0>` (Z = +1) |
| `1` | `qt.sigmam()` = `|1><0|` | `|1>` (Z = -1), as in `SpinQRC` |

**Initial state.** `|0...0><0...0|`.

## `ConstantPolicy`

A policy decides `(gamma_k, theta_k)` for each step. `run` calls the policy inside its loop, so the values need not be fixed before the run.

| Member | Function |
|---|---|
| `__init__(gamma, theta)` | Stores the two constants as floats. |
| `__call__(k, s_k, prev_features)` | Returns `(gamma, theta)` for step `k`. The arguments are ignored here. |

The call signature is the interface for any policy:

- `k`: step index.
- `s_k`: the input of step `k` (for feedforward policies).
- `prev_features`: the features of step `k - 1`, or `None` at `k = 0` (for feedback policies).

A policy can be any callable with this signature, including a class that keeps its own history.
The policy sees features, not the state. It is called once, before `evolve`, so `gamma_k` and
`theta_k` can depend on the previous step's features but not on the current step's.

## `OnlineSpinQRC`

### Constructor arguments

| Argument | Meaning |
|---|---|
| `n` | Number of spins. |
| `encoding_strength` | Strength `s` of the input encoding. Not used by `"uniform"`. |
| `coupling_strength` | `J_s`: `J_ij` uniform in `[-J_s, J_s]`. |
| `gamma` | Damping rate of the default `ConstantPolicy`. |
| `theta` | Reset angle of the default `ConstantPolicy`. |
| `dt` | Duration of one input step. |
| `probes` | Number `M` of measured spins (then spins `0..M-1`) or an explicit list of spin indices. |
| `model` | `"TiltedTFIM"` or `"TFIM"`. |
| `encoding_method` | `"uniform"` (default), `"one_to_one"`, `"fill"` or `"dense"`. See below. |
| `damping_on` | `"all"` spins or only the `"probes"` are damped. |
| `decay_to` | `0` or `1`: fixed point of the amplitude damping. |
| `seed` | Seed for `J`, `noise_fill` and `Win`. |
| `n_shots` | Shots per step for the features. `inf` gives exact probabilities. |
| `shot_seed` | Seed of the shot sampler. Defaults to `seed`. |

The constructor validates `model`, `encoding_method`, `damping_on`, `decay_to`, the probe indices
and `n_shots`, and raises `ValueError` on bad values.

### Input encodings

`field_strengths` turns one input vector `x` into one field per spin. Inputs are assumed to lie in `[0, 1]`.

| `encoding_method` | `data_dim` | Field |
|---|---|---|
| `"uniform"` | 1 (scalar) | `h (1 + x)` on every spin. This is the encoding of Sannia et al. (see module docstring). |
| `"one_to_one"` | `<= n` | `field_i = h + 2 s (x_i - 1/2)` for `i < data_dim`; other spins keep `h`. |
| `"fill"` | `<= n` | Each input dimension drives a block of `n // data_dim` spins: `h + 2 s (x_d - 1/2) * noise_fill[q]`. Leftover spins keep `h`. |
| `"dense"` | any | `field = h + s * (Win @ 2(x - 1/2))`. `Win` is random, with each row normalised so that its absolute values sum to 1. |

Notes:

- `s` is `encoding_strength`.
- `"uniform"` raises `ValueError` if the input is not a scalar.
- `"dense"` draws `Win` at the first call, so `data_dim` must stay the same afterwards.

### Attributes created in `__init__`

| Attribute | Content |
|---|---|
| `sx_ops`, `sz_ops` | Lists of `X_i` and `Z_i` for every spin, cached. |
| `probe_P0_ops`, `probe_P1_ops`, `probe_E01_ops` | `P0 = |0><0|`, `P1 = |1><1|` and `|0><1|` on every probe. |
| `J`, `noise_fill`, `Win` | Random couplings, random block weights (`"fill"` only), random input weights (`"dense"` only, drawn lazily). |
| `H_static` | The static Hamiltonian. |
| `lowering_ops` | Unit-rate lowering operators of the damped spins. |
| `initial_state` | `|0...0><0...0|`. |
| `probes`, `n_probes` | Sorted probe indices and their number. |
| `n_features`, `feature_labels` | Number and names of the output features. |
| `policy` | The default `ConstantPolicy`. |

### Public methods

| Method | Function |
|---|---|
| `run(data, rho0=None, policy=None, return_states=False)` | Drives the reservoir with a whole input sequence. `data` has shape `(steps, data_dim)`, or `(steps,)` for scalar inputs. Calls `step` for every row and hands the state from one step to the next. Returns `features` of shape `(steps, 2^M - 1)`, plus the list of states after each step's back-action if `return_states=True`. `rho0` defaults to the initial state; `policy` defaults to the constant policy. |
| `step(rho, s_k, gamma_k, theta)` | One input step: `evolve`, then `features`, then `measure_and_reset`. Returns `(features, rho after back-action)`. |
| `evolve(rho, s_k, gamma_k)` | Builds `H = H_static + H_dynamic` from the input, builds one jump operator `sqrt(gamma_k) * lowering_i` per damped spin and calls `qt.mesolve` over `[0, dt]`. Returns the new state. |
| `measure_and_reset(rho, theta)` | Applies the Kraus map to every probe, then re-hermitises the result. |
| `outcome_probabilities(rho)` | Joint Z-basis outcome distribution of the probes (`2^M` entries). It reads the diagonal of `rho` and sums over the non-probe spins. The first probe is the most significant bit. |
| `features(rho)` | All `2^M - 1` Z-strings of the probes, computed from `outcome_probabilities`. With finite `n_shots`, the probabilities are replaced by multinomial frequencies first. |
| `field_strengths(data_step)` | Per-spin field from one input step. See the encoding table. |

### Feature layout

Features are the expectation values of Z-strings on subsets of the probes. They are ordered by subset size, then by position:
singles, then pairs, then triples, up to the product over all probes. For `probes=3` the labels are

```
Z0, Z1, Z2, Z0Z1, Z0Z2, Z1Z2, Z0Z1Z2
```

The labels use the actual spin indices from `probes`. They are stored in `feature_labels`. Every feature lies in `[-1, 1]`.

### Finite shots

With a finite `n_shots`, `features` draws `n_shots` samples from the joint probe distribution (`multinomial`) and uses the empirical frequencies. All Z-strings are computed from the *same* histogram, so
the features are correlated and there are no `nan` columns. Only the readout is noisy. The state is never changed by the sampling.

### Private builders

| Method | Function |
|---|---|
| `_get_initial_state()` | Returns `|0...0><0...0|` as a tensor product of single-qubit `fock_dm(2, 0)`. |
| `_build_operator(operator, i)` | Places a single-qubit operator on spin `i` and identities elsewhere (`qt.tensor`). |
| `_build_static_hamiltonian()` | Builds `sum_i h Z_i + sum_{i<j} J_ij X_i X_j`. |
| `_build_feature_map()` | Precomputes the sign matrix `_z_signs` (shape `(2^M - 1, 2^M)`) that turns an outcome distribution into Z-string expectation values, and the labels. |
| `_build_lowering_operators()` | Builds the unit-rate lowering operators for the damped spins, according to `decay_to` and `damping_on`. |

## `trace_distance`

```python
trace_distance(rho, sigma) -> float
```

Returns `D = 1/2 ||rho - sigma||_1`, using the trace norm of the `Qobj` difference. It is used
for the echo-state-property test: two copies of the reservoir started in different states
and driven by the same input should converge (`D -> 0`).

## Differences to `SpinQRC`

| | `SpinQRC` | `OnlineSpinQRC` |
|---|---|---|
| Protocol | Restarting: `rho` is never disturbed | Online: probes are measured and reset after each step |
| Measured spins | All, in X, Y and Z | Probes only, in Z |
| Features | Local Paulis, optionally two-qubit terms | All Z-strings on the probes |
| Finite shots | Three settings, via `utils/shots.py` | One setting, `multinomial` in `features` |
| Damping | Constant `gamma` for all spins | `gamma_k` from a policy, on all spins or on the probes only |
| Default decay | `sigmam()`, fixed point `|1>` | `decay_to=0`, fixed point `|0>` |
| `J` range | `[-J/2, J/2]` | `[-J_s, J_s]`, which is twice as large for the same seed and strength |
| Encodings | `one_to_one`, `fill`, `dense` | Same, plus `"uniform"` (the default) |

To compare the two directly, use `decay_to=1`, `theta=0` and `coupling_strength` equal to half of `SpinQRC`'s value.

## Known issues and caveats

- **Stale error message.** The scalar-input check in `field_strengths` says `encoding_method='paper'`, but the method is called `"uniform"`.
- **Stale docstring.** `field_strengths` still says it is identical to `SpinQRC._generate_field_strengths`, which no longer holds because of `"uniform"`.
- **`encoding_strength` is ignored** by `"uniform"`.
- **`"uniform"` and `TFIM`.** For `TFIM` the input drives Z, so `h (1 + x)` would go on the Z field. The formula of Sannia et al. is written for the X field of `TiltedTFIM`.
- **Unconditional back-action.** The reset is applied to the whole state regardless of the sampled outcomes. A real online run would condition on its own outcomes.
- **Speed.** `qt.mesolve` is called once per step and rebuilds the Liouvillian each time. The cost grows roughly as `4^n`.
- **One rate for all spins.** `gamma_k` is the same for every damped spin.
