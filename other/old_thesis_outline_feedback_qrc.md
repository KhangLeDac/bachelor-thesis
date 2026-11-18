# Bachelor thesis outline — Input- and state-dependent dissipation in online quantum reservoir computing

## 1. Setting

- Reservoir: N spins, Transverse field Ising Hamiltonian. Input s_k enters via transverse field modulation, constant over an interval τ.
- Dissipation: amplitude damping, rate γ_k, on all spins or on the measured subset only.
- Measurement: projective Pauli-Z measurements on M of N spins after each interval. With S → ∞ copies the dynamics is unconditional: dephasing on the measured spins, followed by an outcome-conditioned rotation by θ_k toward |0⟩ (θ = 0 no reset, θ = π full reset, 0 < θ < π partial reset).
- Readout: ⟨Z_i⟩ of the measured spins at each step (optional temporal multiplexing), linear regression.
- Control knobs per step: γ_k and θ_k. Policies: (A) constant (baseline); (B) feedforward γ_k, θ_k = f(s_k); (C) feedback γ_k, θ_k = g(⟨Z⟩_{k−1}); (D) checks: f applied to a shuffled input stream, and to an independent random stream.
- Benchmarks: Echo State Property (ESP) test (trace distance between two initial states), linear Memory Capacity (MC), Information Processing Capacity (IPC), NARMA-10/20, Mackey–Glass, Lorenz-63.

## 2. Research questions

- RQ1: At constant γ, θ, does the per-step contraction of the online protocol depend on the input s_k?
- RQ2: Does step-wise modulation of γ, θ (B or C) improve capacity (MC, IPC) and task error over the best constant setting (A) at matched mean dissipation? Where in the (γ̄, θ̄, τ, M) landscape?
- RQ3: Is a gain due to equalized/task-matched forgetting, or due to the extra input encoding that a modulated rate provides? Decided by the checks (D) and by stationary vs non-stationary inputs.
- RQ4: Contraction factor of one step (dephasing + partial reset + damping) as a function of (γ, θ, h_x); ESP condition for time-varying γ_k, θ_k (strict contraction on average suffices, even if some steps are non-contractive).
- RQ5 (optional add-on): Can a slow loop driven only by task-free statistics of ⟨Z_i⟩ keep (γ̄, θ̄, τ, h-scale) in the high-capacity band and track imposed parameter drift?

## 3. Work plan

| Weeks | Work package | Output |
|---|---|---|
| 1–4 | WP0 Simulator: Lindblad propagation, dephasing/partial-reset channel, benchmarks, ESP test, readout. Unit tests against known limits (θ = π ≙ Fujii–Nakajima reset; γ = 0, θ = 0 ≙ pure dephasing). | Working code. |
| 4–7 | WP1 Baseline landscape: grid over γ, θ, τ, h_x-scale, M for N = 6–8. Contraction map vs s_k (RQ1). Best constant settings per task. | Landscape figures. |
| 7–14 | WP2 Modulated dissipation: policies B, C vs A at matched mean; controls D; stationary and non-stationary inputs; MC/IPC decomposition (RQ2, RQ3). | Main result figures. |
| 10–16 | WP3 Theory (parallel): single-step contraction factor; averaged-contraction ESP criterion; numerical check (RQ4). | Short analytic section. |
| 16–20 | WP4 Optional IBM hardware demonstration | Figures. |
| 20-26 | Thesis writing | Thesis. |
| optional add-on | WP5 Homeostatic loop (RQ5): statistic ↔ capacity correlation on the WP1 grid; closed loop from random start; drift test (slow drift in J, h_x, γ) fixed vs self-tuned on NARMA/Lorenz. | Add-on section. |


## 4. Reading list

1. Fujii & Nakajima, Phys. Rev. Appl. 8, 024030 (2017) — spin-reservoir QRC, reset-based input.
2. Mujal et al., npj Quantum Inf. 9, 16 (2023) — online QRC with weak and projective measurements.
3. Sannia et al., Quantum 8, 1291 (2024) — dissipation as a resource; universality of dissipative spin reservoirs.
4. Kobayashi, Fujii & Yamamoto, PRX Quantum 5, 040325 (2024) — feedback-driven QRC, feedback-strength phases.
5. Morguí-Sancho, Manzano, Giorgi & Zambrini, arXiv:2607.14311 (2026) — general theory of monitored QRC; ESP/fading-memory criteria.
6. Dambre et al., Sci. Rep. 2, 514 (2012) — information processing capacity.
