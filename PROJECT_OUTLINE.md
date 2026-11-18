# Bachelor thesis outline: input-dependent reset strength in a monitored quantum reservoir

*This outline replaces `thesis_outline_feedback_qrc.md`.*

## 1. Setup

- **Reservoir:** N = 6 spins, Ising Hamiltonian (X_i X_j couplings, transverse field h_z) with random couplings,
  H(u) = h_z Σ Z_i + Σ J_ij X_i X_j + (h_x + ε u) Σ X_i. The input u_t enters through the X field (h_x + ε u) and is
  held for a time τ.
- **Monitoring:** after every step all spins are weakly measured in the Z basis with strength λ ∈ [0, 1]
  (λ = 1 projective), then reset towards |0⟩ with strength p ∈ [0, 1] (p = 1 full reset), outcome not observed.
  The reset makes the dynamics non-unital and sets how fast the reservoir forgets.
- **Readout:** the expectation values ⟨Z_i⟩ of all spins at the current step, estimated from S shots, feed a
  linear readout.
- **Performance measure:** information processing capacity (IPC) at finite S: total, per degree, and per delay
  (linear memory function). The total is bounded by N (≈ N at S = ∞ when the target set covers the memory), so settings are compared
  at finite S; check that the memory function has decayed at the largest target delay.
- **Code:** see [README.md](README.md). The simulation propagates the outcome-averaged density matrix, and the
  finite-S capacity follows from the shot covariance without sampling.

## 2. Research questions

- **RQ1 (baseline).** How does the IPC at finite S depend on λ and p, in total, per degree and per delay? What is
  the best constant (λ, p) for a given S, and how does it move with S? How long is the memory of the reservoir,
  measured by the decay of the trace distance between two initial states?
- **RQ2 (core).** Does an input-dependent reset strength p_t = f(u_t) beat a constant p? Two comparisons:
  (a) against the best constant p (optimised), (b) against the constant p with the same mean. Which parts of the
  IPC change (degrees, delays)?
- **RQ3 (mechanism).** Is a gain due to better-matched forgetting, or due to p(u) acting as an additional input
  channel (the reset writes input-dependent populations)? Controls:
  - ε = 0, so the input enters only through p;
  - p driven by an independent random sequence with the same statistics;
  - constant p with a larger input amplitude ε.
- **Optional.** A reset conditioned on a spin's own shot outcome (nontrivial only for λ < 1).

## 3. Pilot and decision

Before the full RQ2/RQ3 study, run a pilot with one or two simple choices of f (e.g. affine in u, or two levels)
on about 10 devices (coupling seeds). Continue with RQ2/RQ3 as planned if the gain over the best constant p
exceeds the spread over devices and survives at realistic S. Otherwise, reframe the thesis as a characterisation
of the reset strength p(u) as an input encoding, compared with encoding through the X field.

## 4. Reading list

1. K. Fujii and K. Nakajima, *Harnessing disordered-ensemble quantum dynamics for machine learning*,
   Phys. Rev. Applied 8, 024030 (2017). Spin-network quantum reservoir computing.
2. P. Mujal et al., *Time-series quantum reservoir computing with weak and projective measurements*,
   npj Quantum Inf. 9, 16 (2023). Online processing with measurements.
3. A. Sannia, R. Martínez-Peña, M. C. Soriano, G. L. Giorgi and R. Zambrini, *Dissipation as a resource for
   quantum reservoir computing*, Quantum 8, 1291 (2024).
4. J. Dambre, D. Verstraeten, B. Schrauwen and S. Massar, *Information processing capacity of dynamical
   systems*, Sci. Rep. 2, 514 (2012). The IPC.
5. F. Hu et al., *Tackling sampling noise in physical systems for machine learning applications: fundamental
   limits and eigentasks*, Phys. Rev. X 13, 041020 (2023). Capacity at finite shot number.
6. O. Morguí-Sancho, G. Manzano, G. L. Giorgi and R. Zambrini, *General theory of monitored quantum reservoir
   computing*, arXiv:2607.14311 (2026). Weak measurement, reset, and when monitored reservoirs work.
7. (Optional part) K. Kobayashi, K. Fujii and N. Yamamoto, *Feedback-driven quantum reservoir computing for
   time-series analysis*, PRX Quantum 5, 040325 (2024).
