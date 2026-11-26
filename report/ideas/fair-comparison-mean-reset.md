---
title: Fair comparison at fixed mean reset strength
status: adopted
date: 2026-11-26
revised: 2026-11-26
relates: [legendre-basis-g]
code: [qrc/reset.py::Reset, rq2/scan.py]
diary: []
---

## Idea
Every input-dependent response $f$ is compared to the constant RQ1 optimum $p^*_S$ at the *same mean*
reset strength, so a gain comes from how the reset is distributed over inputs, not from more or less
reset on average.

## Formulation
$$\langle p_t\rangle_u = \langle f(u)\rangle_{u\sim\mathcal U(-1,1)} = p^*_S, \qquad
\Delta C = C[f] - C[p^*_S],$$
with $\Delta C$ computed paired (same device seed, same input sequence), $\lambda = \lambda^*_S$ fixed.
$f$ must map $[-1,1]\to[0,1]$; parameters outside the range are excluded, not clipped
(`Reset.a_max`, `Reset.from_fraction`).

## Why
- Separates the effect of the *shape* of the response from a shift of the operating point, which RQ1
  already optimised.
- Paired comparison makes the noise of $\Delta C$ much smaller than that of $C$.

## Alternatives considered
- Free mean — would mix RQ1 (best constant) with RQ2 (best response); kept as an open question.
- Clipping $f$ to $[0,1]$ — the tested function would no longer be the formula it claims to be.

## Open questions / next steps
- Is $\langle p\rangle = p^*_S$ the right fairness condition, or should the mean also be free
  (still compared against $p^*_S$)? Open question for Tobi (`rq2/PLAN.md` §6).
