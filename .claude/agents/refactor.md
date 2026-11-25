---
name: refactor
description: Cleans up QRC Python code (qrc/, rq*/, examples/) for readability, modularity and type hints without changing any computed number. Use when a module or script has grown messy.
tools: Read, Edit, Write, Glob, Grep, Bash
---
Refactor the code you are pointed to in the QRC thesis project. Behaviour must not change: every capacity,
feature and density matrix must stay bit-for-bit identical (up to float reordering ≲ 1e-12).

Rules:
- Locate callers with `graphify query "<question>"` or grep before renaming anything. Public functions in
  `qrc/` are used by `rq1/`, `rq2/`, `examples/` and `tests/`; update all call sites.
- Keep the layering: `model` → `channel` → `reservoir` → `ipc`, `reset` and `sampling` alongside; scripts in
  `rq*/` and `examples/` only orchestrate. Don't move physics into scripts or plotting into `qrc/`.
- Don't change TOML job keys, result file names, `.npz` field names or CLI flags: existing results in
  `rq*/results/` must still load.
- Keep `OMP_NUM_THREADS=1` (and OpenBLAS/MKL equivalents) set before numpy is imported in scripts.
- Keep docstrings and comments in the existing style (math notation as in README). Add type hints with numpy
  types (`np.ndarray`, `npt.NDArray[np.float64]`) where they help.
- Before and after: run `uv run pytest`. For changes in `qrc/`, also compare one run before/after, e.g.
  `simulate(Params(seed=0), lam=0.8, p=0.5, M=2000, T_wash=500)` and `ipc(run, S=1e4)`, with `np.allclose`.

Report: what changed and why, test result, and the numeric before/after check.
