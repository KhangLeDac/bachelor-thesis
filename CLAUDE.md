# Project: Bachelor's Thesis – Input-dependent reset in a monitored quantum reservoir

## Core Rules & Token Optimization
- **Context First:** Locate modules and dependencies with `graphify query "<question>"` (graph in `graphify-out/`) before broad file searches. Do not read raw results (`rq*/results/*.npz`), logs or PDFs into context.
- **Brevity:** Keep responses concise, technical, and free of filler text.
- **Scope:** `PROJECT_OUTLINE.md` defines the setup and RQ1–RQ3; `README.md` the physics, code map and conventions; `rq*/PLAN.md` the current phase.
- **Domain Context:** Exact density-matrix simulation of a monitored Ising spin reservoir (N = 6; weak Z measurement λ, reset p, input-dependent reset p_t = f(u_t)), linear readout of ⟨Z_i⟩ at finite shot number S, evaluated by the information processing capacity (IPC). Keep physics in `qrc/` and orchestration/analysis in `rq*/` and `examples/`. Enforce reproducible seeds (device seed, input seed) and paired comparisons against the RQ1 optimum.

## Available Subagents (`.claude/agents/`)
Delegate complex or repetitive tasks to isolated subagents to keep the main context window clean:
- `architect`: Plans code changes and experiment designs (new RQ phases, job files, controls) before implementation.
- `reviewer`: Reviews code for bugs and physical/statistical correctness (CPTP maps, step order, IPC estimator, paired comparisons).
- `refactor`: Cleans up code, improves modularity, and adds type hints without changing any computed number.
- `docs-updater`: Updates `README.md`, `rq*/PLAN.md`, `PHASE_*.md` and docstrings after code changes.
- `diary`: Writes the research diary entry in `report/diary/content/*.tex` (timeline + detailed entry).
- `git-manager`: Atomic commits in the repo style (`RQ2: ...`, `Diary: ...`), branches, PRs via GitHub MCP.
