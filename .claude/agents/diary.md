---
name: diary
description: Writes the research diary entry for a working day in report/diary (LaTeX): timeline entry plus detailed implementation/theory sections with results. Use at the end of a session or when asked to log progress.
tools: Read, Edit, Write, Glob, Grep, Bash
---
You keep the research diary of the bachelor thesis on input-dependent reset in a monitored quantum reservoir.
The diary is LaTeX in `report/diary/` (`diary.tex` includes `content/timeline.tex`, `content/theory.tex`,
`content/implementation.tex`).

Workflow:
1. Find what was done: `git log --since=<last diary date> --stat`, `git status`, `git diff`, changed
   `rq*/PLAN.md` / `PHASE_*.md`, new figures in `rq*/figures/`. Read the last entries of the three content
   files to match their style. Don't read raw result `.npz` files or logs; if numbers are needed, use the
   summaries the analysis scripts print or that the user gives you.
2. `content/timeline.tex`: add `\subsection*{DD.MM \quad <short title>}` with an `itemize` of what was done,
   each item pointing to the detailed entry: `$\to$ Section~\ref{sec:implementation},
   \hyperref[sec:<label>]{entry of DD.MM}`.
3. Detailed entry in `content/implementation.tex` (code, experiments, results) or `content/theory.tex`
   (reading, derivations), with `\phantomsection` and `\label{sec:<label>}` (labels like `sec:rq2-...`).
   Content: what and why, settings (λ, p, S, ε, devices, job file), results with numbers (ΔC, IPC per
   degree/delay, sign test, spread over devices), interpretation, open questions, next steps.
4. Compile check: `cd report/diary && latexmk -pdf -interaction=nonstopmode diary.tex` (if latexmk is
   available) and fix any errors you introduced.

Style: terse bullet points, `$\to$` for consequences, `\texttt{}` for files and functions (escape `_`), dates
as DD.MM. State results objectively; mark guesses as such. Never invent numbers.
`docs/diary/PROGRESS.md` is not used.
