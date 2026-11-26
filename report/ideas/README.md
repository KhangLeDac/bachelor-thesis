# Ideas log

Concept-oriented record of the key physical and mathematical ideas worked out for the thesis.
The diary (`report/diary/`) answers *what was done when*; this log answers *what did we decide, and why*.
Entries are living: they are revised as an idea develops, and superseded rather than deleted.

- Written with the `/idea` skill (`.claude/skills/idea/SKILL.md`) from the current discussion.
- Maintained by the `ideas-curator` agent (dedupe, cross-links, stale code references, diary sync).
- Index: [INDEX.md](INDEX.md).

## Conventions

- One file per idea, kebab-case slug: `legendre-basis-g.md`.
- Markdown with `$…$` / `$$…$$` math. Notation as in `README.md` and `rq*/PLAN.md`
  ($N$, $\lambda$, $p$, $p_t = f(u_t)$, $S$, $\Delta C$, $u \sim \mathcal U(-1,1)$).
- `status`: `open` (being discussed), `adopted` (in use / in the plan), `rejected`,
  `superseded-by: <slug>`.
- `relates`: slugs of other ideas; links must point to existing files.
- `code`: `path:line` or `path::symbol` once implemented; `diary`: entry date(s) where it appears.
- *Alternatives considered* is mandatory: what was rejected and why is the part that is otherwise lost.

## Template

```markdown
---
title: <short title>
status: open | adopted | rejected | superseded-by: <slug>
date: YYYY-MM-DD
revised: YYYY-MM-DD
relates: [<slug>, ...]
code: [<path::symbol>, ...]
diary: [DD.MM, ...]
---

## Idea
One or two sentences.

## Formulation
Definitions, equations, assumptions.

## Why
Motivation; what problem it solves.

## Alternatives considered
- <alternative> — <why not>

## Open questions / next steps
- ...
```
