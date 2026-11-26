---
name: idea
description: Record a key physical or mathematical idea from the current discussion in the ideas log (report/ideas/). Use when the user types /idea, says "keep track of this", "note this idea", or when a discussion settles a definition, derivation, parametrisation or design decision worth keeping.
argument-hint: "[slug or topic hint]"
---
Record the idea we just worked out in `report/ideas/`. You run in the main conversation: the discussion
itself is the primary source. Conventions and template: `report/ideas/README.md`.

1. **Identify the idea.** Use `$ARGUMENTS` as a hint if given; otherwise take the most recent idea that
   was settled or substantially developed. If the discussion holds several distinct ideas, list them and
   ask which to record (or record each in its own file). Do not record trivia, code mechanics or results
   that belong in the diary.
2. **Check for an existing entry.** Read `report/ideas/INDEX.md`; grep `report/ideas/` for key terms.
   If the hint names an existing idea (e.g. "legendre"), everything in the request — observations,
   results, interpretations, follow-up questions about it — goes **into that entry** (a dated
   subsection in *Formulation*, new items in *Open questions*), never into a new file.
   If an entry covers the idea, **update it** (bump `revised`, extend sections, keep earlier reasoning,
   move outdated parts to *Alternatives considered*). If the new idea replaces an old one, set the old
   status to `superseded-by: <new-slug>` instead of deleting it.
3. **Write the entry** from the template:
   - keep equations exact as derived in the discussion; notation as in `README.md` / `rq*/PLAN.md`;
   - *Why* and *Alternatives considered* in the user's reasoning, including rejected options and the
     reason — this is the part that is otherwise lost;
   - `code:` only references that exist (check with grep), `diary:` only existing entry dates;
   - `relates:` and in-text links as relative Markdown links `[title](slug.md)` to existing files.
   - Concise: bullets and equations, no narrative of the conversation.
4. **Update `INDEX.md`** (one row per idea: link, status, revised date, topic RQ1/RQ2/RQ3/general),
   and add the back-link in the `relates:` of related entries.
5. **Report** the file path(s) and a 2–3 line summary of what was recorded or changed. Do not commit;
   commits go through the `git-manager` agent (`Ideas: <short title>`).

If the idea is still uncertain, record it with `status: open` and put the uncertainty under
*Open questions*. Never invent derivations that were not part of the discussion; mark gaps as open.
