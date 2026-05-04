---
name: compact
description: Synthesize session findings, recent changes, and reviewer feedback into the project wiki at docs/wiki/. Use at end of session, before /clear, or periodically as hygiene to ensure knowledge compounds across sessions.
---

# /compact

Compact what this session learned into `docs/wiki/` so the next session that needs this context can recover it without re-discovering. Wiki pages are durable understanding — not changelogs, not session logs. Each page reads as current truth, written for a future reader who has no memory of how it got that way.

Test for inclusion: would a future agent need this fact to make a good decision? If no, it doesn't belong.

## Scope: focused vs. session-wide

Two invocation modes:

- **Focused** (chained from `/verify` on APPROVE) — operate only on the change just approved. Pull the diff, the wiki pages whose topics overlap with the changed files, and any session notes referencing them. Don't load the full session.
- **Session-wide** (manual, end of session) — sweep all unmerged findings using the steps below.

In focused mode, skip Step 1's broad scans. Read only the inputs the change actually touches.

## Step 1: Know what's already in the wiki

You already hold the context of what happened — what was decided, what was tried, what was non-obvious. Don't re-derive that from `git log` or `git diff`. Synthesis comes from your own memory of the session.

What you do need to read is the existing wiki, so you update pages instead of duplicating them:

```bash
ls docs/wiki/*.md 2>/dev/null
```

Read `docs/wiki/index.md` and any page whose topic overlaps with the change at hand.

## Step 2: Map findings to wiki pages

| Type of finding | Wiki destination |
|---|---|
| Architecture decision | `docs/wiki/{subsystem}.md` → `## Key decisions` |
| Pattern discovery | `docs/wiki/patterns.md` or domain-specific page |
| Constraint identified | `docs/wiki/{domain}.md` → `## Gotchas` |
| Bug rationale / fix | `docs/wiki/{subsystem}.md` → `## Gotchas` |
| Reviewer recurring issue | `docs/wiki/{domain}.md` → `## Gotchas` |

## Step 3: Update or create pages

**New page** (`docs/wiki/{topic}.md`, kebab-case):

```markdown
# {Topic}

{Synthesized understanding — what's true now, not how we got here.}

## Key decisions
- {Decision}: {why, what alternatives were rejected}

## Gotchas
- {Non-obvious issue}: {how to avoid}

## Related
- [{other-topic}](other-topic.md) — {relationship}
```

**Existing page**: rewrite sections to incorporate new knowledge. Don't append — synthesize.

Constraints:
- Each page ≤100 lines. Split if larger.
- Every page has a `## Related` section.
- When updating a page, check pages it links to. Update them if affected.

## Step 4: Update the index

Update `docs/wiki/index.md` with any new pages, grouped by domain (not alphabetically).

## Step 5: Lint pass

- `CLAUDE.md` still under 150 lines? Commands still correct?
- Wiki pages over 100 lines? Split.
- Orphan pages not linked from index or any other page?
- `> CONTRADICTION:` markers that can now be resolved?

Fix issues in place.

## Step 6: Log

Append one line to `docs/log.md`:

```
YYYY-MM-DD — Compact: {summary of what was merged/created/cleaned}
```

## Report

Tell the user:
- Wiki pages created or updated
- Contradictions resolved (or flagged)
- Stale content cleaned
- Open questions needing user input
