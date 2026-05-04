---
name: compact
description: Synthesize session findings, recent changes, and reviewer feedback into the project wiki at docs/wiki/. Use at end of session, before /clear, or periodically as hygiene to ensure knowledge compounds across sessions.
---

# /compact

Compact what this session learned into `docs/wiki/` so the next session can recover it without re-discovering. Wiki pages are durable understanding — not changelogs, not session logs. Each page reads as current truth, written for a future reader who has no memory of how it got that way.

Test for inclusion: would a future agent need this fact to make a good decision? If no, it doesn't belong.

## Scope: focused vs. session-wide

- **Focused** (chained from `/verify` on APPROVE) — operate only on the change just approved. Pull the diff, wiki pages overlapping the changed files, and session notes referencing them. Don't load the full session.
- **Session-wide** (manual, end of session) — sweep all unmerged findings.

In focused mode, skip Step 1's broad scans. Read only inputs the change actually touches.

After compact, invoke `/clear` to reset the orchestrator. Verify → compact → clear is the unit boundary that keeps the orchestrator clean between routing decisions.

## Step 1: Read the existing wiki

Don't re-derive findings from `git log`/`diff` — synthesize from session memory. Read existing wiki only to avoid duplicating: `ls docs/wiki/*.md`, `docs/wiki/index.md`, and any page whose topic overlaps with the change at hand.

## Step 2: Map findings to wiki pages

| Type of finding | Wiki destination |
|---|---|
| Architecture decision | `docs/wiki/{subsystem}.md` → `## Key decisions` |
| Pattern discovery | `docs/wiki/patterns.md` or domain-specific page |
| Constraint identified | `docs/wiki/{domain}.md` → `## Gotchas` |
| Bug rationale / fix | `docs/wiki/{subsystem}.md` → `## Gotchas` |
| Reviewer recurring issue | `docs/wiki/{domain}.md` → `## Gotchas` |

## Step 3: Update or create pages

You synthesize content from session context (decisions, summaries, findings). The actual write goes through analyst with the user-approval gate.

For each page that needs updating:

1. Present to user verbatim:
   ```
   Page: docs/wiki/{topic}.md
   Proposed content: <synthesized text>
   Save? (yes / iterate / discard)
   ```
2. Wait for user response.
3. On "yes": user invokes `/approve-wiki`. Then dispatch `analyst`:
   > Write/update `docs/wiki/{topic}.md` with this content: [paste synthesized text].
4. The `wiki-gate.sh` hook permits the write; flag clears post-write (one-shot).

Repeat per page. Each page requires its own `/approve-wiki`.

Page structure: one-paragraph synthesized summary (current truth, not history), `## Key decisions`, `## Gotchas`, `## Related`.

Constraints:
- Each page ≤100 lines. Split if larger.
- Every page has a `## Related` section.
- When updating a page, check pages it links to. Update them if affected.

## Step 4: Finalize

- **Update `docs/wiki/index.md`**: same gate flow as Step 3 — synthesize index changes, present, user invokes `/approve-wiki`, dispatch analyst.
- **Lint** via state queries: `wc -l CLAUDE.md` (≤150?), `find docs/wiki -name '*.md' -exec wc -l {} +` (any over 100?), check for orphan pages and unresolved `> CONTRADICTION:` markers. For any fixes, dispatch analyst (each fix needs its own approval).
- **Log append**: `echo "$(date -I) — Compact: {summary}" >> docs/log.md` — orchestrator runs this directly (Bash redirect to non-wiki paths is unblocked).

## Report

- Wiki pages created or updated
- Contradictions resolved (or flagged)
- Stale content cleaned
- Open questions needing user input
