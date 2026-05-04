# Compact

Triggers the context-keeper to synthesize accumulated knowledge into the project wiki.
Merges session summaries, reviewer findings, experiment results, and agent memory into
`docs/wiki/` pages — then runs a lint pass to clean up contradictions and stale references.

Use this when context is getting long, before `/clear`, or at the end of a work session
to ensure nothing is lost.

## When to use

- Before running `/clear` to reset context — compact first so findings persist
- After a long session with multiple tasks
- After autoresearch runs to synthesize experiment findings
- When you notice agents re-discovering things from previous sessions
- Periodically as hygiene (e.g., end of day)

## Workflow

### Step 1: Gather unmerged knowledge

Collect everything that hasn't been synthesized into the wiki yet:

```bash
# Recent session summaries
ls -t docs/sessions/*.md 2>/dev/null | head -10

# Recent git activity for context
git log --oneline -20

# Current wiki state
ls docs/wiki/ 2>/dev/null

# Current knowledge log
tail -20 docs/log.md 2>/dev/null
```

Read the recent session summaries. Read `docs/wiki/index.md` to understand what wiki
pages already exist.

### Step 2: Identify gaps

Compare what's in the session summaries and recent git history against what's already
in the wiki. Look for:

- New topics that don't have wiki pages yet
- Existing pages that are missing recent findings
- Reviewer findings or bug fixes that revealed gotchas not yet documented
- Architecture decisions made but not recorded
- Experiment results not yet synthesized

### Step 3: Update the wiki

For each gap found, either create a new page or update an existing one.

**Creating a new page** (`docs/wiki/{topic}.md`):
```markdown
# {Topic}

{Synthesized understanding from all available sources.}

## Key decisions
- {Decision}: {why, what alternatives were rejected}

## Gotchas
- {Non-obvious things that caused bugs or confusion}

## Related
- [{Other topic}](other-topic.md) — {relationship}
```

**Updating an existing page**: Rewrite sections to incorporate new knowledge. The page
should read as current truth, not as a changelog. Don't just append — synthesize.

**Cross-references**: When updating a page, check all pages listed in its `## Related`
section. Update them too if the new information affects them. Add new cross-references
if the update connects to pages not yet linked.

### Step 4: Update the index

Update `docs/wiki/index.md` to include any new pages. Group pages by domain, not
alphabetically.

### Step 5: Mistake pattern detection

Read `docs/wiki/mistakes.md`. Analyze logged mistakes for systemic patterns:

1. **Type clustering**: Are 3+ mistakes sharing the same error type (e.g., `missing-context`,
   `wrong-assumption`)? This signals a pipeline problem, not an agent problem.
   - `missing-context` cluster → orchestrator briefs need more context, or wiki is incomplete
   - `wrong-assumption` cluster → researcher isn't verifying claims against code
   - `breaking-change` cluster → reviewer needs to grep for callers more aggressively
   - `stale-context` cluster → wiki pages are outdated, run lint pass

2. **Domain clustering**: Are 3+ mistakes in the same domain? That area needs:
   - A dedicated wiki page if one doesn't exist
   - Richer gotchas if the page exists but mistakes keep happening
   - A path-scoped rule in `.claude/rules/` for automated enforcement

3. **Escalation audit**: Are any mistakes still at **gotcha** tier but recurring?
   Escalate to **rule** or **hook**. The goal is: every recurring mistake eventually
   becomes automated prevention.

4. **Root cause depth**: Are root cause chains consistently shallow ("wrong output")?
   Push for deeper analysis — the systemic cause is what prevents recurrence.

Report patterns found and escalation recommendations.

### Step 6: Lint pass

Audit all documentation for health:

1. **CLAUDE.md**: Still accurate? Under 150 lines? Commands still correct?
2. **`.claude/rules/`**: Any rules referencing deleted files or functions? Any orphan
   rules whose path globs match nothing?
3. **Wiki pages**: Any pages over 100 lines that should be split? Any orphan pages
   not linked from index or other pages? Any `CONTRADICTION` flags that can now be
   resolved?
4. **Cross-references**: Grep wiki pages for topics mentioned but not linked.
5. **Mistakes log**: Any mistakes marked "pending" escalation that can now be resolved?

Fix issues in-place.

### Step 7: Log and summarize

Append to `docs/log.md`:
```
YYYY-MM-DD — Compact: {summary of what was merged/created/cleaned}
```

Report to the user:
- Wiki pages created or updated
- Contradictions found and resolved (or flagged)
- Stale content cleaned up
- Any open questions that need user input
