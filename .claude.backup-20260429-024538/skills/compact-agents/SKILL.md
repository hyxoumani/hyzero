# Compact Agents

Drains agent-accumulated knowledge into the context-keeper's wiki. Each agent builds
up its own memory and findings during work — this skill synthesizes all of it into
`docs/wiki/` pages so the knowledge compounds across sessions and is available to
every agent, not siloed in one agent's memory.

The difference from `/compact`:
- `/compact` merges **session summaries and experiment results** into the wiki
- `/compact-agents` merges **agent-specific memory and findings** into the wiki

## When to use

- After a series of research tasks where the researcher has built up agent-memory files
- When agents are re-discovering things that another agent already found
- Before archiving or clearing agent memory
- As part of end-of-session hygiene (run after `/compact`)
- When onboarding — to make sure all prior agent knowledge is in the wiki

## Workflow

### Step 1: Inventory agent knowledge

Scan all agent knowledge sources:

```bash
# All agent memory
echo "=== Agent Memory ==="
for dir in .claude/agent-memory/*/; do
  [ ! -d "$dir" ] && continue
  AGENT=$(basename "$dir")
  FILES=$(ls "$dir"*.md 2>/dev/null | grep -v README | wc -l)
  [ "$FILES" -gt 0 ] && echo "$AGENT: $FILES files"
  ls -la "$dir"*.md 2>/dev/null | grep -v README
done

# Recent researcher context summaries in plans
echo "=== Research Docs ==="
find docs/plans -name "research.md" 2>/dev/null

# Reviewer findings (in git log)
echo "=== Recent Reviews ==="
git log --oneline --all --grep="review\|REQUEST_CHANGES\|REJECT" -10 2>/dev/null

# Tester patterns (recurring failures)
echo "=== Recent Test Failures ==="
git log --oneline --all --grep="test.*fail\|fix.*test" -10 2>/dev/null

# Wiki current state for comparison
echo "=== Wiki Pages ==="
ls docs/wiki/*.md 2>/dev/null
```

Read each agent memory file in `.claude/agent-memory/*/` (all agents, not just researcher).
Read `docs/wiki/index.md` to understand existing wiki coverage.

### Step 2: Map knowledge to wiki pages

For each piece of agent knowledge, determine where it belongs:

| Agent knowledge | Wiki destination |
|----------------|-----------------|
| Researcher: architecture findings | `docs/wiki/{subsystem}.md` |
| Researcher: pattern discoveries | `docs/wiki/patterns.md` or domain-specific page |
| Researcher: constraint identification | `docs/wiki/{domain}.md` → Gotchas section |
| Orchestrator: integration decisions | `docs/wiki/{feature}.md` → Key decisions |
| Orchestrator: debugging traces | `docs/wiki/{subsystem}.md` → Gotchas section |
| Context-keeper: rule rationale | `docs/wiki/{domain}.md` → Key decisions |
| Planner: rejected approaches | `docs/wiki/{feature}.md` → Key decisions |
| Reviewer: recurring issues | `docs/wiki/{domain}.md` → Gotchas section |
| Tester: flaky test patterns | `docs/wiki/testing.md` → Known flakes |
| Implementer: workaround discoveries | `docs/wiki/{subsystem}.md` → Gotchas |

### Step 3: Synthesize into wiki

For each mapping identified:

**If wiki page exists**: Read the page. Merge the agent knowledge into the appropriate
section. Rewrite to synthesize — don't append raw findings. Update the `## Related`
section if the new knowledge connects to other pages.

**If wiki page doesn't exist**: Create it with the standard structure:

```markdown
# {Topic}

{Synthesized understanding from agent findings.}

## Key decisions
- {Decision}: {why, alternatives rejected}

## Gotchas
- {Non-obvious issues discovered by agents}

## Related
- [{Related topic}](related-topic.md) — {relationship}
```

**Cross-reference check**: For every page updated or created, read all pages in its
`## Related` section. Update them if the new knowledge affects them. Add new
cross-references if connections were discovered.

### Step 4: Reconcile contradictions

Compare agent knowledge against existing wiki content. If an agent found something
that contradicts the wiki:

- If the agent finding is verified (backed by code, tests, or recent git history):
  update the wiki to reflect the new truth
- If unclear which is correct: flag it with `> ⚠️ CONTRADICTION: {description}`
  and note both claims with their sources

### Step 5: Update agent memory status

After merging any agent's memory into the wiki, add a note at the bottom of each
processed agent-memory file:

```markdown
---
_Merged into wiki: YYYY-MM-DD — see docs/wiki/{page}.md_
```

This prevents re-processing the same findings. Do NOT delete agent memory files —
they serve as the raw source record.

### Step 6: Update index and log

Update `docs/wiki/index.md` with any new pages created.

Append to `docs/log.md`:
```
YYYY-MM-DD — Compact-agents: merged {N} agent memory files into wiki. Pages created: {list}. Pages updated: {list}.
```

### Step 7: Report

Summarize to the user:
- Agent memory files processed
- Wiki pages created or updated
- Contradictions found (resolved or flagged)
- Knowledge gaps identified (topics agents explored that still need deeper wiki coverage)
