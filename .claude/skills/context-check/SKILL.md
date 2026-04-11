# Context Check

Audits what each agent loads into its context window and reports sizes, staleness,
and redundancy. Use this to understand where context budget is being spent and
whether agents are loading the right knowledge.

## When to use

- When agents seem slow or are hitting context limits
- When you suspect agents are loading stale or redundant information
- After adding wiki pages — to verify agents will pick them up
- As a diagnostic when agents produce poor results (maybe they're missing context)
- Before optimizing CLAUDE.md or wiki page sizes

## Workflow

### Step 1: Measure always-loaded context

These files load into every agent's context automatically:

```bash
echo "=== Always-Loaded Context ==="

# CLAUDE.md — loaded every session
if [ -f CLAUDE.md ]; then
  LINES=$(wc -l < CLAUDE.md)
  BYTES=$(wc -c < CLAUDE.md)
  echo "CLAUDE.md: ${LINES} lines, ${BYTES} bytes"
  [ "$LINES" -gt 150 ] && echo "  ⚠️ Over 150-line limit"
fi

# PRINCIPLES.md — loaded every session
if [ -f .claude/PRINCIPLES.md ]; then
  LINES=$(wc -l < .claude/PRINCIPLES.md)
  BYTES=$(wc -c < .claude/PRINCIPLES.md)
  echo "PRINCIPLES.md: ${LINES} lines, ${BYTES} bytes"
fi

# Rules without paths (always loaded)
echo ""
echo "=== Always-Loaded Rules ==="
for f in .claude/rules/*.md; do
  [ ! -f "$f" ] && continue
  if ! grep -q "^paths:" "$f" 2>/dev/null; then
    LINES=$(wc -l < "$f")
    BYTES=$(wc -c < "$f")
    echo "$(basename "$f"): ${LINES} lines, ${BYTES} bytes (always loaded — no paths filter)"
  fi
done

echo ""
echo "=== Path-Scoped Rules ==="
for f in .claude/rules/*.md; do
  [ ! -f "$f" ] && continue
  if grep -q "^paths:" "$f" 2>/dev/null; then
    LINES=$(wc -l < "$f")
    PATHS=$(grep "^  - " "$f" | sed 's/.*- "//' | tr -d '"' | paste -sd, -)
    echo "$(basename "$f"): ${LINES} lines — triggers on: ${PATHS}"
  fi
done
```

### Step 2: Measure wiki context

```bash
echo ""
echo "=== Wiki Pages ==="
TOTAL_WIKI_LINES=0
TOTAL_WIKI_BYTES=0
if [ -d docs/wiki ]; then
  for f in docs/wiki/*.md; do
    [ ! -f "$f" ] && continue
    LINES=$(wc -l < "$f")
    BYTES=$(wc -c < "$f")
    TOTAL_WIKI_LINES=$((TOTAL_WIKI_LINES + LINES))
    TOTAL_WIKI_BYTES=$((TOTAL_WIKI_BYTES + BYTES))
    STATUS=""
    # Check staleness — last modified vs last commit touching the related code
    DAYS_OLD=$(( ($(date +%s) - $(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo 0)) / 86400 ))
    [ "$DAYS_OLD" -gt 14 ] && STATUS=" ⚠️ ${DAYS_OLD} days old"
    [ "$LINES" -gt 100 ] && STATUS="${STATUS} ⚠️ over 100-line limit"
    echo "$(basename "$f"): ${LINES} lines, ${BYTES} bytes${STATUS}"
  done
  echo "TOTAL wiki: ${TOTAL_WIKI_LINES} lines, ${TOTAL_WIKI_BYTES} bytes"
fi
```

### Step 3: Measure agent-specific context

```bash
echo ""
echo "=== Agent Memory (Researcher) ==="
TOTAL_MEM_LINES=0
if [ -d .claude/agent-memory/researcher ]; then
  for f in .claude/agent-memory/researcher/*.md; do
    [ ! -f "$f" ] && continue
    [[ "$(basename "$f")" == "README.md" ]] && continue
    LINES=$(wc -l < "$f")
    TOTAL_MEM_LINES=$((TOTAL_MEM_LINES + LINES))
    MERGED=""
    grep -q "Merged into wiki" "$f" && MERGED=" (merged)"
    echo "$(basename "$f"): ${LINES} lines${MERGED}"
  done
  echo "TOTAL agent memory: ${TOTAL_MEM_LINES} lines"
fi

echo ""
echo "=== Session Summaries ==="
if [ -d docs/sessions ]; then
  COUNT=$(ls docs/sessions/*.md 2>/dev/null | wc -l)
  TOTAL_LINES=0
  for f in docs/sessions/*.md; do
    [ ! -f "$f" ] && continue
    TOTAL_LINES=$((TOTAL_LINES + $(wc -l < "$f")))
  done
  echo "${COUNT} sessions, ${TOTAL_LINES} total lines"
  # Check for unmerged sessions
  echo "Most recent 5:"
  ls -t docs/sessions/*.md 2>/dev/null | head -5 | while read f; do
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
  done
fi

echo ""
echo "=== Plans ==="
if [ -d docs/plans ]; then
  for d in docs/plans/*/; do
    [ ! -d "$d" ] && continue
    PLAN_LINES=0
    for f in "$d"*.md; do
      [ ! -f "$f" ] && continue
      PLAN_LINES=$((PLAN_LINES + $(wc -l < "$f")))
    done
    echo "$(basename "$d"): ${PLAN_LINES} lines"
  done
fi
```

### Step 4: Per-agent context profile

Build a report showing what each agent would load for a typical task:

**Orchestrator** (opus):
- Always: CLAUDE.md + PRINCIPLES.md + agent definition
- Per-task: wiki index + relevant wiki pages
- Budget impact: {estimate}

**Researcher** (haiku):
- Always: CLAUDE.md + PRINCIPLES.md + agent definition
- Per-task: agent-memory files + wiki index + relevant wiki pages
- Budget impact: {estimate}

**Planner** (sonnet):
- Always: CLAUDE.md + PRINCIPLES.md + agent definition
- Per-task: context summary + relevant wiki pages
- Budget impact: {estimate}

**Implementer** (sonnet):
- Always: CLAUDE.md + PRINCIPLES.md + agent definition + path-scoped rules
- Per-task: context summary + subtask + relevant wiki pages
- Budget impact: {estimate}

**Reviewer** (sonnet):
- Always: CLAUDE.md + PRINCIPLES.md + agent definition + path-scoped rules
- Per-task: diff + context summary + relevant wiki pages
- Budget impact: {estimate}

**Context-keeper** (haiku):
- Always: CLAUDE.md + PRINCIPLES.md + agent definition
- Per-task: session summaries + wiki pages + results.tsv + log.md
- Budget impact: {estimate}

Read each agent definition file to get accurate line counts:

```bash
echo ""
echo "=== Agent Definitions ==="
for f in .claude/agents/*.md; do
  [ ! -f "$f" ] && continue
  LINES=$(wc -l < "$f")
  BYTES=$(wc -c < "$f")
  echo "$(basename "$f"): ${LINES} lines, ${BYTES} bytes"
done
```

### Step 5: Identify issues

Report:
- **Bloated**: Any file over its size limit (CLAUDE.md > 150 lines, rules > 50, wiki > 100)
- **Stale**: Wiki pages or agent memory not updated in > 14 days
- **Unmerged**: Agent memory files not yet synthesized into wiki (missing "Merged into wiki" marker)
- **Redundant**: Information duplicated between CLAUDE.md, rules, and wiki pages
- **Missing**: Agents that should be reading wiki pages but the relevant pages don't exist
- **Orphaned**: Wiki pages not referenced by index.md or any other page

### Step 6: Recommendations

Based on findings, suggest:
- Files to trim or split
- Agent memory to compact (recommend running `/compact-agents`)
- Sessions to merge into wiki (recommend running `/compact`)
- Rules to consolidate or remove
- Wiki pages to create for uncovered domains
