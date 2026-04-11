---
description: Bootstrap the development framework into a new project. Scans the codebase, populates CLAUDE.md, creates path-scoped rules, and configures hooks.
---

# Bootstrap

You are bootstrapping the solo-dev framework into this project. This is a first-run
setup that discovers project context and configures the development environment.

Read `.claude/PRINCIPLES.md` first. These principles govern all work in this project.

## Step 1: Check framework integrity

Verify that the framework is complete:

```bash
echo "=== Framework Check ==="
[ -f .claude/framework.json ] && echo "OK: framework.json" || echo "MISSING: framework.json"
[ -d .claude/agents ] && echo "OK: agents/" || echo "MISSING: agents/"
[ -d .claude/skills ] && echo "OK: skills/" || echo "MISSING: skills/"
[ -d .claude/hooks ] && echo "OK: hooks/" || echo "MISSING: hooks/"
[ -d .claude/rules ] && echo "OK: rules/" || echo "MISSING: rules/"
[ -d .claude/commands ] && echo "OK: commands/" || echo "MISSING: commands/"
[ -f .claude/PRINCIPLES.md ] && echo "OK: PRINCIPLES.md" || echo "MISSING: PRINCIPLES.md"
[ -f .claude/CLAUDE.md.template ] && echo "OK: CLAUDE.md.template" || echo "MISSING: CLAUDE.md.template"
```

If anything is MISSING, stop and tell the user to copy the complete `.claude/` framework directory.

## Step 2: Check for existing bootstrap

```bash
grep -q "BOOTSTRAPPED" CLAUDE.md 2>/dev/null && echo "ALREADY_BOOTSTRAPPED" || echo "FRESH"
```

If ALREADY_BOOTSTRAPPED: ask the user via AskUserQuestion:
- A) Re-bootstrap (rescan and overwrite CLAUDE.md project sections)
- B) Skip (keep current config, just verify hooks)

If they choose B, jump to Step 9.

## Step 3: Make hooks executable

```bash
chmod +x .claude/hooks/*.sh
```

## Step 4: Discover the project

Scan the codebase. Run these commands and read the output:

```bash
# Project structure
find . -maxdepth 3 -type f \
  -not -path './.git/*' \
  -not -path './node_modules/*' \
  -not -path './target/*' \
  -not -path './.claude/*' \
  -not -path './venv/*' \
  -not -path './__pycache__/*' | head -80

# Package/build files
cat README.md 2>/dev/null | head -50
cat package.json 2>/dev/null | jq '.scripts' 2>/dev/null
cat Cargo.toml 2>/dev/null | head -30
cat pyproject.toml 2>/dev/null | head -30
cat Makefile 2>/dev/null | head -30
cat CMakeLists.txt 2>/dev/null | head -20
cat go.mod 2>/dev/null | head -10

# Git state
git log --oneline -10 2>/dev/null
git branch -a 2>/dev/null

# Test discovery
find . -name '*.test.*' -o -name '*_test.*' -o -name 'test_*' -o -name '*_spec.*' 2>/dev/null | head -20

# Existing conventions
ls .editorconfig .prettierrc .eslintrc* .rustfmt.toml .clang-format clippy.toml 2>/dev/null
```

## Step 5: Identify key information

From the scan, determine:
- **Stack**: Languages, frameworks, key dependencies
- **Build command**: How to compile/build (e.g., `cargo build`, `npm run build`)
- **Test command**: How to run tests (e.g., `cargo test`, `pytest`)
- **Lint/format command**: How to lint (e.g., `cargo clippy`, `eslint .`)
- **Run command**: How to run the project (e.g., `cargo run`, `npm start`)
- **Entry points**: Main source directories
- **Read-only files**: Lock files, generated files, vendored deps

## Step 6: Confirm with user

Present the detected config via AskUserQuestion:

> Here's what I detected:
> - **Stack**: {detected}
> - **Build**: `{detected}`
> - **Test**: `{detected}`
> - **Lint**: `{detected}`
> - **Run**: `{detected}`
> - **Read-only**: {detected}
>
> A) Looks correct, proceed
> B) Let me correct some values

If B, ask for the corrected values.

## Step 7: Generate CLAUDE.md

Copy `.claude/CLAUDE.md.template` to `./CLAUDE.md`. Replace all `_TBD_` placeholders
with the confirmed values from Step 6.

For the Architecture section, write a brief outline based on the directory structure
and source files discovered. Keep it factual (module names, what each contains). The
user and agents will flesh it out over time.

## Step 8: Create language-specific rules

Based on detected stack, create or update rules in `.claude/rules/`:

**Rust** → update `.claude/rules/rust.md` with project-specific conventions found in
the codebase (formatting config, key types, module layout).

**Python** → create `.claude/rules/python.md`:
```yaml
---
paths:
  - "**/*.py"
---
```
With conventions detected from pyproject.toml, setup.cfg, or existing code style.

**TypeScript/JavaScript** → create `.claude/rules/typescript.md` or `javascript.md`.

**Go** → create `.claude/rules/go.md`.

**C/C++** → create `.claude/rules/cpp.md`.

Only create rules for languages actually present in the project.

## Step 9: Configure review gates

Ask the user via AskUserQuestion:

> The framework includes two AI review gates powered by headless Sonnet:
>
> - **Per-edit review** (`sonnet-review-gate.sh`) — reviews every Write/Edit call. Thorough but slower.
> - **Commit review** (`commit-review-gate.sh`) — reviews the full staged diff at commit time. One review per commit.
>
> Both use tool-augmented Sonnet that can read project files, conventions, and grep for callers.
>
> Which review mode do you want?
>
> A) **Commit only** (recommended) — review at commit time, no per-edit overhead
> B) **Per-edit only** — review every file change, no commit gate
> C) **Both** — maximum coverage, slower workflow
> D) **None** — disable AI review gates entirely

Based on the user's choice, update `.claude/settings.json`:

- **A (commit only)**: Remove the `sonnet-review-gate.sh` hook entry from the `Write|Edit|MultiEdit` matcher. Keep `commit-review-gate.sh` under the `Bash` matcher.
- **B (per-edit only)**: Add `sonnet-review-gate.sh` to the `Write|Edit|MultiEdit` hooks (after readonly-guard, with timeout 130000). Remove `commit-review-gate.sh` from the `Bash` matcher.
- **C (both)**: Add `sonnet-review-gate.sh` to the `Write|Edit|MultiEdit` hooks (after readonly-guard, with timeout 130000). Keep `commit-review-gate.sh` under the `Bash` matcher.
- **D (none)**: Remove `sonnet-review-gate.sh` from `Write|Edit|MultiEdit` if present. Remove `commit-review-gate.sh` from `Bash` matcher.

After updating settings.json, also update the "Review System" section in CLAUDE.md to reflect which gates are active.

## Step 10: Configure metric (optional)

Ask the user via AskUserQuestion:

> The autoresearch skill needs a metric to optimize. Configure now or skip?
>
> If configuring, provide:
> - Metric name (e.g., `val_bpb`, `test_pass_rate`, `latency_p99_ms`)
> - Direction: lower is better / higher is better
> - Run command (e.g., `uv run train.py`, `cargo test`)
> - Extract command (e.g., `grep "^val_bpb:" run.log`)
> - Time budget per experiment (default: 300s)
>
> A) Configure metric now
> B) Skip (can configure later by editing CLAUDE.md)

If A, update the Metric section of CLAUDE.md with their values.

## Step 11: Verify hooks

Test that all hooks are functional:

```bash
echo "=== Hook Verification ==="

# Test readonly-guard with a file from Read-only list
echo '{"tool_input":{"file_path":"Cargo.lock","content":"test"}}' | bash .claude/hooks/readonly-guard.sh 2>/dev/null
RG_EXIT=$?
[ $RG_EXIT -eq 2 ] && echo "OK: readonly-guard blocks protected files" || echo "WARN: readonly-guard may not be working (exit $RG_EXIT)"

# Test readonly-guard allows normal files
echo '{"tool_input":{"file_path":"src/main.rs","content":"test"}}' | bash .claude/hooks/readonly-guard.sh 2>/dev/null
RG_EXIT=$?
[ $RG_EXIT -eq 0 ] && echo "OK: readonly-guard allows normal files" || echo "WARN: readonly-guard blocking normal files (exit $RG_EXIT)"

# Check sonnet-review-gate dependency
which claude >/dev/null 2>&1 && echo "OK: claude CLI available for sonnet-review-gate" || echo "WARN: claude CLI not found — sonnet-review-gate will auto-approve"

# Check test command works
echo "Testing: {test_command}..."
```

Run the detected test command with a short timeout to verify it works. Report pass/fail.

## Step 12: Create .gitignore files for Claude-generated artifacts

Create two `.gitignore` files — one inside `.claude/` for Claude-internal generated
files, and one update to the project root `.gitignore` for project-level outputs.

### `.claude/.gitignore` (Claude-internal)

Create `.claude/.gitignore` with these patterns:

```bash
cat > .claude/.gitignore << 'GITIGNORE'
# Local settings (per-developer)
settings.local.json

# Agent memory (session-generated, not portable)
agent-memory/
GITIGNORE

echo "OK: .claude/.gitignore created"
```

### Project `.gitignore` (project-level outputs)

Append Claude output patterns to the project's `.gitignore`. Only add entries that
aren't already present:

```bash
touch .gitignore

for pattern in \
  "docs/sessions/" \
  "results.tsv" \
  "run.log" \
  "evals/results/" \
  "*.log"; do
  grep -qxF "$pattern" .gitignore 2>/dev/null || echo "$pattern" >> .gitignore
done

echo "OK: .gitignore updated with Claude output patterns"
```

## Step 13: Create directories

```bash
mkdir -p docs/plans docs/sessions docs/wiki
```

Create the wiki index:

```bash
cat > docs/wiki/index.md << 'WIKI_INDEX'
# Project Wiki

Knowledge base maintained by the context-keeper. Pages are synthesized from
session findings, reviewer feedback, experiments, and architectural decisions.

## Pages

_No pages yet. The context-keeper will populate this as the project evolves._
WIKI_INDEX
```

## Step 14: Commit

Stage framework files, wiki, and both `.gitignore` files:

```bash
git add CLAUDE.md .gitignore .claude/.gitignore \
  .claude/framework.json .claude/PRINCIPLES.md \
  .claude/CLAUDE.md.template \
  .claude/agents/ .claude/hooks/ .claude/rules/ \
  .claude/commands/ .claude/skills/ .claude/settings.json \
  docs/ docs/wiki/
git commit -m "chore: bootstrap solo-dev framework"
```

Do NOT stage `.claude/settings.local.json` or `.claude/agent-memory/`.

## Step 15: Summary

Print what was configured:
- Stack and commands detected
- Rules files created
- Hooks status (working/warning)
- Metric configured or skipped
- `.gitignore` files created (`.claude/.gitignore` + project root)

## Step 16: Next steps

Tell the user bootstrap is complete, then prompt them to relaunch with the orchestrator:

> Bootstrap complete. To start working with the full agent pipeline, relaunch with:
>
> ```
> claude --agent orchestrator
> ```
