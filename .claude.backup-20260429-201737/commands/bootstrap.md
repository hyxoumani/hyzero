---
description: First-run setup. Discover the project, populate CLAUDE.md, make hooks executable, and verify the framework is ready.
---

# /bootstrap

You are bootstrapping a new project with this template. Read `.claude/PRINCIPLES.md` first — every action follows from those 4 principles.

## Step 1: Framework integrity

```bash
echo "=== Framework Check ==="
[ -f .claude/PRINCIPLES.md ] && echo "OK: PRINCIPLES.md" || echo "MISSING: PRINCIPLES.md"
[ -f .claude/CLAUDE.md.template ] && echo "OK: CLAUDE.md.template" || echo "MISSING: CLAUDE.md.template"
[ -f .claude/framework.json ] && echo "OK: framework.json" || echo "MISSING: framework.json"
[ -f .claude/settings.json ] && echo "OK: settings.json" || echo "MISSING: settings.json"
[ -d .claude/agents ] && echo "OK: agents/" || echo "MISSING: agents/"
[ -d .claude/skills ] && echo "OK: skills/" || echo "MISSING: skills/"
[ -d .claude/hooks ] && echo "OK: hooks/" || echo "MISSING: hooks/"
[ -d .claude/rules ] && echo "OK: rules/" || echo "MISSING: rules/"
```

If anything is MISSING, stop and tell the user to copy the complete `.claude/` directory.

## Step 2: Already bootstrapped?

```bash
grep -q "BOOTSTRAPPED" CLAUDE.md 2>/dev/null && echo "ALREADY_BOOTSTRAPPED" || echo "FRESH"
```

If ALREADY_BOOTSTRAPPED, ask via AskUserQuestion: re-bootstrap (overwrite project sections) or skip (just verify).

## Step 3: Make hooks executable

```bash
chmod +x .claude/hooks/*.sh
```

## Step 4: Discover the project

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
cat go.mod 2>/dev/null | head -10

# Git state
git log --oneline -10 2>/dev/null
git branch -a 2>/dev/null

# Existing conventions
ls .editorconfig .prettierrc .eslintrc* .rustfmt.toml .clang-format clippy.toml 2>/dev/null
```

## Step 5: Identify config

From the scan, determine:

- **Stack**: languages, frameworks, key dependencies
- **Build command**
- **Test command**
- **Lint/format command**
- **Run command**
- **Read-only files**: lock files, generated files, vendored deps

## Step 6: Confirm with user

Present the detected config via AskUserQuestion. Let the user correct values before they get written to `CLAUDE.md`.

## Step 7: Generate CLAUDE.md

Copy `.claude/CLAUDE.md.template` to `./CLAUDE.md`. Replace `_TBD_` placeholders with the confirmed values. Write the test command inside a fenced bash block with a `# Test` comment on the line above — `test-gate.sh` parses this format.

For Architecture, write a brief outline based on the directory structure. Keep it factual (module names, what each contains). Detail will accumulate over time via `/compact`.

## Step 8: Create language-specific rules

If the stack includes Rust/Python/TypeScript/JavaScript/Go/C++, create `.claude/rules/{lang}.md` with:

```yaml
---
paths:
  - "**/*.{ext}"
---
# {Language} Conventions

- {Convention from detected config files}
```

Only create rules for languages actually present. Rules load only when matching files are touched, so cost is bounded.

## Step 9: Verify

```bash
echo "=== Hook Verification ==="

# Safety-net blocks rm -rf /
echo '{"tool_input":{"command":"rm -rf /"}}' | bash .claude/hooks/safety-net.sh 2>/dev/null
[ $? -eq 2 ] && echo "OK: safety-net blocks destructive commands" || echo "WARN: safety-net not blocking"

# Safety-net allows normal commands
echo '{"tool_input":{"command":"ls -la"}}' | bash .claude/hooks/safety-net.sh 2>/dev/null
[ $? -eq 0 ] && echo "OK: safety-net allows normal commands" || echo "WARN: safety-net over-blocking"

# Test command works
echo "Testing: $(grep -A1 '^# Test' CLAUDE.md | tail -1)"
```

Run the detected test command with a short timeout to verify it works.

## Step 10: .gitignore

Create `.claude/.gitignore`:

```
settings.local.json
```

Append to project `.gitignore` (skip duplicates):

```
docs/sessions/
runs/
results.tsv
*.log
.claude/settings.local.json
```

## Step 11: Create directories

```bash
mkdir -p docs/plans docs/sessions docs/wiki

cat > docs/wiki/index.md << 'EOF'
# Project Wiki

Knowledge base maintained via `/compact`. Pages are synthesized from session findings, reviewer feedback, and architectural decisions.

## Pages

_No pages yet. Run `/compact` at the end of a session to populate._
EOF
```

## Step 12: Commit

```bash
git add CLAUDE.md .gitignore .claude/.gitignore .claude/ docs/
git commit -m "chore: bootstrap claude-template framework"
```

## Step 13: Summary

Print:
- Stack and commands detected
- Rules files created
- Hooks status
- Where to start: just run `claude` and ask for what you need. Use `/plan-and-develop` for medium-large work, `/verify` after changes, `/compact` end of session.
