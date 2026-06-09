# Development Workflow & Framework

hyzero runs on a thin-orchestrator workflow: the main Claude session decides
*what to do* but never reads source >50 lines, edits files, or runs
content-spilling Bash. All source-touching work happens inside subagents —
`analyst` for read-only research and review, `developer` for worktree-isolated
edits — which return ≤20-line summaries. PreToolUse hooks (`read-route.sh`,
`edit-route.sh`, `bash-route.sh`) enforce the boundary so the orchestrator's
context stays small and routing stays honest. The authoritative spec is
`CLAUDE.md` (Workflow + Orchestrator Principles).

## Roles (`.claude/agents/`)

- **Orchestrator (main session).** Receives prompts, clarifies ambiguity,
  dispatches subagents, sequences multi-step work, presents summaries. Cannot Read
  source >50 lines, cannot Edit/Write/MultiEdit, cannot run content-spilling Bash
  (`cat`, `head`, `tail`, `less`, `more`, `git diff`/`show`/`blame`/`log -p`,
  `grep -A/-B/-C`, `awk`, `sed`, `tee`). Allowed Bash is state-query only:
  `git status`, `git log --oneline`, `git rev-parse`, `git branch`, `ls`, `find`,
  `wc -l`, `grep -l`, plus the `Glob` tool.
- **`analyst` (read-only).** Investigates, summarizes, reviews diffs. Tools: Read,
  Grep, Glob, Bash. Returns a ≤20-line synthesis; bulky artifacts go to
  `runs/{timestamp}/`. May write `docs/wiki/{topic}.md` only when paired with the
  user-set `.claude/state/wiki-approved` flag, and plan/research artifacts under
  `docs/plans/{feature}/` freely.
- **`developer` (worktree-isolated).** Implements per brief; cannot design, plan,
  or expand scope. Tools: Read, Write, Edit, Bash, Grep, Glob. Operates in a
  sibling git worktree on a `feat/`, `fix/`, or `refactor/` branch — cannot see
  the orchestrator's main worktree filesystem.

## Hooks (`.claude/hooks/`)

- `read-route.sh` (PreToolUse Read) — orchestrator may not Read source >50 lines;
  subagents (those with an `agent_id` in stdin) are exempt.
- `edit-route.sh` (PreToolUse Edit/Write/MultiEdit) — orchestrator may not edit;
  it dispatches `developer`. Subagents exempt.
- `bash-route.sh` (PreToolUse Bash) — orchestrator may not run content-spilling
  commands. Subagents exempt.
- `wiki-gate.sh` (Pre/PostToolUse Edit/Write/MultiEdit on `docs/wiki/*`) — blocks
  unless `.claude/state/wiki-approved` exists; clears the flag after a successful
  write (one-shot).
- `wiki-gate-bash.sh` (Pre/PostToolUse Bash) — same gate for shell redirects
  (`>`, `>>`, `tee`) into `docs/wiki/*`; applies in all contexts (no agent
  exemption).
- `test-gate.sh` (Stop) — runs the project's test command on session end; exit 2
  keeps Claude looping until tests pass.
- `commit-review-gate.sh` (PreToolUse Bash on `git commit*`) — full staged-diff
  review; exit 2 denies the commit.
- `auto-format.sh` (PostToolUse Write/Edit/MultiEdit) — runs `rustfmt` (and
  analogous formatters) on the changed file. Always exits 0 (advisory).
- `agent-trace.sh`, `safety-net.sh`, `test-safety-net.sh` — tracing and safety
  rails on destructive operations and test-state hygiene.

## Skills (`.claude/skills/`)

- `/verify` — analyst reviews the dirty diff against `.claude/rules/`, iterating
  with developer until APPROVE; the user only sees the final APPROVE.
- `/compact` — synthesize session findings into the wiki (focused after a
  `/verify` APPROVE; session-wide at end of session).
- `/autoloop` — autonomous propose → implement → verify → benchmark → keep/discard
  loop against a user-specified baseline metric.
- `/plan-and-develop` — multi-phase planning for changes touching several files;
  analyst writes `docs/plans/{feature}/{plan,research}.md` first.

`/approve-wiki` (sets the one-shot wiki-approved flag) and `/bootstrap` (one-time
project setup that fills `CLAUDE.md`) are referenced by the workflow but are not
standalone skill directories.

## Typical Flow

1. **Research** → orchestrator dispatches `analyst` → ≤20-line summary.
2. **Implementation** → orchestrator dispatches `developer` (worktree-isolated on
   `feat/{name}` / `fix/{name}`) → developer edits, runs `cargo test`, returns a
   summary with branch name + test status.
3. **Verification** → orchestrator invokes `/verify` → analyst reviews the diff
   against `.claude/rules/`, iterating with developer on REJECT until APPROVE.
4. **Wiki update (user-gated)** → on APPROVE the orchestrator asks; if the user
   types `/approve-wiki`, the flag is set, then `/compact` dispatches analyst to
   write `docs/wiki/{topic}.md`.
5. **Commit** → user-explicit only; `commit-review-gate.sh` reviews the staged
   diff before the commit lands.

## Conventions (`.claude/rules/`)

- `git.md` — commit format `{scope}: {description}` (lowercase, imperative); one
  logical change per commit; branches not main; never force-push main;
  worktree branches `feat/`, `fix/`, `refactor/`.
- `testing.md` — behavior-named tests, flaky 3× policy, regression tests must fail
  without the fix (see [Testing Procedures](testing.md)).

## Gotchas

- **Developer worktree isolation.** Sibling worktrees cannot see the main
  worktree's filesystem (or each other's). Work prototyped in a worktree and
  reaped before commit is lost. Use `analyst` (runs against main's filesystem) for
  cross-worktree visibility.
- **Wiki-gate covers Bash redirects too.** `wiki-gate-bash.sh` closes the
  heredoc/redirect bypass; both gates require `/approve-wiki` per write (one-shot).
- **The wiki-approved flag is one-shot.** It is consumed (deleted) after each
  successful `docs/wiki/*` write; re-arm before the next write.
- **No agent-memory directory.** Durable knowledge lives only in `docs/wiki/`,
  updated through the `/verify` → `/approve-wiki` → `/compact` flow.

## Related

- [`CLAUDE.md`](../../CLAUDE.md) — project-root Workflow + Orchestrator Principles
- [`.claude/agents/`](../../.claude/agents/), [`.claude/hooks/`](../../.claude/hooks/), [`.claude/skills/`](../../.claude/skills/), [`.claude/rules/`](../../.claude/rules/)
- [Testing Procedures](testing.md) — the test-gate and conventions
