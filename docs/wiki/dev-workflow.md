# Development Workflow & Framework

The hyzero repository runs on a thin-orchestrator workflow: the main Claude session decides *what to do* but never reads source >50 lines, edits files, or runs content-spilling Bash. All source-touching work happens inside subagents — `analyst` for read-only research and review, `developer` for worktree-isolated edits — which return ≤20-line summaries. PreToolUse hooks (`read-route.sh`, `edit-route.sh`, `bash-route.sh`) enforce the boundary so the orchestrator's context stays small and routing stays honest. This page documents the current architecture; for the historical four-agent (researcher/implementer/verifier/orchestrator) framework, see the note at the top of `mistakes.md`.

## Roles

- **Orchestrator (main session).** Receives user prompts, clarifies ambiguity, dispatches subagents, sequences multi-step work, presents subagent summaries to the user. Cannot Read source >50 lines, cannot Edit/Write/MultiEdit, cannot run `cat`/`head`/`tail`/`less`/`more`/`git diff`/`git show`/`git blame`/`git log -p`/`grep -A/-B/-C`/`awk`/`sed`/`tee`. Allowed Bash is state-query only: `git status`, `git log --oneline`, `git rev-parse`, `git branch`, `ls`, `find`, `wc -l`, `grep -l`.
- **`analyst` (read-only).** Investigates, summarizes, reviews diffs. Tools: Read, Grep, Glob, Bash. Returns ≤20-line synthesized answer; bulky artifacts go to `runs/{timestamp}/proposal.md`. Exception: may write `docs/wiki/{topic}.md` when wiki-write brief is paired with the user-set `.claude/state/wiki-approved` flag, and may write plan/research artifacts under `docs/plans/{feature-name}/` freely.
- **`developer` (worktree-isolated).** Implements changes per brief; cannot design, plan, or expand scope. Tools: Read, Write, Edit, Bash, Grep, Glob. Operates in a sibling git worktree on a `feat/`, `fix/`, or `refactor/` branch — cannot see the orchestrator's main worktree filesystem.

## Hooks

- `read-route.sh` (PreToolUse Read) — orchestrator may not Read source files >50 lines; subagents (those with `agent_id` in stdin) are exempt.
- `edit-route.sh` (PreToolUse Edit/Write/MultiEdit) — orchestrator may not edit; dispatches `developer` instead. Subagents exempt.
- `bash-route.sh` (PreToolUse Bash) — orchestrator may not run content-spilling commands. Subagents exempt.
- `wiki-gate.sh` (Pre/PostToolUse Edit/Write/MultiEdit on `docs/wiki/*`) — blocks unless `.claude/state/wiki-approved` flag exists; clears the flag after a successful write (one-shot).
- `wiki-gate-bash.sh` (Pre/PostToolUse Bash) — same gate for shell redirects (`>`, `>>`, `tee`) into `docs/wiki/*`. Closes the matcher gap; applies in all contexts (no agent exemption).
- `test-gate.sh` (Stop) — runs the project's test command on session end; exit 2 keeps Claude looping until tests pass.
- `commit-review-gate.sh` (PreToolUse Bash on `git commit*`) — full staged-diff review by a tool-augmented Sonnet reviewer; exit 2 denies the commit and the orchestrator must fix.
- `auto-format.sh` (PostToolUse Write/Edit/MultiEdit) — runs `rustfmt` (and analogous formatters) on the changed file. Always exits 0 (advisory).
- `safety-net.sh` / `test-safety-net.sh` — safety rails on destructive operations and test-state hygiene.

## Skills

- `/verify` — analyst reviews the dirty diff against rules, iterates with developer until APPROVE; the user only sees the final APPROVE.
- `/approve-wiki` — sets the one-shot `.claude/state/wiki-approved` flag so the next wiki write is permitted.
- `/compact` — synthesize session findings into the wiki (focused mode after `/verify` APPROVE; session-wide mode at end of session).
- `/autoloop` — autonomous propose → implement → verify → benchmark → keep/discard loop against a user-specified baseline metric.
- `/plan-and-develop` — four-phase planning workflow for changes touching 3+ files; analyst writes `docs/plans/{feature-name}/{plan,research}.md` first.
- `/bootstrap` — one-time project setup; fills in `CLAUDE.md` from the template.

## Typical flow

1. **Research question** → orchestrator dispatches `analyst` → analyst returns a ≤20-line summary; bulky artifacts in `runs/{timestamp}/proposal.md`.
2. **Implementation** → orchestrator dispatches `developer` (worktree-isolated on `feat/{name}` or `fix/{name}`) → developer edits, runs `cargo test`, returns a ≤20-line summary with branch name + test status.
3. **Verification** → orchestrator invokes `/verify` → analyst reviews the diff against `.claude/rules/`, iterates with developer internally on REJECT until APPROVE.
4. **Wiki update (user-gated)** → on APPROVE, orchestrator presents summary + asks; if user types `/approve-wiki`, the flag is set; then `/compact` (focused) dispatches analyst to synthesize into `docs/wiki/{topic}.md`.
5. **Commit** → user-explicit only; `commit-review-gate.sh` reviews the staged diff before the commit lands.

## Gotchas

- **Developer worktree isolation.** Sibling git worktrees cannot see the main worktree's filesystem (or each other's). A developer dispatch operating in `feat/foo` cannot read files that exist only in main's working tree. If something is prototyped in a worktree and the worktree is reaped before commit, the work is gone — see `elo-evaluation.md` for the canonical example. Use `analyst` (which runs against main's filesystem) for any cross-worktree visibility.
- **Wiki-gate covers Bash redirects.** Originally `wiki-gate.sh` only gated `Write`/`Edit`/`MultiEdit`; `wiki-gate-bash.sh` was added to close the heredoc/redirect bypass. Both must be approved via `/approve-wiki` before each write. The flag is one-shot.
- **Analyst return contract drift.** Analyst is supposed to return ≤20 lines, but realistic synthesized reports often run longer. Treat the budget as aspirational; favor pointers and `runs/{timestamp}/proposal.md` references over inlining content.
- **No agent-memory directory.** The previous `.claude/agent-memory/{role}/{topic}.md` system is gone; durable knowledge lives in `docs/wiki/` and is updated through the `/verify` → `/approve-wiki` → `/compact` flow only.

## Related

- [`CLAUDE.md`](../../CLAUDE.md) — project-root spec for Workflow + Orchestrator Principles
- [`.claude/agents/{analyst,developer}.md`](../../.claude/agents/) — agent definitions
- [`.claude/hooks/`](../../.claude/hooks/) — route hooks, gates, formatters
- [`.claude/skills/`](../../.claude/skills/) — verify, autoloop, compact, plan-and-develop
- [Project Roadmap](project-roadmap.md) — current state, baseline score
