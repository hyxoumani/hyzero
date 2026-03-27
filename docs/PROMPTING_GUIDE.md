# Prompting Guide — Lessons from Refactor Session

## Prompt Template

```
Goal: [specific outcome]
Scope: [what to touch, what NOT to touch]
Execution: [subagents ok / explain first / I want to approve each change]
Permissions: [subagents can edit+bash freely / ask me each time]
Status: [update me after each task / just tell me when done]
```

## Example

```
Refactor the chess game logic in board.rs. Fix all bugs that prevent
compilation and gameplay. Add missing en passant. Add draw rules
(50-move, threefold repetition, insufficient material).
Use subagents freely — grant them edit and bash.
Update CLAUDE.md status table after each task.
```

## What Worked

- Clear scope: "refactor game logic, validate moves, add en passant and niche rules"
- Requesting subagent strategy shaped how work got parallelized
- Scoping severity: "only fix medium/high bugs" saved time
- Asking for explanations before applying caught real questions early

## What to Improve

1. **Split bundled tasks.** "Refactor + validate + add EP + add draw rules" is 4 things. Prioritize or separate them so you control the order.

2. **Set permissions upfront.** Subagents got blocked because Edit/Bash permissions weren't pre-approved. Use `/allowed-tools` or project settings before launching parallel work.

3. **Be specific about existing state.** "Analyze the current .md plan" when no plan exists causes confusion. Say what exists vs. what you want created.

4. **Subagents can't pause for questions.** Clicking "no" on a permission prompt kills the subagent. If you want to understand changes before they're applied, tell the main agent "explain each change before making it" — don't rely on interrupting subagents.

5. **Set expectations proactively, not reactively.** Decide upfront: "I want to approve each change" OR "let subagents run freely." Avoids back-and-forth mid-session.

6. **Request status cadence once.** Instead of asking for summaries multiple times, say upfront: "after each phase, give me a status summary before proceeding."

---

## General LLM Prompting Principles

### Getting Better Answers

1. **State constraints upfront, not after.** Front-load all requirements before implementation starts. A bullet list of must-haves saves rounds of revision. "Build a server" then later "oh, also send bitboards every move" causes rewrites.

2. **Ask for failure modes.** After any design decision, ask "how does this break?" or "what's the worst case?" LLMs are biased toward the happy path. Forcing failure analysis is where you get the most value.

3. **Use plans for complex work.** For anything touching 3+ files, start with a plan rather than jumping into implementation. It forces structured thinking before code.

4. **Be specific about what you don't know.** "I've written Go for ten years but this is my first time touching async Rust" gets fundamentally different explanations than a generic question. More context about your knowledge = less wasted time.

5. **Challenge confident-sounding answers.** LLMs sound equally confident whether right or wrong. The more technical the claim, the more you should probe.

6. **Ask targeted review questions.** Don't ask "review this code." Ask "what bugs would cause this to fail silently in production?" or "what happens if two clients send MOVE at the exact same instant?"

7. **One task per message when possible.** Confirming each task before moving on keeps quality high. Dumping 5 unrelated requests in one message drops quality on all of them.

### The Meta-Principle

LLMs are most useful when treated as a collaborator you need to manage, not an oracle. Direct the reasoning, verify the output, and don't let the model drive the architecture decisions — you should make those, with the model providing analysis to inform them.

---

## Lessons from Architecture/Design Session

### What Worked

1. **Spec before implement.** Insisting on designing the Python neural network layer before building Rust infrastructure caught 7 interface mismatches (flat arrays vs structs, hidden state needing a channels field, dual action encoding, etc.). This saved potentially days of rework. Always spec cross-boundary interfaces before either side writes code.

2. **Incremental deepening.** The session followed: concept discussion → architecture doc → task breakdown → execution plan. Each phase built on verified understanding from the previous one. This is far more effective than jumping straight to "build MCTS."

3. **Asking conceptual questions first.** Questions like "does the value function affect PUCT?" and "are discarded branches used in training?" built real understanding before committing to implementation. You can't direct architecture if you don't understand the concepts.

4. **Stopping premature execution.** Saying "No, do not start executing" when the model tried to jump to coding prevented wasted work. The model is biased toward action — you need to be the one who decides when design is done.

5. **Requesting explanations of downstream effects.** "Can you explain the Rust infra changes from speccing Python?" surfaced implications that wouldn't have been obvious otherwise. Always ask "what does this decision change elsewhere?"

### What to Improve

7. **Lead with your mental model, not corrections.** You knew inference and training needed separate queues but stated it as a correction ("we are missing a queue") after the initial design. Front-loading your requirements as a checklist — "must have: separate inference queue, separate training queue, disk storage" — is faster than course-correcting.

8. **Narrow open-ended prompts.** "Let's think more about the Rust side" required multiple rounds to figure out what specifically to think about. Compare: "What data structures do we need for the replay buffer, and how does the self-play coordinator manage concurrency?" — same intent, one round.

9. **Split multi-file documentation updates.** "Add to architecture.md + update CLAUDE.md + create task doc" was three operations. Doing them one at a time lets you verify each before the next. This matters especially for docs that reference each other — if ARCHITECTURE.md has an error, it propagates to the task doc.

10. **State design constraints as requirements, not preferences.** "I think option A seems pretty simple" is weaker than "Use option A because continuous self-play avoids stop-the-world pauses." The latter gives the model your reasoning, which helps it make consistent follow-on decisions.

11. **Use the discussion phase to stress-test.** You did this well with MCTS concepts, but less with infrastructure decisions. For each design choice, explicitly ask: "What breaks if we do it this way? What's the worst failure mode?" — especially for concurrency and inter-process communication.

### Design Session Template

For architecture/design work specifically, this structure works well:

```
Phase 1 — Understand concepts (discussion, no code)
  "Explain X. How does Y work? What happens when Z fails?"

Phase 2 — Spec interfaces (document, no code)
  "Write the types/signatures for the boundary between A and B.
   What changes on side A if we spec side B this way?"

Phase 3 — Task breakdown (plan, no code)
  "Break this into tasks. Which are independent? What's the
   critical path? Create a task doc with verify steps."

Phase 4 — Execute (code, with checkpoints)
  "Run tasks with subagents. Update status after each.
   Stop and tell me if anything deviates from the spec."
```
