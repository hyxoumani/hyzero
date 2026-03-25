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
