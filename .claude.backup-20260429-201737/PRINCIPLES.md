# Engineering Principles

Constraints every agent follows. Non-negotiable.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State assumptions explicitly when the task is ambiguous.
- If two interpretations are plausible, present both rather than picking silently.
- If you can't explain *why* something is broken, you haven't found the bug.
- Ask before doing destructive or hard-to-reverse work.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features that weren't asked for.
- No abstractions for hypothetical future use.
- 10 obvious lines beat 200 lines of cleverness.
- Three similar lines beat a premature helper.
- Would a senior engineer call this overcomplicated? If yes, cut.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Match existing style exactly. Same naming, error handling, indentation.
- Don't refactor adjacent code. A bug fix is not a cleanup pass.
- Remove only what your change made obsolete — not pre-existing dead code.
- Stay in scope. Out-of-scope needs get reported, not made.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

- Convert tasks into measurable goals: tests that pass, commands that exit 0, output that matches.
- Never claim done without running the verification.
- Paste the output. "Should work" is not evidence.
- After 3 failed attempts on the same fix, stop and report what you tried.

---

These bias toward caution over speed. For trivial tasks, use judgment.
