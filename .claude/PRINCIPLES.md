# Engineering Principles

Rules every agent follows. Non-negotiable.

## 1. Spec before implement

Never write code without understanding the interfaces first. Catch mismatches
in the design phase, not after 500 lines of implementation. Ask: what are the
inputs, outputs, error cases, and integration points?

## 2. Root cause or nothing

Don't fix symptoms. If you can't explain WHY something is broken, you haven't
found the bug yet. Three failed hypotheses means stop and escalate.

## 3. Simplicity criterion

Simpler code that achieves the same result always wins. An improvement that adds
ugly complexity is not worth it. An improvement from deleting code is always worth it.
10-line obvious fix > 200-line abstraction. Three similar lines > premature helper.

## 4. Minimal diff

Achieve the goal with the fewest files touched and lines changed. Don't refactor
adjacent code. Don't add features that weren't asked for. Don't "improve" things
outside scope. A bug fix doesn't need surrounding code cleaned up.

## 5. Verify, don't trust

Never say "this should work." Run the tests. Reproduce the bug. Confirm the fix.
If you can't verify it, don't ship it. Paste the output.

## 6. Keep or discard, no half-states

Every change either improves the codebase and stays, or it doesn't and gets reverted.
No commented-out code, no TODO-someday, no "we'll fix this later" left in the diff.
Experimental work lives on branches.

## 7. Compound knowledge

When you learn something non-obvious (a gotcha, a pattern, a constraint), persist it.
Rules files, CLAUDE.md, doc comments. The next session should never re-discover
what this session already learned.

## 8. Autonomous by default

Don't stop to ask unless genuinely stuck. If the path is clear, take it. If a decision
is reversible, make it. Only escalate on: ambiguous requirements, destructive actions,
architectural choices that are hard to undo.

## 9. Search before building

Before introducing a pattern, library, or approach: does the language/framework have
a built-in? Is there an established solution? Only build custom when you have a concrete
reason the standard approach doesn't work. Name the reason.

## 10. Errors are for the next reader

Error messages, comments, and commit messages exist for the person (or agent) who reads
them next. Be specific: name the file, the function, the line. Say what went wrong and
what to do about it. "Something failed" helps nobody.
