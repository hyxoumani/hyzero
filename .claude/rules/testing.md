---
paths:
  - "**/*test*"
  - "**/*spec*"
  - "**/tests/**"
---

# Testing Conventions

- Test names describe behavior, not implementation ("rejects_invalid_move" not "test_func_3")
- One assertion per test when practical
- New code paths require tests before merging
- Flaky tests: run 3x, pass 2/3 = flaky (flag it, don't fail the suite), 0/3 = real failure
- Test failures block session completion via the test-gate hook
- Regression tests must fail without the fix and pass with it
