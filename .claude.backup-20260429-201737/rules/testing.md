# Testing Conventions

- Test names describe behavior, not implementation (`rejects_invalid_move`, not `test_func_3`).
- One assertion per test when practical.
- New code paths require tests before merging.
- Flaky test policy: run 3x. Pass 2/3 = flaky (flag, continue). 0/3 = real failure.
- Test failures block session completion via the `test-gate` hook.
- Regression tests must fail without the fix and pass with it.
