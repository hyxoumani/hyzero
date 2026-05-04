---
paths:
  - "**/*.py"
  - "python/pyproject.toml"
---
# Python Conventions

- Target Python 3.10+. Use `from __future__ import annotations` only when needed for forward refs.
- PyTorch is the only NN framework. Models live under `python/hyzero/`.
- Numpy arrays cross the PyO3 boundary — preserve dtype/shape contracts; do not silently `.astype()`.
- Tests use `pytest`, run from `python/` (`cd python && pytest`).
- No new top-level dependencies without updating `python/pyproject.toml`.
- Prefer pure functions in trainer/model code; reserve side effects (I/O, logging) for the top level.
