---
paths:
  - "**/*.py"
---
# Python Conventions

- Python package lives in `python/hyzero/`. Run tests with `cd python && pytest`.
- Requires Python 3.10+. Use modern type hints (PEP 604 unions, etc.).
- PyTorch tensors: always specify dtype and device explicitly.
- NumPy arrays used for PyO3 bridge — keep shapes documented in docstrings.
- Dependencies: torch, numpy. Dev: pytest.
- Follow existing code style — no auto-formatter configured.
