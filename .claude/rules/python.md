---
paths:
  - "python/**/*.py"
---
# Python conventions for hyzero

- Use `torch.nn.functional.relu()` instead of shared `nn.ReLU(inplace=True)` on residual paths — inplace on skip connections can corrupt gradients.
- `bias=False` on all Conv2d layers that precede BatchNorm — BN's affine params subsume the bias.
- Add `.gitignore` with `__pycache__/`, `*.pyc`, `*.egg-info/`, `.pytest_cache/` before running `pip install -e .` or `pytest` in any new Python package.
- All model `forward()` methods must document input/output tensor shapes in comments.
- Tests use `torch.no_grad()` context and CPU-only random tensors.
