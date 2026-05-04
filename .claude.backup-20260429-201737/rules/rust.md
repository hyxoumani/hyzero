---
paths:
  - "**/*.rs"
  - "Cargo.toml"
---
# Rust Conventions

- Edition 2021. Prefer `cargo fmt` defaults; do not introduce a custom `rustfmt.toml`.
- Run `cargo clippy` before declaring done. Fix or `#[allow]` with justification.
- Use `Result<T, E>` for fallible APIs; reserve `panic!`/`unwrap`/`expect` for invariants and tests.
- Keep `unsafe` out of new code unless required by FFI; document the invariant in a one-line comment.
- PyO3 boundary: convert Python errors at the FFI edge; do not let `PyErr` leak into Rust core types.
- `tokio` is the only async runtime — do not pull in alternatives.
- Tests live next to code in `#[cfg(test)] mod tests`. Slow/perft tests use `#[ignore]`.
