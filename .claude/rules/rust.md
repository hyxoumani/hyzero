---
paths:
  - "**/*.rs"
---
# Rust Conventions

- Run `cargo check` before finishing any task to verify compilation.
- Run `cargo clippy` to catch common mistakes.
- Binary targets live in `src/bin/`, library code in `src/`.
- Tests go in `#[cfg(test)] mod tests` blocks at the bottom of each module file.
- Prefer `thiserror` for library error types, `anyhow` for binary error types.
- Use `async-trait` for async trait methods when needed.

<!-- /bootstrap appends project-specific Rust conventions below this line -->
