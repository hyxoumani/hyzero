# Project Wiki

Knowledge base maintained by the context-keeper. Pages are synthesized from
session findings, reviewer feedback, experiments, and architectural decisions.

## Project
- [Project Roadmap](project-roadmap.md) — current state, baseline score, next batches
- [Development Roadmap](../plans/next-steps/roadmap.md) — detailed 5-batch plan with files and rationale

## Engine Core
- [Chess Engine](chess-engine.md) — board representation, move generation, validation gotchas
- [Special Moves & Draw Rules](special-moves-draws.md) — castling, en passant, promotion, game termination

## Learning Pipeline
- [MCTS & Self-Play](mcts-selfplay.md) — tree search, coordinator, checkpoints, evaluation task
- [Neural Networks](neural-networks.md) — MuZero h/g/f networks, tensor shapes, training plan

## Integration
- [Rust-Python Integration](rust-python-integration.md) — FFI boundary, data flow, PyO3 status

## Development
- [Testing Procedures](testing.md) — commands, cross-validation, perft CLI, edge cases
- [Dev Workflow & Framework](dev-workflow.md) — orchestration, agents, baseline scoring
- [Mistakes Log](mistakes.md) — agent failure cases with root cause analysis, error classification, escalation tiers
