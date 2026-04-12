# Session Log

2026-04-11 — Task 25 complete: Training loop with MuZero loss, K-step unroll, checkpointing. Reviewer found reward loss K/K+1 bug, torch.load deprecation.
2026-04-11 — Task 27 complete: Fixed MCTS value negation in MCTSTree::backpropagate() (two-player zero-sum). Wiki gotcha updated.
2026-04-11 — Task 28 complete: PyO3 integration with PyO3Backend, PyTrainingThread, batch assembly with zero-padding, weight sync. Updated CLAUDE.md architecture, all wiki pages.
2026-04-11 — Task 29 complete: End-to-end validation with full MuZero loop (5 games, 13 training steps, loss 8.52→7.04). Fixed: Dirichlet noise, num_simulations (200→50), max game length (300), multi-step training, loss logging, checkpointing, batch timeout. Metric defined: training_loss, extract via scripts/e2e_test.sh 120. Gotchas documented: Dirichlet CPU overhead, game length scaling (3-4x longer), stdout buffering in cargo run. New infrastructure: scripts/e2e_test.sh, scripts/run_experiment.sh, logs/. Updated CLAUDE.md (metric section, Task 29 row, scripts in architecture), project-roadmap.md (E2E marked DONE, new risks), mcts-selfplay.md (Dirichlet noise section, expanded gotchas).
