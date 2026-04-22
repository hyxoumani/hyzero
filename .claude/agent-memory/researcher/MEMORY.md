# Researcher Agent Memory

Findings from research phases conducted by researcher agents.

- [Chess Engine Foundation](chess_engine_foundation.md) — board representation, move generation, special moves, action encoding, known bugs (action_to_move, position_hash)
- [Perft Position 6 Investigation](perft_d5_bug.md) — position 6 D1 correct at 42 (not 46); calculate_pins missing enemy queen; engine passes D1-D4 for pos6
- [Startpos D5 Overcount Root Cause](startpos_d5_overcount.md) — +8 from nodes+=1 terminal branch in perft(); 8 Fool's Mate paths at ply 4; fix: always recurse, drop terminal branch
- [Engine Validation Audit](engine_validation_audit.md) — full perft coverage gap analysis; stalemate/castling bug; edge-case list; timing estimates for all positions/depths
- [Training Infrastructure SOTA](training_infra_sota.md) — SOTA worker/checkpoint/eval patterns mapped to hyzero; coordinator simplification, rolling checkpoint window, eval vs RandomEvaluator
- [Trainer LR Schedule](trainer_lr_schedule.md) — optimizer at trainer.py:55-59, model_version is step counter, no scheduler exists; LambdaLR design for cosine+warmup with checkpoint persistence
- [Representation Overhaul (Batch 1)](representation-overhaul.md) — full call graph + file:line map for 19→103 planes, 4096→4672 actions, legal-move masking; hardcoded literals list; checkpoint wipe requirement; underpromo index decision needed
- [MCTS Tree Reuse (Batch 2)](mcts-tree-reuse.md) — ownership model (Option::take safe), hidden-state grounding fix, search-level masking already done in MCTSNode::new; file:line map; score model +0.4 conservative
- [Recency-Weighted Replay](recency-replay.md) — bias sample_batch toward recent model_version games via exp decay; file:line map; lambda=0.1 default; expected +0.1 to +0.25 score delta
- [Decisive Ratio Root Cause](decisive-ratio-root-cause.md) — value targets are MCTS Q not bootstrapped outcome; full experiment table incl. Dirichlet fix (6.78 baseline); next experiment = outcome-based value targets (+1.5 to +2.5 expected)
- [Dual-Model Evaluation](dual_model_eval.md) — play_game_dual design, EvaluationTask extension, win_rate_vs_random formula, time-budget analysis, Phase 2 (checkpoint opponent) deferred
- [Material Signal](material_signal.md) — outcome convention (White-absolute), material counting pattern, adjudication state machine, stale wiki finding, test breakage at test_play_game_completes
- [Color Augmentation](color_augmentation.md) — full 103-plane map, rank-mirror formula, action/policy flip transforms, augmentation site (training.rs assemble_batch_arrays), stale wiki (19 vs 103 planes)
- [Tablebase Supervision](tablebase_supervision.md) — TB injection design; batch mix insertion point (top of train_batch); reward_probe.py missing (must write encoder from scratch); python-chess available but not in pyproject.toml
