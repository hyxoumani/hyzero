# Project Roadmap

Current state and next steps for hyzero. Updated 2026-05-04.

## What's Done

- **Tasks 1-16**: Chess engine — bitboards, magic move gen, all special moves, validation
- **Tasks 17-23**: MCTS tree search, self-play pipeline, inference batching, replay buffer
- **Tasks 24-26**: Python MuZero models (h/g/f networks) with training loop and inference server
- **Task 27**: MCTS value negation fix (two-player zero-sum backpropagation)
- **Task 28**: PyO3 integration — full Rust↔Python bridge
- **Task 29**: E2E validation — 5 games, 13 train steps, loss 8.52→7.04
- **Task 30**: Engine foundation — Zobrist hashing, FEN parser, perft validation, pin fixes
- **Task 31**: Engine validation — perft CLI, cross-validation vs python-chess (53M nodes), 3 stalemate/draw bugs fixed
- **Task 32**: Training infrastructure — coordinator simplification (no semaphore), checkpoint rolling window, resume-from-checkpoint, evaluation task, config consolidation with env vars

**Current state (2026-05-04)**: 141 Rust unit tests pass (10 ignored — perft slow paths and Python-env-required tests), zero clippy warnings. Pipeline running 24h continuous-training blocks via `scripts/run_nonstop.sh` with Gumbel-Top-K root selection + 800/400 sims, mate-pretraining bootstrap, and Stockfish-derived middlegame starts.

## Recent Commits (last 10)

- `5f30ea8` mcts: Gumbel-Top-K + sequential halving root selection
- `51876f0` training: mate-pretraining pipeline + auto-bootstrap in `run_baseline`
- `0e590a4` selfplay: material shaping OFF by default (opt-in only)
- `357b1df` scripts: Stockfish-based middlegame starts + merger
- `0ba4cd8` trainer: tolerate partial-optimizer checkpoints (`pretrain_dynamics.pt`)
- `63f0457` scripts: default resume from `pretrain_dynamics.pt` + full-slate cleanup
- `a6398a1` selfplay: add `HYZERO_RESUME_FROM` to pick resume checkpoint
- `55bfebe` scripts: bake TB + diverse starts into baseline defaults
- `56cb5af` scripts: raise baseline defaults — GPU, 200/100 sims, 8 selfplay slots
- `ee132c4` train: TB supervision infrastructure + canonical MuZero backup + diverse starts

## Baseline (Current)

- **Most recent run**: score **5.8589** (`logs/baseline_score.json`, timestamp 20260428_211338, commit `5f30ea8`, 24h block, 6311 games, 100 848 training steps, last_loss 2.367, avg_game_length 35.4, 0 promotions). Configuration: GPU, Gumbel-Top-K, 800 sims (400 eval), `tablebase_frac=0.45`, mate+TB cache.
- **Score with TB supervision (prior)**: 8.16 (Syzygy 45%, masked loss, biased reinit, 2h run 2026-04-21, **first promotion achieved**)
- **Score without TB (β=0.3 reproducibility)**: 14.51 (commit 63afdbe, 2026-04-15)
- **Note**: Direct comparison between the 5.86 nonstop run and prior 30-min/2h runs is indirect — different durations, sim counts, supervision mixes, and resume sources. The 24h run optimizes for steady-state behavior (loss decrease, supervision fraction stability) rather than the 30-min "score" composite.
- **Variance**: ±3 points observed without TB; two β=0.3 runs yielded 11.63 and 14.51. Beyond noise floor of ±1.5 for single 30-min runs.
- **Formula**: `(8.55 - policy_loss) + (promotions * HYZERO_CHAMPION_SCORE_WEIGHT) - (avg_game_length / 100)`
- **Run command** (30-min controlled): `bash scripts/run_baseline.sh 1800` (delete checkpoints first for fair comparison: `rm -f checkpoints/best*.pt`)
- **Run command** (continuous): `nohup bash scripts/run_nonstop.sh > logs/nonstop_outer.log 2>&1 &`
- **TB config** (2026-04-21, recommended for next phase):
  ```bash
  HYZERO_TABLEBASE_PATH=data/syzygy
  HYZERO_TABLEBASE_CACHE_PATH=data/syzygy/cache_balanced.pkl
  HYZERO_TABLEBASE_FRAC=0.45
  HYZERO_REINIT_VALUE_HEAD=1
  HYZERO_REINIT_VALUE_BIAS=0.3
  bash scripts/run_baseline.sh 1800
  ```
- **Stored at**: `logs/baseline_score.json`
- **Outcome blend (β)**: 0.3 — hard-won from 11-experiment sweep. Deviations regress.
- **Previous baseline**: 6.78 (commit d407281, 2026-04-14 — Dirichlet alpha fix before autoresearch)
- **Metric note**: Formula changed on 2026-04-15. Old formula used `decisive_ratio` (self-play metric, flawed). New formula uses `promotions` (discrete promotion events from eval ladder). Metric extraction must count `grep -c "\[eval\] promoted"` not version tags.

## Recent Direction (2026-04-28 → 2026-05-04)

- **Gumbel-Top-K + sequential halving** (commit `5f30ea8`) replaces PUCT+Dirichlet at the root when `HYZERO_USE_GUMBEL=1`. Provides its own noise; sequential halving allocates visit budget across top-K candidates, well-suited to high sim counts (800/round 1 ≈ 50 sims/cand at K=16). Now the `run_nonstop.sh` default.
- **Mate-pretraining pipeline** (`51876f0`) — `scripts/pretrain_on_mates.py` produces `checkpoints/mate_pretrained.pt`, used as the resume source when `checkpoints/best.pt` is absent. `run_baseline.sh` auto-bootstraps it on first invocation.
- **Material shaping default OFF** (`0e930a4`) — `HYZERO_DISABLE_MATERIAL_SHAPING=1` is now the default. Material proxy is opt-in only, since shaping at scale=5 was implicated in the rook-shuffle exploit (see `mcts-selfplay.md` "Passivity Trap").
- **Stockfish-based middlegame starts** (`357b1df`) — `scripts/build_middlegame_positions_stockfish.py` and `merge_starting_positions.py` produce a diverse set of opening positions to break the closed-loop self-play attractor.
- **Partial-optimizer checkpoints** (`0ba4cd8`) — trainer now loads `pretrain_dynamics.pt` even when its optimizer state is partial, enabling reuse of pretraining artifacts as resume checkpoints.

## What's Next

Detailed roadmap with file lists and rationale: [`docs/plans/next-steps/roadmap.md`](../plans/next-steps/roadmap.md)

| Batch | Focus | Key changes | Status |
|-------|-------|-------------|--------|
| — | Bug fix | Dirichlet noise alpha: 0.03 (Go) → 0.3 (chess) per AlphaZero paper | DONE (+2.65, 6.78) |
| 1 | Representation overhaul | History planes (19→103→102), underpromotion (4096→4672), legal move masking | DONE (superseded by alpha fix) |
| 2 | Search improvements | Tree reuse (DISCARDED: Q-value warm-start regressed), MCTS policy masking, temperature decay, **Gumbel-Top-K + sequential halving (DONE, commit `5f30ea8`)** | partial |
| 3 | Training hardening | LR scheduling, loss rebalance, ~~priority replay~~ (DISCARDED), train ratio, **mate-pretraining (DONE, commit `51876f0`)** | partial |
| 4 | Tactical metric | Puzzle suite, strength evaluator, composite score update | — |
| 5 | UCI protocol | Playable engine, time control, Stockfish benchmarking | — |

## Known Risks

- **Dirichlet noise CPU overhead**: Must use `--release` for e2e/baseline tests
- **Game length**: With β=0.3, ~150 moves average; with material-shaping OFF (current default), ~35 moves on the latest 24h run. Short games can starve value-head signal — monitor that promotions return when continuous training matures.
- **Batch timeout tuning**: 10ms empirical, may need adjustment at higher concurrency
- **GIL contention**: One acquisition per batch; monitor if eval + self-play conflict
- **Eval cycle cost**: 10 eval games at 50 sims takes ~3-4 min, may skip threshold crossings
- **Loss weight amplification destabilizes**: value_loss_weight=5.0 test regressed score 11.63→4.84 (2026-04-15). Do not retry above 2.0; prefer tuning outcome blend β instead.
- **Fast-training paradox**: Experiments with lower policy loss (2.4–2.7) regressed in promotions and play quality. Always validate by promotions and evaluation play, not training loss alone.

## Experimental Results

| Exp | Config | Baseline | Result | Delta | Note |
|-----|--------|----------|--------|-------|------|
| alpha-fix | Dirichlet α: 0.03→0.3 (chess, not Go) | 4.13 | 6.78 | +2.65 | Foundational fix, permanent |
| batch1-rep | History 19→103, underpromotion, masking | 6.78 | (superseded) | — | Nullified by alpha fix timing |
| autoresearch-β | 11-experiment β sweep (outcome blend) | 6.78 | **11.63** (β=0.3) | **+4.85** | Peak of program; closed-loop paradox discovered |
| value-weight=5.0 | Amplify value loss 5x at β=0.3 | 11.63 | 4.84 | −6.79 | Closed-loop instability; do not retry |
| games_per_side=6 | More games per training step | 6.78 | 5.48 | −1.30 | Policy loss 2.41 but 0 promotions (fast-training paradox) |
| β=0.4 | Higher outcome blend | 6.78 | 6.80 | +0.02 | Policy loss 2.63 but destabilized (1 promotion) |
| β=0.5 | Even higher outcome blend | 6.78 | 8.07 | +1.29 | Policy loss 2.45 but modest improvement |
| TB-supervision | Syzygy 3-4-5-man + masked-loss + biased-reinit (45% fraction) | 6.05 | **8.16** | **+2.11** | **First promotion achieved**, distributional collapse broken |
| nonstop-gumbel | 24h block, Gumbel-Top-K + 800/400 sims, TB cache, mate boot | — | **5.86** | (single 24h obs) | 100 848 steps, 6311 games, 0 promotions, last_loss 2.37 |

## Metric Evolution

**2026-04-15 update**: Metric formula fixed (commit 2a273d4). Old formula used `max_champion_version` (checkpoint tag index) instead of actual promotion count. New formula correctly uses `promotions = grep -c "\[eval\] promoted"`.

**Current metric** (`training_score` formula): `(8.55 - policy_loss) + (promotions * 2.0) - (avg_game_length / 100)`
- **Policy loss component**: Network learning (lower better, range ~2.4–4.5)
- **Promotions component**: Real wins in eval ladder (higher better)
- **Game length component**: Search efficiency (lower better, but >100 historically healthy; current short-game regime ~35 needs to be re-evaluated)

**Why promotions, not self-play decisive_ratio?**: As model improves, self-play-vs-self converges to draws (identical play = draws). Earlier autoresearch showed policy_loss improving to 2.4 while decisive_ratio dropped to 0 — metric was optimizing toward weakness. Promotions measure actual wins.

**Measurement noise** (2026-04-14): Baseline exhibits ±1 point variance from eval running only 10 games (binomial variance ±0.15-0.20 per cycle) and training step count varying ±50% between runs. Single-run claims <1.5 points are within noise; marginal changes require multi-run validation.

**Future work** (Phase 4 priority):
- **Phase 4 infra** (multi-run averaging): Each experiment runs 3x, median reported to reduce noise
- If needed, add puzzle-solving suite as supplementary metric (tactical strength)
- Consider tournament vs historical versions (longer eval horizon)

## Related
- [Neural Networks](neural-networks.md) — model architecture
- [MCTS & Self-Play](mcts-selfplay.md) — pipeline architecture, Gumbel-Top-K root selection
- [Replay Subsystem](replay-subsystem.md) — per-ply MCTS replay capture
- [Rust-Python Integration](rust-python-integration.md) — FFI boundary
- [Development Roadmap](../plans/next-steps/roadmap.md) — detailed next steps
