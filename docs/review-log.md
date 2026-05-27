# Code Review Log

Tracks commits reviewed by Claude. Most recent at top.

## ee132c4 — train: TB supervision infrastructure + canonical MuZero backup + diverse starts

**Reviewed:** 2026-05-27
**Reviewer:** Claude (claude-opus-4-7[1m])
**Scope:** Latest squash of `autoresearch/apr13` (23 commits). 51 files changed, ~62k lines added (mostly logs/wiki/PGN — code surface is small). Focus areas:

- `src/mcts/tree.rs` — canonical MuZero backup with edge-reward propagation
- `src/data/encoding.rs` — `encode_action_spatial_for_color`, flip invariants
- `src/py/training.rs` — color augmentation + outcome-blend POV
- `src/selfplay/game_task.rs` — POV-flip sort, diverse starts, terminal reward
- `python/hyzero/training/trainer.py` — TB mixing, diagnostics, value-head reinit
- `python/hyzero/data/tablebase.py` — TB snapshot + trajectory builders
- `python/hyzero/data/board_encoder.py` — Python-side board encoder for TB

**Verdict:** Solid fix to the canonical-MuZero backup; tests pin both the new mating-reward propagation and the zero-reward equivalence with the old code. Color-symmetry plumbing is well-tested. Found one cleanup item and several minor latent concerns documented in the session reply.

**Outstanding bugs/items flagged:**

1. **Leftover debug code** — `python/hyzero/training/trainer.py:652-657` writes to `/tmp/hyzero_diag_probe.txt` every train step. Should be removed.
2. **Inconsistent reward-loss denominator at k=1 vs k>=2** under TB mixing (`trainer.py:631-637`) — k=1 averages over all B; k>=2 over non-TB only.
3. **Pre-existing latent UB**: `Square::from(u8 ≥ 64)` in `src/lib.rs:18-23` transmutes without runtime check. Not introduced here, but `decode_underpromo_action` can produce `to_file` in 0..23 (dead slots). Defensive guard in `action_to_move` (`src/data/encoding.rs:195`) would harden against future callers.
