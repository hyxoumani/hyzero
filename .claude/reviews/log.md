# Review Log

Bug-focused review history. Each entry: date | reviewed range | HEAD at review | findings summary.

## 2026-06-07 — elo-promotion feature

- Range: `0ab40d4..df794b3` (11 commits)
- HEAD at review: `df794b37e72ebf7fcb9be0f42c3a261ce374f843`
- Branch: `claude/modest-rubin-4mp3x`
- Scope: elo math module, archive pool helper, opponent inference plumbing, per-opponent eval task refactor, env-var wiring, baseline candidate_elo extraction
- Verdict: clean on blockers/high; 2 medium, 5 low
- Findings:
  - MEDIUM: PGN game-number collisions across pool opponents (`src/selfplay/evaluation.rs:421-466`)
  - MEDIUM: `LAST_CANDIDATE_ELO` default fallback doesn't catch `"0"` from empty awk (`scripts/run_baseline.sh:218-219`)
  - LOW: opp-handle-None branch consumes version bump without promotion attempt (`src/selfplay/evaluation.rs:341-376`)
  - LOW: cooldown startup notice math is Elo-path only, wrong during bootstrap (`src/bin/selfplay.rs:184`)
  - LOW: cooldown tie semantics — `>=` means cycle that crosses threshold can promote (`src/selfplay/evaluation.rs:526-528`)
  - LOW: candidate_elo resets each cycle (by design per plan, flagged for confirmation) (`src/selfplay/evaluation.rs:264,533`)
  - LOW: elo sequential_table_driven test uses hand-computed f32 constants (cross-platform libm drift risk) (`src/selfplay/elo.rs:79-99`)

## Next review starts from

`df794b37e72ebf7fcb9be0f42c3a261ce374f843` (exclusive).
