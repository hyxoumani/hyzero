# Review Log

Rolling log of bug-focused reviews performed by the scheduled `review-changes` routine. Each entry marks a `next_baseline` sha; the following run reviews commits strictly newer than that sha.

---

## 2026-07-21 — initial baseline

- **Reviewed range**: `5f30ea8^..bde4f9b` (first-run backfill covering the mcts Gumbel-Top-K series and the elo-promotion / ladder-eval feature, none of which had been reviewed by this routine).
- **HEAD at review**: `bde4f9b`
- **Findings**: none. Verified in scope: Gumbel inverse-CDF and root-POV sigma_q sign; sequential-halving budget accounting; elo update math; pool enumeration edge cases (missing dir, unparseable names, current-version exclusion); per-game elo sequencing in the eval task; challenger-score sign flip on the Black side; disjoint bootstrap-vs-pool branching; env-var wiring for the opponent inference server on a dedicated mpsc; `run_baseline.sh` awk pipeline defaults; entropy-loss sign in the trainer.
- **next_baseline**: `bde4f9b` — the next run reviews `bde4f9b..HEAD`.
