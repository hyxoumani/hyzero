# Change-review routine log

Each entry records a run of the scheduled "Review changes & give feedback" routine on branch `claude/modest-rubin-dlle1w`. The HEAD sha is the newest commit covered by that review.

## 2026-11-08 — HEAD bde4f9be00d1c59b648a4f3c8e59d63c9121d99c

Scope: full branch history since `main` diverged (18 commits, feature series: elo-promotion ladder, cosine LR re-apply, policy entropy, Gumbel-Top-K, baseline logging).

Findings (bugs only):

1. CRITICAL — src/selfplay/evaluation.rs:247-251 / champion.rs:72-83 — ladder permanently stalls after first promotion: ChampionStore::promote aliases champion to the live challenger evaluator, so pool branch then excludes the only archive and bootstrap replays challenger-vs-itself (win_rate ≈ 0.5 forever).
2. HIGH — src/selfplay/evaluation.rs:103,144-150 — `champion_backend` stored but never read; no hot-swap of weights ever happens. Root cause of #1.
3. MEDIUM-HIGH — src/selfplay/evaluation.rs:247-251 — pool match excludes current champion, so a regression vs. current champion that beats v-1/v-2/v-3 still clears candidate_elo > 1520 and promotes.
4. MEDIUM — src/bin/selfplay.rs:144-147 — HYZERO_ELO_K_FACTOR / HYZERO_OPPONENT_INITIAL_ELO / HYZERO_PROMOTION_ELO_DELTA accept negative values; -K inverts Elo updates.
5. LOW — scripts/run_baseline.sh:214-218,251 — awk `last+0` of a truncated candidate_elo yields 0; `${LAST_CANDIDATE_ELO:-1500.0}` doesn't guard literal 0 → spurious -75 SCORE penalty.
6. LOW — src/mcts/gumbel.rs:30 — upper clamp `1.0 - 1e-9` rounds to 1.0 in f32; only `rng < 1.0` guards against `ln(0) = -inf`. Brittle.
7. LOW — src/bin/selfplay.rs:411-450 — opponent batcher spawned unconditionally; idles with VRAM on the whole bootstrap era.

Status: reported via scheduled-routine push notification. No fixes applied.
