# Auto-review — 2026-08-14

**Range:** `5f30ea8..bde4f9b` (8 code commits; docs-only 06e6129 and logs-only bde4f9b skipped)
**Branch:** `claude/modest-rubin-s9uf8t` (== origin/main at review time)
**Focus:** bugs only
**Reviewer:** scheduled analyst dispatch

## Findings

### [med] Baseline score uses hardcoded 1500 instead of configured opponent initial Elo

- **File:** `scripts/run_baseline.sh:251`
- **Detail:** The candidate-Elo term in the SCORE formula compares against a literal `1500.0`, but the opponent initial rating is user-configurable via `HYZERO_OPPONENT_INITIAL_ELO`.
- **Scenario:** With `HYZERO_OPPONENT_INITIAL_ELO=1200`, a challenger that ties every match sits at `candidate_elo=1200`. The score gains a bogus `(1200 - 1500) * 0.05 = -15` penalty despite the challenger performing exactly at expectation.
- **Fix:** read the env var (or the emitted `opponent_initial_elo` field) rather than the literal.

### [low] Spurious "unexpected archive deletion" WARN on normal single-archive startup

- **File:** `src/selfplay/evaluation.rs:277-281`
- **Scenario:** Resume from `best_v001.pt` with no other archives. `champion_version = 1`, and `latest_archive_versions(dir, 1, 3)` returns `[]` because v1 is excluded from the pool. Every eval cycle then logs "pool empty despite champion_version=1 > 0" as if archives were deleted.
- **Fix:** compare pool_size against total archives actually present on disk (or against `champion_version - 1`), not just `champion_version > 0`.

### [low] PGN "Game N" tag collides across pool opponents within one cycle

- **File:** `src/selfplay/evaluation.rs:424,462`
- **Scenario:** With `pool_size=K`, `"Eval Cycle X Game 1"` appears `K` times in `logs/eval_games.pgn` — only the White/Black label disambiguates entries from different opponents.
- **Fix:** index as `(opp_i * 2 * gps) + game_idx + 1` or include the opponent version in the game tag.

## Non-bugs (checked and ruled out)

- Elo math: `expected_score`, `update_rating`, sequential fold, symmetry, larger-loss-when-favored — correct at `src/selfplay/elo.rs:16-26`.
- `challenger_perspective = -game_outcome` sign flip on Black-side games — correct at `evaluation.rs:328-333, 468-478`.
- `opponent_server_handle` `Mutex` is only locked inside `Python::attach` and released before any `.await`; no lock-across-await.
- Cooldown: `>=` comparison plus `== 0` short-circuit, counter reset on promotion, `total_games_since_last_promotion` accumulates through both bootstrap and pool branches.

## Cursor

Next scheduled review should start at commit **after** `bde4f9b`.
