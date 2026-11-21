# Review log

Durable log of code reviews performed by scheduled `review changes` routine. Each entry = one review pass. Newest first.

## Legend

- Range `A..B` is inclusive of B and exclusive of A (git-rev-list convention).
- "Scope" names the logical feature audited; code files listed are where the audit focused.
- "Findings" = count by severity. Full findings live in the Claude Code session transcript linked from the entry.

---

## 2026-11-21 — elo-ladder-eval & champion-pool promotion

- **Range:** `7b53e5d..bde4f9b` (inclusive). Commits:
  - `7b53e5d` selfplay: add elo math module
  - `9ddee3a` selfplay: add archive pool enumeration helper
  - `2a38e77` selfplay: plumb opponent inference server for elo ladder
  - `a93b077` selfplay: refactor eval task to per-opponent elo ladder
  - `0c35f8f` selfplay: wire elo ladder env vars + startup notices
  - `9450e38` baseline: extract candidate_elo from ladder_match and add to score
  - `924f6be` selfplay: apply cargo fmt to new elo-promotion code
  - `df794b3` selfplay: fix doc-list indentation in eval task run() doc
  - `06e6129` docs: restructure wiki into fresh topic pages
  - `bde4f9b` logs: refresh baseline, eval, and self-play artifacts
- **Scope:** Elo math, ladder match pairing, archive pool enumeration, opponent inference server plumbing, promotion gating, env-var wiring, Python↔Rust boundary on the eval task.
- **Files audited:** `src/selfplay/{elo,pool,evaluation,champion,game_task}.rs`, `src/bin/selfplay.rs`, docs under `docs/wiki/{elo-ladder-eval,champion-pool-promotion}.md`.
- **Findings:** 3 HIGH, 5 MEDIUM, 6 LOW. No fixes applied — review-only scope.
- **Top finding (HIGH):** `src/selfplay/champion.rs:72-112` — `promote()` replaces the evaluator Arc but never calls `.swap()` on the stored `champion_backend`, so the champion inference batcher never loads promoted weights. Effect: "challenger vs champion" ladder games play challenger-vs-self.
- **Other HIGH:**
  - `src/selfplay/champion.rs:44-50` / `src/bin/selfplay.rs:488-492` — pre-existing on-disk `best_v{NNN}.pt` archives are not seeded into `ChampionStore.archive_files` on restart; archive pruning and pool enumeration misbehave.
  - `src/selfplay/evaluation.rs:379-381` — `opponents_label` built before loop; load-failed opponents still appear in `ladder_match opponents=` log line while `candidate_elo` stays at 1500.
- **Session:** https://claude.ai/code/session_01HtPkiuySSTUdJavtY7dooj

---
