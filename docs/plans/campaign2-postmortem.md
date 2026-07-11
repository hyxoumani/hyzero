# Campaign 2 Post-Mortem — `runs/auto-20260706-100435`

Endgame-conversion campaign, 2026-07-06 → 07-11. Successor to campaign 1
(`runs/auto-20260702-151405`, 43 iters) and the June signal-starvation run.
Current-truth record; supersedes optimistic language in the iter-4/5/6 rows of
`results.tsv` and in `docs/wiki/conversion-levers.md`. Read alongside
`docs/wiki/conversion-campaign-202607.md` (2026-07-06 launch section).

## 1. Executive summary

Seven executed iterations plus two audit probes (rows 6a/6b) over five days;
iter-8 (clean baseline) is in flight on 07-11 and iter-9 (fresh curriculum) is
pending. The campaign's headline result — a **Stockfish-demonstration policy
supervision "breakthrough," iter-4 at 24.2% conversion vs the 14.2% baseline
(+10pp, ~4.4σ)** — is **RETRACTED**.

A contamination audit found the demo generator was seeded from the probe fixture
files, so the net was trained on the exact positions it was later scored on. The
clean-split 2×2 settles it:

| demo split | probe = trained starts | probe = held-out starts |
|------------|------------------------|-------------------------|
| **contaminated** (probe-overlapping) | 23–25% (iter-4 24.2, iter-6 23.3, deep 24.7) | **4.0%** |
| **clean** (bulk-only, 0 overlap) | **1.7%** | **0.7%** |

Score survives only when the probed starts were trained on, and only when the
demos overlap the probe — the signature of pure memorization, not learned
technique. The clean-split arm (iter-7) actually **degraded** below its own
pre-demo substrate (1.7% vs 5.8%). **Conversion remains unsolved.** What stands
is negative knowledge (levers falsified) and infrastructure.

## 2. Timeline

| iter | lever | probe(s) | verdict |
|------|-------|----------|---------|
| 0 | baseline = campaign-1 tip `7522646` | 34/240 = **14.2%** | reference |
| 1a | MLH_CAP=30 12h retrain, bonus **OFF** (control) | 30/240 = 12.5% | control |
| 1b | MLH bonus=0.2, q_thr=0.8, cap=30 | 30/240 = 12.5% | discard — 0 moves changed vs control |
| 1d | MLH diagnostic bonus=5.0, q_thr=0 | 29/240 = 12.1% | discard — wiring live, m-signal flat → **MLH exhausted** |
| 2 | from-scratch rep-planes (102→110) + adj-gate, 24h | 26/240 = 10.8% | discard — labels truthful but **value collapsed to draw prior** (loss pinned 1.09–1.17) |
| 3 | TB-steep FRAC=0.75 from-scratch (categorical) | 14/240 = 5.8% on v594 | discard-truncated — killed 5.5h on TB-OVERFIT forensic; **value-distill falsified on categorical head** |
| 4 | SF-demo policy supervision FRAC=0.25, 8k pos, 12h from v594 | 58/240 = **24.2%** | ~~keep / NEW BEST~~ **RETRACTED** (contaminated; TB defaults also rode along) |
| 5 | dose-response 52k demos FRAC=0.4, TB-off, 11h | 44/240 = 18.3% | discard — dose does **not** scale (below iter-4) |
| 6 | three-stream (demos×3 + SF-labeled corpus) FRAC=0.3, ~12h | 56/240 = 23.3% | discard — tied iter-4 (<1σ); 3 ladder promotions (first live 110-plane pool) |
| 6a | audit: deep-probe on **trained** starts | 74/300 = 24.7% | audit-contaminated |
| 6b | audit: **held-out** probe (150 fresh starts, 0 overlap) | 12/300 = **4.0%** | audit-holdout — iters 4–6 predominantly memorization |
| 7 | CLEAN-SPLIT bulk-only demos (0 overlap) FRAC=0.25, 12h from pre-demo v594 | primary 4/240 = **1.7%**, held-out 2/300 = 0.7% | discard — demos teach **zero transferable technique**; memorization locked |
| 8 | clean baseline (MLH_CAP=30, resume from three-stream, TB defaults pinned OFF) | — | in flight (07-11) |
| 9 | fresh-curriculum retrain | — | pending |

## 3. Root-cause work that stands

The diagnostic conclusions from the campaign remain valid and are the campaign's
real yield. They were reached by numeric probe, not speculation, and several were
established by falsification (a lever proven inert, not merely untried):

- **Terminal-blind MuZero tree.** Below the root the tree cannot see true
  terminals: `tree.rs top_k(64)` children are never empty, so mate/stalemate are
  invisible except at root children. Root-child grounding (committed, `c71a7e0`)
  makes defenders exploit hangs/forced draws exactly, but attacker-side avoidance
  still needs value/policy — terminals are only visible at depth 2 from the
  attacker's root. **Partial fix; not a conversion cure.**
- **Draw-prior value collapse.** With truthful labels the curriculum runs ~84%
  draw targets; the categorical value head collapses onto the draw prior. Value
  loss pinned at **1.09–1.17 is the collapse signature, not a fit** (iter-2). This
  is why "truthful labels + repetition planes" is NECESSARY but INSUFFICIENT
  alone.
- **Temperature artifact.** Short curriculum games spent their decisive phase
  under decayed temperature, corrupting the target moves;
  `HYZERO_CURRICULUM_TEMPERATURE_MOVES=2` (committed `c3bf814`) holds temp≈1.0
  through the decisive window. A data-quality fix, not a conversion lever.

Falsifications recorded this campaign:

- **Moves-left head — validly exhausted.** Campaign-1 iter-42/43 probes were
  *vacuous* (`HYZERO_MLH_SEARCH_BONUS` was never set — the head existed but was
  never read in search). Iter-1 wired it live and smoke-tested it: control 12.5 /
  bonus-0.2 12.5 (zero moves changed) / factor-5.0 gate-0 12.1. The `m`-signal is
  flat; the head is inert in-game. **Lesson: smoke-test a lever at an extreme
  value before spending GPU.**
- **Value-level TB distillation — falsified on BOTH heads.** June (tanh
  wash-through) and iter-3 (categorical, FRAC 0.75 steepened cache) both fit the
  value loss with bimodal ±0.9 target histograms (mixing provably active) yet
  produced **zero game transfer**: cm_count frozen, KQvK value probes 0.37–0.57
  against the ~1.0 needed. The *mechanism*, not the head shape, is the dead end.
- **Dose scaling — falsified.** 8k→52k demos (iter-5, FRAC 0.4) fell to 18.3%,
  below iter-4; the three-stream (iter-6) only recovered the regression to a tie.
  More demonstration data does not compound.

## 4. Contamination post-mortem

**How it happened.** The SF-demo generator was seeded, for convenience, from the
same fixture files that define the conversion probe (the 120 fixed won-endgame
starts). Every demonstrated position was therefore also a scored position. The
"breakthrough" measured recall of trained positions, not conversion skill.

**How it was caught.** The `probe_deep` deep-probe scored *too well* on trained
starts (iter-6a, 24.7%) — an implausible jump given the flat plateau history. An
audit then ran a **held-out probe** (iter-6b: 150 fresh starts, 0 overlap, avg
19.8 plies): conversion fell to **4.0%**. Tracing the discrepancy exposed the
generator's seeding from the probe fixtures. Iter-7 confirmed causally by
retraining on **bulk-only** demos with zero probe overlap: 1.7% seen / 0.7%
held-out, *below* the 5.8% pre-demo substrate — i.e. the deep demos actively
**degrade** shallow skill while transferring none.

**New standing rules (permanent):**

1. **Fixtures never seed training data.** Probe/eval start files are off-limits to
   any demo, curriculum, or corpus generator. Every generator's start set must be
   provably disjoint from the probe.
2. **A held-out fixture is permanent.** The held-out probe (150 fresh starts) is
   now a standing second metric; any claimed gain must appear on held-out starts,
   not just trained ones.
3. **Smoke-test every lever before a GPU run** (the MLH lesson, generalized): an
   extreme-value sanity check first; a control arm always; adopt on
   mechanism + held-out direction, never on a single seen-set number.

## 5. Infrastructure delivered

Even with a null scientific result, the campaign hardened the toolchain:

- **Guarded iteration runner** (`run_iter_guarded.sh`, commits `0cc3cad`,
  `20ce90d`): resume snapshotting + automatic final-checkpoint backup (works
  around `run_baseline.sh:214` deleting `model_v*.pt` at startup), a heartbeat
  **watchdog** (900s stall limit), an **rc-guard** that probes only on a clean
  exit, and a **POOL_DEAD** signal surfaced when all pool members fail to load
  (commit `99c73fa`) instead of a silent RandomEvaluator fallback.
- **Revived 110-plane ladder.** iter-6 produced the **first live 110-plane pool
  promotions since 06-10** (3 promotions) — the repetition-plane architecture is
  now trainable end-to-end.
- **Detached-run pattern.** Harness/session restarts kill the guarded runner
  (watchdog, auto-probe, backup) while the `setsid` training group survives and
  finishes silently; standing procedure is now to check for an orphaned finished
  run (newest `model_v*.pt` mtime vs run log), then back up + probe by hand.
- **Data generation stack.** PGN ingest for external-corpus warm-start (commit
  `5178519`), plus the SF **demo** and **curriculum** generators
  (`pgn_cache_sf_bulk.pkl`, ~1500 starts / 6 classes) — now usable *because* the
  contamination rule forces disjoint start sets.
- **Held-out fixture** (`data/probe_deep_starts.txt`, 150 starts, 15–45 plies) as
  the permanent anti-memorization metric.
- **Anti-memorization curriculum.** iter-8's `make_decisive_starts` builds an
  ~8,165-start (100k-FEN) curriculum with mate-puzzle + near-mate mixing to break
  the seen-set overfit.

## 6. Open questions & campaign-3 framing

The central unknown is now **memorization vs generalization**: the net can
memorize demonstrated conversions but transfers ~none to unseen positions of the
same class. Directions:

- **Radius study (running).** Vary the distance between trained starts and probed
  starts to measure how far, if at all, learned technique generalizes off a
  demonstrated position.
- **Clean baselines pending.** iter-8 (clean baseline, TB defaults pinned OFF,
  resumed from the three-stream ckpt) and iter-9 (fresh-curriculum from-scratch)
  re-establish an honest, un-contaminated floor before any new lever.
- **Capacity / architecture hypotheses.** If the plateau is a *representation*
  limit, candidates are more channels/blocks, and finally landing the deferred
  repetition/rule-50 plane work end-to-end (partly live at 110 planes now).
- **Engine-assisted alternatives.** KataGo-style play-to-terminal with cheap
  downweighted searches, or an lc0-style MLH search bonus **gated on Q ≳ 0.9 and
  active in self-play generation** (not just at probe time) — the campaign only
  ever tested the bonus at probe time, where it was inert.

**Bottom line.** Three campaigns, 130+ iterations, conversion still unsolved. The
one lever that ever moved in-game behavior (policy-level SF demonstrations) is now
known to move it only via memorization. Campaign 3 must prove any gain on
held-out starts or it does not count.
</content>
</invoke>
