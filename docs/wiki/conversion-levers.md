# Conversion Levers

The conversion problem — teaching hyzero to actually *finish* won positions rather
than shuffle indefinitely — closes **UNSOLVED** after three campaigns (~100
iterations, 2026-06 → 07-12). The honest metric is **generalized conversion ~1–3%
on the held-out fixture** (`data/probe_holdout_starts_150.txt`), now PRIMARY
because the old 120-start probe is contaminated for any demo-trained lineage (see
retraction below). Every **data-side** lever family is flat on that fixture:
demonstrations (memorization only), curriculum variety (erases recall +
overconfidence, builds nothing), pure truthful selfplay, value-level TB
distillation (both heads), and value calibration (label smoothing). Search scaling
also fails — overconfident net 4.0→0.0 @400 sims AND calibrated net 1.3→0.0 @400 —
so it is not a value-trust problem alone. Convergent conclusion: the constraint is
**architectural** (a hidden-state MuZero-style search cannot execute long forced
sequences without real-board grounding below root). Attack structure, not data.

## Lever ledger

- **Material adjudication (margin 12)** — KEEP, gated. Densifies decisive labels
  (best_ref 6.795→8.610) but pays +1 for unconverted shuffles, poisoning the
  curriculum. Keep only with a curriculum-start exemption: `initial_fen.is_some()`
  gate so fixed-start won positions are exempt.
- **Truthful labels + repetition planes (102→110, lc0-style 1/history)** —
  NECESSARY but INSUFFICIENT alone. iter-2 (24h from-scratch) reached only 10.8%.
  With ~84% draw targets the categorical value head collapses to the draw prior;
  value loss pinned at 1.09–1.17 *is* the collapse signature, not a fit.
- **Moves-left head** — VALIDLY EXHAUSTED. iter-42/43 probes were vacuous
  (`HYZERO_MLH_SEARCH_BONUS` was never set). July-2 iter-1 wired it live: control
  12.5 / bonus-0.2 12.5 (zero moves changed) / factor-5.0 gate-0 12.1 — signal is
  flat. Lesson: always smoke-test a lever at an extreme value before spending GPU.
- **Value-level TB distillation** — FALSIFIED ON BOTH HEADS. June (tanh
  wash-through) and July-2 iter-3 (categorical, FRAC 0.75 steepened cache) both
  fit the value loss with bimodal ±0.9 `tgt_hist` (mixing provably active) yet
  produced ZERO game transfer: cm_count frozen, kqk_value probes 0.37–0.57 vs the
  ~1.0 needed. The *mechanism*, not the head, is the dead end.
- **Curriculum variety (8165 unique starts)** — FALSIFIED as a conversion lever,
  HIGH diagnostic value. Held-out 1.3 flat, yet it *erased* both failure modes it
  was meant to fix: memorization (d0 exact-start +46→0) AND value overconfidence
  (0.94→0.33). Variety washes out recall and calibration artifacts but builds no
  transferable technique — conversion value nil.
- **Value calibration (label-smooth 0.05)** — FALSIFIED. Held-out 2.0 flat.
  Smoothing the categorical value targets does not unlock finishing.
- **Root-child terminal grounding (mcts)** — PARTIAL. Defenders now exploit hangs
  and forced draws exactly, but attacker-side avoidance still needs value/policy:
  terminals are only visible at depth 2 from the attacker's root.
- **Curriculum temperature window (`HYZERO_CURRICULUM_TEMPERATURE_MOVES=2`)** —
  WORKS. Holds temp≈1.0 sampling through the decisive phase of short curriculum
  games. A data-quality fix, not a conversion lever per se.
- **Policy-level SF demonstrations** — **RETRACTED as a conversion lever**
  (2026-07-10 contamination audit). The demo generator was seeded from the probe
  start files. 2×2 audit: trained-starts 23–25% vs held-out 4.0% (demo net);
  clean-split retrain 1.7% / 0.7% — pure position-keyed memorization, ZERO
  transferable technique. Deep demos actively DEGRADE shallow conversion. The
  hang-rate improvement is partially real (non-demo starts) but unstable. The
  earlier "24.2% / only lever that changed behavior" claim was this leak.
- **Pure truthful selfplay (12h)** — NO generalization gain. iter-8 held-out 2.7%
  flat vs 4.0 baseline. More honest data alone does not break the plateau.
- **Search scaling** — DEAD ON BOTH NETS. Overconfident net INVERTS: 400 sims
  scored 0/150 held-out vs 4.0% @100 (more search amplifies flat off-manifold
  overconfidence; radius agreement 46%→12% while value stays 0.94→0.92). Calibrated
  net still dead: 1.3 @100 → 0.0 @400. Not a value-trust problem alone — the
  architectural read now supersedes the earlier "calibration is binding" hypothesis.
- **Memorization radius** — local generalization exists but does not compound:
  +46pts exact start / +20 one square off / +10 two squares off; sequences do not
  chain into finished mates.

## Gotchas

- Probe fixtures must NEVER seed training-data generators — the 07-08 "24.2%
  breakthrough" was exactly this bug. Audit train/test overlap before believing
  any metric gain.
- `scripts/run_baseline.sh:214` runs `find checkpoints -name 'model_v*.pt' -delete`
  at startup — resume ONLY via `run_iter_guarded.sh`'s snapshot guard (which also
  auto-backs-up final ckpts).
- MuZero tree is terminal-blind below root: `tree.rs` `top_k(64)` children are never
  empty, so true terminals are invisible; grounding is only possible at root children.
- `HYZERO_TB_SUPERVISION_GRADED` defaults to 1 and silently re-joins plain WDL over
  pre-steepened caches — set it to 0 when using `cache_tb_dtz_steep.pkl`.
- `conversion_probe.sh` deletes its game logs (mktemp trap), so probe arms cannot be
  move-diffed after the fact.
- MLH bonus requires `HYZERO_MOVES_LEFT_HEAD=1` and a matching `HYZERO_MLH_CAP` at
  probe time.
- Probe starts are shallow (SF avg 4.8 plies); `data/probe_deep_starts.txt` (150
  starts, 15–45 plies) exists for a graded successor metric.
- Harness/session restart kills the guarded runner (watchdog, auto-probe, final-ckpt
  backup) while the setsid training group survives and completes silently. After any
  restart: check for an orphaned finished run (newest `model_v*.pt` mtime vs run log),
  then backup + probe by hand.
- `run_baseline.sh` env defaults leak into any run not explicitly overriding them —
  e.g. `HYZERO_TB_POLICY_WEIGHT=0.5` and `HYZERO_TB_SUPERVISION_GRADED=1` rode along
  in a PGN-mixing run as a confound. Pin every supervision-relevant env var in
  `EXTRA_ENV`, including the ones intended OFF.

## Related

- [[conversion-campaign-202607]]
- [[mcts]]
- [[training-signal]]
- [[board-encoding]]
- [[run-history]]
