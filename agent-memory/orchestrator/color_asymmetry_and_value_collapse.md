# Color Asymmetry + Value Loss Collapse Investigation (2026-04-18)

## Symptoms

1. **Value loss collapse**: value loss falls to ~0.005-0.09 within the first
   ~50 training steps and stays there — essentially no value signal.
2. **Color asymmetry**: trained networks produce games where ~97% end in black's
   favor (outcome < -0.5 under tanh(material_delta/5) adjudication). Pre-fix
   run at commit 2edb194 with β=0.3 showed 0/100+ white wins.

## Bugs identified and fixed

### Bug 1 — Off-by-one in dynamics action indexing (commit 5e201cc)

`assemble_batch_arrays` was feeding `steps[k+1].action` into the dynamics
network when it should have been `steps[k].action` (the action that transitions
s_k → s_{k+1}). The StepRecord convention is: `steps[t].action` is pushed
BEFORE `board.process_move`, so `steps[t].action` IS a_t.

Effect: dynamics network `g(h_k, a_{k+1})` learned a shifted transition.
Regression test in `src/py/training.rs::test_dynamics_action_uses_step_k_not_step_kplus1`.

### Bug 2 — Value/reward target sign under color augmentation (commit b012944)

Under `apply_flip`, `flip_obs_planes` mirrors the observation (POV →
OPPOSITE POV), so training targets (`step.root_value` and outcome) must be
negated to match. The prior formula used
`effective_outcome = -game_outcome` paired with
`root_side_sign = sign_of(!steps[0].white_to_move)`; algebraically
`effective_outcome * root_side_sign * ply_flip == game_outcome *
original_root_side_sign * ply_flip` — i.e., INVARIANT under flip. And
`step.root_value` was never flipped at all.

Net effect: 50% of samples had wrong-sign value targets → loss averaged
to near-zero with no learnable direction.

Fix: multiply both `root_value` and `outcome_in_step_perspective` by a
uniform `flip_sign` = ±1. Regression test:
`test_value_target_sign_under_flip_matches_observation_pov`.

## Color symmetry of game-logic / MCTS / encoding

Ruled out as asymmetry sources via:
- `test_random_play_color_symmetry_audit` (N=2000 random vs random): White
  102 (5.1%), Black 77 (3.85%), Draws 1821 — slight white edge (initiative).
- `test_encode_board_initial_position_symmetry` — byte-identical encoding
  for both starting POVs.
- Grepped `Color::White | white_to_move==` through mcts/, selfplay/,
  data/ — no asymmetric control flow.

## Post-fix run (in progress, PID 153480, started 2026-04-18 22:53 UTC)

At commit 1daa560 with both fixes + β=0.3. After 37 games:
- white_wins: 2
- black_wins: 31
- draws: 4

Improvement over pre-fix (0 white / 100+ black) but still skewed 84% toward
black. Indicates a remaining factor — hypothesis: value-loss collapse persists
at low levels (0.005-0.08), so MCTS Q-values are still near-uniform and
moves are driven mostly by the policy prior, which tends toward "passive"
flank moves. Under passive play, the side that must COMMIT first (white)
tends to get punished.

## Remaining hypotheses to test

1. **Policy entropy / Dirichlet noise scaling**: if exploration is too
   high at the root, white's committed moves get random-selected from
   genuinely bad options.
2. **Value head saturation**: tanh output may be driven to one extreme by
   BatchNorm's running mean during eval; eval is in `torch.no_grad()` and
   `self.f.eval()` so BN uses running stats — those stats may have drifted
   during training.
3. **Residual POV bug in per-step `root_value` sign for mid-trajectory
   samples**: need to verify step.root_value is always in step-k-side POV
   for k > 0 as well (not just root).
4. **Adjudication asymmetry** (very unlikely — `compute_material_diff` is
   `white - black`, no sign bias).

## Next steps

- Monitor current run to T=60 min, T=120 min for outcome distribution shift.
- If asymmetry persists at 24% or lower (black_wins/total), bug 2 was the
  primary cause and we accept the remaining gap as real chess dynamics.
- If asymmetry stays above ~60%, hunt for bug 3+.
