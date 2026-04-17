# Attempt 3 Game Pattern Analysis (adj threshold=12)

Analysis of 787 eval games + 2410 self-play games from the completed 8-hour run.

## Headline findings

- **99.5% of eval games are DRAWS** (4 real checkmates in 787 games = 0.5%)
- **89.5% of games contain a rook shuffle** (≥5 cycles of a1↔b1 or a8↔b8)
- **77% of White first moves are `b1a3`** (the classic passivity opener)
- **90% of Black first moves are `a7a5`** (passive a-pawn push)
- **346 promotions occur in games** (mostly queens) but none lead to checkmate

**The score of 9.34 is NOT a measure of chess skill** — it's a measure of adjudication-based material wins in self-play. Eval games reveal the model can't actually play chess.

## Checkmate rates across attempts

| Attempt | Config | Games | Mates | Rate |
|---|---|---|---|---|
| 1 | no adj, no material | 31 | 7 | **22.6%** |
| 2 | no adj + material-at-cap | 72 | 7 | **9.7%** |
| 3 | **adj=12 + material-at-cap** | 787 | 4 | **0.5%** |

Adjudication at threshold 12 reduced the real checkmate rate by 20× compared to attempt 1. The model optimizes for "don't lose material for 10 plies" instead of "put king in mate."

## Pattern: "Shuffle with promotions"

Typical game structure (from Cycle 1 Game 5):
```
1. b1a3 a7a5 2. a1b1 b7b5 3. b1a1 d7d5 4. a1b1 e7e6 5. g1f3 d8f6 ...
[pieces develop minimally, pawns push]
[a few captures happen, some pawns promote to queens]
[ending:]
... b7a6 b1a1 a6c4 a1b1 c4b3 b1a1 b3c2 a1a2 c2d1q a2a1 d1c2 a1a2 c2d1q a2a1 d1c2
```

The model:
1. Opens with knight-to-corner (b1a3)
2. Starts rook shuffle (a1↔b1) 
3. Occasionally pushes pawns or moves other pieces
4. Pawn eventually promotes to queen
5. Continues shuffling with the queen instead of mating
6. Game ends by repetition/50-move/insufficient material → draw

## Pattern: the 4 real checkmate games

Of 787 eval games, only 4 ended in real checkmate:
1. **Cycle 1 Game 1**: challenger v1 vs Random v0 (v0 mated challenger) — 0-1
2. **Cycle 1 Game 2**: Random v0 vs challenger v1 (v0 mated challenger) — 1-0
3. **Cycle 1 Game 8**: Random v0 vs challenger v1 (v0 mated challenger) — 0-1 (wait, that's Black mating, so challenger Black beat Random)
4. **Cycle 77 Game 3**: challenger v1959 vs Random-era champion v1 (challenger mated) — 1-0

**3 of 4 checkmates involved Random v0** (either Random mating challenger, or challenger mating Random). After v1 becomes trained, games become shuffles.

Game 4 (cycle 77) is the one where **challenger v1959 actually mated trained champion v1** via 43 moves, playing normal-ish chess: `1. g1f3 ... developing pieces ... 43. f4g5` checkmate.

## Promotion statistics

- 346 total promotions across all games (0.63% of moves)
- Queen promotions: 295 (85%)
- Underpromotions: 51 (15%)
- Knight: 17, Bishop: 19, Rook: 15 (surprisingly balanced)

Underpromotions aren't rare — the action space fix from Phase 1 is being used. But queens dominate as expected.

## Self-play asymmetry (from self-play adjudications)

- 380 adjudications total across 2410 self-play games (16% rate at end)
- Black wins 372 of 380 adjudications (98%)
- White wins 8 of 380 (2%)

Persistent Black-win bias in self-play. Color augmentation (50% flip) balances TRAINING data but the underlying self-play dynamics favor Black via first-mover-waste + plane-101-asymmetry.

## Implications

1. **Adjudication at any threshold corrupts chess learning**. Even threshold=12 (intended to fire only on crushing positions) trains the value head to optimize for material accumulation, not positional understanding.

2. **The 9.34 score is a mirage**. It's reward hacking — adjudications in self-play produce easy ±1 signals that the value head can learn from, but those signals don't generalize to "play good chess."

3. **Attempt 2's lower score (~5) with higher real-checkmate-rate (9.7%) is ARGUABLY a better model**. It hadn't learned to exploit adjudication, so it played more like real chess (even if worse at the score metric).

4. **Plane 101 is likely a major contributor to asymmetric Black-wins-in-self-play**. Network uses plane 101 to distinguish colors and has learned color-specific strategies from random init, reinforced by biased self-play data.

## Recommendation

Per user suggestion: **remove adjudication entirely**. Accept lower score metric in the short-term. The real test is whether **checkmate rate trends upward over training cycles**. If we see cycle-10 checkmate rate > cycle-1 checkmate rate, the model IS learning chess — even if score stays modest.

Consider tracking checkmate rate by cycle as the *actual* success metric.
