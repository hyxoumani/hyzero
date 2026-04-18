#!/usr/bin/env python3
"""Compare experiments from logs/experiments/*_results.json."""
import json, glob, sys, os

files = sorted(glob.glob("logs/experiments/*_results.json"))
if not files:
    print("No experiment results yet."); sys.exit(0)

def fmt(v, pad):
    if v is None: return ("-").rjust(pad)
    if isinstance(v, float): return f"{v:.4f}".rjust(pad)
    return str(v).rjust(pad)

rows = []
for f in files:
    r = json.load(open(f))
    rows.append(r)

headers = ["label","steps","last_v","pol_last","val_last","decis%","games","avg_len","top_p","ent","nvis","score"]
print(" ".join(h.rjust(10) for h in headers))
print(" ".join("-"*10 for _ in headers))
for r in rows:
    # Prefer the new decisive_ratio from [game_outcome] traces; fall back to PGN counts.
    if r.get("decisive_ratio") is not None and r.get("n_outcomes",0) > 0:
        d_ratio = r["decisive_ratio"] * 100
    else:
        outs = r.get("pgn_outcomes", {})
        total = sum(outs.values())
        decisive = total - outs.get("1/2-1/2", 0)
        d_ratio = (decisive/total*100) if total else 0.0
    row = [
        r["label"][:10], r.get("n_train_steps","-"), r.get("last_version","-"),
        r.get("policy_loss",{}).get("last"), r.get("value_loss",{}).get("last"),
        round(d_ratio,1), r.get("n_outcomes", r.get("games_total","-")), r.get("avg_game_length","-"),
        r.get("mcts_top_p_mean","-"), r.get("mcts_entropy_mean","-"),
        r.get("mcts_nvisited_mean","-"), r.get("training_score","-"),
    ]
    print(" ".join(fmt(c, 10) for c in row))
