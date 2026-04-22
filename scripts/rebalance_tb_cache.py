#!/usr/bin/env python3
"""Rebalance a Syzygy tablebase position cache so +1/-1 sample counts match.

Reads the existing cache, splits by target_value sign, takes min(len_pos, len_neg)
samples from each signed bucket (exact count balance), keeps all draws, shuffles,
and writes to a new file.

Usage:
    python3 scripts/rebalance_tb_cache.py

Environment variables:
    HYZERO_TABLEBASE_CACHE_PATH: Input cache path. Default: data/syzygy/cache.pkl.
    HYZERO_BALANCED_CACHE_PATH:  Output cache path. Default: data/syzygy/cache_balanced.pkl.

Output:
    Prints: "Rebalanced cache: N_pos=X, N_neg=X, N_zero=Y, total=Z"
"""

from __future__ import annotations

import os
import pickle
import random
import sys
from dataclasses import dataclass


# ─── TBSample dataclass (mirrors python/hyzero/data/tablebase.py) ────────────

@dataclass
class TBSample:
    fen: str
    target_value: float
    mating_actions: list[int]
    optimal_actions: list[int]
    all_legal_actions: list[int]


# ─── Config ───────────────────────────────────────────────────────────────────

IN_PATH = os.environ.get("HYZERO_TABLEBASE_CACHE_PATH", "data/syzygy/cache.pkl")
OUT_PATH = os.environ.get("HYZERO_BALANCED_CACHE_PATH", "data/syzygy/cache_balanced.pkl")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[rebalance] Loading cache from {IN_PATH!r} ...")
    if not os.path.isfile(IN_PATH):
        print(f"ERROR: Input cache not found: {IN_PATH!r}", file=sys.stderr)
        sys.exit(1)

    # Load cache — handle pickle class mismatch (built by __main__.TBSample).
    with open(IN_PATH, "rb") as f:
        raw = pickle.load(f)

    # Normalise to local TBSample regardless of pickle origin.
    samples: list[TBSample] = [
        TBSample(
            fen=item.fen,
            target_value=float(item.target_value),
            mating_actions=list(item.mating_actions),
            optimal_actions=list(item.optimal_actions),
            all_legal_actions=list(item.all_legal_actions),
        )
        for item in raw
    ]

    print(f"[rebalance] Loaded {len(samples)} samples.")

    # Split by target_value sign.
    list_pos  = [s for s in samples if s.target_value > 0.0]
    list_neg  = [s for s in samples if s.target_value < 0.0]
    list_zero = [s for s in samples if s.target_value == 0.0]

    n_keep = min(len(list_pos), len(list_neg))
    if n_keep == 0:
        print("ERROR: One of the signed buckets is empty — nothing to balance.", file=sys.stderr)
        sys.exit(1)

    balanced_pos  = random.sample(list_pos, n_keep)
    balanced_neg  = random.sample(list_neg, n_keep)

    combined = balanced_pos + balanced_neg + list_zero
    random.shuffle(combined)

    n_pos  = len(balanced_pos)
    n_neg  = len(balanced_neg)
    n_zero = len(list_zero)
    total  = len(combined)

    print(f"[rebalance] Before: N_pos={len(list_pos)}, N_neg={len(list_neg)}, N_zero={len(list_zero)}, total={len(samples)}")
    print(f"[rebalance] After:  N_pos={n_pos}, N_neg={n_neg}, N_zero={n_zero}, total={total}")

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[rebalance] Writing balanced cache to {OUT_PATH!r} ...")
    with open(OUT_PATH, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Rebalanced cache: N_pos={n_pos}, N_neg={n_neg}, N_zero={n_zero}, total={total}")


if __name__ == "__main__":
    main()
