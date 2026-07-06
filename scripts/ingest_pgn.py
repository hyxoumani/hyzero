#!/usr/bin/env python3
"""Thin CLI wrapper around ``hyzero.data.pgn_ingest`` for external-corpus warm-start.

Reads a PGN file, replays each game, and writes a pickled ``list[PGNTrajectory]``
cache whose batch-dict rows match the trainer's tablebase-mixer schema. Point the
trainer at the output via ``HYZERO_PGN_CACHE_PATH`` + ``HYZERO_PGN_FRAC``.

Usage:
    python3 scripts/ingest_pgn.py corpus.pgn data/pgn/warmstart.pkl \
        --k-steps 5 --min-elo 2000 --max-games 5000 --stats

Policy label smoothing and value decay are env-tunable:
    HYZERO_PGN_POLICY_SMOOTH  (default 0.0)
    HYZERO_PGN_VALUE_DISCOUNT (default 1.0)
"""

from __future__ import annotations

import os
import sys

# Ensure hyzero package is importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from hyzero.data.pgn_ingest import main

if __name__ == "__main__":
    raise SystemExit(main())
