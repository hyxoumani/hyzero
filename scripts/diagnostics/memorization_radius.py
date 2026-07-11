#!/usr/bin/env python3
"""Memorization-radius study on a demo-trained net.

Measures how far a supervised demo-trained network's move agreement with
Stockfish decays as trained probe positions are perturbed off the training
manifold. A memorized (rather than generalizing) net agrees with SF on the
exact trained starts but decays sharply on nearby perturbations; a net that
learned the underlying skill decays gently.

The signal is isolated by differencing against a pre-demo control checkpoint:
    delta(ring) = demo_top1(ring) - control_top1(ring)
The control has the same general endgame skill but never saw the demos, so a
positive delta that shrinks with perturbation radius is the memorization
fingerprint.

Perturbation rings (each start, winning side to move):
    d0  exact trained position.
    d1  defender (losing) king shifted 1 square.
    d2  additionally attacker (winning) king shifted 1 square.
    d3  a winning (attacker non-king) piece shifted 1-2 squares.
Every perturbed position is kept only if it is legal AND Stockfish still scores
it as a decisive win for the side to move (movetime 30ms). Agreement is the
net policy argmax vs the Stockfish best move (movetime 100ms).

The checkpoints here were trained with a categorical value head and the
moves-left head; this script forces the matching head config so the state_dict
loads cleanly. Torch runs on CPU (device='cpu').

Usage:
    python3 scripts/diagnostics/memorization_radius.py \
        --demo checkpoints/backup_iter_iter-6c_threestream_resume_final.pt \
        --control checkpoints/backup_iter3_tbdistill_v594.pt \
        --starts data/probe_won_starts_120.txt --num-starts 50
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

# These checkpoints carry a categorical value head + moves-left head; force the
# matching head config BEFORE importing config-consuming modules so the loaded
# state_dict shapes line up.
os.environ.setdefault("HYZERO_VALUE_HEAD", "categorical")
os.environ.setdefault("HYZERO_MOVES_LEFT_HEAD", "1")

import numpy as np
import chess
import chess.engine

from hyzero.config import DEFAULT_CONFIG
from hyzero.inference.server import InferenceServer
from hyzero.data.board_encoder import (
    encode_board_python,
    action_from_move,
    NUM_ACTIONS,
    NUM_BASE_ACTIONS,
)

RINGS = ["d0", "d1", "d2", "d3"]


# ─── Checkpoint loading ────────────────────────────────────────────────────────

def load_server(ckpt_path: str, device: str) -> InferenceServer:
    """Load an InferenceServer from a checkpoint on the given device."""
    srv = InferenceServer(dict(DEFAULT_CONFIG), device=device)
    with open(ckpt_path, "rb") as handle:
        srv.load_weights(handle.read())
    return srv


# ─── POV action flip (mirrors src/data/encoding.rs::flip_action) ────────────────

def _flip_sq(sq: int) -> int:
    return (7 - sq // 8) * 8 + (sq % 8)


def _flip_action(action: int) -> int:
    """Rank-mirror an action index; underpromo indices are flip-invariant."""
    if action < NUM_BASE_ACTIONS:
        return _flip_sq(action // 64) * 64 + _flip_sq(action % 64)
    return action


# ─── Net move + value ──────────────────────────────────────────────────────────

def net_top1(srv: InferenceServer, board: chess.Board):
    """Return (argmax legal move, scalar value) for the net on this board.

    The network operates in side-to-move (POV) space, so for Black the legal
    action indices are rank-mirrored (as the Rust selfplay path does) before
    building the legal mask, and the argmax POV index maps back to the real
    absolute move.
    """
    is_black = board.turn == chess.BLACK
    legal = list(board.legal_moves)
    if not legal:
        return None, 0.0
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    pov_to_move: dict[int, chess.Move] = {}
    for mv in legal:
        abs_a = action_from_move(mv, board)
        pov_a = _flip_action(abs_a) if is_black else abs_a
        mask[pov_a] = True
        pov_to_move[pov_a] = mv
    obs = encode_board_python(board).astype(np.float32)[None]
    out = srv.root_setup_batch(obs, mask[None])
    policy = out[1][0]
    value = float(out[2][0])
    best_pov = int(policy.argmax())
    return pov_to_move.get(best_pov), value


# ─── Stockfish helpers ─────────────────────────────────────────────────────────

def sf_is_won(engine: chess.engine.SimpleEngine, board: chess.Board,
              movetime_ms: int, cp_threshold: int = 800) -> bool:
    """True if Stockfish still scores the position as a decisive win for the
    side to move (mate or >= cp_threshold centipawns)."""
    try:
        info = engine.analyse(board, chess.engine.Limit(time=movetime_ms / 1000.0))
    except Exception:
        return False
    score = info["score"].pov(board.turn)
    if score.is_mate():
        return score.mate() > 0
    cp = score.score()
    return cp is not None and cp >= cp_threshold


def sf_best_move(engine: chess.engine.SimpleEngine, board: chess.Board,
                 movetime_ms: int):
    """Return Stockfish's best move at the given movetime, or None."""
    try:
        result = engine.play(board, chess.engine.Limit(time=movetime_ms / 1000.0))
    except Exception:
        return None
    return result.move


# ─── Perturbation generators ───────────────────────────────────────────────────

def _king_adjacent(sq: int) -> list[int]:
    """The up-to-8 board squares a king step away from sq."""
    out = []
    f, r = sq % 8, sq // 8
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                out.append(nr * 8 + nf)
    return out


def _within(sq: int, radius: int) -> list[int]:
    """Board squares within Chebyshev `radius` of sq (excluding sq itself)."""
    out = []
    f, r = sq % 8, sq // 8
    for df in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if df == 0 and dr == 0:
                continue
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                out.append(nr * 8 + nf)
    return out


def _move_piece(board: chess.Board, from_sq: int, to_sq: int) -> chess.Board:
    """Copy the board and relocate the piece on from_sq to (empty) to_sq,
    preserving side to move."""
    b = board.copy()
    piece = b.piece_at(from_sq)
    b.remove_piece_at(from_sq)
    b.set_piece_at(to_sq, piece)
    b.ep_square = None
    return b


def gen_ring(board: chess.Board, ring: str, engine, rng: random.Random,
             verify_ms: int, target: int) -> list[chess.Board]:
    """Generate up to `target` legal, still-won perturbations for the ring."""
    attacker = board.turn
    defender = not attacker
    dk = board.king(defender)
    ak = board.king(attacker)
    winning_pieces = [
        sq for sq in chess.SQUARES
        if (p := board.piece_at(sq)) is not None
        and p.color == attacker and p.piece_type != chess.KING
    ]

    candidates: list[chess.Board] = []

    if ring == "d1":
        for dst in _king_adjacent(dk):
            if board.piece_at(dst) is None:
                candidates.append(_move_piece(board, dk, dst))
    elif ring == "d2":
        for ddst in _king_adjacent(dk):
            if board.piece_at(ddst) is not None:
                continue
            b1 = _move_piece(board, dk, ddst)
            for adst in _king_adjacent(ak):
                if adst == ddst or b1.piece_at(adst) is not None:
                    continue
                candidates.append(_move_piece(b1, ak, adst))
    elif ring == "d3":
        for psq in winning_pieces:
            for dst in _within(psq, 2):
                if board.piece_at(dst) is None:
                    candidates.append(_move_piece(board, psq, dst))
    else:
        raise ValueError(f"unknown ring {ring}")

    rng.shuffle(candidates)
    accepted: list[chess.Board] = []
    for cand in candidates:
        if not cand.is_valid() or cand.is_game_over():
            continue
        if sf_is_won(engine, cand, verify_ms):
            accepted.append(cand)
        if len(accepted) >= target:
            break
    return accepted


# ─── Study driver ──────────────────────────────────────────────────────────────

def run_study(demo_path: str, control_path: str, starts_path: str,
              num_starts: int, device: str, seed: int,
              verify_ms: int, best_ms: int, per_ring: int) -> dict:
    rng = random.Random(seed)
    with open(starts_path) as handle:
        all_fens = [ln.strip() for ln in handle if ln.strip()]
    rng.shuffle(all_fens)

    demo = load_server(demo_path, device)
    control = load_server(control_path, device)
    engine = chess.engine.SimpleEngine.popen_uci(
        os.environ.get("HYZERO_MG_STOCKFISH_BIN", "stockfish")
    )

    # Per-ring tallies: hits/total for each net, plus demo value samples.
    tally = {r: {"demo_hit": 0, "ctrl_hit": 0, "n": 0, "demo_val": []}
             for r in RINGS}

    used = 0
    try:
        for fen in all_fens:
            if used >= num_starts:
                break
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            if not board.is_valid() or board.is_game_over():
                continue
            # The exact start must itself be a decisive win to anchor d0.
            if not sf_is_won(engine, board, verify_ms):
                continue
            used += 1

            for ring in RINGS:
                if ring == "d0":
                    positions = [board]
                else:
                    positions = gen_ring(board, ring, engine, rng,
                                         verify_ms, per_ring)
                for pos in positions:
                    sf_mv = sf_best_move(engine, pos, best_ms)
                    if sf_mv is None:
                        continue
                    demo_mv, demo_v = net_top1(demo, pos)
                    ctrl_mv, _ = net_top1(control, pos)
                    tally[ring]["n"] += 1
                    tally[ring]["demo_hit"] += int(demo_mv == sf_mv)
                    tally[ring]["ctrl_hit"] += int(ctrl_mv == sf_mv)
                    tally[ring]["demo_val"].append(demo_v)
    finally:
        engine.quit()

    rows = {}
    for r in RINGS:
        t = tally[r]
        n = t["n"]
        demo_pct = 100.0 * t["demo_hit"] / n if n else 0.0
        ctrl_pct = 100.0 * t["ctrl_hit"] / n if n else 0.0
        val_mean = float(np.mean(t["demo_val"])) if t["demo_val"] else 0.0
        rows[r] = {
            "n": n,
            "demo_top1": demo_pct,
            "control_top1": ctrl_pct,
            "delta": demo_pct - ctrl_pct,
            "demo_value_mean": val_mean,
        }

    return {
        "demo_ckpt": os.path.basename(demo_path),
        "control_ckpt": os.path.basename(control_path),
        "starts_file": os.path.basename(starts_path),
        "num_starts_used": used,
        "verify_ms": verify_ms,
        "best_ms": best_ms,
        "per_ring": per_ring,
        "seed": seed,
        "rows": rows,
    }


# ─── Reporting ─────────────────────────────────────────────────────────────────

_HDR = f"{'ring':<5} {'n':>5} {'demo%':>8} {'ctrl%':>8} {'delta':>8} {'demoVal':>9}"


def format_table(result: dict) -> str:
    lines = [_HDR, "-" * len(_HDR)]
    for r in RINGS:
        row = result["rows"][r]
        lines.append(
            f"{r:<5} {row['n']:>5} {row['demo_top1']:>8.1f} "
            f"{row['control_top1']:>8.1f} {row['delta']:>+8.1f} "
            f"{row['demo_value_mean']:>9.3f}"
        )
    return "\n".join(lines)


def write_report(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = result["rows"]
    d0d = r["d0"]["delta"]
    d3d = r["d3"]["delta"]
    with open(path, "w") as fh:
        fh.write("# Memorization-radius study\n\n")
        fh.write(f"- Demo checkpoint: `{result['demo_ckpt']}`\n")
        fh.write(f"- Control checkpoint: `{result['control_ckpt']}`\n")
        fh.write(f"- Starts: `{result['starts_file']}` "
                 f"(used {result['num_starts_used']})\n")
        fh.write(f"- SF verify movetime: {result['verify_ms']}ms; "
                 f"best-move movetime: {result['best_ms']}ms\n")
        fh.write(f"- Variants/ring/start target: {result['per_ring']}; "
                 f"seed {result['seed']}\n\n")
        fh.write("## Top-1 agreement vs Stockfish\n\n")
        fh.write("| ring | n | demo top-1 % | control top-1 % | "
                 "delta (demo-ctrl) | demo value mean |\n")
        fh.write("|------|---|--------------|-----------------|"
                 "-------------------|-----------------|\n")
        for k in RINGS:
            row = r[k]
            fh.write(f"| {k} | {row['n']} | {row['demo_top1']:.1f} | "
                     f"{row['control_top1']:.1f} | {row['delta']:+.1f} | "
                     f"{row['demo_value_mean']:.3f} |\n")
        fh.write("\n## Interpretation\n\n")
        fh.write(
            "The delta column isolates the demo-training memorization signal "
            "from general endgame skill (shared by the control). A large "
            "positive delta at d0 that collapses toward zero by d3 indicates "
            "the demo net memorized the exact trained starts rather than "
            "learning a transferable conversion policy. A roughly flat delta "
            "across rings indicates genuine generalization of the demo signal.\n\n"
        )
        fh.write(
            f"Observed delta shrinks from {d0d:+.1f} pts (d0, exact) to "
            f"{d3d:+.1f} pts (d3, winning-piece shift). "
        )
        if d0d - d3d > 5.0:
            fh.write("The decay of the delta with perturbation radius is the "
                     "memorization fingerprint: the demo advantage is "
                     "largely manifold-local.\n\n")
        else:
            fh.write("The delta does not decay materially with radius, "
                     "consistent with the demo signal generalizing beyond the "
                     "exact trained starts.\n\n")
        fh.write("The demo value-head mean per ring probes off-manifold "
                 "confidence collapse: a falling mean from d0 to d3 means the "
                 "net grows less certain as positions leave the training "
                 "manifold.\n")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Memorization-radius study.")
    parser.add_argument(
        "--demo",
        default="checkpoints/backup_iter_iter-6c_threestream_resume_final.pt",
    )
    parser.add_argument(
        "--control",
        default="checkpoints/backup_iter3_tbdistill_v594.pt",
    )
    parser.add_argument("--starts", default="data/probe_won_starts_120.txt")
    parser.add_argument("--num-starts", type=int, default=50)
    parser.add_argument("--per-ring", type=int, default=6)
    parser.add_argument("--verify-ms", type=int, default=30)
    parser.add_argument("--best-ms", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report",
        default="runs/auto-20260706-100435/memorization_radius_report.md",
    )
    args = parser.parse_args(argv)

    result = run_study(
        args.demo, args.control, args.starts, args.num_starts,
        args.device, args.seed, args.verify_ms, args.best_ms, args.per_ring,
    )
    print(format_table(result))
    write_report(result, args.report)
    print(f"\nReport written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
