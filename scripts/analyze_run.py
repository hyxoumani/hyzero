#!/usr/bin/env python3
"""Analyze a hyzero training run log against Phase 1 success criteria.

Tracks metric evolution over the run (not just final state), which is critical
for long runs where the network may go through multiple learning phases.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path


def parse_training_steps(content: str) -> list[dict]:
    """Parse every training step's loss values."""
    pattern = re.compile(
        r"\[py_training\] step (\d+): total=([\d.]+) policy=([\d.]+) value=([\d.]+) "
        r"reward=([\d.]+) \(v(\d+)\)"
    )
    return [
        {
            "step": int(m.group(1)),
            "total": float(m.group(2)),
            "policy": float(m.group(3)),
            "value": float(m.group(4)),
            "reward": float(m.group(5)),
            "version": int(m.group(6)),
        }
        for m in pattern.finditer(content)
    ]


def parse_eval_cycles(content: str) -> list[dict]:
    pattern = re.compile(
        r"\[eval\] v(\d+) cycle=(\d+) ladder_wins=(\d+) ladder_draws=(\d+) "
        r"ladder_losses=(\d+) win_rate=([\d.]+) champion_version=(\d+)"
    )
    return [
        {
            "challenger_version": int(m.group(1)),
            "cycle": int(m.group(2)),
            "wins": int(m.group(3)),
            "draws": int(m.group(4)),
            "losses": int(m.group(5)),
            "win_rate": float(m.group(6)),
            "champion_version": int(m.group(7)),
        }
        for m in pattern.finditer(content)
    ]


def parse_games(content: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"Game received: (\d+) steps", content)]


def summarize_phase(steps: list[dict], phase_label: str) -> str:
    if not steps:
        return f"    {phase_label}: no data"
    policy_start = steps[0]["policy"]
    policy_end = steps[-1]["policy"]
    value_start = steps[0]["value"]
    value_end = steps[-1]["value"]
    return (
        f"    {phase_label}: policy {policy_start:.3f}→{policy_end:.3f} "
        f"(Δ={policy_end - policy_start:+.3f}), "
        f"value {value_start:.4f}→{value_end:.4f} "
        f"(Δ={value_end - value_start:+.4f})"
    )


def analyze_pgn(pgn_path: str) -> dict:
    if not Path(pgn_path).exists():
        return {"pgn_available": False}

    content = Path(pgn_path).read_text()
    games_raw = content.split("[Event")[1:]

    results: Counter = Counter()
    white_first_moves: Counter = Counter()
    challenger_white_wins = 0
    champion_white_wins = 0
    draws = 0
    # By cycle
    cycle_pattern = re.compile(r"Eval Cycle (\d+)")
    first_move_by_cycle: dict = {}

    for game_text in games_raw:
        r_m = re.search(r'\[Result "([^"]+)"\]', game_text)
        w_m = re.search(r'\[White "([^"]+)"\]', game_text)
        cycle_m = cycle_pattern.search(game_text)
        if not (r_m and w_m):
            continue

        result = r_m.group(1)
        white_is_challenger = "challenger" in w_m.group(1)
        results[result] += 1

        # First-move extraction
        for line in game_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("1. "):
                tokens = stripped.split()
                if len(tokens) >= 2:
                    white_first_moves[tokens[1]] += 1
                    if cycle_m:
                        cycle = int(cycle_m.group(1))
                        first_move_by_cycle.setdefault(cycle, Counter())[tokens[1]] += 1
                break

        # Win attribution
        if result == "1-0":
            if white_is_challenger:
                challenger_white_wins += 1
            else:
                champion_white_wins += 1
        elif result == "0-1":
            # Black won → opponent of White
            if white_is_challenger:
                champion_white_wins += 1
            else:
                challenger_white_wins += 1
        else:
            draws += 1

    return {
        "pgn_available": True,
        "total_games": len(games_raw),
        "results": dict(results),
        "distinct_white_first_moves": len(white_first_moves),
        "top_white_first_moves": white_first_moves.most_common(5),
        "first_move_by_cycle": {
            k: dict(v.most_common(3)) for k, v in first_move_by_cycle.items()
        },
        "challenger_wins_net": challenger_white_wins,
        "champion_wins_net": champion_white_wins,
        "draws": draws,
    }


def print_report(log_path: str, pgn_path: str):
    content = Path(log_path).read_text()
    steps = parse_training_steps(content)
    evals = parse_eval_cycles(content)
    games = parse_games(content)
    adjudications = len(re.findall(r"\[selfplay\] adjudicated", content))
    promotions = len(re.findall(r"\[eval\] promoted", content))
    errors = len(re.findall(r"(?i)error|panic", content))

    print(f"Log: {log_path}")
    print("=" * 70)
    print("  Run summary")
    print("=" * 70)
    print(f"Games:             {len(games)}")
    print(f"Training steps:    {len(steps)}")
    print(f"Adjudications:     {adjudications}  (expect 0 post-Fix 3)")
    print(f"Eval cycles:       {len(evals)}")
    print(f"Promotions:        {promotions}")
    print(f"Errors/panics:     {errors}")
    print()

    if games:
        avg_len = sum(games) / len(games)
        print(f"Avg game length:   {avg_len:.1f} plies (max {max(games)}, min {min(games)})")
    print()

    if steps:
        # Phase breakdown: first quarter, last quarter, middle
        n = len(steps)
        print("  Training loss evolution")
        print("-" * 70)
        print(summarize_phase(steps[: n // 4 or 1], "first 25%"))
        print(summarize_phase(steps[n // 4 : 3 * n // 4], "middle 50%"))
        print(summarize_phase(steps[3 * n // 4 :], "last 25%"))
        print()
        # Check for dead value head
        recent = steps[-50:] if len(steps) >= 50 else steps
        avg_recent_value = sum(s["value"] for s in recent) / len(recent)
        if avg_recent_value < 0.005:
            print(f"  ⚠  value loss near zero ({avg_recent_value:.4f}) — may indicate dead value head")
        print()

    if evals:
        print("  Eval cycles")
        print("-" * 70)
        print(f"  {'cycle':>5s} {'v':>4s} {'champ':>5s} {'W':>3s} {'D':>3s} {'L':>3s} {'WR':>6s}")
        for e in evals[:20]:
            print(
                f"  {e['cycle']:>5d} {e['challenger_version']:>4d} "
                f"{e['champion_version']:>5d} {e['wins']:>3d} "
                f"{e['draws']:>3d} {e['losses']:>3d} {e['win_rate']:>6.3f}"
            )
        if len(evals) > 20:
            print(f"  ... ({len(evals) - 20} more cycles)")
        print()

    pgn = analyze_pgn(pgn_path)
    if pgn.get("pgn_available") and pgn["total_games"] > 0:
        print("  PGN analysis")
        print("-" * 70)
        print(f"Total games:       {pgn['total_games']}")
        print(f"Results:           {pgn['results']}")
        net = pgn["challenger_wins_net"] - pgn["champion_wins_net"]
        print(
            f"Challenger wins:   {pgn['challenger_wins_net']} | "
            f"Champion wins: {pgn['champion_wins_net']} | "
            f"Draws: {pgn['draws']} (Δ={net:+d})"
        )
        print(f"Distinct White 1st moves: {pgn['distinct_white_first_moves']}")
        for mv, cnt in pgn["top_white_first_moves"]:
            print(f"    {mv}: {cnt}")

        # First-move evolution by cycle
        if pgn.get("first_move_by_cycle"):
            print()
            print("  White 1st move by eval cycle (ideal: changes over time as network learns)")
            cycles_sorted = sorted(pgn["first_move_by_cycle"].keys())
            for c in cycles_sorted[-10:]:  # last 10 cycles
                moves = pgn["first_move_by_cycle"][c]
                top_mv = max(moves, key=moves.get)
                print(f"    cycle {c:>3d}: {top_mv} (cycle top move)")
        print()

    # Score
    if steps and games:
        final_policy = steps[-1]["policy"]
        avg_len = sum(games) / len(games)
        score = (8.55 - final_policy) + (promotions * 2.0) - (avg_len / 100)
        print(f"Estimated score:   {score:.2f}")
        print(f"  (formula: (8.55 - {final_policy:.3f}) + ({promotions} × 2.0) - ({avg_len:.1f}/100))")
        print()

    print("=" * 70)
    print("  Phase 1 Success Criteria")
    print("=" * 70)
    checks = []
    if steps and games:
        final_policy = steps[-1]["policy"]
        avg_len = sum(games) / len(games)
        score = (8.55 - final_policy) + (promotions * 2.0) - (avg_len / 100)
        checks.append(("score > 8 (vs 3.66 broken baseline)", score > 8))
    checks.append(("adjudications == 0 (Fix 3)", adjudications == 0))
    checks.append(("promotions >= 2", promotions >= 2))
    checks.append(("zero errors/panics", errors == 0))
    if pgn.get("pgn_available"):
        chal = pgn["challenger_wins_net"]
        champ = pgn["champion_wins_net"]
        total_decisive = chal + champ
        if total_decisive > 0:
            ratio_ok = abs(chal - champ) <= max(5, 0.4 * total_decisive)
            checks.append(
                (f"challenger/champion wins balanced ({chal}:{champ})", ratio_ok)
            )
    if steps and len(steps) > 50:
        recent = steps[-50:]
        avg_value = sum(s["value"] for s in recent) / len(recent)
        mid = steps[len(steps) // 2]
        value_changing = abs(avg_value - mid["value"]) > 0.003
        checks.append(
            (
                f"value loss showing dynamics (recent {avg_value:.4f} vs mid {mid['value']:.4f})",
                value_changing,
            )
        )

    for desc, ok in checks:
        print(f"  {'✓' if ok else '✗'} {desc}")
    print()


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    pgn_path = sys.argv[2] if len(sys.argv) > 2 else "logs/eval_games.pgn"

    if not log_path:
        candidates = sorted(Path("logs").glob("training_*.log"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("No training log found", file=sys.stderr)
            sys.exit(1)
        log_path = str(candidates[-1])

    print_report(log_path, pgn_path)
