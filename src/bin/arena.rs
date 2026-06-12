//! Head-to-head match tool: play N games between two FROZEN checkpoints.
//!
//! Usage:
//! ```text
//! cargo run --release --bin arena -- \
//!     --model-a <ckpt.pt> --model-b <ckpt.pt> --games <N> \
//!     [--sims 100] [--starts data/starting_positions.txt] [--concurrency 4]
//! ```
//!
//! Two independent, frozen Python `InferenceServer`s are constructed (one per
//! checkpoint), each fronted by its own `InferenceBatcher` + `ChannelEvaluator`,
//! mirroring the dual-model eval path in `src/selfplay/evaluation.rs`. No trainer
//! and no replay buffer are involved — the weights never change during a match.
//!
//! Mirrored-pair fairness: `N/2` starting FENs are taken from the starts file
//! (the first `N/2` lines, deterministic). Each FEN is played TWICE with colors
//! swapped (A-as-White then B-as-White) and wins are tallied PER MODEL, so a
//! white-advantage bias cancels out. Adjudication / termination reuse whatever
//! `play_game_dual_from` already does (move cap, eval adjudication via the
//! HYZERO_EVAL_ADJUDICATE* gates). Draws are counted as draws.
//!
//! Output (one machine-readable line at the end, to stdout):
//! ```text
//! [arena] model_a=<path> model_b=<path> games=<N> a_wins=<X> draws=<Y> b_wins=<Z>
//! ```

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tokio::sync::{mpsc, Semaphore};

use hyzero::game::board_from_fen;
use hyzero::mcts::evaluator::Evaluator;
use hyzero::py::PyO3Backend;
use hyzero::selfplay::game_task::{play_game_dual_from, DualGameOutcome, GameConfig};
use hyzero::selfplay::{BatcherConfig, ChannelEvaluator, InferenceBatcher};
use hyzero::PrecomputedItems;

/// Parsed arena CLI arguments.
#[derive(Debug, Clone, PartialEq)]
struct ArenaArgs {
    model_a: String,
    model_b: String,
    games: usize,
    sims: u32,
    starts: String,
    concurrency: usize,
}

/// One mirrored-pair game slot: which FEN index to play, and whether model A
/// holds the White pieces this game. Per pair, exactly two slots are emitted —
/// `a_is_white = true` then `false` — so both models play each opening from both
/// colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GameSlot {
    fen_index: usize,
    a_is_white: bool,
}

/// Outcome of a single arena game from MODEL A's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameScore {
    AWin,
    Draw,
    BWin,
}

/// Build the mirrored-pair schedule for `games` total games over `num_fens`
/// available starting FENs.
///
/// Emits `games` slots: `games / 2` mirrored pairs, each pair using one FEN index
/// played twice (A-as-White then B-as-White). FEN indices cycle 0..num_fens so a
/// short starts file still fills an odd or large `games` count deterministically.
/// When `games` is odd the trailing single game is dropped (mirrored pairs only),
/// so the returned length is `2 * (games / 2)`.
///
/// Pure + deterministic: no RNG, no I/O — unit-tested in `tests`.
fn build_schedule(games: usize, num_fens: usize) -> Vec<GameSlot> {
    let pairs = games / 2;
    let mut slots = Vec::with_capacity(pairs * 2);
    for pair in 0..pairs {
        // Cycle through available FENs when there are fewer than `pairs`.
        let fen_index = if num_fens == 0 { 0 } else { pair % num_fens };
        slots.push(GameSlot { fen_index, a_is_white: true });
        slots.push(GameSlot { fen_index, a_is_white: false });
    }
    slots
}

/// Convert a White-perspective `DualGameOutcome` into MODEL A's score, given
/// whether A played White in that game. `game_outcome` is +1 White win, -1 Black
/// win, 0 draw (and tanh-shaped values never occur here — eval adjudication
/// yields exactly ±1 or 0). The ±0.5 thresholds mirror the eval ladder's tally.
fn score_for_a(outcome: &DualGameOutcome, a_is_white: bool) -> GameScore {
    // Re-express the White-absolute outcome from A's perspective.
    let a_perspective = if a_is_white {
        outcome.game_outcome
    } else {
        -outcome.game_outcome
    };
    if a_perspective > 0.5 {
        GameScore::AWin
    } else if a_perspective < -0.5 {
        GameScore::BWin
    } else {
        GameScore::Draw
    }
}

/// Parse the arena CLI args from an iterator of raw arguments (excluding argv[0]).
///
/// Required: `--model-a`, `--model-b`, `--games`. Optional: `--sims` (default
/// 100), `--starts` (default `data/starting_positions.txt`), `--concurrency`
/// (default 4). Returns a human-readable error string on missing/invalid input.
fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<ArenaArgs, String> {
    let mut model_a: Option<String> = None;
    let mut model_b: Option<String> = None;
    let mut games: Option<usize> = None;
    let mut sims: u32 = 100;
    let mut starts: String = "data/starting_positions.txt".to_string();
    let mut concurrency: usize = 4;

    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--model-a" => {
                model_a = Some(it.next().ok_or("--model-a requires a value")?);
            }
            "--model-b" => {
                model_b = Some(it.next().ok_or("--model-b requires a value")?);
            }
            "--games" => {
                let v = it.next().ok_or("--games requires a value")?;
                games = Some(v.parse().map_err(|_| format!("invalid --games: {v}"))?);
            }
            "--sims" => {
                let v = it.next().ok_or("--sims requires a value")?;
                sims = v.parse().map_err(|_| format!("invalid --sims: {v}"))?;
            }
            "--starts" => {
                starts = it.next().ok_or("--starts requires a value")?;
            }
            "--concurrency" => {
                let v = it.next().ok_or("--concurrency requires a value")?;
                concurrency = v.parse().map_err(|_| format!("invalid --concurrency: {v}"))?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let model_a = model_a.ok_or("missing required --model-a <ckpt.pt>")?;
    let model_b = model_b.ok_or("missing required --model-b <ckpt.pt>")?;
    let games = games.ok_or("missing required --games <N>")?;
    if games < 2 {
        return Err("--games must be >= 2 (mirrored pairs)".to_string());
    }
    if concurrency == 0 {
        return Err("--concurrency must be >= 1".to_string());
    }

    Ok(ArenaArgs {
        model_a,
        model_b,
        games,
        sims,
        starts,
        concurrency,
    })
}

/// Read the first `count` non-empty FEN lines from `path`. Fewer than `count`
/// is fine — `build_schedule` cycles over whatever is available.
fn load_starts(path: &str, count: usize) -> Result<Vec<String>, String> {
    let contents =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read starts file {path}: {e}"))?;
    let fens: Vec<String> = contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .take(count)
        .map(|l| l.to_string())
        .collect();
    if fens.is_empty() {
        return Err(format!("starts file {path} contained no FENs"));
    }
    Ok(fens)
}

/// Build one frozen inference evaluator from a checkpoint path: instantiate a
/// fresh Python `InferenceServer`, load the checkpoint weights into it, wrap it
/// in a `PyO3Backend`, spawn a dedicated `InferenceBatcher`, and return a
/// `ChannelEvaluator` fronting that batcher. The weights are loaded once and
/// never changed — the model is frozen for the whole match.
///
/// Mirrors the champion-server construction in `src/bin/selfplay.rs`.
fn build_frozen_evaluator(
    ckpt_path: &str,
    device: &str,
    batcher_config: BatcherConfig,
) -> Arc<dyn Evaluator> {
    let ckpt_bytes = std::fs::read(ckpt_path)
        .unwrap_or_else(|e| panic!("[arena] failed to read checkpoint {ckpt_path}: {e}"));

    let (server, hidden_channels): (Py<PyAny>, usize) = Python::attach(|py| {
        let config_obj = PyModule::import(py, "hyzero.config")
            .expect("hyzero Python package not found — ensure it is installed")
            .getattr("DEFAULT_CONFIG")
            .expect("DEFAULT_CONFIG missing from hyzero.config")
            .into_pyobject(py)
            .expect("into_pyobject failed");
        let hc: usize = config_obj
            .cast::<PyDict>()
            .expect("DEFAULT_CONFIG is not a dict")
            .get_item("hidden_channels")
            .expect("hidden_channels lookup failed")
            .expect("hidden_channels not in DEFAULT_CONFIG")
            .extract()
            .expect("hidden_channels is not a usize");
        let config_unbound = config_obj.unbind();
        let cls = PyModule::import(py, "hyzero.inference.server")
            .expect("hyzero.inference.server not found")
            .getattr("InferenceServer")
            .expect("InferenceServer class not found");
        let srv: Py<PyAny> = cls
            .call1((config_unbound, device))
            .expect("InferenceServer() constructor failed")
            .unbind();
        (srv, hc)
    });

    // Load the frozen checkpoint weights.
    Python::attach(|py| {
        let py_bytes = PyBytes::new(py, &ckpt_bytes);
        server
            .call_method1(py, "load_weights", (py_bytes,))
            .unwrap_or_else(|e| panic!("[arena] failed to load weights from {ckpt_path}: {e}"));
    });

    let (tx, rx) = mpsc::channel(256);
    let backend = Box::new(PyO3Backend::new(server, hidden_channels));
    let mut batcher = InferenceBatcher::new(rx, backend, batcher_config);
    tokio::spawn(async move {
        batcher.run().await;
    });

    Arc::new(ChannelEvaluator::with_channels(tx, hidden_channels))
}

#[tokio::main]
async fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[arena] error: {e}");
            eprintln!(
                "usage: arena --model-a <ckpt.pt> --model-b <ckpt.pt> --games <N> \
                 [--sims 100] [--starts data/starting_positions.txt] [--concurrency 4]"
            );
            std::process::exit(2);
        }
    };

    let device = std::env::var("HYZERO_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    eprintln!(
        "[arena] device={device} sims={} concurrency={} games={}",
        args.sims, args.concurrency, args.games
    );

    // N/2 mirrored pairs → take the first N/2 FENs (deterministic).
    let num_pairs = args.games / 2;
    let fens = match load_starts(&args.starts, num_pairs) {
        Ok(f) => Arc::new(f),
        Err(e) => {
            eprintln!("[arena] error: {e}");
            std::process::exit(2);
        }
    };
    eprintln!("[arena] loaded {} starting FEN(s) from {}", fens.len(), args.starts);

    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());

    // Two FROZEN inference backends — one per model.
    let batcher_config = BatcherConfig {
        max_batch_size: 32,
        batch_timeout_ms: 10,
    };
    let eval_a = build_frozen_evaluator(&args.model_a, &device, batcher_config.clone());
    let eval_b = build_frozen_evaluator(&args.model_b, &device, batcher_config.clone());

    // Eval-style game config: PUCT (no root noise inside play_game_dual_from),
    // eval temperature (hardcoded 0.01), and the eval adjudication gates so a
    // material lead at the move cap decides an otherwise-drawn game.
    let game_config = GameConfig {
        num_simulations: args.sims,
        exploration_constant: 1.5,
        temperature_moves: 0,
        replay_dir: None,
        adjudicate_at_cap: eval_adjudicate_enabled(),
        adjudication_material_margin: eval_adjudication_margin(),
    };

    let schedule = build_schedule(args.games, fens.len());
    let total = schedule.len();

    // Concurrency: the per-model batchers batch across in-flight games, so
    // running several games at once improves GPU utilization. A semaphore bounds
    // the number of simultaneous games to --concurrency.
    let sem = Arc::new(Semaphore::new(args.concurrency));
    let mut handles = Vec::with_capacity(total);

    for (game_idx, slot) in schedule.into_iter().enumerate() {
        let sem = sem.clone();
        let fens = fens.clone();
        let precomputed = precomputed.clone();
        let eval_a = eval_a.clone();
        let eval_b = eval_b.clone();
        let game_config = game_config.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.expect("semaphore closed");
            let fen = &fens[slot.fen_index];
            let (board, side_to_move) = match board_from_fen(fen, precomputed.clone()) {
                Ok((b, stm, _fullmove)) => (b, stm),
                Err(e) => {
                    eprintln!("[arena] WARN: skipping unparseable FEN {fen:?}: {e}");
                    return None;
                }
            };

            // Assign White/Black evaluators for this mirrored-pair game.
            let (white_eval, black_eval) = if slot.a_is_white {
                (eval_a.clone(), eval_b.clone())
            } else {
                (eval_b.clone(), eval_a.clone())
            };

            let outcome = play_game_dual_from(
                white_eval,
                black_eval,
                game_config,
                board,
                side_to_move,
                Some(fen.clone()),
            )
            .await;

            let score = score_for_a(&outcome, slot.a_is_white);
            // Per-pair progress line (stderr — stdout is reserved for the result).
            let white_label = if slot.a_is_white { "A" } else { "B" };
            eprintln!(
                "[arena] game {}/{} fen_index={} white={} moves={} termination={} score_a={:?}",
                game_idx + 1,
                total,
                slot.fen_index,
                white_label,
                outcome.num_moves,
                outcome.termination,
                score,
            );
            Some(score)
        });
        handles.push(handle);
    }

    let mut a_wins = 0usize;
    let mut draws = 0usize;
    let mut b_wins = 0usize;
    for handle in handles {
        match handle.await {
            Ok(Some(GameScore::AWin)) => a_wins += 1,
            Ok(Some(GameScore::Draw)) => draws += 1,
            Ok(Some(GameScore::BWin)) => b_wins += 1,
            Ok(None) => {} // skipped (unparseable FEN)
            Err(e) => eprintln!("[arena] WARN: game task failed: {e}"),
        }
    }

    // The single machine-readable result line (stdout).
    println!(
        "[arena] model_a={} model_b={} games={} a_wins={} draws={} b_wins={}",
        args.model_a, args.model_b, total, a_wins, draws, b_wins,
    );
}

/// Eval-side adjudication gate, matching `src/selfplay/evaluation.rs`: ON by
/// default unless `HYZERO_EVAL_ADJUDICATE` is "0"/"false"/"no"/empty.
fn eval_adjudicate_enabled() -> bool {
    match std::env::var("HYZERO_EVAL_ADJUDICATE") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => true,
    }
}

/// Material lead required to adjudicate a non-checkmate terminal as decisive,
/// matching `src/selfplay/evaluation.rs`: `HYZERO_EVAL_ADJ_MARGIN`, default 5
/// (clamped to >= 1).
fn eval_adjudication_margin() -> i32 {
    std::env::var("HYZERO_EVAL_ADJ_MARGIN")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|&m| m >= 1)
        .unwrap_or(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Required flags parse into the expected struct; optional flags take their
    /// documented defaults when omitted.
    #[test]
    fn parse_args_applies_defaults_for_optional_flags() {
        let a = parse_args(args(&[
            "--model-a", "a.pt", "--model-b", "b.pt", "--games", "10",
        ]))
        .expect("parse ok");
        assert_eq!(
            a,
            ArenaArgs {
                model_a: "a.pt".to_string(),
                model_b: "b.pt".to_string(),
                games: 10,
                sims: 100,
                starts: "data/starting_positions.txt".to_string(),
                concurrency: 4,
            }
        );
    }

    /// All flags (including the optionals) parse to the supplied values.
    #[test]
    fn parse_args_reads_all_flags() {
        let a = parse_args(args(&[
            "--model-a", "x.pt", "--model-b", "y.pt", "--games", "8", "--sims", "50",
            "--starts", "s.txt", "--concurrency", "2",
        ]))
        .expect("parse ok");
        assert_eq!(a.sims, 50);
        assert_eq!(a.starts, "s.txt");
        assert_eq!(a.concurrency, 2);
    }

    /// A missing required flag is a parse error, not a silent default.
    #[test]
    fn parse_args_rejects_missing_required_flag() {
        let err = parse_args(args(&["--model-a", "a.pt", "--games", "4"]));
        assert!(err.is_err(), "missing --model-b must error");
    }

    /// `--games` below 2 cannot form a mirrored pair and is rejected.
    #[test]
    fn parse_args_rejects_games_below_two() {
        let err = parse_args(args(&[
            "--model-a", "a.pt", "--model-b", "b.pt", "--games", "1",
        ]));
        assert!(err.is_err(), "games < 2 must error");
    }

    /// The schedule emits `2 * (games/2)` slots: one mirrored pair per FEN with
    /// A-as-White followed by B-as-White, both sharing the same fen_index.
    #[test]
    fn build_schedule_emits_mirrored_pairs() {
        let slots = build_schedule(4, 2);
        assert_eq!(slots.len(), 4, "4 games = 2 mirrored pairs = 4 slots");
        // Pair 0: fen 0, A-white then B-white.
        assert_eq!(slots[0], GameSlot { fen_index: 0, a_is_white: true });
        assert_eq!(slots[1], GameSlot { fen_index: 0, a_is_white: false });
        // Pair 1: fen 1, A-white then B-white.
        assert_eq!(slots[2], GameSlot { fen_index: 1, a_is_white: true });
        assert_eq!(slots[3], GameSlot { fen_index: 1, a_is_white: false });
    }

    /// Every mirrored pair plays each color exactly once per model, so across the
    /// whole schedule A holds White in exactly half the games.
    #[test]
    fn build_schedule_is_color_balanced() {
        let slots = build_schedule(20, 7);
        let a_white = slots.iter().filter(|s| s.a_is_white).count();
        let a_black = slots.iter().filter(|s| !s.a_is_white).count();
        assert_eq!(a_white, a_black, "A must play White and Black equally often");
        assert_eq!(slots.len(), 20);
    }

    /// An odd `games` count drops the unpaired trailing game (mirrored pairs only).
    #[test]
    fn build_schedule_drops_odd_trailing_game() {
        let slots = build_schedule(5, 10);
        assert_eq!(slots.len(), 4, "5 games → 2 pairs → 4 slots (trailing dropped)");
    }

    /// FEN indices cycle when there are fewer FENs than pairs, so a short starts
    /// file still fills a large schedule deterministically.
    #[test]
    fn build_schedule_cycles_fens_when_short() {
        let slots = build_schedule(8, 2); // 4 pairs, 2 FENs
        let indices: Vec<usize> = slots.iter().step_by(2).map(|s| s.fen_index).collect();
        assert_eq!(indices, vec![0, 1, 0, 1], "pair FENs cycle 0,1,0,1");
    }

    fn outcome(game_outcome: f32) -> DualGameOutcome {
        DualGameOutcome {
            game_outcome,
            num_moves: 1,
            moves: vec![],
            termination: "move-cap".to_string(),
            starting_fen: None,
        }
    }

    /// When A is White, a White win (+1) is an A win; a Black win (-1) is a B win.
    #[test]
    fn score_for_a_when_a_is_white() {
        assert_eq!(score_for_a(&outcome(1.0), true), GameScore::AWin);
        assert_eq!(score_for_a(&outcome(-1.0), true), GameScore::BWin);
        assert_eq!(score_for_a(&outcome(0.0), true), GameScore::Draw);
    }

    /// When A is Black, the White-perspective outcome flips: a White win (+1) is a
    /// B win, a Black win (-1) is an A win. This is the per-model (not per-color)
    /// tally that mirrored pairs require.
    #[test]
    fn score_for_a_when_a_is_black() {
        assert_eq!(score_for_a(&outcome(1.0), false), GameScore::BWin);
        assert_eq!(score_for_a(&outcome(-1.0), false), GameScore::AWin);
        assert_eq!(score_for_a(&outcome(0.0), false), GameScore::Draw);
    }
}
