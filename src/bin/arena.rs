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
//! Mirrored-pair fairness: `N/2` starting FENs are sampled from the starts file
//! by a deterministic, seedless STRIDE over the whole file (index
//! `floor(i * L / (N/2))` for `i in 0..N/2`, `L` = non-empty line count), so the
//! selection spans the entire file rather than just its head. Each FEN is played
//! TWICE with colors swapped (A-as-White then B-as-White) and wins are tallied
//! PER MODEL, so a white-advantage bias cancels out. Adjudication / termination
//! reuse whatever `play_game_dual_from` already does (move cap, eval adjudication
//! via the HYZERO_EVAL_ADJUDICATE* gates). Draws are counted as draws.
//!
//! Output (one machine-readable line at the end, to stdout):
//! ```text
//! [arena] model_a=<path> model_b=<path> games=<N> a_wins=<X> draws=<Y> b_wins=<Z>
//! ```

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tokio::sync::{mpsc, Semaphore};

use hyzero::data::{encode_board, flip_action, move_to_action, ActionIndex};
use hyzero::game::board_from_fen;
use hyzero::game::perft::get_legal_moves_for_perft;
use hyzero::mcts::evaluator::Evaluator;
use hyzero::py::PyO3Backend;
use hyzero::selfplay::game_task::{play_game_dual_from, DualGameOutcome, GameConfig};
use hyzero::selfplay::{BatcherConfig, ChannelEvaluator, InferenceBatcher};
use hyzero::{Color, PrecomputedItems};

/// Standard chess starting position FEN, used for the cheap weight-load
/// fingerprint probe (a single eval through the same ChannelEvaluator used to
/// play). Distinct checkpoints must yield distinct fingerprints.
const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Parsed arena CLI arguments.
#[derive(Debug, Clone, PartialEq)]
struct ArenaArgs {
    model_a: String,
    model_b: String,
    games: usize,
    sims: u32,
    starts: String,
    concurrency: usize,
    device: String,
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
/// (default 4), `--device` (default: `HYZERO_DEVICE` env, else `cuda` — matching
/// `scripts/run_baseline.sh` and `src/bin/selfplay.rs`). The `--device` flag
/// overrides the env. Returns a human-readable error string on missing/invalid
/// input.
fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<ArenaArgs, String> {
    let mut model_a: Option<String> = None;
    let mut model_b: Option<String> = None;
    let mut games: Option<usize> = None;
    let mut sims: u32 = 100;
    let mut starts: String = "data/starting_positions.txt".to_string();
    let mut concurrency: usize = 4;
    // Default device: HYZERO_DEVICE env, falling back to "cuda" (matching
    // scripts/run_baseline.sh `DEVICE=${HYZERO_DEVICE:-cuda}`). A `--device` flag
    // overrides it.
    let mut device: String = std::env::var("HYZERO_DEVICE").unwrap_or_else(|_| "cuda".to_string());

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
            "--device" => {
                device = it.next().ok_or("--device requires a value")?;
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
        device,
    })
}

/// Deterministic, seedless stride indices selecting `count` positions spread over
/// `line_count` available lines: `floor(i * line_count / count)` for `i in
/// 0..count`. This spans the WHOLE file (head-to-tail) instead of taking only the
/// first `count` lines, so e.g. 100 FENs sample evenly across a 100k-line file.
///
/// When `count >= line_count`, every line is selected (and, for `count >
/// line_count`, later indices repeat the tail — `build_schedule` already cycles
/// FENs, so duplicate selections are harmless). `count == 0` or `line_count == 0`
/// yields an empty vector. Pure + deterministic — unit-tested in `tests`.
fn stride_indices(line_count: usize, count: usize) -> Vec<usize> {
    if count == 0 || line_count == 0 {
        return Vec::new();
    }
    (0..count)
        .map(|i| (i * line_count) / count)
        .collect()
}

/// Sample `count` non-empty FEN lines from `path` by deterministic stride over the
/// whole file (see `stride_indices`). Returns the sampled FENs plus the total
/// non-empty line count `L`, so the caller can log the sampling provenance. Fewer
/// available lines than `count` is fine — `build_schedule` cycles over whatever is
/// returned.
fn load_starts(path: &str, count: usize) -> Result<(Vec<String>, usize), String> {
    let contents =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read starts file {path}: {e}"))?;
    let lines: Vec<&str> = contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return Err(format!("starts file {path} contained no FENs"));
    }
    let line_count = lines.len();
    let fens: Vec<String> = stride_indices(line_count, count)
        .into_iter()
        .map(|idx| lines[idx].to_string())
        .collect();
    Ok((fens, line_count))
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

/// Cheap weight-load fingerprint: evaluate the standard starting position once
/// through the SAME `ChannelEvaluator` used to play, and return the root value
/// (side-to-move POV). Two different checkpoints must produce different values;
/// identical checkpoints produce identical ones — so a silent `load_weights`
/// failure (both models sharing one set of weights) is caught by comparing the
/// two fingerprints. Reuses `encode_board` + the perft legal-move generator and
/// the same Black-POV `flip_action` canonicalization that `play_game_dual_from`
/// applies, so the probe exercises the real inference path.
async fn fingerprint_evaluator(
    evaluator: &Arc<dyn Evaluator>,
    precomputed: Arc<PrecomputedItems>,
) -> f32 {
    let (board, side_to_move, _fullmove) = board_from_fen(START_FEN, precomputed)
        .unwrap_or_else(|e| panic!("[arena] failed to parse START_FEN for fingerprint: {e}"));

    let observation = encode_board(&board, side_to_move, &[]);

    // Legal moves in current-player POV, matching the play path's flip + sort.
    let raw_moves = get_legal_moves_for_perft(&board, side_to_move);
    let mut legal_actions: Vec<ActionIndex> = raw_moves
        .iter()
        .map(|mv| {
            let a = move_to_action(mv);
            if side_to_move == Color::Black {
                flip_action(a as usize) as ActionIndex
            } else {
                a
            }
        })
        .collect();
    legal_actions.sort_unstable();

    let mut legal_mask = vec![false; hyzero::data::NUM_ACTIONS];
    for &a in &legal_actions {
        legal_mask[a as usize] = true;
    }

    let (_hidden_state, _policy, value) = evaluator.root_setup(&observation, &legal_mask).await;
    value
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
                 [--sims 100] [--starts data/starting_positions.txt] [--concurrency 4] \
                 [--device cuda]"
            );
            std::process::exit(2);
        }
    };

    let device = args.device.clone();
    eprintln!(
        "[arena] device={device} sims={} concurrency={} games={}",
        args.sims, args.concurrency, args.games
    );

    // N/2 mirrored pairs → sample N/2 FENs by deterministic stride over the whole
    // starts file (not just its head), so the openings span the full file.
    let num_pairs = args.games / 2;
    let (fens, line_count) = match load_starts(&args.starts, num_pairs) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[arena] error: {e}");
            std::process::exit(2);
        }
    };
    let fens = Arc::new(fens);
    eprintln!(
        "[arena] loaded {} starting FEN(s) from {} ({} non-empty lines, deterministic stride sampling)",
        fens.len(),
        args.starts,
        line_count,
    );

    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());

    // Two FROZEN inference backends — one per model.
    let batcher_config = BatcherConfig {
        max_batch_size: 32,
        batch_timeout_ms: 10,
    };
    let eval_a = build_frozen_evaluator(&args.model_a, &device, batcher_config.clone());
    let eval_b = build_frozen_evaluator(&args.model_b, &device, batcher_config.clone());

    // Weight-load diagnostic: one cheap eval of the start position per model.
    // Distinct checkpoints MUST show distinct fingerprints; identical files show
    // identical ones — guarding against a silent `load_weights` failure that would
    // leave both models effectively sharing weights.
    let fp_a = fingerprint_evaluator(&eval_a, precomputed.clone()).await;
    let fp_b = fingerprint_evaluator(&eval_b, precomputed.clone()).await;
    eprintln!("[arena] model_a fingerprint value={fp_a:.6}");
    eprintln!("[arena] model_b fingerprint value={fp_b:.6}");
    if args.model_a != args.model_b && fp_a == fp_b {
        eprintln!(
            "[arena] WARN: distinct checkpoints produced identical fingerprints \
             ({fp_a:.6}) — possible silent load_weights failure"
        );
    }

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
    /// documented defaults when omitted. The device default reads process env
    /// (`HYZERO_DEVICE`, else "cuda") and so is exercised via `--device` overrides
    /// rather than asserted here.
    #[test]
    fn parse_args_applies_defaults_for_optional_flags() {
        let a = parse_args(args(&[
            "--model-a", "a.pt", "--model-b", "b.pt", "--games", "10",
        ]))
        .expect("parse ok");
        assert_eq!(a.model_a, "a.pt");
        assert_eq!(a.model_b, "b.pt");
        assert_eq!(a.games, 10);
        assert_eq!(a.sims, 100);
        assert_eq!(a.starts, "data/starting_positions.txt");
        assert_eq!(a.concurrency, 4);
    }

    /// All flags (including the optionals) parse to the supplied values.
    #[test]
    fn parse_args_reads_all_flags() {
        let a = parse_args(args(&[
            "--model-a", "x.pt", "--model-b", "y.pt", "--games", "8", "--sims", "50",
            "--starts", "s.txt", "--concurrency", "2", "--device", "cpu",
        ]))
        .expect("parse ok");
        assert_eq!(a.sims, 50);
        assert_eq!(a.starts, "s.txt");
        assert_eq!(a.concurrency, 2);
        assert_eq!(a.device, "cpu");
    }

    /// `--device` overrides whatever the env/default would supply.
    #[test]
    fn parse_args_device_flag_overrides_default() {
        let a = parse_args(args(&[
            "--model-a", "a.pt", "--model-b", "b.pt", "--games", "4", "--device", "cuda:1",
        ]))
        .expect("parse ok");
        assert_eq!(a.device, "cuda:1");
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

    /// Sampling `count` from `line_count` lines spans the whole file head-to-tail
    /// via `floor(i * line_count / count)` — not just the first `count` lines.
    #[test]
    fn stride_indices_spans_whole_file() {
        // 4 samples over 100 lines → 0, 25, 50, 75 (full-file spread, not 0..4).
        assert_eq!(stride_indices(100, 4), vec![0, 25, 50, 75]);
    }

    /// The stride formula matches the documented `floor(i * L / count)` exactly,
    /// including the truncating division on a non-divisible split.
    #[test]
    fn stride_indices_uses_floor_division() {
        // 3 samples over 10 lines → floor(0), floor(10/3=3), floor(20/3=6).
        assert_eq!(stride_indices(10, 3), vec![0, 3, 6]);
    }

    /// The first sampled index is always 0 and the last is strictly inside the
    /// file (`< line_count`), so no index ever overruns the available lines.
    #[test]
    fn stride_indices_stay_in_bounds() {
        let idx = stride_indices(100_000, 100);
        assert_eq!(idx.len(), 100);
        assert_eq!(idx[0], 0);
        assert!(idx.iter().all(|&i| i < 100_000), "every index in [0, L)");
        // The 100-sample stride over 100k lines reaches the file's tail.
        assert_eq!(*idx.last().unwrap(), 99_000);
    }

    /// The sampler is deterministic and seedless: identical inputs yield identical
    /// indices across calls.
    #[test]
    fn stride_indices_is_deterministic() {
        assert_eq!(stride_indices(777, 13), stride_indices(777, 13));
    }

    /// A degenerate request (zero samples or empty file) yields no indices rather
    /// than panicking on a divide-by-zero.
    #[test]
    fn stride_indices_handles_empty_inputs() {
        assert!(stride_indices(0, 5).is_empty(), "empty file → no indices");
        assert!(stride_indices(50, 0).is_empty(), "zero samples → no indices");
    }

    /// When `count` exceeds the line count, every line is reachable and the
    /// indices are non-decreasing (later samples repeat the tail). `build_schedule`
    /// already cycles FENs, so duplicate selections are harmless.
    #[test]
    fn stride_indices_when_count_exceeds_lines() {
        let idx = stride_indices(3, 6);
        assert_eq!(idx.len(), 6);
        assert_eq!(idx[0], 0);
        assert!(idx.iter().all(|&i| i < 3), "every index in [0, 3)");
        assert!(
            idx.windows(2).all(|w| w[0] <= w[1]),
            "indices are non-decreasing"
        );
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
