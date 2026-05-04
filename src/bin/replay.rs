//! Replay viewer.
//!
//! `cargo run --bin replay -- <file.replay>` loads a per-ply MCTS dump produced
//! by self-play (when `HYZERO_REPLAY_DIR` was set) and steps through the game
//! interactively, showing the board on the left and the MCTS root statistics
//! on the right at each ply.
//!
//! Keys: ←/→ step, Home/End jump, q/Esc quit, space alias for →.

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    queue,
    style::{Color as TColor, Print, ResetColor, SetForegroundColor},
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};

use hyzero::data::{flip_action, ActionIndex, ReplayFile, NUM_BASE_ACTIONS};
use hyzero::game::fen::board_from_fen;
use hyzero::game::{GameBoard, Player};
use hyzero::mcts::puct::puct_score_detail;
use hyzero::{Color, Piece, PieceType, PrecomputedItems};

const TABLE_ROWS: usize = 12;

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: replay <file.replay>");
            std::process::exit(2);
        }
    };

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    let replay: ReplayFile = match bincode::deserialize(&bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("deserialize {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());

    if let Err(e) = run_tui(&path, &replay, precomputed) {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        eprintln!("tui error: {e}");
        std::process::exit(1);
    }
}

fn run_tui(
    path: &std::path::Path,
    replay: &ReplayFile,
    precomputed: Arc<PrecomputedItems>,
) -> io::Result<()> {
    let mut stdout = io::stdout();
    terminal::enable_raw_mode()?;
    crossterm::execute!(stdout, EnterAlternateScreen, cursor::Hide)?;

    let n_plies = replay.steps.len();
    let mut cur: usize = 0; // 0..=n_plies
    let mut board = build_board(replay, cur, &precomputed);

    let result = loop {
        if let Err(e) = render(&mut stdout, path, replay, cur, &board) {
            break Err(e);
        }
        match event::read()? {
            Event::Key(KeyEvent {
                code,
                kind,
                modifiers,
                ..
            }) if kind == KeyEventKind::Press => match code {
                KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => break Ok(()),
                KeyCode::Right | KeyCode::Char(' ') | KeyCode::Char('l') => {
                    if cur < n_plies {
                        cur += 1;
                        board = build_board(replay, cur, &precomputed);
                    }
                }
                KeyCode::Left | KeyCode::Char('h') => {
                    if cur > 0 {
                        cur -= 1;
                        board = build_board(replay, cur, &precomputed);
                    }
                }
                KeyCode::Home => {
                    cur = 0;
                    board = build_board(replay, cur, &precomputed);
                }
                KeyCode::End => {
                    cur = n_plies;
                    board = build_board(replay, cur, &precomputed);
                }
                _ => {}
            },
            Event::Resize(_, _) => {} // re-renders on next iter
            _ => {}
        }
        // Drain any pending events quickly so we don't fall behind on key repeat.
        while event::poll(Duration::from_millis(0))? {
            let _ = event::read()?;
        }
    };

    let _ = terminal::disable_raw_mode();
    let _ = crossterm::execute!(stdout, cursor::Show, LeaveAlternateScreen);
    result
}

/// Rebuild the board from the starting position and replay `n` plies onto it.
/// Easier than implementing reverse moves; chess games are short enough that
/// rebuild is cheap on every step.
fn build_board(replay: &ReplayFile, n: usize, precomputed: &Arc<PrecomputedItems>) -> GameBoard {
    let (mut board, mut side_to_move) = match replay.starting_fen.as_deref() {
        Some(fen) => match board_from_fen(fen, precomputed.clone()) {
            Ok((b, s, _)) => (b, s),
            Err(_) => default_board(precomputed),
        },
        None => default_board(precomputed),
    };

    for (ply, rec) in replay.steps.iter().take(n).enumerate() {
        // Action stored in current-player POV (flipped for Black). Un-flip
        // before converting to absolute-coordinate notation.
        let absolute = if side_to_move == Color::Black {
            flip_action(rec.action as usize) as ActionIndex
        } else {
            rec.action
        };
        let mv = action_to_uci(absolute, side_to_move);
        if board.process_move(&mv, side_to_move, ply).is_err() {
            // Replay corruption: stop applying further moves but keep showing
            // whatever board state we reached.
            break;
        }
        side_to_move = match side_to_move {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };
    }
    board
}

fn default_board(precomputed: &Arc<PrecomputedItems>) -> (GameBoard, Color) {
    let p1 = Player::init_player(true);
    let p2 = Player::init_player(false);
    (
        GameBoard::init_game_board(precomputed.clone(), p1, p2),
        Color::White,
    )
}

fn render(
    stdout: &mut io::Stdout,
    path: &std::path::Path,
    replay: &ReplayFile,
    cur: usize,
    board: &GameBoard,
) -> io::Result<()> {
    queue!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;

    let n_plies = replay.steps.len();
    let outcome = format_outcome(replay.game_outcome, replay.is_draw);
    let header = format!(
        "{}  v{}  outcome={}  c_puct={:.2}  ply {}/{}",
        path.file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        replay.model_version,
        outcome,
        replay.c_puct,
        cur,
        n_plies,
    );
    queue!(stdout, Print(header), cursor::MoveToNextLine(1))?;
    queue!(stdout, cursor::MoveToNextLine(1))?;

    let board_lines = render_board(&board.board_snapshot());

    let detail_lines = if cur < n_plies {
        render_move_detail(&replay.steps[cur], replay.c_puct)
    } else {
        vec![format!("(end of game — {})", outcome)]
    };

    let max_lines = board_lines.len().max(detail_lines.len());
    for i in 0..max_lines {
        let left = board_lines.get(i).map(String::as_str).unwrap_or("");
        let right = detail_lines.get(i).map(String::as_str).unwrap_or("");
        let line = format!("{left:<22}  {right}");
        queue!(stdout, Print(line), cursor::MoveToNextLine(1))?;
    }

    queue!(stdout, cursor::MoveToNextLine(1))?;
    queue!(
        stdout,
        SetForegroundColor(TColor::DarkGrey),
        Print("←/→ step   Home/End jump   q quit"),
        ResetColor,
        cursor::MoveToNextLine(1),
    )?;

    stdout.flush()
}

/// 8 ranks of the board, top = rank 8. Returns 9 lines: file header + 8 ranks.
fn render_board(snapshot: &[Option<Piece>; 64]) -> Vec<String> {
    let mut out = Vec::with_capacity(9);
    out.push("  a b c d e f g h    ".to_string());
    for rank in (0..8).rev() {
        let mut row = format!("{} ", rank + 1);
        for file in 0..8 {
            let sq = rank * 8 + file;
            row.push(piece_glyph(snapshot[sq]));
            row.push(' ');
        }
        row.push_str(&format!("{}", rank + 1));
        out.push(row);
    }
    out.push("  a b c d e f g h    ".to_string());
    out
}

fn piece_glyph(p: Option<Piece>) -> char {
    match p {
        None => '.',
        Some(Piece { piece_type, color }) => match (color, piece_type) {
            (Color::White, PieceType::Pawn) => 'P',
            (Color::White, PieceType::Knight) => 'N',
            (Color::White, PieceType::Bishop) => 'B',
            (Color::White, PieceType::Rook) => 'R',
            (Color::White, PieceType::Queen) => 'Q',
            (Color::White, PieceType::King) => 'K',
            (Color::Black, PieceType::Pawn) => 'p',
            (Color::Black, PieceType::Knight) => 'n',
            (Color::Black, PieceType::Bishop) => 'b',
            (Color::Black, PieceType::Rook) => 'r',
            (Color::Black, PieceType::Queen) => 'q',
            (Color::Black, PieceType::King) => 'k',
        },
    }
}

/// Lines describing the move at `cur`: chosen move, root value, MCTS table.
fn render_move_detail(rec: &hyzero::data::ReplayRecord, c_puct: f32) -> Vec<String> {
    let mut out = Vec::with_capacity(TABLE_ROWS + 4);

    let side = if rec.white_to_move { "White" } else { "Black" };
    let mover_color = if rec.white_to_move {
        Color::White
    } else {
        Color::Black
    };
    let chosen_abs = if rec.white_to_move {
        rec.action
    } else {
        flip_action(rec.action as usize) as ActionIndex
    };
    let chosen_uci = action_to_uci(chosen_abs, mover_color);

    out.push(format!(
        "{} to move — plays {}  (root V={:+.3})",
        side, chosen_uci, rec.root_value
    ));

    let n = rec.legal_moves.len();
    let parent_visits: u32 = rec.child_visits.iter().sum::<u32>() + 1;

    // Rank slots by visit count, descending. Tie-break by prior to keep the
    // ordering stable across runs.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        rec.child_visits[b].cmp(&rec.child_visits[a]).then_with(|| {
            rec.priors[b]
                .partial_cmp(&rec.priors[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    out.push(format!(
        "{:<7}{:>6}{:>6}{:>8}{:>8}{:>8}{:>8}",
        "move", "N", "N%", "P", "Q", "U", "PUCT"
    ));
    out.push("─".repeat(51));

    let total_visits: u32 = rec.child_visits.iter().sum();
    let total_visits_f = (total_visits.max(1)) as f32;

    let take = TABLE_ROWS.min(n);
    for &i in order.iter().take(take) {
        let abs_action = if rec.white_to_move {
            rec.legal_moves[i]
        } else {
            flip_action(rec.legal_moves[i] as usize) as ActionIndex
        };
        let uci = action_to_uci(abs_action, mover_color);
        let n_i = rec.child_visits[i];
        let p_i = rec.priors[i];
        let q_i = rec.q_values[i];
        let (q_out, u_out, total) = puct_score_detail(q_i, p_i, parent_visits, n_i, c_puct);
        let pct = 100.0 * (n_i as f32) / total_visits_f;
        let mark = if rec.legal_moves[i] == rec.action {
            '*'
        } else {
            ' '
        };
        out.push(format!(
            "{}{:<6}{:>6}{:>5.1}%{:>8.3}{:>8.3}{:>8.3}{:>8.3}",
            mark, uci, n_i, pct, p_i, q_out, u_out, total
        ));
    }

    if n > take {
        out.push(format!("  ... ({} more legal moves)", n - take));
    }

    out
}

fn format_outcome(o: f32, is_draw: bool) -> &'static str {
    if is_draw {
        "draw"
    } else if o > 0.5 {
        "1-0"
    } else if o < -0.5 {
        "0-1"
    } else {
        "draw"
    }
}

/// Convert an `ActionIndex` (in absolute coordinates) plus the moving color
/// into UCI coordinate notation (e.g. `"e2e4"`, `"a7a8q"`, `"e7d8n"`).
///
/// Mirrors the private encoder in `selfplay::game_task` — kept here so the
/// viewer binary doesn't need a non-public dependency.
fn action_to_uci(action: ActionIndex, color: Color) -> String {
    if action as usize >= NUM_BASE_ACTIONS {
        let offset = action as usize - NUM_BASE_ACTIONS;
        let piece_idx = offset / 192;
        let remainder = offset % 192;
        let from_file = (remainder / 24) as u8;
        let to_file_slot = (remainder % 24) as u8;
        let to_file = to_file_slot.min(7);

        let (from_rank_char, to_rank_char) = if color == Color::White {
            ('7', '8')
        } else {
            ('2', '1')
        };
        let from_file_char = (b'a' + from_file) as char;
        let to_file_char = (b'a' + to_file) as char;
        let suffix = match piece_idx {
            0 => 'n',
            1 => 'b',
            2 => 'r',
            _ => 'q',
        };
        return format!(
            "{}{}{}{}{}",
            from_file_char, from_rank_char, to_file_char, to_rank_char, suffix
        );
    }

    let from_sq = (action / 64) as u8;
    let to_sq = (action % 64) as u8;
    let from_file = (b'a' + from_sq % 8) as char;
    let from_rank = (b'1' + from_sq / 8) as char;
    let to_file = (b'a' + to_sq % 8) as char;
    let to_rank = (b'1' + to_sq / 8) as char;

    let from_rank_num = from_sq / 8;
    let to_rank_num = to_sq / 8;
    let is_promotion = (color == Color::White && from_rank_num == 6 && to_rank_num == 7)
        || (color == Color::Black && from_rank_num == 1 && to_rank_num == 0);
    if is_promotion {
        format!("{}{}{}{}q", from_file, from_rank, to_file, to_rank)
    } else {
        format!("{}{}{}{}", from_file, from_rank, to_file, to_rank)
    }
}
