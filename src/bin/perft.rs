/// perft — CLI for perft node counting and move enumeration.
///
/// Usage:
///   perft [--divide|--moves|--status] <FEN> [depth]
///
/// Modes:
///   (default)  Print total node count at the given depth.
///   --divide   Print per-move node counts and total.
///   --moves    Print all legal moves in UCI format, one per line, sorted.
///   --status   Print game termination status for the given FEN.
use std::sync::Arc;

use hyzero::game::fen::board_from_fen;
use hyzero::game::perft::{get_legal_moves_for_perft, perft};
use hyzero::game::Move;
use hyzero::{Color, PieceType, PrecomputedItems};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse mode flag and remaining positional args.
    let mut divide = false;
    let mut moves_only = false;
    let mut status_only = false;
    let mut positional: Vec<&str> = Vec::new();

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "--divide" => divide = true,
            "--moves" => moves_only = true,
            "--status" => status_only = true,
            other => positional.push(other),
        }
    }

    if positional.is_empty() {
        eprintln!("Usage: perft [--divide|--moves|--status] <FEN> [depth]");
        std::process::exit(1);
    }

    let fen = positional[0];

    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());

    let (mut board, color, _fullmove) = match board_from_fen(fen, precomputed.clone()) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("FEN parse error: {}", e);
            std::process::exit(1);
        }
    };

    if status_only {
        println!("{}", board.game_status(color));
        return;
    }

    if moves_only {
        // --moves: enumerate legal moves in UCI format, sorted.
        let mut legal = get_legal_moves_for_perft(&board, color);
        let mut uci_moves: Vec<String> = legal.drain(..).map(move_to_uci).collect();
        uci_moves.sort();
        for m in uci_moves {
            println!("{}", m);
        }
        return;
    }

    // Both default and --divide modes require a depth argument.
    if positional.len() < 2 {
        eprintln!("Usage: perft [--divide] <FEN> <depth>");
        std::process::exit(1);
    }
    let depth: u32 = match positional[1].parse() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Invalid depth: {}", positional[1]);
            std::process::exit(1);
        }
    };

    if divide {
        // --divide: print per-move counts and total.
        let legal = get_legal_moves_for_perft(&board, color);
        let next_color = opponent(color);
        let turn_count = if color == Color::White { 0 } else { 1 };

        let mut total = 0u64;
        let mut rows: Vec<(String, u64)> = Vec::with_capacity(legal.len());
        for mv in &legal {
            let mut new_board = board.clone();
            new_board.compute_turn_items(turn_count, *mv);
            let count = if depth > 1 {
                perft(&new_board, next_color, depth - 1, &precomputed)
            } else {
                1
            };
            total += count;
            rows.push((move_to_uci(*mv), count));
        }
        // Sort by move string for deterministic output.
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for (uci, count) in rows {
            println!("{}: {}", uci, count);
        }
        println!("Total: {}", total);
    } else {
        // Default: print total node count only.
        let count = perft(&board, color, depth, &precomputed);
        println!("{}", count);
    }
}

/// Convert a `Move` to UCI notation (e.g. `e2e4`, `e1g1`, `a7a8q`).
fn move_to_uci(mv: Move) -> String {
    let from = mv.from as u8;
    let to = mv.to as u8;
    let from_str = square_to_str(from);
    let to_str = square_to_str(to);
    if let Some(promo) = mv.promotion_piece_type {
        let promo_char = piece_to_char(promo);
        format!("{}{}{}", from_str, to_str, promo_char)
    } else {
        format!("{}{}", from_str, to_str)
    }
}

/// Convert a square index (0=a1, 63=h8) to algebraic notation.
fn square_to_str(sq: u8) -> String {
    let file = (b'a' + sq % 8) as char;
    let rank = (b'1' + sq / 8) as char;
    format!("{}{}", file, rank)
}

/// Lowercase promotion piece character.
fn piece_to_char(pt: PieceType) -> char {
    match pt {
        PieceType::Queen => 'q',
        PieceType::Rook => 'r',
        PieceType::Bishop => 'b',
        PieceType::Knight => 'n',
        _ => '?',
    }
}

/// Return the opponent color.
fn opponent(color: Color) -> Color {
    if color == Color::White {
        Color::Black
    } else {
        Color::White
    }
}
