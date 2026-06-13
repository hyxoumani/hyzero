//! Root forced-mate solver.
//!
//! Exhaustive AND/OR (DFS) search over the engine's REAL move generator: does the
//! side to move have a forced checkmate within `max_plies` plies?
//!
//!   mate-in-1 → 1 ply, mate-in-2 → 3 plies, mate-in-3 → 5 plies.
//!
//! The mover (OR node) needs ONE move such that ALL opponent replies (AND node)
//! lead to mate within the remaining plies. Checkmate / stalemate / draw semantics
//! are taken verbatim from `GameBoard::compute_turn_items` (via `apply_move` here):
//! a stalemate is NOT a mate, and only `GameResult::Checkmate(mover)` counts as a win.
//!
//! Safety caps: `max_plies` bounds depth; a node-count cap (`MAX_NODES`) bounds the
//! total positions explored so a middlegame position cannot blow up — on overflow
//! the search returns `None` (no mate proven) rather than running unbounded.

use super::Move;
use crate::game::board::{GameBoard, GameResult};
use crate::{BitIterator, CastleOption, Color, PieceType, Square};

/// Node-count safety cap. Sparse endgames stay far below this; middlegame
/// positions hit it and the solver bails out with `None`.
const MAX_NODES: u64 = 200_000;

/// Search-time mutable counter so all recursion levels share one budget.
struct NodeBudget {
    used: u64,
    cap: u64,
}

impl NodeBudget {
    /// Consume one node. Returns `false` once the cap is exceeded.
    #[inline]
    fn spend(&mut self) -> bool {
        self.used += 1;
        self.used <= self.cap
    }
}

/// Find a forced mate for the side to move in at most `max_plies` plies.
///
/// `max_plies` is counted in half-moves: mate-in-1 = 1, mate-in-2 = 3, etc.
/// Returns `Some(mv)` for a move that forces mate within the bound, or `None`
/// if no forced mate is proven (including on node-cap overflow or `max_plies == 0`).
pub fn find_forced_mate(board: &GameBoard, color: Color, max_plies: u32) -> Option<Move> {
    if max_plies == 0 {
        return None;
    }
    let mut budget = NodeBudget {
        used: 0,
        cap: MAX_NODES,
    };
    find_mate_move(board, color, max_plies, &mut budget)
}

/// OR node: the mover searches for a single move that forces mate within
/// `plies_left`. Returns the mating move when one exists.
fn find_mate_move(
    board: &GameBoard,
    color: Color,
    plies_left: u32,
    budget: &mut NodeBudget,
) -> Option<Move> {
    if plies_left == 0 {
        return None;
    }
    for mv in legal_moves(board, color) {
        if !budget.spend() {
            return None; // node-cap overflow: abandon search, prove nothing
        }
        let next = apply_move(board, mv, color);
        match next.result() {
            // The mover just delivered checkmate — immediate win.
            GameResult::Checkmate(winner) if winner == color => return Some(mv),
            // Opponent to move and the position is still going: this move forces
            // mate only if EVERY reply is still mated within the remaining plies.
            GameResult::Ongoing => {
                if plies_left >= 2
                    && opponent_is_lost(&next, opposite(color), plies_left - 1, budget)
                {
                    return Some(mv);
                }
            }
            // Any other terminal (stalemate / draw / opponent mate) is not a win
            // for the mover; this move does not force mate.
            _ => {}
        }
    }
    None
}

/// AND node: returns true iff the side to move (`color`, the defender) is lost —
/// i.e. it has at least one reply and EVERY reply lets the attacker force mate
/// within `plies_left`. A defender with no legal moves here is already terminal
/// (handled by the caller via `result()`), so this only runs on ongoing positions.
fn opponent_is_lost(
    board: &GameBoard,
    color: Color,
    plies_left: u32,
    budget: &mut NodeBudget,
) -> bool {
    let replies = legal_moves(board, color);
    if replies.is_empty() {
        // No legal reply in an ongoing position should not happen (it would be a
        // terminal), but treat it conservatively as not-forced-mate.
        return false;
    }
    for mv in replies {
        if !budget.spend() {
            return false; // node-cap overflow: cannot prove the defender is lost
        }
        let next = apply_move(board, mv, color);
        match next.result() {
            // Defender's move delivered mate against the attacker — escape.
            GameResult::Checkmate(winner) if winner == color => return false,
            GameResult::Ongoing => {
                // Attacker (the original side) must still force mate from here.
                if find_mate_move(&next, opposite(color), plies_left - 1, budget).is_none() {
                    return false; // this reply escapes the mate
                }
            }
            // Any non-checkmate terminal (stalemate / draw) is an escape: the
            // defender avoided being mated.
            _ => return false,
        }
    }
    true
}

#[inline]
fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

/// Clone the board and apply `mv` (made by `color`) using the engine's real
/// `compute_turn_items`, so checkmate / stalemate / draw detection is identical
/// to actual play. `compute_turn_items` decides which side to evaluate for
/// checkmate from the `count` parity (even → White just moved), so we pass a
/// `count` whose parity matches `color`.
fn apply_move(board: &GameBoard, mv: Move, color: Color) -> GameBoard {
    let mut next = board.clone();
    // Even count → White just moved (engine checks Black for mate); odd → Black moved.
    let count = if color == Color::White { 0 } else { 1 };
    next.compute_turn_items(count, mv);
    next
}

/// Enumerate every legal move for `color` as `Move` structs (the real movegen).
///
/// Mirrors `selfplay::game_task::get_legal_moves` but yields `Move`s instead of
/// action indices: pseudo-legal moves from `get_move_mask`, validated by
/// `validate_move`, including the four promotion piece types, en passant, and
/// both castles.
fn legal_moves(board: &GameBoard, color: Color) -> Vec<Move> {
    let mut moves = Vec::new();
    let combined = board.white_pieces | board.black_pieces;

    for sq in 0..64usize {
        let piece = match board.board_arr[sq] {
            Some(p) if p.color == color => p,
            _ => continue,
        };

        let move_mask = board.get_move_mask(
            sq,
            color,
            piece.piece_type,
            combined,
            board.white_pieces,
            board.black_pieces,
        );

        for to_sq in BitIterator::new(move_mask) {
            // Never generate a king capture. Kings can never be legally captured;
            // the engine's check detection has a known gap (a checked king is
            // allowed to "escape" along the checking ray, since occupancy isn't
            // cleared at the king's old square), which in deep AND/OR lines can let
            // a king-capturing reply slip past `validate_move`. Applying such a move
            // yields a king-less position that panics the engine's pin calculation.
            // Filtering king-destination squares here keeps the search on legal,
            // panic-free lines without altering the engine's mate semantics.
            if let Some(captured) = board.board_arr[to_sq] {
                if captured.piece_type == PieceType::King {
                    continue;
                }
            }
            let from = Square::from(sq as u8);
            let to = Square::from(to_sq as u8);
            let to_rank = to_sq / 8;
            let is_promotion =
                piece.piece_type == PieceType::Pawn && (to_rank == 7 || to_rank == 0);
            let en_passant = piece.piece_type == PieceType::Pawn
                && board.en_passant_target == Some(to_sq)
                && (sq % 8 != to_sq % 8);

            if is_promotion {
                for &promo in &[
                    PieceType::Queen,
                    PieceType::Knight,
                    PieceType::Bishop,
                    PieceType::Rook,
                ] {
                    let candidate = Move {
                        from,
                        to,
                        promotion_piece_type: Some(promo),
                        castle_option: None,
                        en_passant: false,
                    };
                    if board.validate_move(
                        candidate,
                        color,
                        combined,
                        board.white_pieces,
                        board.black_pieces,
                    ) {
                        moves.push(candidate);
                    }
                }
            } else {
                let candidate = Move {
                    from,
                    to,
                    promotion_piece_type: None,
                    castle_option: None,
                    en_passant,
                };
                if board.validate_move(
                    candidate,
                    color,
                    combined,
                    board.white_pieces,
                    board.black_pieces,
                ) {
                    moves.push(candidate);
                }
            }
        }

        if piece.piece_type == PieceType::King {
            for &castle_opt in &[CastleOption::Kingside, CastleOption::Queenside] {
                let to_file: u8 = match castle_opt {
                    CastleOption::Kingside => 6,
                    CastleOption::Queenside => 2,
                };
                let king_rank = if color == Color::White { 0u8 } else { 7u8 };
                let to_sq_castle = king_rank * 8 + to_file;
                let candidate = Move {
                    from: Square::from(sq as u8),
                    to: Square::from(to_sq_castle),
                    promotion_piece_type: None,
                    castle_option: Some(castle_opt),
                    en_passant: false,
                };
                if board.validate_move(
                    candidate,
                    color,
                    combined,
                    board.white_pieces,
                    board.black_pieces,
                ) {
                    moves.push(candidate);
                }
            }
        }
    }

    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::fen::board_from_fen;
    use crate::PrecomputedItems;
    use std::sync::Arc;

    fn precomputed() -> Arc<PrecomputedItems> {
        Arc::new(PrecomputedItems::begin_precomputing())
    }

    /// Parse a FEN and return (board, side_to_move).
    fn from_fen(fen: &str) -> (GameBoard, Color) {
        let (board, side, _full) =
            board_from_fen(fen, precomputed()).expect("test FEN should parse");
        (board, side)
    }

    fn move_str(mv: &Move) -> String {
        let f = u8::from(mv.from);
        let t = u8::from(mv.to);
        format!(
            "{}{}{}{}",
            (b'a' + f % 8) as char,
            (b'1' + f / 8) as char,
            (b'a' + t % 8) as char,
            (b'1' + t / 8) as char,
        )
    }

    /// Verify the returned move actually delivers checkmate for `color`,
    /// per the engine's own `compute_turn_items` result.
    fn move_delivers_mate(board: &GameBoard, mv: Move, color: Color) -> bool {
        let next = apply_move(board, mv, color);
        next.result() == GameResult::Checkmate(color)
    }

    /// All FENs below are verified against the ENGINE's own checkmate/stalemate
    /// detection (`compute_turn_items`), not an external oracle, so the tests
    /// assert exactly the semantics the solver relies on.

    #[test]
    fn finds_mate_in_one_queen_box() {
        // Engine-true mate-in-1: White Kf6, Qg6; Black Kh8. Qg6-g7 is checkmate
        // (king-supported queen, no escape, no ray-escape ambiguity).
        // FEN: 7k/8/5KQ1/8/8/8/8/8 w - - 0 1
        let (board, side) = from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1");
        assert_eq!(side, Color::White);
        let mv = find_forced_mate(&board, side, 1).expect("mate-in-1 should be found");
        assert_eq!(move_str(&mv), "g6g7", "mating move should be Qg7#");
        assert!(move_delivers_mate(&board, mv, side));
    }

    #[test]
    fn finds_mate_in_two_at_three_plies_but_not_one() {
        // Engine-true mate-in-2: White Kf6, Qe6; Black Kh8. Not a mate-in-1, but a
        // forced mate within 3 plies. The solver must find it at 3 plies and the
        // returned move must structurally force mate (verified by re-search).
        // FEN: 7k/8/4QK2/8/8/8/8/8 w - - 0 1
        let (board, side) = from_fen("7k/8/4QK2/8/8/8/8/8 w - - 0 1");
        assert_eq!(side, Color::White);
        assert!(
            find_forced_mate(&board, side, 1).is_none(),
            "position is not mate-in-1"
        );
        let mv = find_forced_mate(&board, side, 3).expect("mate-in-2 should be found at 3 plies");
        // The returned first move must be legal.
        assert!(
            legal_moves(&board, side)
                .iter()
                .any(|m| move_str(m) == move_str(&mv)),
            "returned move must be legal"
        );
        // After the solver's move, with Black to reply, Black must be lost within
        // the remaining 2 plies — i.e. every Black reply is still forced-mated.
        let after = apply_move(&board, mv, side);
        assert_eq!(after.result(), GameResult::Ongoing);
        let mut budget = NodeBudget {
            used: 0,
            cap: MAX_NODES,
        };
        assert!(
            opponent_is_lost(&after, Color::Black, 2, &mut budget),
            "after the solver's move, black must be lost within 2 plies"
        );
    }

    #[test]
    fn no_false_positive_on_stalemate_trap() {
        // Stalemate-in-1 trap: White Kf6, Qd5; Black Kh8 (white to move). The
        // tempting queen move Qd5-f7 STALEMATES the black king rather than mating
        // it, and no white move mates in one. A correct solver must NOT report a
        // mate-in-1 here (a stalemate is not a mate).
        // FEN: 7k/8/5K2/3Q4/8/8/8/8 w - - 0 1
        let (board, side) = from_fen("7k/8/5K2/3Q4/8/8/8/8 w - - 0 1");
        assert_eq!(side, Color::White);
        // Confirm the trap exists: Qf7 is engine-classified as Stalemate, not mate.
        let trap = legal_moves(&board, side)
            .into_iter()
            .find(|m| move_str(m) == "d5f7")
            .expect("Qd5-f7 should be legal");
        assert_eq!(
            apply_move(&board, trap, side).result(),
            GameResult::Stalemate,
            "Qd5-f7 must stalemate, not mate"
        );
        // The solver must not be fooled into reporting a mate-in-1.
        assert!(
            find_forced_mate(&board, side, 1).is_none(),
            "no mate-in-1: the tempting queen move only stalemates"
        );
    }

    #[test]
    fn disabled_returns_none_without_search() {
        // max_plies == 0 must return None even though an engine-true mate-in-1
        // exists in this position (the solver short-circuits before searching).
        let (board, side) = from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1");
        assert!(
            find_forced_mate(&board, side, 0).is_none(),
            "max_plies==0 disables the solver"
        );
    }

    #[test]
    fn node_cap_overflow_returns_none() {
        // A dense middlegame with a tiny node budget must bail out gracefully
        // instead of returning a false mate.
        let (board, side) =
            from_fen("r1bqkbnr/pppppppp/2n5/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 1");
        let mut budget = NodeBudget { used: 0, cap: 50 };
        assert!(
            find_mate_move(&board, side, 7, &mut budget).is_none(),
            "tiny node budget must abort with None, not a false mate"
        );
        assert!(
            budget.used >= budget.cap,
            "budget should have been exhausted"
        );
    }
}
