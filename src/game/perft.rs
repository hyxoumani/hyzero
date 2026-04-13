use std::sync::Arc;

use crate::game::{GameBoard, Move};
use crate::{BitIterator, CastleOption, Color, PieceType, PrecomputedItems, Square};

/// Count leaf nodes at `depth` from the given position.
///
/// Returns the number of nodes at the given depth, useful for validating
/// move generation correctness against known-correct perft values.
#[allow(clippy::only_used_in_recursion)]
pub fn perft(
    board: &GameBoard,
    color: Color,
    depth: u32,
    precomputed: &Arc<PrecomputedItems>,
) -> u64 {
    if depth == 0 {
        return 1;
    }

    let legal_moves = get_legal_moves_for_perft(board, color);

    if depth == 1 {
        return legal_moves.len() as u64;
    }

    let mut nodes = 0u64;
    let next_color = if color == Color::White {
        Color::Black
    } else {
        Color::White
    };
    // turn_count parity: 0 = white to move, 1 = black to move
    // compute_turn_items(count, mv) interprets count.is_multiple_of(2) as white just moved
    // so when white is moving we pass 0 (even), when black is moving we pass 1 (odd)
    let turn_count = if color == Color::White { 0 } else { 1 };

    for mv in legal_moves {
        let mut new_board = board.clone();
        new_board.compute_turn_items(turn_count, mv);
        nodes += perft(&new_board, next_color, depth - 1, precomputed);
    }

    nodes
}

/// Collect all legal moves for the given color, including all promotion types.
pub fn get_legal_moves_for_perft(board: &GameBoard, color: Color) -> Vec<Move> {
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
            if piece.piece_type == PieceType::Pawn {
                let to_rank = to_sq / 8;

                // Promotion
                if to_rank == 7 || to_rank == 0 {
                    for promo_type in [
                        PieceType::Queen,
                        PieceType::Rook,
                        PieceType::Bishop,
                        PieceType::Knight,
                    ] {
                        let mv = Move {
                            from: Square::from(sq as u8),
                            to: Square::from(to_sq as u8),
                            promotion_piece_type: Some(promo_type),
                            castle_option: None,
                            en_passant: false,
                        };
                        if board.validate_move(
                            mv,
                            color,
                            combined,
                            board.white_pieces,
                            board.black_pieces,
                        ) {
                            moves.push(mv);
                        }
                    }
                    continue;
                }

                // En passant: the pawn move mask already includes the EP square when applicable.
                // Detect it by a diagonal move to the EP target.
                if let Some(ep_target) = board.en_passant_target {
                    if to_sq == ep_target && sq % 8 != to_sq % 8 {
                        let mv = Move {
                            from: Square::from(sq as u8),
                            to: Square::from(to_sq as u8),
                            promotion_piece_type: None,
                            castle_option: None,
                            en_passant: true,
                        };
                        if board.validate_move(
                            mv,
                            color,
                            combined,
                            board.white_pieces,
                            board.black_pieces,
                        ) {
                            moves.push(mv);
                        }
                        continue;
                    }
                }
            }

            // Normal move (also covers 1-square king moves)
            let mv = Move {
                from: Square::from(sq as u8),
                to: Square::from(to_sq as u8),
                promotion_piece_type: None,
                castle_option: None,
                en_passant: false,
            };
            if board.validate_move(mv, color, combined, board.white_pieces, board.black_pieces) {
                moves.push(mv);
            }
        }

        // Castling: get_move_mask returns only 1-square king moves, so we add
        // castling candidates explicitly (same approach as selfplay/game_task.rs).
        if piece.piece_type == PieceType::King {
            let king_rank = if color == Color::White { 0u8 } else { 7u8 };
            for &castle_opt in &[CastleOption::Kingside, CastleOption::Queenside] {
                let to_file: u8 = match castle_opt {
                    CastleOption::Kingside => 6,
                    CastleOption::Queenside => 2,
                };
                let to_sq_castle = king_rank * 8 + to_file;
                let mv = Move {
                    from: Square::from(sq as u8),
                    to: Square::from(to_sq_castle),
                    promotion_piece_type: None,
                    castle_option: Some(castle_opt),
                    en_passant: false,
                };
                if board.validate_move(mv, color, combined, board.white_pieces, board.black_pieces)
                {
                    moves.push(mv);
                }
            }
        }
    }

    moves
}

/// Print per-move node counts for debugging perft mismatches.
#[allow(dead_code)]
fn perft_divide(board: &GameBoard, color: Color, depth: u32, precomputed: &Arc<PrecomputedItems>) {
    let legal_moves = get_legal_moves_for_perft(board, color);
    let next_color = if color == Color::White {
        Color::Black
    } else {
        Color::White
    };
    let turn_count = if color == Color::White { 0 } else { 1 };
    let mut total = 0u64;
    for mv in &legal_moves {
        let mut new_board = board.clone();
        new_board.compute_turn_items(turn_count, *mv);
        let count = if depth > 1 {
            perft(&new_board, next_color, depth - 1, precomputed)
        } else {
            1
        };
        total += count;
        println!(
            "{}{}: {}",
            square_name(mv.from as u8),
            square_name(mv.to as u8),
            count
        );
    }
    println!("Total: {}", total);
}

fn square_name(sq: u8) -> String {
    let file = (b'a' + sq % 8) as char;
    let rank = (b'1' + sq / 8) as char;
    format!("{}{}", file, rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::fen::board_from_fen;

    fn precomputed() -> Arc<PrecomputedItems> {
        Arc::new(PrecomputedItems::begin_precomputing())
    }

    fn perft_fen(fen: &str, depth: u32) -> u64 {
        let pc = precomputed();
        let (board, color, _) = board_from_fen(fen, pc.clone()).unwrap();
        perft(&board, color, depth, &pc)
    }

    // ---- Starting position ----

    #[test]
    fn test_perft_startpos_d1() {
        assert_eq!(
            perft_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                1
            ),
            20
        );
    }

    #[test]
    fn test_perft_startpos_d2() {
        assert_eq!(
            perft_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                2
            ),
            400
        );
    }

    #[test]
    fn test_perft_startpos_d3() {
        assert_eq!(
            perft_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                3
            ),
            8902
        );
    }

    #[test]
    #[ignore]
    fn test_perft_startpos_d4() {
        assert_eq!(
            perft_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                4
            ),
            197_281
        );
    }

    #[test]
    #[ignore]
    fn test_perft_startpos_d5() {
        assert_eq!(
            perft_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                5
            ),
            4_865_609
        );
    }

    // ---- Kiwipete ----

    #[test]
    fn test_perft_kiwipete_d1() {
        assert_eq!(
            perft_fen(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                1
            ),
            48
        );
    }

    #[test]
    fn test_perft_kiwipete_d2() {
        assert_eq!(
            perft_fen(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                2
            ),
            2039
        );
    }

    #[test]
    #[ignore]
    fn test_perft_kiwipete_d3() {
        assert_eq!(
            perft_fen(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                3
            ),
            97_862
        );
    }

    // ---- Position 3 ----

    #[test]
    fn test_perft_pos3_d1() {
        assert_eq!(
            perft_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1),
            14
        );
    }

    #[test]
    fn test_perft_pos3_d2() {
        assert_eq!(
            perft_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2),
            191
        );
    }

    #[test]
    fn test_perft_pos3_d3() {
        assert_eq!(
            perft_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3),
            2812
        );
    }

    // ---- Position 5 ----

    #[test]
    fn test_perft_pos5_d1() {
        assert_eq!(
            perft_fen(
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                1
            ),
            44
        );
    }

    #[test]
    fn test_perft_pos5_d2() {
        assert_eq!(
            perft_fen(
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                2
            ),
            1486
        );
    }
}
