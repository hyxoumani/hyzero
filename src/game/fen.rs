use std::collections::HashMap;
use std::sync::Arc;

use crate::game::board::{GameBoard, GameResult};
use crate::game::playerobj::Player;
use crate::game::Move;
use crate::{Color, Piece, PieceType, PrecomputedItems};

/// Parse a FEN string into a `GameBoard`, the active color, and the fullmove number.
///
/// Returns `Ok((board, side_to_move, fullmove_number))` on success, or an error
/// string describing the first parse failure.
pub fn board_from_fen(
    fen: &str,
    precomputed: Arc<PrecomputedItems>,
) -> Result<(GameBoard, Color, u32), String> {
    let fields: Vec<&str> = fen.split_whitespace().collect();
    if fields.len() < 6 {
        return Err(format!("FEN must have 6 fields, got {}", fields.len()));
    }

    // ---- Field 1: piece placement ----
    let mut board_arr = [None::<Piece>; 64];
    let mut white_pieces_bb: [u64; 6] = [0u64; 6];
    let mut black_pieces_bb: [u64; 6] = [0u64; 6];
    let mut white_own_board = [None::<Piece>; 64];
    let mut black_own_board = [None::<Piece>; 64];

    let ranks: Vec<&str> = fields[0].split('/').collect();
    if ranks.len() != 8 {
        return Err(format!(
            "Piece placement must have 8 ranks, got {}",
            ranks.len()
        ));
    }

    for (rank_idx, rank_str) in ranks.iter().enumerate() {
        // FEN rank 0 (leftmost in split) corresponds to board rank 7 (rank 8 in chess)
        let board_rank = 7 - rank_idx;
        let mut file: usize = 0;
        for ch in rank_str.chars() {
            if file > 8 {
                return Err(format!("Rank {} overflows: too many squares", rank_idx));
            }
            if let Some(skip) = ch.to_digit(10) {
                file += skip as usize;
            } else {
                let (piece_type, color) = char_to_piece(ch)?;
                let sq = board_rank * 8 + file;
                let piece = Piece { piece_type, color };
                board_arr[sq] = Some(piece);
                match color {
                    Color::White => {
                        white_pieces_bb[piece_type as usize] |= 1u64 << sq;
                        white_own_board[sq] = Some(piece);
                    }
                    Color::Black => {
                        black_pieces_bb[piece_type as usize] |= 1u64 << sq;
                        black_own_board[sq] = Some(piece);
                    }
                }
                file += 1;
            }
        }
        if file != 8 {
            return Err(format!(
                "Rank {} has {} squares, expected 8",
                rank_idx, file
            ));
        }
    }

    let white_combined: u64 = white_pieces_bb.iter().fold(0u64, |acc, &b| acc | b);
    let black_combined: u64 = black_pieces_bb.iter().fold(0u64, |acc, &b| acc | b);

    let player1 = Player {
        color: Color::White,
        pieces_bb: white_pieces_bb,
        pieces: white_combined,
        own_board: white_own_board,
    };
    let player2 = Player {
        color: Color::Black,
        pieces_bb: black_pieces_bb,
        pieces: black_combined,
        own_board: black_own_board,
    };

    // ---- Field 2: active color ----
    let side_to_move = match fields[1] {
        "w" => Color::White,
        "b" => Color::Black,
        other => return Err(format!("Unknown active color '{}'", other)),
    };

    // ---- Field 3: castling availability ----
    let castling_str = fields[2];
    let white_kingside = castling_str.contains('K');
    let white_queenside = castling_str.contains('Q');
    let black_kingside = castling_str.contains('k');
    let black_queenside = castling_str.contains('q');

    // ---- Field 4: en passant target ----
    let en_passant_target: Option<usize> = if fields[3] == "-" {
        None
    } else {
        let ep_chars: Vec<char> = fields[3].chars().collect();
        if ep_chars.len() < 2 {
            return Err(format!("Invalid en passant field '{}'", fields[3]));
        }
        let ep_file = ep_chars[0] as usize - 'a' as usize;
        let ep_rank = ep_chars[1] as usize - '1' as usize;
        Some(ep_rank * 8 + ep_file)
    };

    // ---- Field 5: halfmove clock ----
    let halfmove_clock: u32 = fields[4]
        .parse()
        .map_err(|_| format!("Invalid halfmove clock '{}'", fields[4]))?;

    // ---- Field 6: fullmove number ----
    let fullmove_number: u32 = fields[5]
        .parse()
        .map_err(|_| format!("Invalid fullmove number '{}'", fields[5]))?;

    // ---- Compute Zobrist hash ----
    let zt = &precomputed.zobrist;
    let mut zobrist_hash = 0u64;
    for (sq, slot) in board_arr.iter().enumerate() {
        if let Some(piece) = slot {
            zobrist_hash ^= zt.piece_sq[piece.color as usize][piece.piece_type as usize][sq];
        }
    }
    if white_kingside {
        zobrist_hash ^= zt.castling[0];
    }
    if white_queenside {
        zobrist_hash ^= zt.castling[1];
    }
    if black_kingside {
        zobrist_hash ^= zt.castling[2];
    }
    if black_queenside {
        zobrist_hash ^= zt.castling[3];
    }
    if let Some(ep_sq) = en_passant_target {
        zobrist_hash ^= zt.en_passant_file[ep_sq % 8];
    }
    if side_to_move == Color::Black {
        zobrist_hash ^= zt.side_to_move;
    }

    // ---- Assemble the board ----
    let combined_pieces = white_combined | black_combined;
    let mut board = GameBoard {
        player1,
        player2,
        board_arr,
        white_pieces: white_combined,
        precomputed_items: precomputed,
        black_pieces: black_combined,
        combined_pieces,
        in_check: false,
        white_pins: 0u64,
        black_pins: 0u64,
        game_result: GameResult::Ongoing,
        white_kingside,
        white_queenside,
        black_kingside,
        black_queenside,
        last_move: Move::default(),
        en_passant_target,
        halfmove_clock,
        position_history: HashMap::new(),
        zobrist_hash,
    };
    // Count the initial position as the first occurrence
    board.position_history.insert(board.zobrist_hash, 1);

    Ok((board, side_to_move, fullmove_number))
}

impl GameBoard {
    /// Emit the first four FEN fields (piece placement, active color, castling
    /// availability, en passant target) space-joined into a "normalized FEN".
    ///
    /// The halfmove clock and fullmove number are intentionally omitted so the
    /// key is clock-invariant: two positions that differ only in their move
    /// counters produce the same normfen. This matches the normfen emitted by
    /// `scripts/export_tb_wdl.py`, which reconstructs the ep field from the raw
    /// ep target square (python-chess `board.ep_square`) rather than the
    /// legality-filtered `board.fen()` field, so both sides agree on the key.
    ///
    /// `side_to_move` is supplied explicitly because the board does not store it.
    pub fn to_normfen(&self, side_to_move: Color) -> String {
        // ---- Field 1: piece placement (rank 8 down to rank 1) ----
        let mut placement = String::new();
        for board_rank in (0..8).rev() {
            let mut empty = 0u32;
            for file in 0..8 {
                let sq = board_rank * 8 + file;
                match self.board_arr[sq] {
                    Some(piece) => {
                        if empty > 0 {
                            placement.push_str(&empty.to_string());
                            empty = 0;
                        }
                        placement.push(piece_to_char(piece));
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                placement.push_str(&empty.to_string());
            }
            if board_rank > 0 {
                placement.push('/');
            }
        }

        // ---- Field 2: active color ----
        let color = if side_to_move == Color::White {
            "w"
        } else {
            "b"
        };

        // ---- Field 3: castling availability ----
        let mut castling = String::new();
        if self.white_kingside {
            castling.push('K');
        }
        if self.white_queenside {
            castling.push('Q');
        }
        if self.black_kingside {
            castling.push('k');
        }
        if self.black_queenside {
            castling.push('q');
        }
        if castling.is_empty() {
            castling.push('-');
        }

        // ---- Field 4: en passant target (raw target square) ----
        let ep = match self.en_passant_target {
            Some(sq) => {
                let file = (b'a' + (sq % 8) as u8) as char;
                let rank = (b'1' + (sq / 8) as u8) as char;
                format!("{file}{rank}")
            }
            None => "-".to_string(),
        };

        format!("{placement} {color} {castling} {ep}")
    }
}

/// Map a piece to its FEN character (uppercase White, lowercase Black).
fn piece_to_char(piece: Piece) -> char {
    let c = match piece.piece_type {
        PieceType::Pawn => 'p',
        PieceType::Knight => 'n',
        PieceType::Bishop => 'b',
        PieceType::Rook => 'r',
        PieceType::Queen => 'q',
        PieceType::King => 'k',
    };
    if piece.color == Color::White {
        c.to_ascii_uppercase()
    } else {
        c
    }
}

fn char_to_piece(ch: char) -> Result<(PieceType, Color), String> {
    let color = if ch.is_uppercase() {
        Color::White
    } else {
        Color::Black
    };
    let piece_type = match ch.to_ascii_lowercase() {
        'p' => PieceType::Pawn,
        'n' => PieceType::Knight,
        'b' => PieceType::Bishop,
        'r' => PieceType::Rook,
        'q' => PieceType::Queen,
        'k' => PieceType::King,
        other => return Err(format!("Unknown piece character '{}'", other)),
    };
    Ok((piece_type, color))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn precomputed() -> Arc<PrecomputedItems> {
        Arc::new(PrecomputedItems::begin_precomputing())
    }

    #[test]
    fn test_fen_starting_position() {
        let (board, color, fullmove) = board_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            precomputed(),
        )
        .unwrap();
        assert_eq!(color, Color::White);
        assert_eq!(fullmove, 1);
        assert!(board.white_kingside);
        assert!(board.white_queenside);
        assert!(board.black_kingside);
        assert!(board.black_queenside);
        assert_eq!(board.en_passant_target, None);
        assert_eq!(board.halfmove_clock, 0);
        // Verify white pawns on rank 2
        assert_eq!(
            board.player1.pieces_bb[PieceType::Pawn as usize],
            0x000000000000FF00
        );
        // Verify black pawns on rank 7
        assert_eq!(
            board.player2.pieces_bb[PieceType::Pawn as usize],
            0x00FF000000000000
        );
    }

    #[test]
    fn test_fen_midgame() {
        // Italian Game position
        let (board, color, _) = board_from_fen(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            precomputed(),
        )
        .unwrap();
        assert_eq!(color, Color::White);
        assert_eq!(board.halfmove_clock, 4);
        // White bishop on c4 = square 26 (rank 3, file 2)
        assert!(board.board_arr[26].is_some());
        assert_eq!(board.board_arr[26].unwrap().piece_type, PieceType::Bishop);
        assert_eq!(board.board_arr[26].unwrap().color, Color::White);
        // White knight on f3 = square 21 (rank 2, file 5)
        assert!(board.board_arr[21].is_some());
        assert_eq!(board.board_arr[21].unwrap().piece_type, PieceType::Knight);
    }

    #[test]
    fn test_fen_castling_partial() {
        let (board, _, _) =
            board_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1", precomputed()).unwrap();
        assert!(board.white_kingside);
        assert!(!board.white_queenside);
        assert!(!board.black_kingside);
        assert!(board.black_queenside);
    }

    #[test]
    fn test_fen_en_passant() {
        let (board, _, _) = board_from_fen(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            precomputed(),
        )
        .unwrap();
        // d6 = rank 5, file 3 = 5*8+3 = 43
        assert_eq!(board.en_passant_target, Some(43));
    }

    #[test]
    fn test_fen_black_to_move() {
        let (_, color, _) = board_from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            precomputed(),
        )
        .unwrap();
        assert_eq!(color, Color::Black);
    }

    /// `to_normfen` reproduces exactly the first four fields of the source FEN.
    #[test]
    fn normfen_reproduces_first_four_fen_fields() {
        let fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
        let (board, color, _) = board_from_fen(fen, precomputed()).unwrap();
        let expected: String = fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ");
        assert_eq!(board.to_normfen(color), expected);
    }

    /// The halfmove and fullmove clock fields are ignored: two positions that
    /// differ only in their move counters produce the SAME normfen.
    #[test]
    fn normfen_ignores_clock_fields() {
        let base = "4k3/8/4K3/8/8/8/4R3/8 w - -";
        let (b1, c1, _) = board_from_fen(&format!("{base} 0 1"), precomputed()).unwrap();
        let (b2, c2, _) = board_from_fen(&format!("{base} 37 99"), precomputed()).unwrap();
        assert_eq!(b1.to_normfen(c1), b2.to_normfen(c2));
        assert_eq!(b1.to_normfen(c1), base);
    }

    /// The en passant target survives the round-trip as its raw square.
    #[test]
    fn normfen_preserves_en_passant_square() {
        let fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
        let (board, color, _) = board_from_fen(fen, precomputed()).unwrap();
        assert_eq!(
            board.to_normfen(color),
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6"
        );
    }
}
