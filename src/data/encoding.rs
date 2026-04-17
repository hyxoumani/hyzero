use super::types::{ActionIndex, BoardObservation, BoardSnapshot, NUM_ACTIONS, NUM_BASE_ACTIONS};
use crate::game::{GameBoard, Move};
use crate::{BitIterator, CastleOption, Color, Piece, PieceType, Square};

/// Encode a GameBoard into a BoardObservation for the representation network.
///
/// Produces 103 planes: 8 positions × 12 piece planes + 7 game-state planes.
///
/// # Arguments
/// * `board`        — current board state
/// * `side_to_move` — whose turn it is
/// * `history`      — slice of up to 7 past `BoardSnapshot`s, oldest first.
///   If fewer than 7 are provided, missing positions are all-zeros.
pub fn encode_board(
    board: &GameBoard,
    side_to_move: Color,
    history: &[BoardSnapshot],
) -> BoardObservation {
    let mut obs = BoardObservation::default();

    let is_black = side_to_move == Color::Black;

    // Current-player perspective (AlphaZero/MuZero convention):
    // Planes 0-5:  Current player's pieces
    // Planes 6-11: Opponent's pieces
    // When Black to move, rank-mirror all square indices and swap which player's
    // bitboards go into each plane group.
    let (my_player, opp_player) = if is_black {
        (&board.player2, &board.player1)
    } else {
        (&board.player1, &board.player2)
    };

    // Planes 0-5: current player's pieces
    for pt in 0..6 {
        let bb = my_player.pieces_bb[pt];
        for sq in BitIterator::new(bb) {
            let esq = if is_black { flip_sq(sq) } else { sq };
            obs.planes[pt * 64 + esq] = 1.0;
        }
    }
    // Planes 6-11: opponent's pieces
    for pt in 0..6 {
        let bb = opp_player.pieces_bb[pt];
        for sq in BitIterator::new(bb) {
            let esq = if is_black { flip_sq(sq) } else { sq };
            obs.planes[(pt + 6) * 64 + esq] = 1.0;
        }
    }

    // Planes 12-95: Past positions (up to 7), each encoded as 12 piece planes.
    // history[0] is the oldest position provided; we place it at the earliest available slot.
    // If history has N entries (N <= 7), they fill planes 12..12+N*12; remainder is zeros.
    // Same perspective flip applied: current player's pieces in planes 0-5 of each slot.
    for (i, snap) in history.iter().enumerate() {
        let plane_base = (1 + i) * 12; // position slot 1..=7
        let (my_bb, opp_bb) = if is_black {
            (&snap.black_pieces_bb, &snap.white_pieces_bb)
        } else {
            (&snap.white_pieces_bb, &snap.black_pieces_bb)
        };
        // Current player's pieces (6 planes)
        for (pt, &bb) in my_bb.iter().enumerate() {
            for sq in BitIterator::new(bb) {
                let esq = if is_black { flip_sq(sq) } else { sq };
                obs.planes[(plane_base + pt) * 64 + esq] = 1.0;
            }
        }
        // Opponent's pieces (6 planes)
        for (pt, &bb) in opp_bb.iter().enumerate() {
            for sq in BitIterator::new(bb) {
                let esq = if is_black { flip_sq(sq) } else { sq };
                obs.planes[(plane_base + 6 + pt) * 64 + esq] = 1.0;
            }
        }
    }

    // Planes 96-99: Castling rights — current player's castling in 96-97, opponent's in 98-99.
    let (p1_ks, p1_qs, p2_ks, p2_qs) = if is_black {
        (
            board.black_kingside,
            board.black_queenside,
            board.white_kingside,
            board.white_queenside,
        )
    } else {
        (
            board.white_kingside,
            board.white_queenside,
            board.black_kingside,
            board.black_queenside,
        )
    };
    let castling = [p1_ks, p1_qs, p2_ks, p2_qs];
    for (i, &has_right) in castling.iter().enumerate() {
        if has_right {
            let plane_offset = (96 + i) * 64;
            for sq in 0..64 {
                obs.planes[plane_offset + sq] = 1.0;
            }
        }
    }

    // Plane 100: En passant target (one-hot, current position only).
    // Flip the EP square when Black to keep it in current-player coordinate space.
    if let Some(ep_sq) = board.en_passant_target {
        let encoded_ep = if is_black { flip_sq(ep_sq) } else { ep_sq };
        obs.planes[100 * 64 + encoded_ep] = 1.0;
    }

    // Plane 101: Side to move (all 1.0 if white, all 0.0 if black).
    // Tells the network which color the current player is.
    if side_to_move == Color::White {
        let plane_offset = 101 * 64;
        for sq in 0..64 {
            obs.planes[plane_offset + sq] = 1.0;
        }
    }

    // Plane 102: Halfmove clock (normalized by 100)
    let clock_val = board.halfmove_clock as f32 / 100.0;
    let plane_offset = 102 * 64;
    for sq in 0..64 {
        obs.planes[plane_offset + sq] = clock_val;
    }

    obs
}

/// Snapshot the current board into a lightweight `BoardSnapshot` for the history buffer.
pub fn board_to_snapshot(board: &GameBoard) -> BoardSnapshot {
    BoardSnapshot {
        white_pieces_bb: board.player1.pieces_bb,
        black_pieces_bb: board.player2.pieces_bb,
    }
}

/// Underpromotion encoding (actions 4096..4671):
///
/// For non-queen promotions, the action index is:
///   action = NUM_BASE_ACTIONS + piece_idx * 192 + from_file * 24 + to_file_slot
///
/// where:
///   piece_idx:    0 = Knight, 1 = Bishop, 2 = Rook
///   from_file:    0-7 (file of the promoting pawn)
///   to_file_slot: encodes the destination file relative to from_file:
///                 0-7:   to_file = 0-7 (straight-ahead, only slot 0 == from_file is legal)
///                 8-15:  reserved (not used, network learns to suppress)
///                 16-23: reserved (not used, network learns to suppress)
///
/// In practice only 3 destinations per from_file are legal:
///   from_file (straight), from_file-1 (left capture), from_file+1 (right capture).
/// We encode to_file directly (0-7) in to_file_slot so round-tripping is clean.
/// The 576 slots include many illegal (from_file, to_file) combinations — the
/// network learns to suppress them, just as it does for illegal base moves.
fn encode_underpromo_action(piece_type: PieceType, from_sq: u8, to_sq: u8) -> ActionIndex {
    let piece_idx: usize = match piece_type {
        PieceType::Knight => 0,
        PieceType::Bishop => 1,
        PieceType::Rook => 2,
        _ => panic!("encode_underpromo_action: unexpected piece type"),
    };
    let from_file = (from_sq % 8) as usize;
    let to_file = (to_sq % 8) as usize;
    (NUM_BASE_ACTIONS + piece_idx * 192 + from_file * 24 + to_file) as ActionIndex
}

/// Decode an underpromotion action (>= NUM_BASE_ACTIONS) into its components.
/// Returns (piece_type, from_file, to_file).
fn decode_underpromo_action(action: ActionIndex) -> (PieceType, u8, u8) {
    let offset = action as usize - NUM_BASE_ACTIONS;
    let piece_idx = offset / 192;
    let remainder = offset % 192;
    let from_file = (remainder / 24) as u8;
    let to_file = (remainder % 24) as u8;

    let piece_type = match piece_idx {
        0 => PieceType::Knight,
        1 => PieceType::Bishop,
        2 => PieceType::Rook,
        _ => panic!("decode_underpromo_action: invalid piece_idx {piece_idx}"),
    };
    (piece_type, from_file, to_file)
}

/// Encode a Move as an ActionIndex.
///
/// For queen promotions and non-promotion moves: `from_sq * 64 + to_sq` (base range 0..4095).
/// For knight/bishop/rook promotions: underpromotion encoding in range 4096..4671.
pub fn move_to_action(mv: &Move) -> ActionIndex {
    match mv.promotion_piece_type {
        Some(PieceType::Knight) | Some(PieceType::Bishop) | Some(PieceType::Rook) => {
            encode_underpromo_action(mv.promotion_piece_type.unwrap(), mv.from as u8, mv.to as u8)
        }
        _ => (mv.from as u16) * 64 + (mv.to as u16),
    }
}

/// Decode an ActionIndex back to a Move, using board context to detect castling and en passant.
///
/// For actions in the underpromotion range (>= NUM_BASE_ACTIONS), reconstructs the move
/// from the encoded from_file and to_file, inferring the correct rank from the color.
pub fn action_to_move(action: ActionIndex, board: &GameBoard, color: Color) -> Move {
    // Handle underpromotion range
    if action as usize >= NUM_BASE_ACTIONS {
        let (piece_type, from_file, to_file) = decode_underpromo_action(action);
        // Determine from_rank and to_rank based on color
        let (from_rank, to_rank): (u8, u8) = if color == Color::White {
            (6, 7) // White pawn promotes from rank 7 (sq 48-55) to rank 8 (sq 56-63)
        } else {
            (1, 0) // Black pawn promotes from rank 2 (sq 8-15) to rank 1 (sq 0-7)
        };
        let from_sq = from_rank * 8 + from_file;
        let to_sq = to_rank * 8 + to_file;
        return Move {
            from: Square::from(from_sq),
            to: Square::from(to_sq),
            promotion_piece_type: Some(piece_type),
            castle_option: None,
            en_passant: false,
        };
    }

    let from_sq = (action / 64) as u8;
    let to_sq = (action % 64) as u8;
    let from_file = (from_sq % 8) as i8;
    let to_file = (to_sq % 8) as i8;
    let to_rank = to_sq / 8;

    let piece: Option<Piece> = board.board_arr[from_sq as usize];

    // Castling detection: King moves two files
    let castle_option = if piece.map(|p| p.piece_type) == Some(PieceType::King)
        && (to_file - from_file).abs() == 2
    {
        if to_file > from_file {
            Some(CastleOption::Kingside)
        } else {
            Some(CastleOption::Queenside)
        }
    } else {
        None
    };

    // En passant detection: Pawn moves diagonally to the ep target square
    let en_passant = castle_option.is_none()
        && piece.map(|p| p.piece_type) == Some(PieceType::Pawn)
        && (to_file - from_file).abs() == 1
        && board.en_passant_target == Some(to_sq as usize);

    // Promotion detection: Pawn reaches back rank (only when not castling or ep)
    let promotion_piece_type = if castle_option.is_none()
        && !en_passant
        && piece.map(|p| p.piece_type) == Some(PieceType::Pawn)
        && (to_rank == 7 || to_rank == 0)
    {
        Some(PieceType::Queen)
    } else {
        None
    };

    Move {
        from: Square::from(from_sq),
        to: Square::from(to_sq),
        promotion_piece_type,
        castle_option,
        en_passant,
    }
}

/// Encode an action as 3 spatial planes for the dynamics network.
/// Plane 0: source square one-hot (8x8)
/// Plane 1: destination square one-hot (8x8)
/// Plane 2: promotion flag (all 1s if promotion, all 0s otherwise)
///
/// For underpromotion actions (>= NUM_BASE_ACTIONS), the from/to squares are derived
/// from the encoded from_file and to_file. Since these always involve a promotion,
/// the promotion flag plane is always set for underpromotion actions.
pub fn encode_action_spatial(action: ActionIndex) -> [f32; 3 * 64] {
    let mut planes = [0.0f32; 3 * 64];

    let (from_sq, to_sq, is_promo) = if action as usize >= NUM_BASE_ACTIONS {
        // Underpromotion: decode file indices; use rank 6→7 (white perspective) for spatial encoding
        let (_piece_type, from_file, to_file) = decode_underpromo_action(action);
        let from_sq = 6 * 8 + from_file as usize; // rank 7 (0-indexed 6)
        let to_sq = 7 * 8 + to_file as usize; // rank 8 (0-indexed 7)
        (from_sq, to_sq, true)
    } else {
        let from_sq = (action / 64) as usize;
        let to_sq = (action % 64) as usize;
        let to_rank = to_sq / 8;
        let is_promo = to_rank == 7 || to_rank == 0;
        (from_sq, to_sq, is_promo)
    };

    // Plane 0: source square
    planes[from_sq] = 1.0;
    // Plane 1: destination square
    planes[64 + to_sq] = 1.0;
    // Plane 2: promotion flag
    if is_promo {
        for sq in 0..64 {
            planes[128 + sq] = 1.0;
        }
    }

    planes
}

/// Total action space size.
pub fn num_actions() -> usize {
    NUM_ACTIONS
}

// ─── Color-augmentation helpers ───────────────────────────────────────────────

/// Flip a square index rank-wise: rank 0↔7, 1↔6, etc.
///
/// `sq = rank * 8 + file`; the flipped square is `(7 - rank) * 8 + file`.
pub(crate) fn flip_sq(sq: usize) -> usize {
    (7 - sq / 8) * 8 + (sq % 8)
}

/// Flip a base action index (`from_sq * 64 + to_sq`) under rank mirror.
pub(crate) fn flip_base_action(a: usize) -> usize {
    flip_sq(a / 64) * 64 + flip_sq(a % 64)
}

/// Flip any action index (handles both base 0..4095 and underpromo 4096..4671).
///
/// Underpromotion indices are invariant — they encode files, not ranks, and the
/// rank is inferred from color at decode time; so the same index is used for
/// both White and Black underpromotions.
pub(crate) fn flip_action(action: usize) -> usize {
    if action < NUM_BASE_ACTIONS {
        flip_base_action(action)
    } else {
        action
    }
}

/// Rank-mirror a single 64-element plane block: read from `src`, write to `dst`.
fn rank_mirror_plane(src: &[f32], dst: &mut [f32]) {
    for sq in 0..64 {
        dst[(7 - sq / 8) * 8 + (sq % 8)] = src[sq];
    }
}

/// Flip all 103 observation planes for color augmentation.
///
/// Produces a new 6592-element Vec representing the board from the opponent's
/// perspective:
/// - For each of the 8 history groups (12 planes each): swap White (0-5) ↔ Black
///   (6-11) within the group, then rank-mirror every 64-element plane.
/// - Castling planes: 96↔98 and 97↔99 (constant-fill, no rank mirror needed).
/// - En-passant plane 100: rank-mirrored.
/// - Side-to-move plane 101: `1.0 - value` for every square.
/// - Halfmove clock plane 102: unchanged.
pub(crate) fn flip_obs_planes(obs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(
        obs.len(),
        103 * 64,
        "flip_obs_planes: expected 103*64 elements"
    );
    let mut out = vec![0.0f32; 103 * 64];

    // 8 history groups × 12 planes each (current + 7 past)
    for group in 0..8usize {
        let base = group * 12;
        // Within each group: swap White piece planes (0-5) ↔ Black piece planes (6-11),
        // and rank-mirror each plane.
        for pt in 0..6usize {
            // White side goes to Black slot, rank-mirrored
            let src_w = &obs[(base + pt) * 64..(base + pt) * 64 + 64];
            let dst_b = &mut out[(base + 6 + pt) * 64..(base + 6 + pt) * 64 + 64];
            rank_mirror_plane(src_w, dst_b);

            // Black side goes to White slot, rank-mirrored
            let src_b_start = (base + 6 + pt) * 64;
            let dst_w_start = (base + pt) * 64;
            // Avoid overlapping borrows by using index-based copy
            for sq in 0..64 {
                out[dst_w_start + (7 - sq / 8) * 8 + (sq % 8)] = obs[src_b_start + sq];
            }
        }
    }

    // Castling planes: constant-fill, swap in pairs (no rank mirror needed)
    // 96 (W kingside) ↔ 98 (B kingside)
    // 97 (W queenside) ↔ 99 (B queenside)
    out[96 * 64..97 * 64].copy_from_slice(&obs[98 * 64..99 * 64]);
    out[97 * 64..98 * 64].copy_from_slice(&obs[99 * 64..100 * 64]);
    out[98 * 64..99 * 64].copy_from_slice(&obs[96 * 64..97 * 64]);
    out[99 * 64..100 * 64].copy_from_slice(&obs[97 * 64..98 * 64]);

    // Plane 100: en passant target — rank-mirror the one-hot square
    let ep_src = &obs[100 * 64..101 * 64];
    let ep_dst = &mut out[100 * 64..101 * 64];
    rank_mirror_plane(ep_src, ep_dst);

    // Plane 101: side-to-move — flip constant fill (all-1.0 → all-0.0, and vice versa)
    for i in 0..64 {
        out[101 * 64 + i] = 1.0 - obs[101 * 64 + i];
    }

    // Plane 102: halfmove clock — unchanged
    out[102 * 64..103 * 64].copy_from_slice(&obs[102 * 64..103 * 64]);

    out
}

/// Flip the 3-plane (192-element) spatial action encoding under rank mirror.
///
/// Plane 0 (source) and Plane 1 (dest): rank-mirror the one-hot square.
/// Plane 2 (promotion flag): constant-fill, unchanged.
pub(crate) fn flip_action_planes(planes: &[f32; 192]) -> [f32; 192] {
    let mut out = [0.0f32; 192];
    for sq in 0..64 {
        let fsq = flip_sq(sq);
        out[fsq] = planes[sq]; // plane 0: source
        out[64 + fsq] = planes[64 + sq]; // plane 1: dest
    }
    // Plane 2: promotion flag — copy as-is (constant fill)
    out[128..192].copy_from_slice(&planes[128..192]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{NUM_OBS_PLANES, NUM_UNDERPROMO_ACTIONS};
    use crate::game::{GameBoard, Move, Player};
    use crate::{Color, Piece, PieceType, PrecomputedItems, Square};
    use std::sync::Arc;

    fn make_board() -> GameBoard {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let p1 = Player::init_player(true);
        let p2 = Player::init_player(false);
        GameBoard::init_game_board(precomputed, p1, p2)
    }

    #[test]
    fn test_action_to_move_normal() {
        // e2e4: from=12 (E2), to=28 (E4), action = 12*64+28 = 796
        let board = make_board();
        let mv = action_to_move(796, &board, Color::White);
        assert_eq!(mv.from, Square::E2);
        assert_eq!(mv.to, Square::E4);
        assert!(mv.castle_option.is_none());
        assert!(!mv.en_passant);
        assert!(mv.promotion_piece_type.is_none());
    }

    #[test]
    fn test_action_to_move_castling_kingside() {
        // King e1->g1: from=4 (E1), to=6 (G1), action = 4*64+6 = 262
        let mut board = make_board();
        // Clear f1 (5) and g1 (6) for the castling path
        board.board_arr[5] = None;
        board.board_arr[6] = None;
        let mv = action_to_move(262, &board, Color::White);
        assert_eq!(mv.castle_option, Some(crate::CastleOption::Kingside));
        assert!(!mv.en_passant);
        assert!(mv.promotion_piece_type.is_none());
    }

    #[test]
    fn test_action_to_move_castling_queenside() {
        // King e1->c1: from=4 (E1), to=2 (C1), action = 4*64+2 = 258
        let mut board = make_board();
        // Clear b1 (1), c1 (2), d1 (3)
        board.board_arr[1] = None;
        board.board_arr[2] = None;
        board.board_arr[3] = None;
        let mv = action_to_move(258, &board, Color::White);
        assert_eq!(mv.castle_option, Some(crate::CastleOption::Queenside));
        assert!(!mv.en_passant);
        assert!(mv.promotion_piece_type.is_none());
    }

    #[test]
    fn test_action_to_move_en_passant() {
        // White pawn on e5 (36) captures to d6 (43), ep target = 43
        let mut board = make_board();
        board.board_arr[36] = Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::White,
        });
        board.en_passant_target = Some(43);
        let action = 36 * 64 + 43; // 2347
        let mv = action_to_move(action, &board, Color::White);
        assert!(mv.en_passant);
        assert!(mv.castle_option.is_none());
        assert!(mv.promotion_piece_type.is_none());
    }

    #[test]
    fn test_action_to_move_promotion() {
        // White pawn on e7 (52) moves to e8 (60), action = 52*64+60 = 3388
        let mut board = make_board();
        board.board_arr[52] = Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::White,
        });
        let mv = action_to_move(52 * 64 + 60, &board, Color::White);
        assert_eq!(mv.promotion_piece_type, Some(PieceType::Queen));
        assert!(mv.castle_option.is_none());
        assert!(!mv.en_passant);
    }

    #[test]
    fn test_action_to_move_no_ep_when_square_differs() {
        // White pawn on e5 (36) moves straight to e6 (44), ep target = 43 (d6) — no ep
        let mut board = make_board();
        board.board_arr[36] = Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::White,
        });
        board.en_passant_target = Some(43);
        let mv = action_to_move(36 * 64 + 44, &board, Color::White);
        assert!(!mv.en_passant);
    }

    #[test]
    fn test_move_to_action_knight_promotion() {
        // White pawn e7 (sq 52) → e8 (sq 60) with Knight promotion
        let mv = Move {
            from: Square::E7,
            to: Square::E8,
            promotion_piece_type: Some(PieceType::Knight),
            castle_option: None,
            en_passant: false,
        };
        let action = move_to_action(&mv);
        assert!(
            action as usize >= NUM_BASE_ACTIONS
                && (action as usize) < NUM_BASE_ACTIONS + NUM_UNDERPROMO_ACTIONS,
            "knight promotion action {action} out of underpromo range"
        );
    }

    #[test]
    fn test_move_to_action_bishop_promotion() {
        // White pawn e7 → e8 with Bishop promotion
        let mv = Move {
            from: Square::E7,
            to: Square::E8,
            promotion_piece_type: Some(PieceType::Bishop),
            castle_option: None,
            en_passant: false,
        };
        let action = move_to_action(&mv);
        assert!(
            action as usize >= NUM_BASE_ACTIONS
                && (action as usize) < NUM_BASE_ACTIONS + NUM_UNDERPROMO_ACTIONS,
            "bishop promotion action {action} out of underpromo range"
        );
    }

    #[test]
    fn test_move_to_action_rook_promotion() {
        // White pawn e7 → e8 with Rook promotion
        let mv = Move {
            from: Square::E7,
            to: Square::E8,
            promotion_piece_type: Some(PieceType::Rook),
            castle_option: None,
            en_passant: false,
        };
        let action = move_to_action(&mv);
        assert!(
            action as usize >= NUM_BASE_ACTIONS
                && (action as usize) < NUM_BASE_ACTIONS + NUM_UNDERPROMO_ACTIONS,
            "rook promotion action {action} out of underpromo range"
        );
    }

    #[test]
    fn test_move_to_action_queen_promotion_stays_in_base_range() {
        // Queen promotion must stay in base (from*64+to) range
        let mv = Move {
            from: Square::E7,
            to: Square::E8,
            promotion_piece_type: Some(PieceType::Queen),
            castle_option: None,
            en_passant: false,
        };
        let action = move_to_action(&mv);
        assert!(
            (action as usize) < NUM_BASE_ACTIONS,
            "queen promotion action {action} should be < NUM_BASE_ACTIONS"
        );
    }

    #[test]
    fn test_action_to_move_knight_underpromo_roundtrip_white() {
        // Encode then decode: white pawn e7 → e8 knight underpromotion
        let original = Move {
            from: Square::E7,
            to: Square::E8,
            promotion_piece_type: Some(PieceType::Knight),
            castle_option: None,
            en_passant: false,
        };
        let action = move_to_action(&original);
        let board = make_board();
        let decoded = action_to_move(action, &board, Color::White);

        assert_eq!(decoded.promotion_piece_type, Some(PieceType::Knight));
        assert_eq!(decoded.from, Square::E7);
        assert_eq!(decoded.to, Square::E8);
        assert!(decoded.castle_option.is_none());
        assert!(!decoded.en_passant);
    }

    #[test]
    fn test_action_to_move_rook_underpromo_roundtrip_white() {
        // Encode then decode: white pawn e7 → e8 rook underpromotion
        let original = Move {
            from: Square::E7,
            to: Square::E8,
            promotion_piece_type: Some(PieceType::Rook),
            castle_option: None,
            en_passant: false,
        };
        let action = move_to_action(&original);
        let board = make_board();
        let decoded = action_to_move(action, &board, Color::White);

        assert_eq!(decoded.promotion_piece_type, Some(PieceType::Rook));
        assert_eq!(decoded.from, Square::E7);
        assert_eq!(decoded.to, Square::E8);
    }

    #[test]
    fn test_action_to_move_black_underpromo_knight() {
        // Black pawn h2 (sq 15) → h1 (sq 7) with knight underpromotion
        // h2 = rank 2 (0-indexed 1), file h (7), sq = 1*8+7 = 15
        // h1 = rank 1 (0-indexed 0), file h (7), sq = 0*8+7 = 7
        let original = Move {
            from: Square::H2,
            to: Square::H1,
            promotion_piece_type: Some(PieceType::Knight),
            castle_option: None,
            en_passant: false,
        };
        let action = move_to_action(&original);
        let board = make_board();
        let decoded = action_to_move(action, &board, Color::Black);

        assert_eq!(decoded.promotion_piece_type, Some(PieceType::Knight));
        assert_eq!(decoded.from, Square::H2);
        assert_eq!(decoded.to, Square::H1);
    }

    #[test]
    fn test_different_underpromo_pieces_have_distinct_actions() {
        // Knight, Bishop, Rook promotions for same from/to must have distinct actions
        let mk_mv = |pt: PieceType| Move {
            from: Square::D7,
            to: Square::D8,
            promotion_piece_type: Some(pt),
            castle_option: None,
            en_passant: false,
        };
        let knight_action = move_to_action(&mk_mv(PieceType::Knight));
        let bishop_action = move_to_action(&mk_mv(PieceType::Bishop));
        let rook_action = move_to_action(&mk_mv(PieceType::Rook));

        assert_ne!(
            knight_action, bishop_action,
            "Knight and Bishop promotions must differ"
        );
        assert_ne!(
            bishop_action, rook_action,
            "Bishop and Rook promotions must differ"
        );
        assert_ne!(
            knight_action, rook_action,
            "Knight and Rook promotions must differ"
        );
    }

    #[test]
    fn test_encode_board_plane_count() {
        let board = make_board();
        let obs = encode_board(&board, Color::White, &[]);
        assert_eq!(
            obs.planes.len(),
            NUM_OBS_PLANES * 64,
            "encode_board output length should be NUM_OBS_PLANES * 64 = {}",
            NUM_OBS_PLANES * 64
        );
    }

    #[test]
    fn test_encode_board_no_history_planes_are_zero() {
        let board = make_board();
        let obs = encode_board(&board, Color::White, &[]);
        // Planes 12-95 (past positions) should all be zero when no history provided
        for plane in 12..96 {
            for sq in 0..64 {
                assert_eq!(
                    obs.planes[plane * 64 + sq],
                    0.0,
                    "plane {plane} sq {sq} should be 0 with empty history"
                );
            }
        }
    }

    #[test]
    fn test_encode_board_with_history_plane_matches_snapshot() {
        let board = make_board();
        let snap = board_to_snapshot(&board);
        // Provide one snapshot (past position 1 → planes 12-23)
        let obs = encode_board(&board, Color::White, std::slice::from_ref(&snap));

        // Planes 12-17 should match white pieces of the snapshot
        for pt in 0..6 {
            let bb = snap.white_pieces_bb[pt];
            for sq in 0..64usize {
                let expected = if (bb >> sq) & 1 == 1 { 1.0f32 } else { 0.0f32 };
                let plane = 12 + pt;
                assert_eq!(
                    obs.planes[plane * 64 + sq],
                    expected,
                    "white piece plane {plane} sq {sq} mismatch"
                );
            }
        }
    }

    // ── Color-augmentation helper tests ──────────────────────────────────────

    #[test]
    fn test_flip_sq_known_squares() {
        // A1 (sq 0, rank 0, file 0) → A8 (sq 56, rank 7, file 0)
        assert_eq!(flip_sq(0), 56, "A1 should flip to A8");
        // H1 (sq 7, rank 0, file 7) → H8 (sq 63, rank 7, file 7)
        assert_eq!(flip_sq(7), 63, "H1 should flip to H8");
        // H8 (sq 63) → H1 (sq 7)
        assert_eq!(flip_sq(63), 7, "H8 should flip to H1");
        // flip is its own inverse for all squares
        for sq in 0..64 {
            assert_eq!(
                flip_sq(flip_sq(sq)),
                sq,
                "flip_sq round-trip failed at sq {sq}"
            );
        }
    }

    #[test]
    fn test_flip_action_base_e2e4() {
        // e2e4: from=E2=sq12 (rank1,fileE=4), to=E4=sq28 (rank3,fileE=4)
        // action = 12*64+28 = 796
        // After flip: from_flip = flip_sq(12) = (7-1)*8+4 = 52 (E7),
        //             to_flip   = flip_sq(28) = (7-3)*8+4 = 36 (E5)
        // flipped action = 52*64+36 = 3364
        let action = 12 * 64 + 28;
        let flipped = flip_action(action);
        assert_eq!(flipped, flip_sq(12) * 64 + flip_sq(28));
        assert_eq!(flip_sq(12), 52);
        assert_eq!(flip_sq(28), 36);
        assert_eq!(flipped, 52 * 64 + 36);
    }

    #[test]
    fn test_flip_action_underpromo_invariant() {
        // Underpromotion actions (>= 4096) must be unchanged
        assert_eq!(flip_action(4096), 4096);
        assert_eq!(flip_action(4671), 4671);
    }

    #[test]
    fn test_flip_action_base_is_own_inverse() {
        // flip_action is its own inverse for all base actions
        for a in 0..NUM_BASE_ACTIONS {
            assert_eq!(
                flip_action(flip_action(a)),
                a,
                "flip_action round-trip failed at action {a}"
            );
        }
    }

    #[test]
    fn test_flip_obs_planes_round_trip() {
        // Build a non-trivial observation: put a white pawn bit at A2 (plane 0, sq 8)
        // and set side-to-move to White (plane 101 = all-1.0).
        let mut obs = vec![0.0f32; 103 * 64];
        obs[0 * 64 + 8] = 1.0; // White pawn at A2
        for sq in 0..64 {
            obs[101 * 64 + sq] = 1.0; // White to move
        }

        let flipped = flip_obs_planes(&obs);
        let round_trip = flip_obs_planes(&flipped);

        for i in 0..103 * 64 {
            assert!(
                (round_trip[i] - obs[i]).abs() < 1e-6,
                "round-trip mismatch at index {i}: expected {}, got {}",
                obs[i],
                round_trip[i]
            );
        }
    }

    #[test]
    fn test_flip_obs_planes_piece_swap() {
        // White pawn at A2: plane 0 (White Pawn), sq 8 (rank 1, file 0)
        // After color flip:
        //   - The piece becomes a Black pawn in plane 6 (Black Pawn)
        //   - A2 rank-mirrors to A7: sq = (7-1)*8+0 = 48
        let mut obs = vec![0.0f32; 103 * 64];
        obs[0 * 64 + 8] = 1.0; // White pawn at A2 (plane 0, sq 8)

        let flipped = flip_obs_planes(&obs);

        // Black pawn at A7 should be set: plane 6, sq 48
        assert!(
            (flipped[6 * 64 + 48] - 1.0).abs() < 1e-6,
            "expected Black pawn at A7 (plane 6, sq 48) after flip, got {}",
            flipped[6 * 64 + 48]
        );
        // White pawn plane should be all-zero after flip
        for sq in 0..64 {
            assert_eq!(
                flipped[0 * 64 + sq],
                0.0,
                "White pawn plane should be zero after flip at sq {sq}"
            );
        }
    }

    #[test]
    fn test_flip_obs_planes_side_to_move_flips() {
        // White to move (plane 101 = all-1.0) → after flip → Black to move (all-0.0)
        let mut obs = vec![0.0f32; 103 * 64];
        for sq in 0..64 {
            obs[101 * 64 + sq] = 1.0;
        }
        let flipped = flip_obs_planes(&obs);
        for sq in 0..64 {
            assert!(
                flipped[101 * 64 + sq].abs() < 1e-6,
                "side-to-move plane should be 0.0 after flip at sq {sq}"
            );
        }

        // Black to move (plane 101 = all-0.0) → after flip → White to move (all-1.0)
        let obs_black = vec![0.0f32; 103 * 64];
        let flipped2 = flip_obs_planes(&obs_black);
        for sq in 0..64 {
            assert!(
                (flipped2[101 * 64 + sq] - 1.0).abs() < 1e-6,
                "side-to-move plane should be 1.0 after flip-from-black at sq {sq}"
            );
        }
    }

    #[test]
    fn test_flip_obs_planes_castling_swap() {
        // Set White kingside (plane 96) and Black queenside (plane 99)
        let mut obs = vec![0.0f32; 103 * 64];
        for sq in 0..64 {
            obs[96 * 64 + sq] = 1.0; // W kingside
            obs[99 * 64 + sq] = 1.0; // B queenside
        }
        let flipped = flip_obs_planes(&obs);
        // W kingside (96) → B kingside (98)
        for sq in 0..64 {
            assert!(
                (flipped[98 * 64 + sq] - 1.0).abs() < 1e-6,
                "B kingside should be set after flip (sq {sq})"
            );
            // W kingside slot should now be 0 (was from B kingside which was 0)
            assert!(
                flipped[96 * 64 + sq].abs() < 1e-6,
                "W kingside slot should be 0 after flip (sq {sq})"
            );
        }
        // B queenside (99) → W queenside (97)
        for sq in 0..64 {
            assert!(
                (flipped[97 * 64 + sq] - 1.0).abs() < 1e-6,
                "W queenside should be set after flip (sq {sq})"
            );
        }
    }
}
