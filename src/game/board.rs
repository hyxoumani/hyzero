use crate::game::Player;
use crate::{Bitboard, Color, PieceType, Square, Piece};
use crate::PrecomputedItems;
use super::Move;
use std::sync::Arc;




#[derive(Debug, Clone)]
pub struct GameBoard {
    pub(crate) player1: Player,
    pub(crate) player2: Player,
    board_arr: [Option<Piece>; 64],
    white_pieces: Bitboard,
    precomputed_items: Arc<PrecomputedItems>,
    black_pieces: Bitboard,
    combined_pieces: Bitboard,
    in_check: bool

}

impl GameBoard{

    pub fn init_game_board(precomputed_items: Arc<PrecomputedItems>) -> Self{
        let mut board = [None; 64];
        let white = |t:PieceType| Some(Piece{piece_type: t, color: Color::White});
        let black = |t:PieceType| Some(Piece{piece_type:t, color: Color::Black});
        board[0] = white(PieceType::Rook);
        board[1] = white(PieceType::Knight);
        board[2] = white(PieceType::Bishop);
        board[3] = white(PieceType::Queen);
        board[4] = white(PieceType::King);
        board[5] = white(PieceType::Bishop);
        board[6] = white(PieceType::Knight);
        board[7] = white(PieceType::Rook);
        for i in 8..16 { board[i] = white(PieceType::Pawn); }
        for i in 48..56 { board[i] = black(PieceType::Pawn); }
        board[56] = black(PieceType::Rook);
        board[57] = black(PieceType::Knight);
        board[58] = black(PieceType::Bishop);
        board[59] = black(PieceType::Queen);
        board[60] = black(PieceType::King);
        board[61] = black(PieceType::Bishop);
        board[62] = black(PieceType::Knight);
        board[63] = black(PieceType::Rook);

        Self{
            player1: Player::new_white(),
            player2: Player::new_black(),
            board_arr: board,
            white_pieces: 0x000000000000FFFF,
            precomputed_items,
            black_pieces: 0xFFFF000000000000,
            combined_pieces: 0x000000000000FFFF | 0xFFFF000000000000,
            in_check: false
        }
    }

    pub fn start_game(&mut self){
        let count = 0;
        loop {
            let mut piece_moved: Move;
            if count % 2 == 0 {
                loop {
                    piece_moved = self.player1.make_move();
                    if self.validate_move(piece_moved, Color::White){
                        return
                    }
                }
            } else {
                    loop {
                    piece_moved = self.player2.make_move();
                    if self.validate_move(piece_moved, Color::Black){
                        return
                    }
                }
            }
            self.compute_turn_items(count);
            count += 1;
        }
    }

    pub fn compute_turn_items(&mut self, count: usize){
        //at the end of each turn recompute where each piece is
        self.white_pieces = self.player1.pieces;
        self.black_pieces = self.player2.pieces;
        self.combined_pieces = self.white_pieces | self.black_pieces;

        //recalculate checks after each turn
    }

    pub fn validate_move(&self, piece_moved: Move, color: Color, board: Bitboard) -> bool {
        //just check if it's valid & not in check
        let mut player = if color == Color::White {
            &self.player1
        } else {
            &self.player2
        };

        let from_idx: Square = piece_moved.from;
        let to_idx: Square = piece_moved.to;
        let king_sq: usize = player.pieces_bb[PieceType::King as usize].trailing_zeros() as usize;
        let mut temp_state = self.clone();
        if 1u64 << to_idx as u64 & temp_state.get_move_mask(from_idx as usize, color, temp_state.board_arr[from_idx as usize].unwrap().piece_type) == 0{
            return false
        }
        temp_state.update_board(from_idx, to_idx);
        let opponent_color = if color == Color::White { Color::Black } else { Color::White };

        if temp_state.get_attackers(king_sq, opponent_color) != 0 {
            return false;
        }

        true
    }

    pub fn update_board(&mut self, from: Square, to: Square) {
        let from_idx = usize::from(from);
        let to_idx = usize::from(to);

        let moving_piece = self.board_arr[from_idx].expect("No piece at 'from' square");
        let attacker_color = moving_piece.color;
        let piece_type = moving_piece.piece_type;

        let from_mask = 1u64 << from_idx;
        let to_mask = 1u64 << to_idx;

        {
            let player = if attacker_color == Color::White {
                &mut self.player1
            } else {
                &mut self.player2
            };

            player.pieces ^= from_mask;                  // Remove from start
            player.pieces |= to_mask;                    // Add to destination
            player.pieces_bb[piece_type as usize] ^= from_mask; 
            player.pieces_bb[piece_type as usize] |= to_mask;
        }

        if let Some(captured_piece) = self.board_arr[to_idx] {
            let opponent = if attacker_color == Color::White {
                &mut self.player2
            } else {
                &mut self.player1
            };

            opponent.pieces &= !to_mask;
            opponent.pieces_bb[captured_piece.piece_type as usize] &= !to_mask;
        }

        // 5. Specialized Pawn Logic: En Passant Target Generation
        if piece_type == PieceType::Pawn {
            let diff = (from_idx as i8 - to_idx as i8).abs();
            if diff == 16 {
                let ep_idx = if attacker_color == Color::White { from_idx + 8 } else { from_idx - 8 };
            }
        }

        self.board_arr[to_idx] = self.board_arr[from_idx].take();
    }

    pub fn get_attackers(&self, sq: usize, attacker_color:Color) -> u64{
        let mut attackers = 0u64;
        
        let mut opponent = if attacker_color == Color::White {
            &self.player1 
        } else {
            &self.player2 
        };
        let mut opp_color = if attacker_color == Color::White{
            Color::Black
        } else {
            Color::White
        };
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Pawn) & opponent.pieces_bb[PieceType::Pawn as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Knight) & opponent.pieces_bb[PieceType::Knight as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::King) & opponent.pieces_bb[PieceType::King as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Rook) & opponent.pieces_bb[PieceType::Rook as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Bishop) & opponent.pieces_bb[PieceType::Bishop as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Queen) & opponent.pieces_bb[PieceType::Queen as usize];

        attackers
    }

    pub fn get_move_mask(&self, sq: usize, color: Color, piece: PieceType) -> u64 {
        let own_pieces = if color == Color::White { self.white_pieces } else { self.black_pieces };
        let enemy_pieces = if color == Color::White { self.black_pieces } else { self.white_pieces };

        match piece {
            PieceType::Pawn => self.get_pawn_moves(sq, color, enemy_pieces),
            PieceType::Knight => self.precomputed_items.knight_moves[sq] & !own_pieces,
            PieceType::King => self.precomputed_items.king_moves[sq] & !own_pieces,
            PieceType::Rook => self.get_sliding_moves(sq, PieceType::Rook) & !own_pieces,
            PieceType::Bishop => self.get_sliding_moves(sq, PieceType::Bishop) & !own_pieces,
            PieceType::Queen => {
                (self.get_sliding_moves(sq, PieceType::Rook) | 
                self.get_sliding_moves(sq, PieceType::Bishop)) & !own_pieces
            }
        }
    }

    /// Helper to encapsulate Magic Bitboard lookups
    fn get_sliding_moves(&self, sq: usize, piece_type: PieceType) -> u64 {
        match piece_type {
            PieceType::Rook => {
                let e = &self.precomputed_items.rook_moves[sq];
                let hash = ( (e.mask & self.combined_pieces).wrapping_mul(e.magic_num) ) >> (64 - e.sig_bits);
                e.magic_table[hash as usize]
            }
            PieceType::Bishop => {
                let e = &self.precomputed_items.bishop_moves[sq];
                let hash = ( (e.mask & self.combined_pieces).wrapping_mul(e.magic_num) ) >> (64 - e.sig_bits);
                e.magic_table[hash as usize]
            }
            _ => 0,
        }
    }

    /// Helper for Pawn Pushes and Attacks
    fn get_pawn_moves(&self, sq: usize, color: Color, enemy_pieces: u64) -> u64 {
        let (pushes, attacks) = match color {
            Color::White => (
                self.precomputed_items.white_pawn_pushes[sq],
                self.precomputed_items.white_pawn_attacks[sq]
            ),
            Color::Black => (
                self.precomputed_items.black_pawn_pushes[sq],
                self.precomputed_items.black_pawn_attacks[sq]
            ),
        };

        // 1. Calculate Pushes (must handle the "blockade" logic)
        let mut valid_pushes = pushes & !self.combined_pieces;
        
        // If the square immediately in front is blocked, the double-push is also blocked
        let push_direction = if color == Color::White { 8i8 } else { -8i8 };
        let one_step_sq = 1u64 << (sq as i8 + push_direction) as usize;
        
        if (one_step_sq & self.combined_pieces) != 0 {
            // This removes the double-jump if the first square is occupied
            let double_step_mask = if color == Color::White { 0xFFFF_0000_0000_0000 } else { 0x0000_0000_0000_FFFF };
            valid_pushes &= !double_step_mask; 
        }

        // 2. Calculate Attacks (only moves if an enemy piece is there)
        let valid_attacks = attacks & enemy_pieces;

        valid_pushes | valid_attacks
    }
}