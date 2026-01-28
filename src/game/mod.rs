pub mod board;
pub mod player;

pub use player::Player;
pub use board::GameBoard;
use crate::{Bitboard, Color, PieceType, Square, Piece};
use crate::PrecomputedItems;
use std::sync::Arc;

pub struct Move{
    pub from: Square,
    pub to: Square,
    pub promotion_piece_type: Option<PieceType>
}



#[derive(Debug, Clone)]
pub struct GameState {
    game_state: GameBoard,
    board_arr: [Option<Piece>; 64],
    precomputed_items: Arc<PrecomputedItems>,
    white_pieces: Bitboard,
    black_pieces: Bitboard,
    combined_pieces: Bitboard
}

impl GameState{
    pub fn init_game(precomputed_items: Arc<PrecomputedItems>) -> Self{
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
            game_state: GameBoard::start_game(),
            board_arr: board,
            precomputed_items,
            white_pieces: 0x000000000000FFFF,
            black_pieces: 0xFFFF000000000000,
            combined_pieces: 0x000000000000FFFF | 0xFFFF000000000000
        }
    }
    pub fn get_game_board(&self) -> &GameBoard {
        &self.game_state
    }

    pub fn compute_turn_items(&self){
        //at the end of each turn recompute where each piece is
        
    }

    pub fn start_game(&mut self){
        let count = 0;
        if count % 2 != 0 {
            loop {
                let piece_moved: Move = self.game_state.player1.make_move();
                if self.validate_move(piece_moved, true){
                    return
                }
            }
            self.compute_turn_items();
        }
    }

    pub fn validate_move(&mut self, move_obj: Move, isWhite: bool) -> bool{
        //use mask found in pre_computed items
        //implement check logic
        //implement enpasse
        let from: Square = move_obj.from;
        let to: Square = move_obj.to;
        let mut temp_state = self.clone();
        
        let mut player = if isWhite {
            &mut temp_state.game_state.player1
        } else {
            &mut temp_state.game_state.player2
        };

        if player.pieces & (1u64 << (from as u64)) == 0 {
            return false;
        }

        match temp_state.board_arr[from as usize]{
            Some(Piece { color: Color::White, piece_type: PieceType::Pawn }) => {
                let from_idx = from as usize;
                // 1. Get all potential pushes from your table (includes +8 and +16)
                let mut valid_pushes = temp_state.precomputed_items.white_pawn_pushes[from_idx];
                // 2. Remove any pushes that land on occupied squares
                valid_pushes &= !temp_state.combined_pieces;
                // 3. THE "JUMP" FIX: If the square directly in front (+8) is occupied, 
                // the pawn cannot reach the +16 square even if it's empty.
                let single_push_sq = 1u64 << (from_idx + 8);
                if (single_push_sq & temp_state.combined_pieces) != 0 {
                    // If the +8 square is blocked, the +16 move is impossible.
                    // We mask out the +16 square (which is just the single_push shifted by 8)
                    valid_pushes &= !(single_push_sq << 8);
                }
                let valid_attacks = temp_state.precomputed_items.white_pawn_attacks[from_idx] & temp_state.black_pieces;
                let valid_moves = valid_pushes | valid_attacks;
                if valid_moves & (1u64 << (to as u64)) == 0 {
                    return false;
                }
            }
            Some(Piece { color: Color::Black, piece_type: PieceType::Pawn }) => {
                let from_idx = from as usize;
                let mut valid_pushes = temp_state.precomputed_items.black_pawn_pushes[from_idx];
                valid_pushes &= !temp_state.combined_pieces;
                let single_push_sq = 1u64 << (from_idx - 8);
                if (single_push_sq & temp_state.combined_pieces) != 0 {
                    valid_pushes &= !(single_push_sq >> 8);
                }

                let valid_attacks = temp_state.precomputed_items.black_pawn_attacks[from_idx] & temp_state.white_pieces;
                let valid_moves = valid_pushes | valid_attacks;
                if valid_moves & (1u64 << (to as u64)) == 0 {
                    return false;
                }
            }
            Some(Piece{piece_type: PieceType::Rook, ..}) => {
                //IMPLEMENT EASY WAY TO GET ALL PIECES
                let entry = &temp_state.precomputed_items.rook_moves[from as usize];
                let blockers = entry.mask & temp_state.combined_pieces;
                let valid_moves = (entry.magic_table[(entry.magic_num.wrapping_mul(blockers) >> (64 - entry.sig_bits)) as usize]) & !player.pieces;
                if valid_moves & (1u64 << (to as u64)) == 0{
                    return false;
                }
            }
            Some(Piece{piece_type: PieceType::Bishop, ..}) => {
                let entry = &temp_state.precomputed_items.bishop_moves[from as usize];
                let blockers = entry.mask & temp_state.combined_pieces;
                let valid_moves = (entry.magic_table[(entry.magic_num.wrapping_mul(blockers) >> (64 - entry.sig_bits)) as usize]) & !player.pieces;
                if valid_moves & (1u64 << (to as u64)) == 0{
                    return false;
                }
            }
            Some(Piece{piece_type: PieceType::Queen, ..}) => {
                let bishop_entry = &temp_state.precomputed_items.bishop_moves[from as usize];
                let bishop_blockers = bishop_entry.mask & temp_state.combined_pieces;
                let bishop_valid_moves = (bishop_entry.magic_table[(bishop_entry.magic_num.wrapping_mul(bishop_blockers) >> (64 - bishop_entry.sig_bits)) as usize]) & !player.pieces;
                let rook_entry = &temp_state.precomputed_items.rook_moves[from as usize];
                let rook_blockers = rook_entry.mask & temp_state.combined_pieces;
                let rook_valid_moves = (rook_entry.magic_table[(rook_entry.magic_num.wrapping_mul(rook_blockers) >> (64 - rook_entry.sig_bits)) as usize]) & !player.pieces;
                let valid_moves = bishop_valid_moves | rook_valid_moves;
                if valid_moves & (1u64 << (to as u64)) == 0{
                    return false;
                }
            }
            Some(Piece{piece_type: PieceType::King, ..}) => {
                let valid_moves = temp_state.precomputed_items.king_moves[from as usize] & !player.pieces;
                if valid_moves & (1u64 << (to as u64)) == 0{
                    return false;
                }
            }
            Some(Piece{piece_type: PieceType::Knight, ..}) => {
                let valid_moves = temp_state.precomputed_items.knight_moves[from as usize] & !player.pieces;
                if valid_moves & (1u64 << (to as u64)) == 0{
                    return false;
                }
            }
            None => {
                return false;
            }

        }
        
        

        temp_state.update_board(from, to);

        if isWhite{if temp_state.is_in_check_white(){ return false}}
        if !isWhite{if temp_state.is_in_check_black(){ return false}}

        *self = temp_state;
        true
    }


    pub fn is_in_check_white(&self) -> bool{
        let white_king: Square = (self.game_state.player1.pieces_bb[PieceType::King as usize].trailing_zeros() as u8).into();
        //check pawns
        if self.precomputed_items.black_pawn_attacks[white_king as usize] & self.game_state.player2.pieces_bb[PieceType::Pawn as usize] != 0{
            return true;
        }
        if self.precomputed_items.knight_moves[white_king as usize] & self.game_state.player2.pieces_bb[PieceType::Knight as usize] != 0{
            return true;
        }
        if self.precomputed_items.king_moves[white_king as usize] & self.game_state.player2.pieces_bb[PieceType::King as usize] != 0{
            return true;
        }
        let bishop_entry = &self.precomputed_items.bishop_moves[white_king as usize];
        let bishop_blockers = bishop_entry.mask & self.combined_pieces;
        let bishop_valid_moves = (bishop_entry.magic_table[(bishop_entry.magic_num.wrapping_mul(bishop_blockers) >> (64 - bishop_entry.sig_bits)) as usize]) & self.game_state.player2.pieces_bb[PieceType::Bishop as usize];
        let rook_entry = &self.precomputed_items.rook_moves[white_king as usize];
        let rook_blockers = rook_entry.mask & self.combined_pieces;
        let rook_valid_moves = (rook_entry.magic_table[(rook_entry.magic_num.wrapping_mul(rook_blockers) >> (64 - rook_entry.sig_bits)) as usize]) & self.game_state.player2.pieces_bb[PieceType::Rook as usize];
        let queen_valid_moves = (bishop_entry.magic_table[(bishop_entry.magic_num.wrapping_mul(bishop_blockers) >> (64 - bishop_entry.sig_bits)) as usize] | rook_entry.magic_table[(rook_entry.magic_num.wrapping_mul(rook_blockers) >> (64 - rook_entry.sig_bits)) as usize]) & self.game_state.player2.pieces_bb[PieceType::Queen as usize];
        if rook_valid_moves != 0{
            return true;
        }
        if bishop_valid_moves != 0{
            return true;
        }
        if queen_valid_moves != 0{
            return true;
        }

        false
    }

     pub fn is_in_check_black(&self) -> bool{
        let black_king: Square = (self.game_state.player2.pieces_bb[PieceType::King as usize].trailing_zeros() as u8).into();
        //check pawns
        if self.precomputed_items.white_pawn_attacks[black_king as usize] & self.game_state.player1.pieces_bb[PieceType::Pawn as usize] != 0{
            return true;
        }
        if self.precomputed_items.knight_moves[black_king as usize] & self.game_state.player1.pieces_bb[PieceType::Knight as usize] != 0{
            return true;
        }
        if self.precomputed_items.king_moves[black_king as usize] & self.game_state.player1.pieces_bb[PieceType::King as usize] != 0{
            return true;
        }
        let bishop_entry = &self.precomputed_items.bishop_moves[black_king as usize];
        let bishop_blockers = bishop_entry.mask & self.combined_pieces;
        let bishop_valid_moves = (bishop_entry.magic_table[(bishop_entry.magic_num.wrapping_mul(bishop_blockers) >> (64 - bishop_entry.sig_bits)) as usize]) & self.game_state.player1.pieces_bb[PieceType::Bishop as usize];
        let rook_entry = &self.precomputed_items.rook_moves[black_king as usize];
        let rook_blockers = rook_entry.mask & self.combined_pieces;
        let rook_valid_moves = (rook_entry.magic_table[(rook_entry.magic_num.wrapping_mul(rook_blockers) >> (64 - rook_entry.sig_bits)) as usize]) & self.game_state.player1.pieces_bb[PieceType::Rook as usize];
        let queen_valid_moves = (bishop_entry.magic_table[(bishop_entry.magic_num.wrapping_mul(bishop_blockers) >> (64 - bishop_entry.sig_bits)) as usize] | rook_entry.magic_table[(rook_entry.magic_num.wrapping_mul(rook_blockers) >> (64 - rook_entry.sig_bits)) as usize]) & self.game_state.player1.pieces_bb[PieceType::Queen as usize];
        if rook_valid_moves != 0{
            return true;
        }
        if bishop_valid_moves != 0{
            return true;
        }
        if queen_valid_moves != 0{
            return true;
        }

        false
    }

    pub fn update_board(&mut self, from: Square, to: Square){
        // given a from find if white or black as well as what type of piece it is
        let piece_temp_from: Piece = self.board_arr[usize::from(from)].unwrap();
        let piece_temp_to: Piece = self.board_arr[usize::from(to)].unwrap();
        let attacking_color = piece_temp_from.color;

        let temp_mask_from = 1u64 << (u8::from(from));
        let temp_mask_to = 1u64 << (u8:: from(to));
        let combined_mask = temp_mask_from | temp_mask_to;

        let mut player = if piece_temp_from.color == Color::White {
            &mut self.game_state.player1
        } else {
            &mut self.game_state.player2
        };

        player.pieces ^= temp_mask_from;

        player.pieces_bb[usize::from(piece_temp_from.piece_type)] ^= combined_mask;
 
        if self.board_arr[usize::from(to)] != None {
            //just have to update bit_boards
            let mut player2 = if attacking_color == Color::White {
                &mut self.game_state.player2
            } else {
                &mut self.game_state.player1
            };
            player2.pieces &= !temp_mask_to;
            player2.pieces_bb[usize::from(piece_temp_to.piece_type)] &= !temp_mask_to;
        } else{
            player.pieces |= temp_mask_to;
        }

        self.board_arr[usize::from(to)] = self.board_arr[usize::from(from)].take()
    }

}
