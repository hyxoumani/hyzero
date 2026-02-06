use crate::game::Player;
use crate::{Bitboard, Color, PieceType, Square, Piece, BitIterator, CastleOption};
use crate::PrecomputedItems;
use super::Move;
use std::sync::Arc;
use crate::create_game_board;



#[derive(Debug, Clone)]
pub struct GameBoard {
    pub(crate) player1: Player,
    pub(crate) player2: Player,
    board_arr: [Option<Piece>; 64],
    white_pieces: Bitboard,
    precomputed_items: Arc<PrecomputedItems>,
    black_pieces: Bitboard,
    combined_pieces: Bitboard,
    in_check: bool,
    white_pins: Bitboard,
    black_pins: Bitboard,
    is_game_over: bool,
    white_kingside: bool,
    white_queenside: bool,
    black_kingside: bool,
    black_queenside: bool,
    last_move: Move,
    is_en_passant: bool

}

impl GameBoard{

    pub fn init_game_board(precomputed_items: Arc<PrecomputedItems>) -> Self{

        Self{
            player1: Player::new_white(),
            player2: Player::new_black(),
            board_arr: create_game_board(),
            white_pieces: 0x000000000000FFFF,
            precomputed_items,
            black_pieces: 0xFFFF000000000000,
            combined_pieces: 0x000000000000FFFF | 0xFFFF000000000000,
            in_check: false,
            white_pins: 0u64,
            black_pins: 0u64,
            is_game_over: false,
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
            last_move: Move::default(),
            is_en_passant: false,
        }
    }

    pub fn start_game(&mut self){
        let count = 0;
        self.is_game_over = false;
        loop {
            if self.is_game_over {break}
            let mut piece_moved: Move;
            if count % 2 == 0 {
                loop {
                    piece_moved = self.player1.make_move();
                    if self.validate_move(piece_moved, Color::White, self.combined_pieces, self.white_pieces, self.black_pieces){
                        return
                    }
                }
            } else {
                    loop {
                    piece_moved = self.player2.make_move();
                    if self.validate_move(piece_moved, Color::Black, self.combined_pieces, self.white_pieces, self.black_pieces){
                        return
                    }
                }
            }
            self.compute_turn_items(count, piece_moved);
            count += 1;

        }
    }

    pub fn compute_turn_items(&mut self, count: usize, piece_moved: Move){
        //at the end of each turn recompute where each piece is
        self.update_board(piece_moved);
        self.white_pieces = self.player1.pieces;
        self.black_pieces = self.player2.pieces;
        self.combined_pieces = self.white_pieces | self.black_pieces;
        self.last_move = piece_moved;
        let (color_to_move, friendly_bits, opponent_bits, opp_color, pins) = if count % 2 == 0 {
            (Color::White, self.white_pieces, self.black_pieces, Color::Black, self.white_pins)
        } else {
            (Color::Black, self.black_pieces, self.white_pieces, Color::White, self.black_pins)
        };
        self.calculate_pins(color_to_move, self.combined_pieces, self.white_pieces, self.black_pieces);
        //recalculate checkmate & stalemate after each turn
        self.is_game_over = self.calculate_checkmate(count);
        if !self.is_game_over {
            self.is_game_over = self.calculate_stalemate(color_to_move, opp_color, self.combined_pieces, friendly_bits, opponent_bits, pins);
        }

    }

    pub fn update_castling(&mut self, piece_moved: Move) {
        let from_idx : u8 = piece_moved.from.into();
        let to_idx : u8 = piece_moved.to.into();

        if from_idx == 4 {  
            self.white_kingside = false;
            self.white_queenside = false;
        } else if from_idx == 60 { 
            self.black_kingside = false;
            self.black_queenside = false;
        }
        if from_idx == 0 || to_idx == 0 { 
            self.white_queenside = false;
        }
        if from_idx == 7 || to_idx == 7 {
            self.white_kingside = false;
        }
        if from_idx == 56 || to_idx == 56 {
            self.black_queenside = false;
        }
        if from_idx == 63 || to_idx == 63 {
            self.black_kingside = false;
        }
    }




    pub fn calculate_pins(&mut self, color: Color, board: Bitboard, white_pieces: Bitboard, black_pieces: Bitboard){
        let mut player1 = &self.player1;
        let mut player2 = &self.player2;
        
        let (king_bits, friendly_bits, enemy_bits, enemy_sliders) = match color {
            Color::White => (
                player1.pieces_bb[PieceType::King as usize] as u64, 
                white_pieces, 
                black_pieces,
                player2.pieces_bb[PieceType::King as usize] | player2.pieces_bb[PieceType::Rook as usize] | player2.pieces_bb[PieceType::Bishop as usize]
            ),
            Color::Black => (
                player2.pieces_bb[PieceType::King as usize] as u64, 
                black_pieces, 
                white_pieces,
                player1.pieces_bb[PieceType::King as usize] | player1.pieces_bb[PieceType::Rook as usize] | player1.pieces_bb[PieceType::Bishop as usize]
            ),
        };

        let king_sq = king_bits.trailing_zeros() as usize;

        // 2. The Closure: Captures the specific context for THIS side
        let find_pin = |attacker_sq: usize| -> Option<u64> {
            // Get precomputed ray between King and Attacker
            let path = self.precomputed_items.rays[king_sq][attacker_sq];
            let blockers = path & board;

            // If exactly one piece blocks the ray and it's our color
            if blockers.count_ones() == 1 && (blockers & friendly_bits) != 0 {
                Some(blockers)
            } else {
                None
            }
        };

        // 3. Execution
        let mut pin_mask = 0u64;
        
        // Only check enemy sliders that are on the same lines as the King
        for attacker_sq in BitIterator::new(enemy_sliders) {
            let attacker_idx = attacker_sq as usize;
            // 1. Get the precomputed exclusive path
            let path = self.precomputed_items.rays[king_sq][attacker_idx];

            // 2. If path is 0, they aren't on a line (or are adjacent)
            // Note: Adjacent sliders are checks, not pins, so we ignore path == 0
            if path == 0 { continue; }

            // 3. Count pieces on the path
            let blockers = path & board;

            if blockers.count_ones() == 1 {
                // 4. If the single blocker is friendly, it's a pin!
                if (blockers & friendly_bits) != 0 {
                    pin_mask |= blockers;
                    
                }
            }
        }

        // 4. Store the result in the appropriate player's state
        match color {
            Color::White => self.white_pins = pin_mask,
            Color::Black => self.black_pins = pin_mask,
        }
        
    }

    pub fn get_king_sq(&mut self, color:Color) -> u64 {
        let player = if color == Color::White {
            &self.player1
        } else {
            &self.player2
        };

        return player.pieces_bb[PieceType::King as usize]
    }

    pub fn calculate_stalemate(&mut self, color: Color, opp_color: Color, combined_bits: u64, friendly_bits: u64, opponent_bits: u64, pins: u64) -> bool{
        let mut move_mask = 0u64;
        for sq in BitIterator::new(friendly_bits){
            let piece_type = self.board_arr[sq as usize].unwrap().piece_type;
            if piece_type != PieceType::King {
                //need to check for pins
                let move_set = self.get_move_mask(sq, color, piece_type, combined_bits, friendly_bits, opponent_bits);
                if  move_set != 0{
                    if 1u64 << sq & pins != 0 {
                        let king_sq = self.get_king_sq(color) as usize;
                        if move_set & self.precomputed_items.lines[king_sq][sq] != 0 {
                            return false
                        }
                        //check pinlines 
                    } else {
                        return false
                    }
                }
            } else {
                for king_move in BitIterator::new(self.get_move_mask(sq, color, piece_type, combined_bits, friendly_bits, opponent_bits)){
                    if self.get_attackers(king_move, opp_color, self.combined_pieces, self.white_pieces, self.black_pieces) == 0{
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn calculate_checkmate(&mut self, count: usize) -> bool{
        let (player, opp_color, color, pins) = if count % 2 == 0 { 
            (&self.player2, Color::Black, Color::White, self.white_pins)
        } else {
            (&self.player1, Color::White, Color::Black, self.black_pins)
        };
        let king_sq = player.pieces_bb[PieceType::King as usize].trailing_zeros() as usize;

        //if in check
        let attackers = self.get_attackers(king_sq, opp_color, self.combined_pieces, self.white_pieces, self.black_pieces);
        if attackers != 0{
            let mut king_moves = self.get_move_mask(king_sq, opp_color, PieceType::King, self.combined_pieces, self.white_pieces, self.black_pieces) & player.pieces;
            while king_moves != 0 {
                let mut king_sq = king_moves.trailing_zeros() as usize;
                if self.get_attackers(king_sq, opp_color, self.combined_pieces, self.white_pieces, self.black_pieces) == 0{
                    return false;
                }
                king_moves &= king_moves - 1; 
            }
        }

        if attackers.count_ones() > 1 {
            return true;
        }

        //get rays
        let ray_mask = self.precomputed_items.rays[king_sq][attackers.trailing_zeros() as usize] | attackers;

        for sq in BitIterator::new(player.pieces){
            let move_mask = self.get_move_mask(sq, color, self.board_arr[sq as usize].unwrap().piece_type, self.combined_pieces, self.white_pieces, self.black_pieces);
            if move_mask & ray_mask != 0 {
                if 1u64 << sq & pins != 0 {
                    if self.precomputed_items.rays[king_sq][sq] & ray_mask != 0 {
                        return false
                    }
                } else {
                    return false
                }
            }
        }

        true
    }

    pub fn validate_move(&self, piece_moved: Move, color: Color, board: Bitboard, white_pieces: Bitboard, black_pieces: Bitboard) -> bool {
        //just check if it's valid & not in check
        //better to instead pass clone pass a board

        let (mut player, castle_kingside, castle_queenside, opp_color) = if color == Color::White {
            (&self.player1, self.white_kingside, self.white_queenside, Color::Black)
        } else {
            (&self.player2, self.black_kingside, self.black_queenside, Color::White)
        };

        let from_idx: Square = piece_moved.from;
        let to_idx: Square = piece_moved.to;
        let king_sq: usize = player.pieces_bb[PieceType::King as usize].trailing_zeros() as usize;
        
        //handle castling
        if let Some(castle_option) = piece_moved.castle_option {
            match castle_option {
                CastleOption::Kingside => if !castle_kingside {return false}, 
                CastleOption::Queenside => if !castle_queenside {return false}
            }

            //get casling side
            if castle_option == CastleOption::Kingside {
                for sq in BitIterator::new(self.precomputed_items.castle_squares[color as usize][castle_option as usize]) {
                    if board & (1u64 << sq) != 0 {
                        return false;
                    }
                    if self.get_attackers(sq, opp_color, board, white_pieces, black_pieces) != 0 {
                        return false
                    }
                }
            }
            return true
        }
        
        //check for enpassant
        if let Some(is_en_passant) = self.last_move.en_passant {
            if is_en_passant {
                
            }
        }


        let mut temp_state = self.clone();
        if 1u64 << to_idx as u64 & temp_state.get_move_mask(from_idx as usize, color, temp_state.board_arr[from_idx as usize].unwrap().piece_type, board, white_pieces, black_pieces) == 0{
            return false
        }
        temp_state.update_board(piece_moved);
        let opponent_color = if color == Color::White { Color::Black } else { Color::White };

        if temp_state.get_attackers(king_sq, opponent_color, board, white_pieces, black_pieces) != 0 {
            return false;
        }

        true
    }

    pub fn update_board(&mut self, piece_moved: Move) {
        let from_idx = usize::from(piece_moved.from);
        let to_idx = usize::from(piece_moved.to);
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
            
            // Update own_board
            player.own_board[to_idx] = player.own_board[from_idx].take();
        }
        
        if let Some(captured_piece) = self.board_arr[to_idx] {
            let opponent = if attacker_color == Color::White {
                &mut self.player2
            } else {
                &mut self.player1
            };
            
            opponent.pieces &= !to_mask;
            opponent.pieces_bb[captured_piece.piece_type as usize] &= !to_mask;
            
            // Clear captured piece from opponent's own_board
            opponent.own_board[to_idx] = None;
        }
        
        // Handle castling
        if let Some(castle) = piece_moved.castle_option {
            let (rook_from, rook_to) = match (castle, attacker_color) {
                (CastleOption::Kingside, Color::White) => (7usize, 5usize),   // h1 -> f1
                (CastleOption::Kingside, Color::Black) => (63usize, 61usize), // h8 -> f8
                (CastleOption::Queenside, Color::White) => (0usize, 3usize),  // a1 -> d1
                (CastleOption::Queenside, Color::Black) => (56usize, 59usize), // a8 -> d8
            };
            
            let rook_from_mask = 1u64 << rook_from;
            let rook_to_mask = 1u64 << rook_to;
            
            let player = if attacker_color == Color::White {
                &mut self.player1
            } else {
                &mut self.player2
            };
            
            // Move rook in bitboards
            player.pieces ^= rook_from_mask;
            player.pieces |= rook_to_mask;
            player.pieces_bb[PieceType::Rook as usize] ^= rook_from_mask;
            player.pieces_bb[PieceType::Rook as usize] |= rook_to_mask;
            
            // Move rook in board_arr and own_board
            self.board_arr[rook_to] = self.board_arr[rook_from].take();
            player.own_board[rook_to] = player.own_board[rook_from].take();
        }
        
        // 5. Specialized Pawn Logic: En Passant Target Generation
        if piece_type == PieceType::Pawn {
            let diff = (from_idx as i8 - to_idx as i8).abs();
            if diff == 16 {
                let ep_idx = if attacker_color == Color::White { from_idx + 8 } else { from_idx - 8 };
            }
        }
        
        self.board_arr[to_idx] = self.board_arr[from_idx].take();
        
        let player = if attacker_color == Color::White {
            &mut self.player1
        } else {
            &mut self.player2
        };
        
        // Update for promotion
        if let Some(piece) = piece_moved.promotion_piece_type {
            player.pieces_bb[PieceType::Pawn as usize] ^= 1u64 << to_idx;
            player.pieces_bb[piece as usize] |= 1u64 << to_idx;
            self.board_arr[to_idx] = Some(Piece{color: attacker_color, piece_type: piece});
            
            // Update own_board with promoted piece
            player.own_board[to_idx] = Some(Piece{color: attacker_color, piece_type: piece});
        }
    }

    pub fn get_attackers(&self, sq: usize, attacker_color:Color, board: Bitboard, white_pieces: Bitboard, black_pieces: Bitboard) -> u64{
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
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Pawn, board, white_pieces, black_pieces) & opponent.pieces_bb[PieceType::Pawn as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Knight, board, white_pieces, black_pieces) & opponent.pieces_bb[PieceType::Knight as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::King, board, white_pieces, black_pieces) & opponent.pieces_bb[PieceType::King as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Rook, board, white_pieces, black_pieces) & opponent.pieces_bb[PieceType::Rook as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Bishop, board, white_pieces, black_pieces) & opponent.pieces_bb[PieceType::Bishop as usize];
        attackers |= self.get_move_mask(sq, opp_color, PieceType::Queen, board, white_pieces, black_pieces) & opponent.pieces_bb[PieceType::Queen as usize];

        attackers
    }

    pub fn get_move_mask(&self, sq: usize, color: Color, piece: PieceType, board: Bitboard, white_pieces: Bitboard, black_pieces: Bitboard) -> u64 {
        let own_pieces = if color == Color::White { white_pieces } else { black_pieces };
        let enemy_pieces = if color == Color::White { black_pieces } else { white_pieces };

        match piece {
            PieceType::Pawn => self.get_pawn_moves(sq, color, enemy_pieces, board),
            PieceType::Knight => self.precomputed_items.knight_moves[sq] & !own_pieces,
            PieceType::King => self.precomputed_items.king_moves[sq] & !own_pieces,
            PieceType::Rook => self.get_sliding_moves(sq, PieceType::Rook, board) & !own_pieces,
            PieceType::Bishop => self.get_sliding_moves(sq, PieceType::Bishop, board) & !own_pieces,
            PieceType::Queen => {
                (self.get_sliding_moves(sq, PieceType::Rook, board) | 
                self.get_sliding_moves(sq, PieceType::Bishop, board)) & !own_pieces
            }
        }
    }

    /// Helper to encapsulate Magic Bitboard lookups
    fn get_sliding_moves(&self, sq: usize, piece_type: PieceType, board: Bitboard) -> u64 {
        match piece_type {
            PieceType::Rook => {
                let e = &self.precomputed_items.rook_moves[sq];
                let hash = ( (e.mask & board).wrapping_mul(e.magic_num) ) >> (64 - e.sig_bits);
                e.magic_table[hash as usize]
            }
            PieceType::Bishop => {
                let e = &self.precomputed_items.bishop_moves[sq];
                let hash = ( (e.mask & board).wrapping_mul(e.magic_num) ) >> (64 - e.sig_bits);
                e.magic_table[hash as usize]
            }
            _ => 0,
        }
    }

    /// Helper for Pawn Pushes and Attacks
    fn get_pawn_moves(&self, sq: usize, color: Color, enemy_pieces: u64, board: Bitboard) -> u64 {
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
        let mut valid_pushes = pushes & !board;
        
        // If the square immediately in front is blocked, the double-push is also blocked
        let push_direction = if color == Color::White { 8i8 } else { -8i8 };
        let one_step_sq = 1u64 << (sq as i8 + push_direction) as usize;
        
        if (one_step_sq & board) != 0 {
            // This removes the double-jump if the first square is occupied
            let double_step_mask = if color == Color::White { 0xFFFF_0000_0000_0000 } else { 0x0000_0000_0000_FFFF };
            valid_pushes &= !double_step_mask; 
        }

        // 2. Calculate Attacks (only moves if an enemy piece is there)
        let valid_attacks = attacks & enemy_pieces;

        valid_pushes | valid_attacks
    }
}