use super::Move;
use crate::create_game_board;
use crate::game::Player;
use crate::PrecomputedItems;
use crate::{BitIterator, Bitboard, CastleOption, Color, Piece, PieceType, Square};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameResult {
    Ongoing,
    Checkmate(Color),
    Stalemate,
    FiftyMoveRule,
    ThreefoldRepetition,
    InsufficientMaterial,
}

#[derive(Debug, Clone)]
pub struct GameBoard {
    pub(crate) player1: Player,
    pub(crate) player2: Player,
    pub(crate) board_arr: [Option<Piece>; 64],
    pub(crate) white_pieces: Bitboard,
    pub(crate) precomputed_items: Arc<PrecomputedItems>,
    pub(crate) black_pieces: Bitboard,
    pub(crate) combined_pieces: Bitboard,
    #[allow(dead_code)]
    pub(crate) in_check: bool,
    pub(crate) white_pins: Bitboard,
    pub(crate) black_pins: Bitboard,
    pub(crate) game_result: GameResult,
    pub(crate) white_kingside: bool,
    pub(crate) white_queenside: bool,
    pub(crate) black_kingside: bool,
    pub(crate) black_queenside: bool,
    pub(crate) last_move: Move,
    pub(crate) en_passant_target: Option<usize>,
    pub(crate) halfmove_clock: u32,
    pub(crate) position_history: HashMap<u64, u8>,
    pub(crate) zobrist_hash: u64,
}

impl GameBoard {
    pub fn init_game_board(
        precomputed_items: Arc<PrecomputedItems>,
        player1: Player,
        player2: Player,
    ) -> Self {
        let board_arr = create_game_board();

        // Compute initial Zobrist hash from scratch
        let mut zobrist_hash = 0u64;
        let zt = &precomputed_items.zobrist;
        for sq in 0..64usize {
            if let Some(piece) = board_arr[sq] {
                let color_idx = piece.color as usize;
                let pt_idx = piece.piece_type as usize;
                zobrist_hash ^= zt.piece_sq[color_idx][pt_idx][sq];
            }
        }
        // White to move — do not XOR side_to_move (side_to_move is XORed only when Black)
        // XOR in all four initial castling rights
        zobrist_hash ^= zt.castling[0]; // WK
        zobrist_hash ^= zt.castling[1]; // WQ
        zobrist_hash ^= zt.castling[2]; // BK
        zobrist_hash ^= zt.castling[3]; // BQ

        Self {
            player1,
            player2,
            board_arr,
            white_pieces: 0x000000000000FFFF,
            precomputed_items,
            black_pieces: 0xFFFF000000000000,
            combined_pieces: 0x000000000000FFFF | 0xFFFF000000000000,
            in_check: false,
            white_pins: 0u64,
            black_pins: 0u64,
            game_result: GameResult::Ongoing,
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
            last_move: Move::default(),
            en_passant_target: None,
            halfmove_clock: 0,
            position_history: HashMap::new(),
            zobrist_hash,
        }
    }

    pub fn start_game(&mut self) {
        let mut count: usize = 0;
        self.game_result = GameResult::Ongoing;
        loop {
            if self.game_result != GameResult::Ongoing {
                break;
            }
            let piece_moved;
            if count.is_multiple_of(2) {
                loop {
                    let candidate = self.player1.make_move();
                    if self.validate_move(
                        candidate,
                        Color::White,
                        self.combined_pieces,
                        self.white_pieces,
                        self.black_pieces,
                    ) {
                        piece_moved = candidate;
                        break;
                    }
                    println!("Invalid move, try again.");
                }
            } else {
                loop {
                    let candidate = self.player2.make_move();
                    if self.validate_move(
                        candidate,
                        Color::Black,
                        self.combined_pieces,
                        self.white_pieces,
                        self.black_pieces,
                    ) {
                        piece_moved = candidate;
                        break;
                    }
                    println!("Invalid move, try again.");
                }
            }
            self.compute_turn_items(count, piece_moved);
            count += 1;
        }

        match self.game_result {
            GameResult::Checkmate(winner) => println!("Checkmate! {:?} wins!", winner),
            GameResult::Stalemate => println!("Draw by stalemate!"),
            GameResult::FiftyMoveRule => println!("Draw by 50-move rule!"),
            GameResult::ThreefoldRepetition => println!("Draw by threefold repetition!"),
            GameResult::InsufficientMaterial => println!("Draw by insufficient material!"),
            GameResult::Ongoing => {}
        }
    }

    pub fn compute_turn_items(&mut self, count: usize, piece_moved: Move) {
        // Detect pawn move / capture BEFORE updating the board
        let moving_piece = self.board_arr[usize::from(piece_moved.from)];
        let is_pawn_move = moving_piece.map(|p| p.piece_type) == Some(PieceType::Pawn);
        let is_capture = self.board_arr[usize::from(piece_moved.to)].is_some()
            || (is_pawn_move && self.en_passant_target == Some(usize::from(piece_moved.to)));

        //at the end of each turn recompute where each piece is
        self.update_board(piece_moved);
        self.update_castling(piece_moved);
        self.white_pieces = self.player1.pieces;
        self.black_pieces = self.player2.pieces;
        self.combined_pieces = self.white_pieces | self.black_pieces;
        self.last_move = piece_moved;

        // Recalculate pins for BOTH sides
        self.calculate_pins(
            Color::White,
            self.combined_pieces,
            self.white_pieces,
            self.black_pieces,
        );
        self.calculate_pins(
            Color::Black,
            self.combined_pieces,
            self.white_pieces,
            self.black_pieces,
        );

        // When count.is_multiple_of(2), white just moved, now check if BLACK is in checkmate/stalemate
        let (color_to_move, friendly_bits, opponent_bits, opp_color, pins) =
            if count.is_multiple_of(2) {
                (
                    Color::Black,
                    self.black_pieces,
                    self.white_pieces,
                    Color::White,
                    self.black_pins,
                )
            } else {
                (
                    Color::White,
                    self.white_pieces,
                    self.black_pieces,
                    Color::Black,
                    self.white_pins,
                )
            };
        //recalculate checkmate & stalemate after each turn
        if self.calculate_checkmate(color_to_move) {
            // The winner is the side that just moved (opposite of color_to_move)
            let winner = if color_to_move == Color::White {
                Color::Black
            } else {
                Color::White
            };
            self.game_result = GameResult::Checkmate(winner);
        }
        if self.game_result == GameResult::Ongoing
            && self.calculate_stalemate(
                color_to_move,
                opp_color,
                self.combined_pieces,
                friendly_bits,
                opponent_bits,
                pins,
            )
        {
            self.game_result = GameResult::Stalemate;
        }

        // 50-move rule
        if is_pawn_move || is_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }
        if self.game_result == GameResult::Ongoing && self.halfmove_clock >= 100 {
            self.game_result = GameResult::FiftyMoveRule;
        }

        // Threefold repetition
        let hash = self.zobrist_hash;
        let count_entry = self.position_history.entry(hash).or_insert(0);
        *count_entry += 1;
        if self.game_result == GameResult::Ongoing && *count_entry >= 3 {
            self.game_result = GameResult::ThreefoldRepetition;
        }

        // Insufficient material
        if self.game_result == GameResult::Ongoing && self.is_insufficient_material() {
            self.game_result = GameResult::InsufficientMaterial;
        }
    }

    pub fn update_castling(&mut self, piece_moved: Move) {
        let from_idx: u8 = piece_moved.from.into();
        let to_idx: u8 = piece_moved.to.into();

        // Snapshot castling rights before modification for Zobrist delta
        let old_wk = self.white_kingside;
        let old_wq = self.white_queenside;
        let old_bk = self.black_kingside;
        let old_bq = self.black_queenside;

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

        // XOR out rights that were just lost: old=true & new=false means it flipped
        let zt = &self.precomputed_items.zobrist;
        if old_wk && !self.white_kingside {
            self.zobrist_hash ^= zt.castling[0];
        }
        if old_wq && !self.white_queenside {
            self.zobrist_hash ^= zt.castling[1];
        }
        if old_bk && !self.black_kingside {
            self.zobrist_hash ^= zt.castling[2];
        }
        if old_bq && !self.black_queenside {
            self.zobrist_hash ^= zt.castling[3];
        }
    }

    pub fn calculate_pins(
        &mut self,
        color: Color,
        board: Bitboard,
        white_pieces: Bitboard,
        black_pieces: Bitboard,
    ) {
        let player1 = &self.player1;
        let player2 = &self.player2;

        let (king_bits, friendly_bits, enemy_sliders) = match color {
            Color::White => (
                player1.pieces_bb[PieceType::King as usize],
                white_pieces,
                player2.pieces_bb[PieceType::King as usize]
                    | player2.pieces_bb[PieceType::Rook as usize]
                    | player2.pieces_bb[PieceType::Bishop as usize],
            ),
            Color::Black => (
                player2.pieces_bb[PieceType::King as usize],
                black_pieces,
                player1.pieces_bb[PieceType::King as usize]
                    | player1.pieces_bb[PieceType::Rook as usize]
                    | player1.pieces_bb[PieceType::Bishop as usize],
            ),
        };

        let king_sq = king_bits.trailing_zeros() as usize;

        // Execution
        let mut pin_mask = 0u64;

        // Only check enemy sliders that are on the same lines as the King
        for attacker_sq in BitIterator::new(enemy_sliders) {
            // 1. Get the precomputed exclusive path
            let path = self.precomputed_items.rays[king_sq][attacker_sq];

            // 2. If path is 0, they aren't on a line (or are adjacent)
            // Note: Adjacent sliders are checks, not pins, so we ignore path == 0
            if path == 0 {
                continue;
            }

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

    pub fn get_king_sq(&mut self, color: Color) -> u64 {
        let player = if color == Color::White {
            &self.player1
        } else {
            &self.player2
        };

        player.pieces_bb[PieceType::King as usize]
    }

    pub fn calculate_stalemate(
        &mut self,
        color: Color,
        opp_color: Color,
        combined_bits: u64,
        friendly_bits: u64,
        opponent_bits: u64,
        pins: u64,
    ) -> bool {
        for sq in BitIterator::new(friendly_bits) {
            let piece_type = self.board_arr[sq].unwrap().piece_type;
            if piece_type != PieceType::King {
                //need to check for pins
                let move_set = self.get_move_mask(
                    sq,
                    color,
                    piece_type,
                    combined_bits,
                    friendly_bits,
                    opponent_bits,
                );
                if move_set != 0 {
                    if (1u64 << sq) & pins != 0 {
                        let king_sq = self.get_king_sq(color).trailing_zeros() as usize;
                        if move_set & self.precomputed_items.lines[king_sq][sq] != 0 {
                            return false;
                        }
                        //check pinlines
                    } else {
                        return false;
                    }
                }
            } else {
                for king_move in BitIterator::new(self.get_move_mask(
                    sq,
                    color,
                    piece_type,
                    combined_bits,
                    friendly_bits,
                    opponent_bits,
                )) {
                    if self.get_attackers(
                        king_move,
                        opp_color,
                        self.combined_pieces,
                        self.white_pieces,
                        self.black_pieces,
                    ) == 0
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn calculate_checkmate(&mut self, color: Color) -> bool {
        // color = the side we're checking for checkmate
        let (player, opp_color, pins) = match color {
            Color::White => (&self.player1, Color::Black, self.white_pins),
            Color::Black => (&self.player2, Color::White, self.black_pins),
        };
        let king_sq = player.pieces_bb[PieceType::King as usize].trailing_zeros() as usize;

        // Find opponent pieces attacking our king
        let attackers = self.get_attackers(
            king_sq,
            opp_color,
            self.combined_pieces,
            self.white_pieces,
            self.black_pieces,
        );

        // Not in check = not checkmate
        if attackers == 0 {
            return false;
        }

        // Check if king can escape
        let mut king_moves = self.get_move_mask(
            king_sq,
            color,
            PieceType::King,
            self.combined_pieces,
            self.white_pieces,
            self.black_pieces,
        );
        while king_moves != 0 {
            let escape_sq = king_moves.trailing_zeros() as usize;
            if self.get_attackers(
                escape_sq,
                opp_color,
                self.combined_pieces,
                self.white_pieces,
                self.black_pieces,
            ) == 0
            {
                return false;
            }
            king_moves &= king_moves - 1;
        }

        // Double check = must move king, and we already checked king can't escape
        if attackers.count_ones() > 1 {
            return true;
        }

        // Single check: can any piece block or capture the attacker?
        let attacker_sq = attackers.trailing_zeros() as usize;
        let ray_mask = self.precomputed_items.rays[king_sq][attacker_sq] | attackers;

        for sq in BitIterator::new(player.pieces) {
            if sq == king_sq {
                continue;
            }
            let piece = match self.board_arr[sq] {
                Some(p) => p,
                None => continue,
            };
            let move_mask = self.get_move_mask(
                sq,
                color,
                piece.piece_type,
                self.combined_pieces,
                self.white_pieces,
                self.black_pieces,
            );
            if move_mask & ray_mask != 0 {
                if (1u64 << sq) & pins != 0 {
                    if self.precomputed_items.lines[king_sq][sq] & ray_mask != 0 {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        true
    }

    pub fn validate_move(
        &self,
        piece_moved: Move,
        color: Color,
        board: Bitboard,
        white_pieces: Bitboard,
        black_pieces: Bitboard,
    ) -> bool {
        //just check if it's valid & not in check
        //better to instead pass clone pass a board

        let (_player, castle_kingside, castle_queenside, opp_color) = if color == Color::White {
            (
                &self.player1,
                self.white_kingside,
                self.white_queenside,
                Color::Black,
            )
        } else {
            (
                &self.player2,
                self.black_kingside,
                self.black_queenside,
                Color::White,
            )
        };

        let from_idx: Square = piece_moved.from;
        let to_idx: Square = piece_moved.to;

        //handle castling
        if let Some(castle_option) = piece_moved.castle_option {
            match castle_option {
                CastleOption::Kingside => {
                    if !castle_kingside {
                        return false;
                    }
                }
                CastleOption::Queenside => {
                    if !castle_queenside {
                        return false;
                    }
                }
            }

            // Check empty squares (must be unoccupied)
            for sq in BitIterator::new(
                self.precomputed_items.castle_empty_squares[color as usize][castle_option as usize],
            ) {
                if board & (1u64 << sq) != 0 {
                    return false;
                }
            }
            // Check path squares (king must not pass through attack)
            for sq in BitIterator::new(
                self.precomputed_items.castle_path_squares[color as usize][castle_option as usize],
            ) {
                if self.get_attackers(sq, opp_color, board, white_pieces, black_pieces) != 0 {
                    return false;
                }
            }
            return true;
        }

        let mut temp_state = self.clone();
        if 1u64 << to_idx as u64
            & temp_state.get_move_mask(
                from_idx as usize,
                color,
                temp_state.board_arr[from_idx as usize].unwrap().piece_type,
                board,
                white_pieces,
                black_pieces,
            )
            == 0
        {
            return false;
        }
        temp_state.update_board(piece_moved);
        let opponent_color = if color == Color::White {
            Color::Black
        } else {
            Color::White
        };

        // Use updated occupancy from temp state
        let temp_white = temp_state.player1.pieces;
        let temp_black = temp_state.player2.pieces;
        let temp_combined = temp_white | temp_black;

        // King may have moved, recalculate its position
        let temp_king_sq = if color == Color::White {
            temp_state.player1.pieces_bb[PieceType::King as usize].trailing_zeros() as usize
        } else {
            temp_state.player2.pieces_bb[PieceType::King as usize].trailing_zeros() as usize
        };

        if temp_state.get_attackers(
            temp_king_sq,
            opponent_color,
            temp_combined,
            temp_white,
            temp_black,
        ) != 0
        {
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

        // --- Zobrist: XOR out old EP file (if any), XOR side-to-move token ---
        {
            let zt = &self.precomputed_items.zobrist;
            if let Some(ep_sq) = self.en_passant_target {
                self.zobrist_hash ^= zt.en_passant_file[ep_sq % 8];
            }
            // Flip side to move
            self.zobrist_hash ^= zt.side_to_move;
        }

        // --- Zobrist: XOR out moving piece from its source square ---
        {
            let zt = &self.precomputed_items.zobrist;
            self.zobrist_hash ^=
                zt.piece_sq[attacker_color as usize][piece_type as usize][from_idx];
        }

        {
            let player = if attacker_color == Color::White {
                &mut self.player1
            } else {
                &mut self.player2
            };

            player.pieces ^= from_mask; // Remove from start
            player.pieces |= to_mask; // Add to destination
            player.pieces_bb[piece_type as usize] ^= from_mask;
            player.pieces_bb[piece_type as usize] |= to_mask;

            // Update own_board
            player.own_board[to_idx] = player.own_board[from_idx].take();
        }

        // --- Zobrist: XOR out captured piece (normal capture) ---
        if let Some(captured_piece) = self.board_arr[to_idx] {
            let opponent_color = if attacker_color == Color::White {
                Color::Black
            } else {
                Color::White
            };
            let zt = &self.precomputed_items.zobrist;
            self.zobrist_hash ^=
                zt.piece_sq[opponent_color as usize][captured_piece.piece_type as usize][to_idx];

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
                (CastleOption::Kingside, Color::White) => (7usize, 5usize), // h1 -> f1
                (CastleOption::Kingside, Color::Black) => (63usize, 61usize), // h8 -> f8
                (CastleOption::Queenside, Color::White) => (0usize, 3usize), // a1 -> d1
                (CastleOption::Queenside, Color::Black) => (56usize, 59usize), // a8 -> d8
            };

            let rook_from_mask = 1u64 << rook_from;
            let rook_to_mask = 1u64 << rook_to;

            // --- Zobrist: XOR rook out of rook_from, into rook_to ---
            {
                let zt = &self.precomputed_items.zobrist;
                self.zobrist_hash ^=
                    zt.piece_sq[attacker_color as usize][PieceType::Rook as usize][rook_from];
                self.zobrist_hash ^=
                    zt.piece_sq[attacker_color as usize][PieceType::Rook as usize][rook_to];
            }

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

        // Handle en passant capture
        if piece_type == PieceType::Pawn && Some(to_idx) == self.en_passant_target {
            let captured_pawn_idx = if attacker_color == Color::White {
                to_idx - 8
            } else {
                to_idx + 8
            };
            let captured_mask = 1u64 << captured_pawn_idx;
            let opponent_color = if attacker_color == Color::White {
                Color::Black
            } else {
                Color::White
            };
            // --- Zobrist: XOR out the EP-captured pawn ---
            {
                let zt = &self.precomputed_items.zobrist;
                self.zobrist_hash ^= zt.piece_sq[opponent_color as usize][PieceType::Pawn as usize]
                    [captured_pawn_idx];
            }
            let opponent = if attacker_color == Color::White {
                &mut self.player2
            } else {
                &mut self.player1
            };
            opponent.pieces &= !captured_mask;
            opponent.pieces_bb[PieceType::Pawn as usize] &= !captured_mask;
            opponent.own_board[captured_pawn_idx] = None;
            self.board_arr[captured_pawn_idx] = None;
        }

        // 5. Specialized Pawn Logic: En Passant Target Generation
        if piece_type == PieceType::Pawn {
            let diff = (from_idx as i8 - to_idx as i8).abs();
            if diff == 16 {
                let new_ep = if attacker_color == Color::White {
                    from_idx + 8
                } else {
                    from_idx - 8
                };
                // --- Zobrist: XOR in new EP file ---
                self.zobrist_hash ^= self.precomputed_items.zobrist.en_passant_file[new_ep % 8];
                self.en_passant_target = Some(new_ep);
            } else {
                self.en_passant_target = None;
            }
        } else {
            self.en_passant_target = None;
        }

        self.board_arr[to_idx] = self.board_arr[from_idx].take();

        let player = if attacker_color == Color::White {
            &mut self.player1
        } else {
            &mut self.player2
        };

        // Update for promotion
        if let Some(piece) = piece_moved.promotion_piece_type {
            // --- Zobrist: XOR out pawn at to_idx, XOR in promoted piece at to_idx ---
            {
                let zt = &self.precomputed_items.zobrist;
                self.zobrist_hash ^=
                    zt.piece_sq[attacker_color as usize][PieceType::Pawn as usize][to_idx];
                self.zobrist_hash ^= zt.piece_sq[attacker_color as usize][piece as usize][to_idx];
            }
            player.pieces_bb[PieceType::Pawn as usize] ^= 1u64 << to_idx;
            player.pieces_bb[piece as usize] |= 1u64 << to_idx;
            self.board_arr[to_idx] = Some(Piece {
                color: attacker_color,
                piece_type: piece,
            });

            // Update own_board with promoted piece
            player.own_board[to_idx] = Some(Piece {
                color: attacker_color,
                piece_type: piece,
            });
        } else {
            // --- Zobrist: XOR in moving piece at destination (non-promotion) ---
            let zt = &self.precomputed_items.zobrist;
            self.zobrist_hash ^= zt.piece_sq[attacker_color as usize][piece_type as usize][to_idx];
        }
    }

    pub fn get_attackers(
        &self,
        sq: usize,
        attacker_color: Color,
        board: Bitboard,
        white_pieces: Bitboard,
        black_pieces: Bitboard,
    ) -> u64 {
        let mut attackers = 0u64;
        let opponent = if attacker_color == Color::White {
            &self.player1
        } else {
            &self.player2
        };
        let opp_color = if attacker_color == Color::White {
            Color::Black
        } else {
            Color::White
        };
        attackers |= self.get_move_mask(
            sq,
            opp_color,
            PieceType::Pawn,
            board,
            white_pieces,
            black_pieces,
        ) & opponent.pieces_bb[PieceType::Pawn as usize];
        attackers |= self.get_move_mask(
            sq,
            opp_color,
            PieceType::Knight,
            board,
            white_pieces,
            black_pieces,
        ) & opponent.pieces_bb[PieceType::Knight as usize];
        attackers |= self.get_move_mask(
            sq,
            opp_color,
            PieceType::King,
            board,
            white_pieces,
            black_pieces,
        ) & opponent.pieces_bb[PieceType::King as usize];
        attackers |= self.get_move_mask(
            sq,
            opp_color,
            PieceType::Rook,
            board,
            white_pieces,
            black_pieces,
        ) & opponent.pieces_bb[PieceType::Rook as usize];
        attackers |= self.get_move_mask(
            sq,
            opp_color,
            PieceType::Bishop,
            board,
            white_pieces,
            black_pieces,
        ) & opponent.pieces_bb[PieceType::Bishop as usize];
        attackers |= self.get_move_mask(
            sq,
            opp_color,
            PieceType::Queen,
            board,
            white_pieces,
            black_pieces,
        ) & opponent.pieces_bb[PieceType::Queen as usize];

        attackers
    }

    pub fn get_move_mask(
        &self,
        sq: usize,
        color: Color,
        piece: PieceType,
        board: Bitboard,
        white_pieces: Bitboard,
        black_pieces: Bitboard,
    ) -> u64 {
        let own_pieces = if color == Color::White {
            white_pieces
        } else {
            black_pieces
        };
        let enemy_pieces = if color == Color::White {
            black_pieces
        } else {
            white_pieces
        };

        match piece {
            PieceType::Pawn => self.get_pawn_moves(sq, color, enemy_pieces, board),
            PieceType::Knight => self.precomputed_items.knight_moves[sq] & !own_pieces,
            PieceType::King => self.precomputed_items.king_moves[sq] & !own_pieces,
            PieceType::Rook => self.get_sliding_moves(sq, PieceType::Rook, board) & !own_pieces,
            PieceType::Bishop => self.get_sliding_moves(sq, PieceType::Bishop, board) & !own_pieces,
            PieceType::Queen => {
                (self.get_sliding_moves(sq, PieceType::Rook, board)
                    | self.get_sliding_moves(sq, PieceType::Bishop, board))
                    & !own_pieces
            }
        }
    }

    /// Helper to encapsulate Magic Bitboard lookups
    fn get_sliding_moves(&self, sq: usize, piece_type: PieceType, board: Bitboard) -> u64 {
        match piece_type {
            PieceType::Rook => {
                let e = &self.precomputed_items.rook_moves[sq];
                let hash = ((e.mask & board).wrapping_mul(e.magic_num)) >> (64 - e.sig_bits);
                e.magic_table[hash as usize]
            }
            PieceType::Bishop => {
                let e = &self.precomputed_items.bishop_moves[sq];
                let hash = ((e.mask & board).wrapping_mul(e.magic_num)) >> (64 - e.sig_bits);
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
                self.precomputed_items.white_pawn_attacks[sq],
            ),
            Color::Black => (
                self.precomputed_items.black_pawn_pushes[sq],
                self.precomputed_items.black_pawn_attacks[sq],
            ),
        };

        // 1. Calculate Pushes (must handle the "blockade" logic)
        let mut valid_pushes = pushes & !board;

        // If the square immediately in front is blocked, the double-push is also blocked
        let push_direction: i8 = if color == Color::White { 8 } else { -8 };
        let one_step = sq as i8 + push_direction;
        if (0..64).contains(&one_step) {
            let one_step_sq = 1u64 << one_step as usize;
            if (one_step_sq & board) != 0 {
                let double_step = sq as i8 + 2 * push_direction;
                if (0..64).contains(&double_step) {
                    let double_step_sq = 1u64 << double_step as usize;
                    valid_pushes &= !double_step_sq;
                }
            }
        }

        // 2. Calculate Attacks (only moves if an enemy piece is there)
        let mut valid_attacks = attacks & enemy_pieces;
        if let Some(ep_sq) = self.en_passant_target {
            if attacks & (1u64 << ep_sq) != 0 {
                valid_attacks |= 1u64 << ep_sq;
            }
        }

        valid_pushes | valid_attacks
    }

    pub fn process_move(
        &mut self,
        move_str: &str,
        color: Color,
        turn_count: usize,
    ) -> Result<(Move, GameResult), String> {
        let player = if color == Color::White {
            &self.player1
        } else {
            &self.player2
        };
        let candidate = player.parse_move(move_str);

        if !self.validate_move(
            candidate,
            color,
            self.combined_pieces,
            self.white_pieces,
            self.black_pieces,
        ) {
            return Err("Invalid move".to_string());
        }

        self.compute_turn_items(turn_count, candidate);
        Ok((candidate, self.game_result))
    }

    pub fn board_snapshot(&self) -> [Option<Piece>; 64] {
        self.board_arr
    }

    pub fn result(&self) -> GameResult {
        self.game_result
    }

    pub fn bitboard_string(&self) -> String {
        format!(
            "wp={:016x} wn={:016x} wb={:016x} wr={:016x} wq={:016x} wk={:016x} bp={:016x} bn={:016x} bb={:016x} br={:016x} bq={:016x} bk={:016x}",
            self.player1.pieces_bb[PieceType::Pawn as usize],
            self.player1.pieces_bb[PieceType::Knight as usize],
            self.player1.pieces_bb[PieceType::Bishop as usize],
            self.player1.pieces_bb[PieceType::Rook as usize],
            self.player1.pieces_bb[PieceType::Queen as usize],
            self.player1.pieces_bb[PieceType::King as usize],
            self.player2.pieces_bb[PieceType::Pawn as usize],
            self.player2.pieces_bb[PieceType::Knight as usize],
            self.player2.pieces_bb[PieceType::Bishop as usize],
            self.player2.pieces_bb[PieceType::Rook as usize],
            self.player2.pieces_bb[PieceType::Queen as usize],
            self.player2.pieces_bb[PieceType::King as usize],
        )
    }

    fn is_insufficient_material(&self) -> bool {
        let white = &self.player1;
        let black = &self.player2;

        if white.pieces_bb[PieceType::Pawn as usize] != 0
            || black.pieces_bb[PieceType::Pawn as usize] != 0
        {
            return false;
        }
        if white.pieces_bb[PieceType::Rook as usize] != 0
            || black.pieces_bb[PieceType::Rook as usize] != 0
        {
            return false;
        }
        if white.pieces_bb[PieceType::Queen as usize] != 0
            || black.pieces_bb[PieceType::Queen as usize] != 0
        {
            return false;
        }

        let white_knights = white.pieces_bb[PieceType::Knight as usize].count_ones();
        let white_bishops = white.pieces_bb[PieceType::Bishop as usize].count_ones();
        let black_knights = black.pieces_bb[PieceType::Knight as usize].count_ones();
        let black_bishops = black.pieces_bb[PieceType::Bishop as usize].count_ones();

        let white_minor = white_knights + white_bishops;
        let black_minor = black_knights + black_bishops;

        // K vs K
        if white_minor == 0 && black_minor == 0 {
            return true;
        }
        // K+minor vs K
        if (white_minor <= 1 && black_minor == 0) || (white_minor == 0 && black_minor <= 1) {
            return true;
        }
        // K+B vs K+B (same colored bishops)
        if white_knights == 0 && black_knights == 0 && white_bishops == 1 && black_bishops == 1 {
            let wb_sq = white.pieces_bb[PieceType::Bishop as usize].trailing_zeros() as usize;
            let bb_sq = black.pieces_bb[PieceType::Bishop as usize].trailing_zeros() as usize;
            let wb_color = (wb_sq / 8 + wb_sq % 8) % 2;
            let bb_color = (bb_sq / 8 + bb_sq % 8) % 2;
            if wb_color == bb_color {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;
    use crate::PrecomputedItems;
    use std::sync::Arc;

    fn make_board() -> GameBoard {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let p1 = Player::init_player(true);
        let p2 = Player::init_player(false);
        GameBoard::init_game_board(precomputed, p1, p2)
    }

    #[test]
    fn test_zobrist_starting_position() {
        let board1 = make_board();
        let board2 = make_board();
        assert_ne!(board1.zobrist_hash, 0);
        assert_eq!(board1.zobrist_hash, board2.zobrist_hash);
    }

    #[test]
    fn test_zobrist_different_positions() {
        let mut board1 = make_board();
        let board2 = make_board();
        // Move e2 to e4
        let mv = Move {
            from: Square::E2,
            to: Square::E4,
            promotion_piece_type: None,
            castle_option: None,
            en_passant: false,
        };
        board1.compute_turn_items(0, mv);
        assert_ne!(board1.zobrist_hash, board2.zobrist_hash);
    }

    #[test]
    fn test_zobrist_castling_rights_differ() {
        // Two boards, one with modified castling rights
        let board1 = make_board();
        let mut board2 = make_board();
        // Manually toggle white kingside castling right and update hash
        let zt = &board2.precomputed_items.zobrist;
        board2.zobrist_hash ^= zt.castling[0]; // XOR out WK
        board2.white_kingside = false;
        assert_ne!(board1.zobrist_hash, board2.zobrist_hash);
    }

    #[test]
    fn test_zobrist_ep_differs() {
        let board1 = make_board();
        let mut board2 = make_board();
        // Manually set an en passant target on file 4 (e-file)
        let zt = &board2.precomputed_items.zobrist;
        board2.en_passant_target = Some(20); // e3
        board2.zobrist_hash ^= zt.en_passant_file[4]; // e-file = index 4
        assert_ne!(board1.zobrist_hash, board2.zobrist_hash);
    }
}
