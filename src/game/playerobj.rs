use crate::{Bitboard, Color, CastleOption, Piece};
use crate::game::Move;
use crate::PieceType;
use rand::Rng;
use crate::create_own_board;
use tokio::net::TcpStream;


#[derive(Debug, Clone)]
pub struct Player {
    pub (crate)color: Color,
    pub (crate)pieces_bb: [u64; 6], // [Pawn, Knight, Bishop, Rook, Queen, King]
    pub (crate)pieces: u64,         // All pieces combined (occupancy)
    pub (crate)own_board: [Option<Piece>; 64],
    stream: TcpStream

}

impl Player {

    pub fn init_player(is_white: bool, stream: TcpStream) -> Self {
        if is_white {
            return Self::new_white();
        }
        return Self::new_black();
    }

    pub fn new_white() -> Self {
        let bbs = [
            0x000000000000FF00, // 0: Pawns
            0x0000000000000042, // 1: Knights
            0x0000000000000024, // 2: Bishops
            0x0000000000000081, // 3: Rooks
            0x0000000000000008, // 4: Queen
            0x0000000000000010, // 5: King
        ];

        Self {
            color: Color::White,
            pieces_bb: bbs,
            pieces: 0x000000000000FFFF, 
            own_board: create_own_board(Color::White)
        }
    }

    pub fn get_piece_type_at(&self, from_idx: u8) -> Option<PieceType> {
        return Some(self.own_board[from_idx as usize].unwrap().piece_type)
    }

    fn parse_move(&self, notation: String) -> Move {
        let chars: Vec<char> = notation.chars().collect();
        
        if chars.len() < 4 {
            panic!("Invalid move notation: {}", notation);
        }
        
        // Parse from square
        let from_file = (chars[0] as u8 - b'a') as u8;
        let from_rank = (chars[1].to_digit(10).unwrap() - 1) as u8;
        let from_idx = from_rank * 8 + from_file;
        
        // Parse to square
        let to_file = (chars[2] as u8 - b'a') as u8;
        let to_rank = (chars[3].to_digit(10).unwrap() - 1) as u8;
        let to_idx = to_rank * 8 + to_file;
        
        // All logic here
        let piece_type = self.get_piece_type_at(from_idx);
        
        // Check for castling
        let castle_option = if piece_type == Some(PieceType::King) {
            let file_diff = (to_file as i8 - from_file as i8).abs();
            if file_diff == 2 {
                if to_file > from_file {
                    Some(CastleOption::Kingside)
                } else {
                    Some(CastleOption::Queenside)
                }
            } else {
                None
            }
        } else {
            None
        };
        
        // Check for promotion
        let promotion_piece_type = if piece_type == Some(PieceType::Pawn) && (to_rank == 7 || to_rank == 0) {
            // If promotion rank and there's a 5th character, use it; otherwise default to Queen
            if chars.len() == 5 {
                match chars[4] {
                    'q' | 'Q' => Some(PieceType ::Queen),
                    'r' | 'R' => Some(PieceType::Rook),
                    'b' | 'B' => Some(PieceType::Bishop),
                    'n' | 'N' => Some(PieceType::Knight),
                    _ => Some(PieceType::Queen), // Default
                }
            } else {
                Some(PieceType::Queen) // Default promotion
            }
        } else {
            None
        };
        
        Move {
            from: from_idx.into(),
            to: to_idx.into(),
            promotion_piece_type,
            castle_option,
            en_passant: None,
        }
    }

    pub fn new_black() -> Self {
        let bbs = [
            0x00FF000000000000, // 0: Pawns
            0x4200000000000000, // 1: Knights
            0x2400000000000000, // 2: Bishops
            0x8100000000000000, // 3: Rooks
            0x0800000000000000, // 4: Queen
            0x1000000000000000, // 5: King
        ];

        Self {
            color: Color::Black,
            pieces_bb: bbs,
            pieces: 0xFFFF000000000000,
            own_board: create_own_board(Color::Black)

        }
    }

    pub fn make_move(&self) -> Move {
        self.parse_move(self.player.get_move())
    }

}