pub mod board;
pub mod player;

pub use player::Player;
pub use board::GameBoard;
use crate::{Bitboard, Color, PieceType, Square, Piece};

#[derive(Debug)]
pub struct GameState {
    game_state: GameBoard,
    board_arr: [Option<Piece>; 64]
}

impl GameState{
    pub fn init_game() -> Self{
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
            board_arr: board
        }
    }
    pub fn get_game_board(&self) -> &GameBoard {
        &self.game_state
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

        self.board_arr[usize::from(to)] = Some(piece_temp_to);



        //2 scenarios
        //1 board is emtpy
        //2 board is not empty, then have to capture piece and re-calculate checks, for now will focus on just updating game states



    }
}
