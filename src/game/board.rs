use crate::game::Player;

pub struct GameBoard {
    player1: Player,
    player2: Player
}

impl GameBoard{
    pub fn start_game()-> Self{
        Self 
        {
            player1: Player::new_white(),
            player2: Player::new_black()
        }  
    }
}