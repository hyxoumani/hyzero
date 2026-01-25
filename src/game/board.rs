use crate::game::Player;
#[derive(Debug)]
pub struct GameBoard {
    pub(crate) player1: Player,
    pub(crate) player2: Player
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