//using this to manage sessions & out of game things
use crate::PrecomputedItems;
use crate::game::GameState;
use std::sync::Arc;

pub struct SessionObj {
    precomputed_items: Arc<PrecomputedItems>
}

impl SessionObj {
    pub fn start_session() -> Self{
        let precomputed_items: Arc<PrecomputedItems> = Arc::new(PrecomputedItems::begin_precomputing());
        //define how long you want the game to go on for
        
        SessionObj {precomputed_items }
    }

    pub fn start_games(&self, num_games: u8){
        for i in 0..num_games{
            let mut game_state: GameState = GameState::init_game_state(Arc::clone(&self.precomputed_items));
            game_state.start_game();

        }
    }
}