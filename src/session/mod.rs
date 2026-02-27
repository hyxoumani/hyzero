//using this to intialize session vars like precomputed items, mcst info, etc
use crate::PrecomputedItems;
use crate::game::GameState;
use std::sync::Arc;

pub struct SessionObj {
    precomputed_items: Arc<PrecomputedItems>,
    
    
}

impl SessionObj {
    pub fn start_session() -> Self{
        //create pre_computed items list
        let precomputed_items: Arc<PrecomputedItems> = Arc::new(PrecomputedItems::begin_precomputing());        
        SessionObj {precomputed_items }
    }

    /*
    pub fn start_games(&self, num_games: u8){
        for i in 0..num_games{
            let mut game_state: GameState = GameState::init_game_state(Arc::clone(&self.precomputed_items));
            game_state.start_game();

        }
    }
        */
}