//using this to intialize session vars like precomputed items, mcst info, etc
use crate::PrecomputedItems;
use crate::game::GameState;
use std::sync::Arc;

pub struct SessionObj {
    precomputed_items: Arc<PrecomputedItems>,
    

}

impl SessionObj {
    pub fn start_session(precomputed_items: Arc<PrecomputedItems>) -> Self{
        //create pre_computed items list
        SessionObj {precomputed_items }
    }
}