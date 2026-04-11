use hyzero::session::SessionObj;
use hyzero::PrecomputedItems;
use std::sync::Arc;

fn main() {

    let precomputed_items = Arc::new(PrecomputedItems::begin_precomputing());
    let _session_obj : SessionObj = SessionObj::start_session(precomputed_items);

    let mut test_vec: Vec<Option<Test>> = Vec::new();

    for i in 0..5 {
        test_vec.push(Some(Test::new(i)));
    }

    for item in test_vec.iter().take(5) {
        println!("{:?}", item.as_ref().unwrap());
    }



}

#[derive(Debug)]
pub struct Test {
    #[allow(dead_code)]
    val: u8
}

impl Test {
    pub fn new (i: u8) -> Self {
        Self {val: i}
    } 
}