use hyzero::init_game;

fn main() {

    init_game();
    
    let mut test_vec: Vec<Option<Test>> = Vec::new();

    for i in 0..5 {
        test_vec.push(Some(Test::new(i)));
    }

    for i in 0..5 {
        println!("{:?}", test_vec[i].as_ref().unwrap());
    }



}

#[derive(Debug)]
pub struct Test {
    val: u8
}

impl Test {
    pub fn new (i: u8) -> Self {
        Self {val: i}
    } 
}