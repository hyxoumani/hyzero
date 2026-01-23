pub mod pawn;

pub struct Test {
    pub attribute: i32,
    pub attribute1: i8
}

impl Test {
    pub fn new(attribute: Test) -> Self {
        Self { 
            attribute: attribute.attribute, 
            attribute1: attribute.attribute1 
        }
    }
}

pub fn test_fn(x: Test) {
    println!("this is a test crate, number : {} number2 : {}", x.attribute, x.attribute1);
}
