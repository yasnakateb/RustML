extern crate serde;
extern crate serde_derive;

use std::process::exit;

mod linear_regression;

fn main() {
    let res = linear_regression::run();
    exit(match res {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}