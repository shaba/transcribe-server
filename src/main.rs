mod config;

use clap::Parser;

fn main() {
    let cfg = config::Config::parse();
    if cfg.verbose {
        println!("{cfg:#?}");
    }
    println!("transcribe-server 0.0.1");
}
