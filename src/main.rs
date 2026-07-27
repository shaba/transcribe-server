mod api;
mod audio;
mod auth;
mod chunk;
mod config;
mod engine;

use clap::Parser;

fn main() {
    let cfg = config::Config::parse();
    if cfg.verbose {
        println!("{cfg:#?}");
    }
    println!("transcribe-server 0.0.1");
}
