use clap::Parser;
use transcribe_server::config::Config;

fn main() {
    let cfg = Config::parse();
    if cfg.verbose {
        println!("{cfg:#?}");
    }
    println!("transcribe-server 0.0.1");
}
