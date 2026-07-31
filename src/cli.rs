use std::process::ExitCode;

use clap::{CommandFactory, Parser};

#[derive(Parser)]
#[command(name = "fnlp", about = "Local Nanbeige4.2-3B NLP toolbox")]
struct Cli;

pub fn cli_main() -> ExitCode {
    let _cli = Cli::parse();
    let mut command = Cli::command();
    let _ = command.print_help();
    println!();
    ExitCode::SUCCESS
}
