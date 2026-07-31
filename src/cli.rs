use std::{io, process::ExitCode};

use clap::{CommandFactory, Parser, Subcommand};

use crate::robot::{self, RobotCommand};

#[derive(Parser)]
#[command(name = "fnlp", about = "Local Nanbeige4.2-3B NLP toolbox")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Agent-facing, versioned NDJSON interface.
    Robot {
        #[command(subcommand)]
        command: RobotSubcommand,
    },
}

#[derive(Subcommand)]
enum RobotSubcommand {
    /// Emit the frozen, versioned robot schema.
    Schema,
    /// Emit an honest unpopulated health skeleton.
    Health,
    /// Emit an honest unpopulated backend inventory skeleton.
    Backends,
}

impl From<RobotSubcommand> for RobotCommand {
    fn from(command: RobotSubcommand) -> Self {
        match command {
            RobotSubcommand::Schema => Self::Schema,
            RobotSubcommand::Health => Self::Health,
            RobotSubcommand::Backends => Self::Backends,
        }
    }
}

pub fn cli_main() -> ExitCode {
    match Cli::parse().command {
        Some(Command::Robot { command }) => {
            let mut stdout = io::stdout().lock();
            let mut stderr = io::stderr().lock();
            match robot::write_command(&mut stdout, &mut stderr, command.into()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("fnlp: robot output failure: {error}");
                    ExitCode::from(1)
                }
            }
        }
        None => {
            let mut command = Cli::command();
            let _ = command.print_help();
            println!();
            ExitCode::SUCCESS
        }
    }
}
