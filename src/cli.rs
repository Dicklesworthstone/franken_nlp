//! Public command dispatch. Existing maintainer commands remain byte-identical
//! in cli_existing.rs; task commands share their root help and argument parser.
use std::{ffi::OsString, io::{self, BufRead, IsTerminal, Write}, process::ExitCode};
use clap::FromArgMatches;
use crate::{error::ErrorCode, redaction_cli::RedactCommand};

mod existing {
    include!("cli_existing.rs");

    pub(super) fn definition() -> clap::Command { Cli::command() }
    pub(super) fn dispatch(args: Vec<OsString>, input: &mut impl BufRead, terminal: bool) -> ExitCode {
        cli_main_with_reader_and_terminal(args, input, terminal)
    }
}

fn definition() -> clap::Command {
    existing::definition().subcommand(crate::redaction_cli::definition())
        .subcommands(crate::text_cli::definitions()).subcommand(crate::text_batch::definition())
}

pub fn cli_main() -> ExitCode {
    let args: Vec<_> = std::iter::once(OsString::from("fnlp")).chain(std::env::args_os().skip(1)).collect();
    let mut command = definition();
    let matches = match command.clone().try_get_matches_from(&args) {
        Ok(matches) => matches,
        Err(error) => {
            if !error.use_stderr() { return if error.print().is_ok() { ExitCode::SUCCESS } else { ErrorCode::Generic.as_process_exit() }; }
            // Never echo arbitrary argv values into redaction diagnostics. Key
            // bytes have no argv option, even on an invalid invocation.
            if args.get(1).and_then(|s| s.to_str()).is_some_and(|s| s == "redact" || s == "batch" || crate::text_cli::TextCommand::recognizes(s)) {
                eprintln!("fnlp: invalid task arguments; run the task command with --help");
            } else { let _ = error.print(); }
            return ErrorCode::Usage.as_process_exit();
        }
    };
    if matches.subcommand().is_none() {
        let mut out = io::stdout().lock();
        return if command.write_help(&mut out).and_then(|()| writeln!(out)).and_then(|()| out.flush()).is_ok() {
            ExitCode::SUCCESS
        } else { ErrorCode::Generic.as_process_exit() };
    }
    let stdin = io::stdin();
    let terminal = stdin.is_terminal();
    let mut input = stdin.lock();
    if let Some(("batch", matches)) = matches.subcommand() {
        let options = match crate::text_batch::BatchCommand::from_arg_matches(matches) {
            Ok(options) => options,
            Err(_) => return ErrorCode::Usage.as_process_exit(),
        };
        return options.run(&mut input, &mut io::stdout().lock(), &mut io::stderr().lock());
    }
    if let Some(("redact", matches)) = matches.subcommand() {
        let options = match RedactCommand::from_arg_matches(matches) {
            Ok(options) => options,
            Err(_) => return ErrorCode::Usage.as_process_exit(),
        };
        return crate::redaction_cli::run(options, &mut input, &mut io::stdout().lock(), &mut io::stderr().lock());
    }
    if let Some((name, matches)) = matches.subcommand() {
        if crate::text_cli::TextCommand::recognizes(name) {
            let task = match crate::text_cli::TextCommand::from_matches(name, matches) {
                Ok(task) => task, Err(_) => return ErrorCode::Usage.as_process_exit(),
            };
            return task.run(&mut input, &mut io::stdout().lock(), &mut io::stderr().lock());
        }
    }
    existing::dispatch(args, &mut input, terminal)
}

#[cfg(test)]
mod task_dispatch_tests {
    use super::*;
    #[test]
    fn new_task_and_existing_commands_share_root_help() {
        let mut root = definition();
        let help = root.render_long_help().to_string();
        for name in ["redact", "tokens", "split", "normalize", "batch", "schema", "robot", "convert", "release", "models"] { assert!(help.contains(name)); }
        assert!(root.try_get_matches_from(["fnlp", "redact", "--rules-only"]).is_ok());
    }
    #[test]
    fn batch_tasks_share_root_dispatch_without_replacing_existing_commands() {
        for task in ["normalize", "split"] {
            assert!(definition().try_get_matches_from(["fnlp", "batch", "--task", task]).is_ok());
        }
    }
    #[test]
    fn existing_schema_arguments_still_parse() {
        assert!(definition().try_get_matches_from(["fnlp", "schema", "check", "-"]).is_ok());
        assert!(definition().try_get_matches_from(["fnlp", "robot", "schema"]).is_ok());
    }
}
