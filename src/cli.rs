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
    let command = existing::definition().subcommand(crate::redaction_cli::definition())
        .subcommands(crate::text_cli::definitions()).subcommand(crate::text_batch::definition())
        .subcommand(crate::candidate_cli::definition());
    #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    let command = command.subcommand(crate::job_cli::definition());
    command
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
            if args.get(1).and_then(|s| s.to_str()).is_some_and(|s| s == "redact" || s == "batch" || s == "job" || s == "candidate" || crate::text_cli::TextCommand::recognizes(s)) {
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
    #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    if let Some(("job", matches)) = matches.subcommand() {
        let options = match crate::job_cli::JobCommand::from_arg_matches(matches) {
            Ok(options) => options,
            Err(_) => return ErrorCode::Usage.as_process_exit(),
        };
        return options.run(&mut io::stdout().lock(), &mut io::stderr().lock());
    }
    // Candidate corpus execution transfers owned IO into the blocking pool.
    // Holding a caller-thread stdio lock here would deadlock its worker.
    if let Some(("candidate", matches)) = matches.subcommand() {
        let options = match crate::candidate_cli::CandidateCommand::from_matches(matches) {
            Ok(options) => options,
            Err(_) => return ErrorCode::Usage.as_process_exit(),
        };
        return options.run_stdio();
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
        for name in ["redact", "tokens", "split", "normalize", "batch", "schema", "robot", "convert", "release", "models", "candidate"] { assert!(help.contains(name)); }
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
    #[test]
    fn candidate_commands_are_explicit_and_do_not_replace_certified_task_names() {
        for task in ["generate", "chat", "ner", "keyphrases", "summarize", "answer"] {
            assert!(definition().try_get_matches_from(["fnlp", "candidate", task,
                "--model", "local.fnlpq", "--memory-mib", "8192"]).is_ok());
        }
    }
    #[test]
    fn native_candidate_batch_is_separate_from_model_free_text_batch() {
        for task in ["ner", "keyphrases", "summarize", "answer"] {
            let matches = definition().try_get_matches_from(["fnlp", "candidate", "batch",
                "--task", task, "--model", "local.fnlpq", "--memory-mib", "8192"]).unwrap();
            let ("candidate", inner) = matches.subcommand().unwrap() else { panic!("wrong route") };
            assert!(crate::candidate_cli::CandidateCommand::from_matches(inner).is_ok());
        }
        assert!(definition().try_get_matches_from(["fnlp", "batch", "--task", "normalize"]).is_ok());
    }
    #[test]
    fn stored_job_commands_are_present_only_with_the_actual_host_and_storage_profile() {
        let supported = cfg!(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")));
        assert_eq!(definition().try_get_matches_from(["fnlp", "job", "schema"]).is_ok(), supported);
    }
}
