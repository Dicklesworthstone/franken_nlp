#![deny(unsafe_code)]

pub mod artifact;
pub mod batch;
pub mod cli;
pub mod error;
pub mod grammar;
pub mod jobs;
pub mod native_engine;
pub mod orchestrator;
pub mod robot;
pub mod storage;
pub mod tasks;
pub mod template;
pub mod textutil;
pub mod tokenizer;
pub mod validation;

pub fn cli_main() -> std::process::ExitCode {
    cli::cli_main()
}
