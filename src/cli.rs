use std::{
    ffi::OsString,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{CommandFactory, Parser, Subcommand};

use crate::{
    error::ErrorCode,
    grammar::{CompileLimits, CompiledSchema, SchemaError, compile_json_schema},
    robot::{self, RobotCommand},
};

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
    /// Compile and inspect the bounded v1 JSON-Schema subset without a model.
    Schema {
        #[command(subcommand)]
        command: SchemaSubcommand,
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

#[derive(Subcommand)]
enum SchemaSubcommand {
    /// Compile + resource-check a schema; `-` reads schema JSON from stdin.
    Check {
        #[arg(value_name = "SCHEMA")]
        schema: PathBuf,
    },
    /// Emit one canonical valid instance; `-` reads schema JSON from stdin.
    Sample {
        #[arg(value_name = "SCHEMA")]
        schema: PathBuf,
    },
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
    let canonical_args = std::iter::once(OsString::from("fnlp")).chain(std::env::args_os().skip(1));
    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    cli_main_with_reader(canonical_args, &mut stdin)
}

/// Runs the CLI with an explicit schema-input reader.
///
/// Keeping the reader at this boundary prevents unit tests from accidentally
/// inheriting the process stdin and waiting for a terminal or pipe to close.
fn cli_main_with_reader(
    args: impl IntoIterator<Item = OsString>,
    schema_input: &mut impl Read,
) -> ExitCode {
    match Cli::parse_from(args).command {
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
        Some(Command::Schema { command }) => run_schema_command_with_reader(command, schema_input),
        None => {
            let mut command = Cli::command();
            let _ = command.print_help();
            println!();
            ExitCode::SUCCESS
        }
    }
}

fn run_schema_command_with_reader(
    command: SchemaSubcommand,
    schema_input: &mut impl Read,
) -> ExitCode {
    let (mode, path) = match command {
        SchemaSubcommand::Check { schema } => ("check", schema),
        SchemaSubcommand::Sample { schema } => ("sample", schema),
    };
    let source = match read_schema_source_from(&path, schema_input) {
        Ok(source) => source,
        Err(error) => {
            eprintln!(
                "SCHEMA mode={mode} RESULT=FAIL pointer=$ keyword=none reason=input-read:{error}"
            );
            return ErrorCode::InputDecodeOrParse.as_process_exit();
        }
    };
    let compiled = match compile_json_schema(&source, CompileLimits::default()) {
        Ok(compiled) => compiled,
        Err(error) => {
            emit_schema_error(mode, &error);
            return ErrorCode::SchemaOrRecipeCompile.as_process_exit();
        }
    };
    if mode == "sample" {
        match compiled.sample_json() {
            Ok(sample) => println!("{sample}"),
            Err(error) => {
                emit_schema_error(mode, &error);
                return ErrorCode::SchemaOrRecipeCompile.as_process_exit();
            }
        }
    }
    emit_schema_success(mode, &compiled);
    ExitCode::SUCCESS
}

fn read_schema_source_from(path: &Path, schema_input: &mut impl Read) -> io::Result<String> {
    if path == Path::new("-") {
        let mut source = String::new();
        schema_input.read_to_string(&mut source)?;
        Ok(source)
    } else {
        fs::read_to_string(path)
    }
}

fn emit_schema_success(mode: &str, compiled: &CompiledSchema) {
    let estimate = compiled.estimate();
    let source_product = if compiled.requires_verbatim_source() {
        "REQUIRES_OQ_17"
    } else {
        "NONE"
    };
    eprintln!(
        "SCHEMA mode={mode} RESULT=PASS states={} transitions={} mask_bytes={} enum_trie_nodes={} number_lexers={} min_output_bytes={} source_product={source_product}",
        estimate.state_count,
        estimate.transition_count,
        estimate.mask_cache_bytes,
        estimate.enum_trie_nodes,
        estimate.number_lexers,
        estimate.minimum_output_bytes,
    );
}

fn emit_schema_error(mode: &str, error: &SchemaError) {
    let keyword = error.keyword().unwrap_or("none");
    eprintln!(
        "SCHEMA mode={mode} RESULT=FAIL pointer={} keyword={keyword} reason={error}",
        error.pointer(),
    );
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, io::Cursor, process::ExitCode};

    use super::cli_main_with_reader;

    #[test]
    fn schema_dash_uses_the_injected_reader() {
        let args = [
            OsString::from("fnlp"),
            OsString::from("schema"),
            OsString::from("check"),
            OsString::from("-"),
        ];
        let mut schema_input = Cursor::new(br#"{"type":"string"}"#);

        assert_eq!(
            cli_main_with_reader(args, &mut schema_input),
            ExitCode::SUCCESS
        );
    }
}
