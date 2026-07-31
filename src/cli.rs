use std::{
    ffi::OsString,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{CommandFactory, Parser, Subcommand};

use crate::{
    artifact::converter::{
        DEFAULT_PANEL_BYTES, ConvertArch, ConverterError, ConvertRequest, prepare_convert_request,
    },
    artifact::package::{PackageRequest, package_model, verify_model_package},
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
    /// Construct or verify an immutable maintainer model-release package.
    Release {
        #[command(subcommand)]
        command: ReleaseSubcommand,
    },
    /// Prepare a bounded, canonical Generic conversion from the pinned source closure.
    Convert {
        /// Directory containing the pinned ten-file source closure.
        #[arg(long)]
        source: PathBuf,
        /// Authenticated source-closure manifest.
        #[arg(long)]
        source_manifest: PathBuf,
        /// Versioned conversion recipe identity.
        #[arg(long)]
        recipe: String,
        /// Canonical artifact target; only `generic` is admitted.
        #[arg(long)]
        arch: String,
        /// Final canonical `.fnlpq` destination; no output is created until the
        /// streaming envelope is available.
        #[arg(short = 'o', long)]
        output: PathBuf,
        /// Bypass the later interactive confirmation step.
        #[arg(long)]
        yes: bool,
        /// Reject all otherwise-unrelated source-directory entries.
        #[arg(long)]
        strict_source_dir: bool,
        /// Reserve stdout for versioned robot events once conversion execution lands.
        #[arg(long)]
        robot: bool,
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

#[derive(Subcommand)]
enum ReleaseSubcommand {
    /// Split a finished canonical artifact into fixed immutable release parts.
    PackageModel {
        /// Finished canonical Generic `.fnlpq` artifact.
        #[arg(long)]
        artifact: PathBuf,
        /// New, previously absent package staging directory.
        #[arg(long)]
        staging_dir: PathBuf,
        /// Versioned `.fnlpq` basename recorded in the release receipt.
        #[arg(long)]
        logical_artifact_name: String,
        /// Converter receipt retained with the release closure.
        #[arg(long)]
        conversion_receipt: PathBuf,
        /// Directory containing exactly the approved three-file license bundle.
        #[arg(long)]
        license_bundle_dir: PathBuf,
    },
    /// Rehash, reassemble, and validate an immutable release package.
    VerifyModelPackage {
        /// Existing package staging directory to verify.
        #[arg(long)]
        staging_dir: PathBuf,
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
        Some(Command::Release { command }) => run_release_command(command),
        Some(Command::Convert {
            source,
            source_manifest,
            recipe,
            arch,
            output,
            yes,
            strict_source_dir,
            robot,
        }) => run_convert_command(
            source,
            source_manifest,
            recipe,
            arch,
            output,
            yes,
            strict_source_dir,
            robot,
        ),
        None => {
            let mut command = Cli::command();
            let _ = command.print_help();
            println!();
            ExitCode::SUCCESS
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_convert_command(
    source_dir: PathBuf,
    source_manifest: PathBuf,
    recipe_id: String,
    arch: String,
    output: PathBuf,
    yes: bool,
    strict_source_dir: bool,
    robot: bool,
) -> ExitCode {
    let arch = match ConvertArch::parse(&arch) {
        Ok(arch) => arch,
        Err(error) => return emit_convert_refusal(error),
    };
    let request = ConvertRequest {
        source_dir,
        source_manifest,
        recipe_id,
        arch,
        output,
        yes,
        strict_source_dir,
        robot,
    };

    match prepare_convert_request(&request, DEFAULT_PANEL_BYTES) {
        Ok(prepared) => {
            eprintln!(
                "CONVERT RESULT=BLOCKED stage=streaming-envelope source-root-sha256={} census-sha256={} tensors={} reason=canonical-v1-writer-requires-an-in-memory-envelope; no-output-created",
                prepared.source.source_root_sha256,
                prepared.census_sha256,
                prepared.census.len(),
            );
            ErrorCode::Generic.as_process_exit()
        }
        Err(error) => emit_convert_refusal(error),
    }
}

fn emit_convert_refusal(error: ConverterError) -> ExitCode {
    eprintln!("CONVERT RESULT=FAIL stage=admission reason={error}");
    match error {
        ConverterError::InvalidConvertArgument { .. } => ErrorCode::Usage.as_process_exit(),
        _ => ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit(),
    }
}

fn run_release_command(command: ReleaseSubcommand) -> ExitCode {
    match command {
        ReleaseSubcommand::PackageModel {
            artifact,
            staging_dir,
            logical_artifact_name,
            conversion_receipt,
            license_bundle_dir,
        } => match package_model(&PackageRequest {
            artifact,
            staging_dir,
            logical_artifact_name,
            conversion_receipt,
            license_bundle_dir,
        }) {
            Ok(report) => {
                eprintln!(
                    "RELEASE_PACKAGE RESULT=PASS staging_dir={} artifact_bytes={} artifact_sha256={} parts={}",
                    report.staging_dir.display(),
                    report.artifact_bytes,
                    report.artifact_sha256,
                    report.parts.len(),
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("RELEASE_PACKAGE RESULT=FAIL reason={error}");
                ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()
            }
        },
        ReleaseSubcommand::VerifyModelPackage { staging_dir } => {
            match verify_model_package(&staging_dir) {
                Ok(report) => {
                    eprintln!(
                        "RELEASE_VERIFY RESULT=PASS staging_dir={} artifact_bytes={} artifact_sha256={} parts={}",
                        report.staging_dir.display(),
                        report.artifact_bytes,
                        report.artifact_sha256,
                        report.parts.len(),
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("RELEASE_VERIFY RESULT=FAIL reason={error}");
                    ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()
                }
            }
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

    use clap::Parser;

    use super::{Cli, Command, cli_main_with_reader};

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

    #[test]
    fn release_package_commands_have_only_explicit_named_paths() {
        let package = Cli::try_parse_from([
            "fnlp",
            "release",
            "package-model",
            "--artifact",
            "/artifacts/model.fnlpq",
            "--staging-dir",
            "/staging/model-v1",
            "--logical-artifact-name",
            "model.fnlpq",
            "--conversion-receipt",
            "/receipts/conversion.json",
            "--license-bundle-dir",
            "/licenses/model-v1",
        ])
        .expect("explicit package command parses");
        assert!(matches!(
            package.command,
            Some(Command::Release {
                command: ReleaseSubcommand::PackageModel { .. }
            })
        ));

        let verify = Cli::try_parse_from([
            "fnlp",
            "release",
            "verify-model-package",
            "--staging-dir",
            "/staging/model-v1",
        ])
        .expect("explicit verify command parses");
        assert!(matches!(
            verify.command,
            Some(Command::Release {
                command: ReleaseSubcommand::VerifyModelPackage { .. }
            })
        ));
    }

    #[test]
    fn convert_reference_invocation_requires_every_named_authority() {
        let convert = Cli::try_parse_from([
            "fnlp",
            "convert",
            "--source",
            "/models/nanbeige-source",
            "--source-manifest",
            "docs/truth-pack/nanbeige4.2-3b.source.json",
            "--recipe",
            "nanbeige42-int8-v1",
            "--arch",
            "generic",
            "-o",
            "nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq",
        ])
        .expect("reference conversion invocation parses");
        assert!(matches!(convert.command, Some(Command::Convert { .. })));
    }
}
