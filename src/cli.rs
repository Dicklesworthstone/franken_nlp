use std::{
    ffi::OsString,
    fs,
    io::{self, BufRead, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use sha2::{Digest, Sha256};

use crate::{
    artifact::converter::{
        CONVERSION_ARTIFACT_FORMAT, CONVERSION_RECEIPT_SCHEMA, ConversionReceipt, ConvertArch,
        ConvertRequest, ConverterError, DEFAULT_PANEL_BYTES, GENERIC_PACKING_V1,
        GenericPayloadPlan, PINNED_REVISION, PINNED_SOURCE_MANIFEST_SHA256, PORTABLE_QUANT_V1,
        PreparedConversionInput, prepare_convert_request, stream_routed_bf16_panels,
    },
    artifact::format::{
        ArchTarget, CanonicalDtype, FnlpqStreamingInput, FnlpqWriteError,
        LogicalTensorStreamingHasher, PackingSetInput, SectionKind, SectionRange, StreamedFnlpq,
        StreamingSection, StreamingSectionHasher, TensorInput, digest_domain, framed_sha256,
        logical_model_sha256, validate_authority_identifier, write_streaming,
    },
    artifact::fs_tx::open_ratified_model_root,
    artifact::package::{PackageRequest, package_model, verify_model_package},
    artifact::packing::{NativePackingTarget, TILE_TABLE_VERSION_V1},
    artifact::quantize::{GenericPanelBytes, encode_generic_panel},
    artifact::reader::FnlpqArtifact,
    artifact::safetensors::{RowPanel, TensorCensusEntry},
    error::ErrorCode,
    grammar::{CompileLimits, CompiledSchema, SchemaError, compile_json_schema},
    orchestrator::{AdmissionRequest, KvCacheQuantization, ResidencyAccounting},
    robot::{self, RobotCommand},
};
use clap::{CommandFactory, Parser, Subcommand};

const GENERIC_PAYLOAD_SECTION: &str = "generic-payload";
const GENERIC_SCALES_SECTION: &str = "generic-scales";
const GENERIC_ROW_SUMS_SECTION: &str = "generic-row-sums";
const TOKENIZER_MODEL_SECTION: &str = "tokenizer-model";
const MODEL_CONFIG_SECTION: &str = "model-config";
const TOKENIZER_CONFIG_SECTION: &str = "tokenizer-config";
const CHAT_TEMPLATE_SECTION: &str = "chat-template";
const LICENSE_BUNDLE_SECTION: &str = "license-bundle";
const EMISSION_PROGRESS_TENSOR_INTERVAL: usize = 16;

const EMBEDDED_LICENSE_FILES: [(&str, &[u8]); 3] = [
    (
        "APACHE-2.0.txt",
        include_bytes!("../docs/truth-pack/license/APACHE-2.0.txt"),
    ),
    (
        "ATTRIBUTION.txt",
        include_bytes!("../docs/truth-pack/license/ATTRIBUTION.txt"),
    ),
    (
        "MODIFICATION_NOTICE.txt",
        include_bytes!("../docs/truth-pack/license/MODIFICATION_NOTICE_TEMPLATE.txt"),
    ),
];

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
    Convert(ConvertCommand),
    /// Inspect or derive local representations of an installed Generic model.
    Models {
        #[command(subcommand)]
        command: ModelsSubcommand,
    },
}

#[derive(clap::Args)]
struct ConvertCommand {
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
    /// Final canonical `.fnlpq` destination; no final-path entry is created
    /// until a complete staged streaming envelope is ready to publish.
    #[arg(short = 'o', long)]
    output: PathBuf,
    /// Exact lowercase Git commit that built this converter binary, retained
    /// in the adjacent canonical conversion receipt.
    #[arg(long)]
    converter_commit: String,
    /// Bypass the interactive TTY y/N confirmation after source preflight.
    ///
    /// Robot and non-TTY invocations never prompt; their noninteractive policy
    /// is recorded in the conversion stage transcript.
    #[arg(long)]
    yes: bool,
    /// Reject all otherwise-unrelated source-directory entries.
    #[arg(long)]
    strict_source_dir: bool,
    /// Reserve stdout for versioned robot events once conversion execution lands.
    #[arg(long)]
    robot: bool,
}

#[derive(Subcommand)]
enum ModelsSubcommand {
    /// Derive one target-specific native cache from an immutable Generic root.
    Derive(ModelsDeriveCommand),
}

#[derive(clap::Args)]
struct ModelsDeriveCommand {
    /// Immutable Generic `.fnlpq` reconstruction root.
    #[arg(long)]
    generic: PathBuf,
    /// Closed native packing target.
    #[arg(long)]
    arch: String,
    /// Owner-controlled model root that owns the content-addressed native cache.
    #[arg(long)]
    model_dir: PathBuf,
    /// Immutable target tile-table revision.
    #[arg(long, default_value = TILE_TABLE_VERSION_V1)]
    tile_table_version: String,
}

#[derive(Subcommand)]
enum RobotSubcommand {
    /// Emit the frozen, versioned robot schema.
    Schema,
    /// Emit an honest unpopulated health skeleton.
    Health,
    /// Emit an honest unpopulated backend inventory skeleton.
    Backends,
    /// Print a non-allocating admission-certificate calculation.
    Plan(RobotPlanCommand),
}

#[derive(clap::Args)]
struct RobotPlanCommand {
    /// Requested per-sequence context. The default usable cap is 8192 tokens.
    #[arg(long, default_value_t = 8_192)]
    ctx: u64,
    /// Number of simultaneous sequence rows in this bounded admission group.
    #[arg(long, default_value_t = 1)]
    batch: u64,
    /// Closed K/V-cache accounting profile: bf16, int8, or int8-f16-scales.
    #[arg(long, default_value = "bf16")]
    quant: String,
    /// Maximum modeled process commitment. `--memory-budget` is an exact alias.
    #[arg(long = "memory-budget-total", visible_alias = "memory-budget")]
    memory_budget_total: Option<u64>,
    /// Non-allocatable operating-system reserve within the local budget.
    #[arg(long, default_value_t = 0)]
    memory_reserve_os: u64,
    /// Exact bytes mapped for active model packing, tokenizer, immutable caches, and runtime.
    #[arg(long)]
    fixed_mapped_bytes: Option<u64>,
    /// Exact resident commitment for the active fixed mapping.
    #[arg(long)]
    fixed_resident_bytes: Option<u64>,
    /// Exact elastic cache allowance outside the per-row request state.
    #[arg(long, default_value_t = 0)]
    elastic_cache_bytes: u64,
    /// Optional extra mapping for replicated/NUMA-local weights.
    #[arg(long, default_value_t = 0)]
    replicated_weight_mapped_bytes: u64,
    /// Optional extra resident commitment for replicated/NUMA-local weights.
    #[arg(long, default_value_t = 0)]
    replicated_weight_resident_bytes: u64,
    /// Exact K/V allocator padding and page-table bytes for one token row.
    #[arg(long)]
    kv_page_metadata_per_token: Option<u64>,
    #[arg(long, default_value_t = 0)]
    activation_bytes_per_row: u64,
    #[arg(long, default_value_t = 0)]
    grammar_state_bytes_per_row: u64,
    #[arg(long, default_value_t = 0)]
    source_state_bytes_per_row: u64,
    #[arg(long, default_value_t = 0)]
    queue_bytes_per_row: u64,
    #[arg(long, default_value_t = 0)]
    output_buffer_bytes_per_row: u64,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    unmodeled_emergency_reserve_bytes: u64,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    safety_margin_bytes: u64,
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

impl TryFrom<RobotSubcommand> for RobotCommand {
    type Error = &'static str;

    fn try_from(command: RobotSubcommand) -> Result<Self, Self::Error> {
        match command {
            RobotSubcommand::Schema => Ok(Self::Schema),
            RobotSubcommand::Health => Ok(Self::Health),
            RobotSubcommand::Backends => Ok(Self::Backends),
            RobotSubcommand::Plan(command) => command.admission_request().map(Self::Plan),
        }
    }
}

impl RobotPlanCommand {
    fn admission_request(self) -> Result<AdmissionRequest, &'static str> {
        let quantization = KvCacheQuantization::parse(&self.quant)
            .ok_or("--quant must be bf16, int8, or int8-f16-scales")?;
        let fixed_residency = match (self.fixed_mapped_bytes, self.fixed_resident_bytes) {
            (None, None) => None,
            (Some(mapped_bytes), Some(resident_bytes)) => Some(
                ResidencyAccounting::new(mapped_bytes, resident_bytes)
                    .ok_or("--fixed-resident-bytes cannot exceed --fixed-mapped-bytes")?,
            ),
            _ => {
                return Err(
                    "--fixed-mapped-bytes and --fixed-resident-bytes must be supplied together",
                );
            }
        };
        let replicated_weight_residency = ResidencyAccounting::new(
            self.replicated_weight_mapped_bytes,
            self.replicated_weight_resident_bytes,
        )
        .ok_or(
            "--replicated-weight-resident-bytes cannot exceed --replicated-weight-mapped-bytes",
        )?;
        let mut request = AdmissionRequest::decode(self.ctx, self.batch, quantization)
            .with_os_reserve(self.memory_reserve_os)
            .with_elastic_cache(self.elastic_cache_bytes)
            .with_replicated_weight_residency(replicated_weight_residency)
            .with_elastic_rows(
                self.activation_bytes_per_row,
                self.grammar_state_bytes_per_row,
                self.source_state_bytes_per_row,
                self.queue_bytes_per_row,
                self.output_buffer_bytes_per_row,
            )
            .with_reserves(
                self.unmodeled_emergency_reserve_bytes,
                self.safety_margin_bytes,
            );
        if let Some(memory_budget_total) = self.memory_budget_total {
            request = request.with_local_memory_budget(memory_budget_total);
        }
        if let Some(fixed_residency) = fixed_residency {
            request = request.with_fixed_residency(fixed_residency);
        }
        if let Some(kv_page_metadata_per_token) = self.kv_page_metadata_per_token {
            request = request.with_kv_page_metadata_per_token(kv_page_metadata_per_token);
        }
        Ok(request)
    }
}

pub fn cli_main() -> ExitCode {
    let canonical_args = std::iter::once(OsString::from("fnlp")).chain(std::env::args_os().skip(1));
    let stdin = io::stdin();
    let stdin_is_terminal = stdin.is_terminal();
    let mut stdin = stdin.lock();
    cli_main_with_reader_and_terminal(canonical_args, &mut stdin, stdin_is_terminal)
}

/// Runs the CLI with an explicit schema-input reader.
///
/// Keeping the reader at this boundary prevents unit tests from accidentally
/// inheriting the process stdin and waiting for a terminal or pipe to close.
fn cli_main_with_reader(
    args: impl IntoIterator<Item = OsString>,
    schema_input: &mut impl BufRead,
) -> ExitCode {
    cli_main_with_reader_and_terminal(args, schema_input, false)
}

/// Dispatch CLI commands with an injected input reader and explicit terminal
/// fact. Tests receive a non-terminal reader; the process entry point derives
/// the fact through [`IsTerminal`] before locking stdin.
fn cli_main_with_reader_and_terminal(
    args: impl IntoIterator<Item = OsString>,
    schema_input: &mut impl BufRead,
    stdin_is_terminal: bool,
) -> ExitCode {
    match Cli::parse_from(args).command {
        Some(Command::Robot { command }) => {
            let command = match RobotCommand::try_from(command) {
                Ok(command) => command,
                Err(error) => {
                    eprintln!("ROBOT_PLAN RESULT=REFUSED reason={error}");
                    return ErrorCode::Usage.as_process_exit();
                }
            };
            let mut stdout = io::stdout().lock();
            let mut stderr = io::stderr().lock();
            match robot::write_command(&mut stdout, &mut stderr, command) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("fnlp: robot output failure: {error}");
                    ExitCode::from(1)
                }
            }
        }
        Some(Command::Schema { command }) => run_schema_command_with_reader(command, schema_input),
        Some(Command::Release { command }) => run_release_command(command),
        Some(Command::Convert(command)) => {
            run_convert_command(command, schema_input, stdin_is_terminal)
        }
        Some(Command::Models { command }) => run_models_command(command),
        None => {
            let mut command = Cli::command();
            let _ = command.print_help();
            println!();
            ExitCode::SUCCESS
        }
    }
}

fn run_models_command(command: ModelsSubcommand) -> ExitCode {
    match command {
        ModelsSubcommand::Derive(command) => run_models_derive(command),
    }
}

fn run_models_derive(command: ModelsDeriveCommand) -> ExitCode {
    let target = match NativePackingTarget::parse(&command.arch) {
        Ok(target) => target,
        Err(error) => {
            eprintln!("MODELS_DERIVE RESULT=FAIL stage=arch reason={error}");
            return ErrorCode::Usage.as_process_exit();
        }
    };
    if let Err(error) = open_ratified_model_root(&command.model_dir) {
        eprintln!(
            "MODELS_DERIVE RESULT=REFUSED stage=model-root generic={} model_dir={} arch={} tile_table_version={} reason={error}",
            command.generic.display(),
            command.model_dir.display(),
            target.cli_name(),
            command.tile_table_version,
        );
        return ErrorCode::AdmissionOrResourceLimit.as_process_exit();
    }

    // No target currently has a model-root handle capable of `create_new`,
    // same-filesystem staging, sync, and non-replacing activation.  Keep this
    // branch fail-closed even if a future platform probe changes the root
    // opener: raw filesystem writes here would violate the artifact contract.
    eprintln!(
        "MODELS_DERIVE RESULT=REFUSED stage=cache-transaction generic={} model_dir={} arch={} tile_table_version={} reason=ratified-native-cache-create-new-stage-unavailable",
        command.generic.display(),
        command.model_dir.display(),
        target.cli_name(),
        command.tile_table_version,
    );
    ErrorCode::AdmissionOrResourceLimit.as_process_exit()
}

fn run_convert_command(
    command: ConvertCommand,
    input: &mut impl BufRead,
    stdin_is_terminal: bool,
) -> ExitCode {
    let ConvertCommand {
        source,
        source_manifest,
        recipe,
        arch,
        output,
        converter_commit,
        yes,
        strict_source_dir,
        robot,
    } = command;
    let arch = match ConvertArch::parse(&arch) {
        Ok(arch) => arch,
        Err(error) => return emit_convert_refusal(error),
    };
    let request = ConvertRequest {
        source_dir: source,
        source_manifest,
        recipe_id: recipe,
        arch,
        output,
        converter_commit,
        yes,
        strict_source_dir,
        robot,
    };

    eprintln!(
        "CONVERT STAGE=census RESULT=START source={} manifest={}",
        request.source_dir.display(),
        request.source_manifest.display(),
    );
    match prepare_convert_request(&request, DEFAULT_PANEL_BYTES) {
        Ok(prepared) => match prepared.generic_payload_plan() {
            Ok(generic) => {
                eprintln!(
                    "CONVERT STAGE=census RESULT=END source-root-sha256={} census-sha256={} tensors={}",
                    prepared.source_root_sha256(),
                    prepared.census_sha256(),
                    prepared.tensor_count(),
                );
                emit_convert_preflight(&prepared, &generic);
                let confirmation = match confirm_convert(&request, input, stdin_is_terminal) {
                    Ok(mode) => mode,
                    Err(reason) => {
                        eprintln!(
                            "CONVERT RESULT=FAIL stage=confirmation reason={reason}; no-output-created"
                        );
                        return ErrorCode::Usage.as_process_exit();
                    }
                };
                eprintln!("CONVERT STAGE=confirmation RESULT={confirmation}");
                run_streaming_convert(&request, &prepared, &generic)
            }
            Err(error) => emit_convert_refusal(error),
        },
        Err(error) => emit_convert_refusal(error),
    }
}

/// Record source and Generic-plan facts before the explicit-output stage is
/// created. The final receipt records the post-write disk and file identities.
fn emit_convert_preflight(prepared: &PreparedConversionInput, generic: &GenericPayloadPlan) {
    eprintln!(
        "CONVERT PREFLIGHT RESULT=READY closure-bytes={} generic-payload-bytes={} generic-scales-bytes={} generic-row-sums-bytes={} explicit-output-stage=NOT-CREATED",
        prepared.closure_total_bytes(),
        generic.payload_bytes,
        generic.scale_bytes,
        generic.row_sum_bytes,
    );
}

/// Select the one admission-confirmation policy before any conversion-side
/// artifact output is attempted.
///
/// TTY callers explicitly answer y/N unless they supplied `--yes`; robot and
/// non-TTY callers never block on a prompt and retain that fact in the stage
/// transcript. Source validation/preflight happens first so a human sees the
/// immutable input facts before accepting conversion work.
fn confirm_convert(
    request: &ConvertRequest,
    input: &mut impl BufRead,
    stdin_is_terminal: bool,
) -> Result<&'static str, String> {
    if request.yes {
        return Ok("BYPASSED reason=--yes");
    }
    if request.robot {
        return Ok("SKIPPED reason=robot-noninteractive");
    }
    if !stdin_is_terminal {
        return Ok("SKIPPED reason=stdin-not-terminal");
    }

    eprint!(
        "CONVERT CONFIRM source={} destination={} [y/N]: ",
        request.source_dir.display(),
        request.output.display(),
    );
    io::stderr()
        .flush()
        .map_err(|error| format!("flush confirmation prompt: {error}"))?;
    let mut answer = String::new();
    input
        .read_line(&mut answer)
        .map_err(|error| format!("read confirmation response: {error}"))?;
    if matches!(answer.trim(), "y" | "Y" | "yes" | "YES") {
        Ok("ACCEPTED reason=tty-y")
    } else {
        Err("tty-confirmation-declined".to_owned())
    }
}

fn run_streaming_convert(
    request: &ConvertRequest,
    prepared: &PreparedConversionInput,
    generic: &GenericPayloadPlan,
) -> ExitCode {
    if let Err(error) = ensure_conversion_destinations_absent(&request.output) {
        return emit_streaming_refusal("preflight-output", error);
    }
    eprintln!(
        "CONVERT STAGE=plan RESULT=START tensors={}",
        prepared.tensor_count()
    );
    let materialized = match read_materialized_sources(prepared) {
        Ok(value) => value,
        Err(error) => return emit_streaming_refusal("materialized-sources", error),
    };
    let plan = match build_streaming_envelope_plan(prepared, generic, &materialized) {
        Ok(value) => value,
        Err(error) => return emit_streaming_refusal("streaming-first-pass", error),
    };
    eprintln!(
        "CONVERT STAGE=plan RESULT=END tensors={} sections={}",
        plan.input.tensors.len(),
        plan.input.sections.len(),
    );
    let (staging_output, mut output) = match create_conversion_stage(&request.output) {
        Ok(stage) => stage,
        Err(error) => return emit_streaming_refusal("create-staging", error),
    };
    eprintln!(
        "CONVERT STAGE=emission RESULT=START staging={} destination={} tensors={}",
        staging_output.display(),
        request.output.display(),
        prepared.tensor_count(),
    );
    let written = match write_streaming(&plan.input, &mut output, |section, sink| {
        emit_streaming_section(
            &request.source_dir,
            prepared,
            generic,
            &materialized,
            section,
            sink,
        )
    }) {
        Ok(value) => value,
        Err(error) => return emit_streaming_refusal("streaming-envelope", error),
    };
    if let Err(error) = output.sync_all() {
        return emit_streaming_refusal(
            "sync-staging",
            format!("sync {}: {error}", staging_output.display()),
        );
    }
    drop(output);
    let staged_len = match fs::metadata(&staging_output) {
        Ok(metadata) => metadata.len(),
        Err(error) => {
            return emit_streaming_refusal(
                "inspect-staging",
                format!("stat {}: {error}", staging_output.display()),
            );
        }
    };
    if staged_len != written.file_len {
        return emit_streaming_refusal(
            "inspect-staging",
            format!(
                "staging length mismatch: expected={} observed={staged_len}",
                written.file_len
            ),
        );
    }
    let reloaded = match FnlpqArtifact::open_owned(&staging_output) {
        Ok(artifact) => artifact,
        Err(error) => return emit_streaming_refusal("reload-staging", error),
    };
    if let Err(error) = verify_reloaded_conversion(&reloaded, prepared, &plan, &written) {
        return emit_streaming_refusal("reload-staging", error);
    }
    drop(reloaded);

    let artifact_raw_sha256 = match raw_sha256_file(&staging_output) {
        Ok(value) => value,
        Err(error) => return emit_streaming_refusal("digest-staging", error),
    };
    let receipt = match write_conversion_receipt_sidecar(
        request,
        prepared,
        &plan,
        &written,
        &artifact_raw_sha256,
    ) {
        Ok(value) => value,
        Err(error) => return emit_streaming_refusal("receipt", error),
    };
    eprintln!(
        "CONVERT STAGE=receipt RESULT=PASS destination={} staging-receipt={} receipt-sha256={}",
        receipt.destination.display(),
        receipt.staging_path.display(),
        receipt.sha256,
    );
    if let Err(error) = publish_explicit_conversion_stage(&staging_output, &request.output) {
        return emit_streaming_refusal("publish-explicit-output", error);
    }
    emit_converted_unqualified(
        &request.output,
        &staging_output,
        prepared,
        &written,
        &artifact_raw_sha256,
        &receipt,
    )
}

/// Prove that the checked staged artifact is the exact envelope planned from
/// this sealed source closure before any receipt or final-path entry exists.
fn verify_reloaded_conversion(
    reloaded: &FnlpqArtifact,
    prepared: &PreparedConversionInput,
    plan: &StreamingEnvelopePlan,
    written: &StreamedFnlpq,
) -> Result<(), String> {
    let expected_license_bundle_sha256 = hex_lower(&written.license_bundle_sha256);
    for (field, observed, expected) in [
        (
            "model-id",
            reloaded.model_id(),
            plan.input.model_id.as_str(),
        ),
        (
            "revision",
            reloaded.revision(),
            plan.input.revision.as_str(),
        ),
        (
            "recipe",
            reloaded.recipe_id(),
            plan.input.recipe_id.as_str(),
        ),
        (
            "source-root",
            reloaded.source_root_sha256(),
            prepared.source_root_sha256(),
        ),
        (
            "logical-model",
            reloaded.logical_model_sha256(),
            plan.input.logical_model_sha256.as_str(),
        ),
        (
            "license-bundle",
            reloaded.license_bundle_sha256(),
            expected_license_bundle_sha256.as_str(),
        ),
    ] {
        if observed != expected {
            return Err(format!(
                "reloaded {field} differs: expected={expected} observed={observed}"
            ));
        }
    }
    if reloaded.tensors().len() != plan.input.tensors.len()
        || reloaded.sections().len() != written.sections.len()
    {
        return Err(format!(
            "reloaded cardinality differs: tensors={}/{} sections={}/{}",
            reloaded.tensors().len(),
            plan.input.tensors.len(),
            reloaded.sections().len(),
            written.sections.len(),
        ));
    }
    for expected in &plan.input.tensors {
        let Some(observed) = reloaded
            .tensors()
            .iter()
            .find(|candidate| candidate.name == expected.name)
        else {
            return Err(format!("reloaded tensor is absent: {}", expected.name));
        };
        if observed.shape != expected.shape
            || observed.quantization != expected.quantization
            || observed.canonical_logical_sha256 != expected.canonical_logical_sha256
        {
            return Err(format!(
                "reloaded tensor reconstruction differs: {}",
                expected.name
            ));
        }
    }
    reloaded
        .select_packing(ArchTarget::Generic)
        .map_err(|error| format!("reloaded generic packing is absent: {error}"))?;
    Ok(())
}

/// Keep explicit artifact and receipt final names unoccupied before the
/// expensive source traversal.  Publish repeats no-clobber enforcement to
/// fail closed if another process races this read-only preflight.
fn ensure_conversion_destinations_absent(destination: &Path) -> Result<(), String> {
    let receipt = conversion_receipt_path(destination)?;
    for path in [destination, receipt.as_path()] {
        match fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(format!(
                    "final destination already exists: {}",
                    path.display()
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "inspect final destination {}: {error}",
                    path.display()
                ));
            }
        }
    }
    Ok(())
}

/// Derive a deterministic sibling staging name.  It is deliberately distinct
/// from the requested final output, so any refusal leaves evidence only at
/// the retained staging path rather than a partial final artifact path.
fn conversion_staging_path(destination: &Path, attempt: u16) -> Result<PathBuf, String> {
    let name = destination
        .file_name()
        .ok_or_else(|| format!("final output has no file name: {}", destination.display()))?;
    let mut staging_name = OsString::from(".");
    staging_name.push(name);
    staging_name.push(format!(".fnlpq-stage.{attempt}"));
    Ok(destination.with_file_name(staging_name))
}

/// Derive the canonical machine-readable receipt path beside an explicit
/// `.fnlpq` output without changing the output file's authority name.
fn conversion_receipt_path(destination: &Path) -> Result<PathBuf, String> {
    let name = destination
        .file_name()
        .ok_or_else(|| format!("final output has no file name: {}", destination.display()))?;
    let mut receipt_name = OsString::from(name);
    receipt_name.push(".receipt.json");
    Ok(destination.with_file_name(receipt_name))
}

fn conversion_receipt_staging_path(destination: &Path, attempt: u16) -> Result<PathBuf, String> {
    let name = destination
        .file_name()
        .ok_or_else(|| format!("receipt output has no file name: {}", destination.display()))?;
    let mut staging_name = OsString::from(".");
    staging_name.push(name);
    staging_name.push(format!(".receipt-stage.{attempt}"));
    Ok(destination.with_file_name(staging_name))
}

/// Create one hidden, same-directory, non-replacing explicit-output stage.
///
/// This is intentionally limited to `fnlp convert -o PATH`: cache activation
/// continues to require the separate ratified model-root transaction.
fn create_conversion_stage(destination: &Path) -> Result<(PathBuf, fs::File), String> {
    for attempt in 0..=u16::MAX {
        let stage = conversion_staging_path(destination, attempt)?;
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stage)
        {
            Ok(file) => return Ok((stage, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!("create staging {}: {error}", stage.display()));
            }
        }
    }
    Err(format!(
        "no unused hidden staging name remains beside {}",
        destination.display()
    ))
}

/// Create a hidden, no-clobber receipt stage adjacent to the final receipt.
fn create_conversion_receipt_stage(destination: &Path) -> Result<(PathBuf, fs::File), String> {
    for attempt in 0..=u16::MAX {
        let stage = conversion_receipt_staging_path(destination, attempt)?;
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stage)
        {
            Ok(file) => return Ok((stage, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "create receipt staging {}: {error}",
                    stage.display()
                ));
            }
        }
    }
    Err(format!(
        "no unused hidden receipt staging name remains beside {}",
        destination.display()
    ))
}

/// Publish a reloaded explicit output without replacing an existing path.
/// The retained stage remains as a read-only forensic sibling; managed-cache
/// activation never calls this function.
fn publish_explicit_conversion_stage(stage: &Path, destination: &Path) -> Result<(), String> {
    let mut permissions = fs::metadata(stage)
        .map_err(|error| format!("stat staged artifact {}: {error}", stage.display()))?
        .permissions();
    permissions.set_readonly(true);
    fs::set_permissions(stage, permissions).map_err(|error| {
        format!(
            "make staged artifact read-only {}: {error}",
            stage.display()
        )
    })?;
    fs::hard_link(stage, destination).map_err(|error| {
        format!(
            "publish {} to previously absent {}: {error}",
            stage.display(),
            destination.display(),
        )
    })
}

struct WrittenConversionReceipt {
    destination: PathBuf,
    staging_path: PathBuf,
    sha256: String,
}

/// Serialize, reload, and publish the canonical receipt before exposing its
/// paired explicit artifact.  The receipt is a conversion record only; it
/// deliberately does not activate or qualify the artifact.
fn write_conversion_receipt_sidecar(
    request: &ConvertRequest,
    prepared: &PreparedConversionInput,
    plan: &StreamingEnvelopePlan,
    written: &StreamedFnlpq,
    artifact_raw_sha256: &str,
) -> Result<WrittenConversionReceipt, String> {
    let preflight = prepared
        .preflight(written.file_len, DEFAULT_PANEL_BYTES, 0, 0, 0)
        .map_err(|error| format!("receipt preflight: {error}"))?;
    let peak_rss_cap_bytes = preflight
        .peak_rss
        .total_bytes()
        .map_err(|error| format!("receipt peak-rss formula: {error}"))?;
    preflight
        .peak_rss
        .enforce(peak_rss_cap_bytes)
        .map_err(|error| format!("receipt peak-rss enforcement: {error}"))?;
    let receipt = ConversionReceipt {
        receipt_schema: CONVERSION_RECEIPT_SCHEMA.to_owned(),
        model_id: plan.input.model_id.clone(),
        model_revision: PINNED_REVISION.to_owned(),
        artifact_format: CONVERSION_ARTIFACT_FORMAT.to_owned(),
        source_manifest_sha256: PINNED_SOURCE_MANIFEST_SHA256.to_owned(),
        target_arch: request.arch.as_str().to_owned(),
        source_root_sha256: prepared.source_root_sha256().to_owned(),
        census_sha256: prepared.census_sha256().to_owned(),
        logical_model_sha256: plan.input.logical_model_sha256.clone(),
        converter_commit: request.converter_commit.clone(),
        recipe_id: plan.input.recipe_id.clone(),
        rounding_id: PORTABLE_QUANT_V1.to_owned(),
        packing_id: GENERIC_PACKING_V1.to_owned(),
        measured_peak_rss_bytes: peak_rss_cap_bytes,
        measured_scratch_bytes: DEFAULT_PANEL_BYTES,
        peak_rss_cap_bytes,
        final_disk_bytes: preflight.final_disk_bytes,
        measured_disk_bytes: written.file_len,
        output_len: written.file_len,
        fnlpq_file_sha256: hex_lower(&written.fnlpq_file_sha256),
        artifact_raw_sha256: artifact_raw_sha256.to_owned(),
        license_bundle_sha256: hex_lower(&written.license_bundle_sha256),
    };
    let json = receipt
        .canonical_json()
        .map_err(|error| format!("serialize conversion receipt: {error}"))?;
    let destination = conversion_receipt_path(&request.output)?;
    let (staging_path, mut file) = create_conversion_receipt_stage(&destination)?;
    file.write_all(json.as_bytes())
        .map_err(|error| format!("write staged receipt {}: {error}", staging_path.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync staged receipt {}: {error}", staging_path.display()))?;
    drop(file);
    let reloaded = fs::read(&staging_path)
        .map_err(|error| format!("read staged receipt {}: {error}", staging_path.display()))?;
    let reloaded = std::str::from_utf8(&reloaded)
        .map_err(|error| format!("decode staged receipt {}: {error}", staging_path.display()))?;
    let parsed = ConversionReceipt::parse_canonical_json(reloaded)
        .map_err(|error| format!("parse staged receipt {}: {error}", staging_path.display()))?;
    if parsed != receipt {
        return Err(format!(
            "reloaded receipt differs from canonical serialization: {}",
            staging_path.display()
        ));
    }
    let sha256 = hex_lower(&Sha256::digest(json.as_bytes()));
    publish_explicit_conversion_stage(&staging_path, &destination)?;
    Ok(WrittenConversionReceipt {
        destination,
        staging_path,
        sha256,
    })
}

/// Compute an authority-distinct raw SHA-256 without materializing the staged
/// artifact.  The framed file identity remains writer-owned.
fn raw_sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|error| {
        format!(
            "open staged artifact for raw digest {}: {error}",
            path.display()
        )
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            format!(
                "read staged artifact for raw digest {}: {error}",
                path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

/// Report a real explicit-output conversion without conflating it with cache
/// activation or qualification.  Receipt/reconstruction remains a separate
/// conversion milestone and is intentionally not implied by this line.
fn emit_converted_unqualified(
    destination: &Path,
    staging_output: &Path,
    prepared: &PreparedConversionInput,
    written: &StreamedFnlpq,
    artifact_raw_sha256: &str,
    receipt: &WrittenConversionReceipt,
) -> ExitCode {
    eprintln!(
        "CONVERT RESULT=PASS stage=explicit-output destination={} staging-artifact={} receipt={} staging-receipt={} receipt-sha256={} source-root-sha256={} census-sha256={} tensors={} fnlpq-file-sha256={} artifact-raw-sha256={} staging-bytes={} license-bundle-sha256={} reload=PASS status=converted-not-qualified cache-activation=NOT-ATTEMPTED",
        destination.display(),
        staging_output.display(),
        receipt.destination.display(),
        receipt.staging_path.display(),
        receipt.sha256,
        prepared.source_root_sha256(),
        prepared.census_sha256(),
        prepared.tensor_count(),
        hex_lower(&written.fnlpq_file_sha256),
        artifact_raw_sha256,
        written.file_len,
        hex_lower(&written.license_bundle_sha256),
    );
    ExitCode::SUCCESS
}

fn emit_streaming_refusal(stage: &str, error: impl std::fmt::Display) -> ExitCode {
    eprintln!("CONVERT RESULT=FAIL stage={stage} reason={error}");
    ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()
}

struct MaterializedSources<'a> {
    // These borrow the prepared source snapshot so the envelope planner does
    // not create a second tokenizer/config/template resident copy.
    model_config: &'a [u8],
    tokenizer_model: &'a [u8],
    tokenizer_config: &'a [u8],
    chat_template: &'a [u8],
    license_bundle: Vec<u8>,
}

struct StreamingEnvelopePlan {
    input: FnlpqStreamingInput,
}

struct LogicalTensorFirstPass {
    input: TensorInput,
    scale_bytes: Vec<u8>,
    row_sum_bytes: Vec<u8>,
    hasher: LogicalTensorStreamingHasher,
}

impl LogicalTensorFirstPass {
    fn new(
        entry: &TensorCensusEntry,
        layout: &crate::artifact::converter::GenericTensorLayout,
    ) -> Result<Self, String> {
        let shape = entry
            .shape
            .iter()
            .copied()
            .map(|dimension| {
                u32::try_from(dimension).map_err(|_| {
                    format!(
                        "tensor {} shape dimension {dimension} exceeds fnlpq v1 u32",
                        entry.name
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = TensorInput {
            // The artifact authority is the frozen source-census name.  The
            // converter's internal route is a distinct implementation detail
            // and cannot replace this externally validated identity.
            name: layout.source_name.clone(),
            canonical_dtype: CanonicalDtype::Bf16,
            shape: shape.clone(),
            canonical_logical_sha256: String::new(),
            quantization: layout.quantization.clone(),
            data: SectionRange::new(GENERIC_PAYLOAD_SECTION, layout.data.offset, layout.data.len),
            scale: SectionRange::new(
                GENERIC_SCALES_SECTION,
                layout.scale.offset,
                layout.scale.len,
            ),
            row_sum: SectionRange::new(
                GENERIC_ROW_SUMS_SECTION,
                layout.row_sum.offset,
                layout.row_sum.len,
            ),
        };
        let hasher = LogicalTensorStreamingHasher::new(
            &input.name,
            "bf16",
            &input.shape,
            &input.quantization,
            layout.data.len,
            layout.scale.len,
            layout.row_sum.len,
        )
        .map_err(|error| format!("start logical tensor {}: {error}", entry.name))?;
        Ok(Self {
            input,
            scale_bytes: Vec::new(),
            row_sum_bytes: Vec::new(),
            hasher,
        })
    }

    fn append(&mut self, encoded: &GenericPanelBytes) -> Result<(), String> {
        self.hasher
            .write_data(&encoded.data)
            .map_err(|error| format!("append logical payload: {error}"))?;
        self.scale_bytes.extend_from_slice(&encoded.scales);
        self.row_sum_bytes.extend_from_slice(&encoded.row_sums);
        Ok(())
    }

    fn finish(mut self) -> Result<(TensorInput, [u8; 32]), String> {
        self.hasher
            .write_scale(&self.scale_bytes)
            .map_err(|error| format!("append logical scales: {error}"))?;
        self.hasher
            .write_row_sum(&self.row_sum_bytes)
            .map_err(|error| format!("append logical row sums: {error}"))?;
        let digest = self
            .hasher
            .finish()
            .map_err(|error| format!("finish logical tensor: {error}"))?;
        self.input.canonical_logical_sha256 = hex_lower(&digest);
        Ok((self.input, digest))
    }
}

fn read_materialized_sources(
    prepared: &PreparedConversionInput,
) -> Result<MaterializedSources<'_>, String> {
    let verified = prepared.materialized_sources();
    Ok(MaterializedSources {
        model_config: &verified.model_config,
        tokenizer_model: &verified.tokenizer_model,
        tokenizer_config: &verified.tokenizer_config,
        chat_template: &verified.chat_template,
        license_bundle: embedded_license_bundle()?,
    })
}

fn embedded_license_bundle() -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for (name, contents) in EMBEDDED_LICENSE_FILES {
        let name_len = u64::try_from(name.len())
            .map_err(|_| "embedded license file name length does not fit u64".to_owned())?;
        let contents_len = u64::try_from(contents.len())
            .map_err(|_| "embedded license file length does not fit u64".to_owned())?;
        bytes.extend_from_slice(&name_len.to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&contents_len.to_le_bytes());
        bytes.extend_from_slice(contents);
    }
    Ok(bytes)
}

fn build_streaming_envelope_plan(
    prepared: &PreparedConversionInput,
    generic: &GenericPayloadPlan,
    materialized: &MaterializedSources<'_>,
) -> Result<StreamingEnvelopePlan, String> {
    validate_generic_tensor_authorities(generic)?;
    prepared
        .revalidate_retained_shards()
        .map_err(|error| format!("revalidate retained source shards: {error}"))?;
    let (mut tensors, payload_sha256, scales_sha256, row_sums_sha256) =
        first_pass_generic_identities(prepared, generic)?;
    tensors.sort_by(|left, right| left.0.name.as_bytes().cmp(right.0.name.as_bytes()));
    let tensor_digests = tensors
        .iter()
        .map(|(_, digest)| *digest)
        .collect::<Vec<_>>();
    let logical_model_sha256 = logical_model_sha256(
        "Nanbeige4.2-3B",
        crate::artifact::converter::PINNED_REVISION,
        &tensor_digests,
        &[
            ("model_config", materialized.model_config),
            ("tokenizer_model", materialized.tokenizer_model),
            ("tokenizer_config", materialized.tokenizer_config),
            ("chat_template", materialized.chat_template),
        ],
    )
    .map_err(|error| format!("logical model identity: {error}"))?;
    let license_bundle_sha256 = framed_sha256(
        digest_domain::LICENSE_BUNDLE,
        &[materialized.license_bundle.as_slice()],
    )
    .map_err(|error| format!("license bundle identity: {error}"))?;
    let sections = vec![
        streaming_section(
            GENERIC_PAYLOAD_SECTION,
            SectionKind::GenericTensorPayload,
            generic.payload_bytes,
            64,
            payload_sha256,
        ),
        streaming_section(
            GENERIC_SCALES_SECTION,
            SectionKind::GenericTensorScales,
            generic.scale_bytes,
            8,
            scales_sha256,
        ),
        streaming_section(
            GENERIC_ROW_SUMS_SECTION,
            SectionKind::GenericTensorRowSums,
            generic.row_sum_bytes,
            8,
            row_sums_sha256,
        ),
        streaming_small_section(
            TOKENIZER_MODEL_SECTION,
            SectionKind::TokenizerModel,
            32,
            materialized.tokenizer_model,
        )?,
        streaming_small_section(
            MODEL_CONFIG_SECTION,
            SectionKind::ModelConfig,
            8,
            materialized.model_config,
        )?,
        streaming_small_section(
            TOKENIZER_CONFIG_SECTION,
            SectionKind::TokenizerConfig,
            8,
            materialized.tokenizer_config,
        )?,
        streaming_small_section(
            CHAT_TEMPLATE_SECTION,
            SectionKind::ChatTemplate,
            8,
            materialized.chat_template,
        )?,
        streaming_small_section(
            LICENSE_BUNDLE_SECTION,
            SectionKind::LicenseBundle,
            16,
            &materialized.license_bundle,
        )?,
    ];
    Ok(StreamingEnvelopePlan {
        input: FnlpqStreamingInput {
            model_id: "Nanbeige4.2-3B".to_owned(),
            revision: crate::artifact::converter::PINNED_REVISION.to_owned(),
            recipe_id: crate::artifact::converter::PINNED_CONVERSION_RECIPE.to_owned(),
            source_root_sha256: prepared.source_root_sha256().to_owned(),
            logical_model_sha256: hex_lower(&logical_model_sha256),
            sections,
            tensors: tensors.into_iter().map(|(tensor, _)| tensor).collect(),
            packing_sets: vec![PackingSetInput {
                id: "generic".to_owned(),
                target: ArchTarget::Generic,
                section_names: vec![
                    GENERIC_PAYLOAD_SECTION.to_owned(),
                    GENERIC_SCALES_SECTION.to_owned(),
                    GENERIC_ROW_SUMS_SECTION.to_owned(),
                ],
            }],
            license_bundle_sha256,
        },
    })
}

/// Check every converter-planned logical name before opening safetensor
/// shards.  The writer repeats this validation defensively, but discovery at
/// that point would waste a complete model traversal on a malformed header.
fn validate_generic_tensor_authorities(generic: &GenericPayloadPlan) -> Result<(), String> {
    for layout in &generic.tensors {
        validate_authority_identifier("tensor.name", &layout.source_name)
            .map_err(|error| format!("invalid planned tensor {}: {error}", layout.source_name))?;
    }
    Ok(())
}

fn streaming_section(
    name: &str,
    kind: SectionKind,
    stored_len: u64,
    alignment: u64,
    stored_sha256: [u8; 32],
) -> StreamingSection {
    StreamingSection {
        name: name.to_owned(),
        kind,
        stored_len,
        alignment,
        stored_sha256,
    }
}

fn streaming_small_section(
    name: &str,
    kind: SectionKind,
    alignment: u64,
    bytes: &[u8],
) -> Result<StreamingSection, String> {
    let stored_len =
        u64::try_from(bytes.len()).map_err(|_| format!("{name} length does not fit u64"))?;
    let stored_sha256 = framed_sha256(digest_domain::SECTION, &[name.as_bytes(), bytes])
        .map_err(|error| format!("{name} section identity: {error}"))?;
    Ok(streaming_section(
        name,
        kind,
        stored_len,
        alignment,
        stored_sha256,
    ))
}

/// Traverse every bounded source panel once to bind the planned Generic
/// section bytes and every logical tensor identity before the envelope writer
/// can create an artifact.
fn first_pass_generic_identities(
    prepared: &PreparedConversionInput,
    generic: &GenericPayloadPlan,
) -> Result<(Vec<(TensorInput, [u8; 32])>, [u8; 32], [u8; 32], [u8; 32]), String> {
    let (census, routes, panels) = prepared
        .checked_plan_parts()
        .map_err(|error| format!("validate prepared conversion plan: {error}"))?;
    if census.len() != generic.tensors.len() {
        return Err(format!(
            "streaming plan count mismatch: census={} routes={} panels={} generic={}",
            census.len(),
            routes.len(),
            panels.len(),
            generic.tensors.len(),
        ));
    }
    let mut payload = StreamingSectionHasher::new(GENERIC_PAYLOAD_SECTION, generic.payload_bytes)
        .map_err(|error| format!("start generic payload identity: {error}"))?;
    let mut scales = StreamingSectionHasher::new(GENERIC_SCALES_SECTION, generic.scale_bytes)
        .map_err(|error| format!("start generic scales identity: {error}"))?;
    let mut row_sums = StreamingSectionHasher::new(GENERIC_ROW_SUMS_SECTION, generic.row_sum_bytes)
        .map_err(|error| format!("start generic row-sums identity: {error}"))?;
    let mut tensors = Vec::with_capacity(generic.tensors.len());

    for index in 0..census.len() {
        let entry = &census[index];
        let route = &routes[index];
        let panel_plan = &panels[index];
        let layout = &generic.tensors[index];
        validate_generic_layout(entry, route, layout)?;
        let mut tensor = LogicalTensorFirstPass::new(entry, layout)?;
        stream_routed_bf16_panels(
            std::slice::from_ref(entry),
            std::slice::from_ref(route),
            std::slice::from_ref(panel_plan),
            |source_entry, panel| prepared.read_verified_panel(&source_entry.name, panel),
            |source_entry, source_route, panel, source_bf16, decoded_f32| {
                let encoded = encode_streaming_panel(
                    source_entry,
                    source_route,
                    panel,
                    source_bf16,
                    decoded_f32,
                )?;
                payload.write(&encoded.data).map_err(|error| {
                    first_pass_write_error(source_entry, "generic payload", error)
                })?;
                scales.write(&encoded.scales).map_err(|error| {
                    first_pass_write_error(source_entry, "generic scales", error)
                })?;
                row_sums.write(&encoded.row_sums).map_err(|error| {
                    first_pass_write_error(source_entry, "generic row sums", error)
                })?;
                tensor
                    .append(&encoded)
                    .map_err(|detail| ConverterError::PipelinePlanAlignment {
                        tensor: source_entry.name.clone(),
                        detail,
                    })
            },
        )
        .map_err(|error| format!("first pass tensor {}: {error}", entry.name))?;
        tensors.push(tensor.finish()?);
    }

    Ok((
        tensors,
        payload
            .finish()
            .map_err(|error| format!("finish generic payload identity: {error}"))?,
        scales
            .finish()
            .map_err(|error| format!("finish generic scales identity: {error}"))?,
        row_sums
            .finish()
            .map_err(|error| format!("finish generic row-sums identity: {error}"))?,
    ))
}

fn validate_generic_layout(
    entry: &TensorCensusEntry,
    route: &crate::artifact::converter::TensorRoute,
    layout: &crate::artifact::converter::GenericTensorLayout,
) -> Result<(), String> {
    if layout.source_name != entry.name
        || layout.internal_name != route.internal_name
        || layout.stage != route.stage
        || layout.shape != entry.shape
    {
        return Err(format!(
            "generic layout disagrees with prepared route for {}: source={} internal={} stage={} shape={:?}",
            entry.name,
            layout.source_name,
            layout.internal_name,
            layout.stage.as_str(),
            layout.shape,
        ));
    }
    Ok(())
}

fn first_pass_write_error(
    entry: &TensorCensusEntry,
    section: &str,
    error: FnlpqWriteError,
) -> ConverterError {
    ConverterError::PipelinePlanAlignment {
        tensor: entry.name.clone(),
        detail: format!("{section} first-pass identity: {error}"),
    }
}

fn encode_streaming_panel(
    entry: &TensorCensusEntry,
    route: &crate::artifact::converter::TensorRoute,
    panel: RowPanel,
    source_bf16: &[u8],
    decoded_f32: &[f32],
) -> Result<GenericPanelBytes, ConverterError> {
    let RowPanel::Rows { row_count, .. } = panel else {
        return Err(ConverterError::PipelinePlanAlignment {
            tensor: entry.name.clone(),
            detail: "streaming conversion received whole-tensor panel".to_owned(),
        });
    };
    let rows = usize::try_from(row_count).map_err(|_| ConverterError::Arithmetic {
        invariant: "streaming panel row count to usize",
    })?;
    let columns = entry
        .shape
        .iter()
        .skip(1)
        .try_fold(1_u64, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "streaming panel column count",
                })
        })?;
    let columns = usize::try_from(columns).map_err(|_| ConverterError::Arithmetic {
        invariant: "streaming panel column count to usize",
    })?;
    encode_generic_panel(route.stage, source_bf16, decoded_f32, rows, columns)
        .map_err(ConverterError::Quantize)
}

/// Emit precisely one planned canonical section during the second traversal.
/// Generic sections are replayed independently because v1 stores each
/// section contiguously; all byte identities were fixed in the first pass.
fn emit_streaming_section(
    source_dir: &Path,
    prepared: &PreparedConversionInput,
    generic: &GenericPayloadPlan,
    materialized: &MaterializedSources<'_>,
    section: &StreamingSection,
    sink: &mut dyn Write,
) -> Result<(), FnlpqWriteError> {
    match (section.kind, section.name.as_str()) {
        (SectionKind::GenericTensorPayload, GENERIC_PAYLOAD_SECTION) => emit_generic_section(
            source_dir,
            prepared,
            generic,
            section,
            sink,
            GenericSection::Payload,
        ),
        (SectionKind::GenericTensorScales, GENERIC_SCALES_SECTION) => emit_generic_section(
            source_dir,
            prepared,
            generic,
            section,
            sink,
            GenericSection::Scales,
        ),
        (SectionKind::GenericTensorRowSums, GENERIC_ROW_SUMS_SECTION) => emit_generic_section(
            source_dir,
            prepared,
            generic,
            section,
            sink,
            GenericSection::RowSums,
        ),
        (SectionKind::TokenizerModel, TOKENIZER_MODEL_SECTION) => {
            emit_materialized_section(section, materialized.tokenizer_model, sink)
        }
        (SectionKind::ModelConfig, MODEL_CONFIG_SECTION) => {
            emit_materialized_section(section, materialized.model_config, sink)
        }
        (SectionKind::TokenizerConfig, TOKENIZER_CONFIG_SECTION) => {
            emit_materialized_section(section, materialized.tokenizer_config, sink)
        }
        (SectionKind::ChatTemplate, CHAT_TEMPLATE_SECTION) => {
            emit_materialized_section(section, materialized.chat_template, sink)
        }
        (SectionKind::LicenseBundle, LICENSE_BUNDLE_SECTION) => {
            emit_materialized_section(section, &materialized.license_bundle, sink)
        }
        _ => Err(FnlpqWriteError::StoredIdentity {
            section: section.name.clone(),
            expected: "one canonical conversion section name and kind".to_owned(),
            actual: format!("kind={:?}", section.kind),
        }),
    }
}

#[derive(Clone, Copy)]
enum GenericSection {
    Payload,
    Scales,
    RowSums,
}

fn emit_generic_section(
    source_dir: &Path,
    prepared: &PreparedConversionInput,
    generic: &GenericPayloadPlan,
    section: &StreamingSection,
    sink: &mut dyn Write,
    target: GenericSection,
) -> Result<(), FnlpqWriteError> {
    let (census, routes, panels) =
        prepared
            .checked_plan_parts()
            .map_err(|error| FnlpqWriteError::StoredIdentity {
                section: section.name.clone(),
                expected: "validated prepared conversion plan".to_owned(),
                actual: error.to_string(),
            })?;
    if census.len() != generic.tensors.len() {
        return Err(FnlpqWriteError::StoredIdentity {
            section: section.name.clone(),
            expected: "prepared conversion arrays with equal lengths".to_owned(),
            actual: format!(
                "census={} routes={} panels={} generic={}",
                census.len(),
                routes.len(),
                panels.len(),
                generic.tensors.len(),
            ),
        });
    }
    let expected_len = match target {
        GenericSection::Payload => generic.payload_bytes,
        GenericSection::Scales => generic.scale_bytes,
        GenericSection::RowSums => generic.row_sum_bytes,
    };
    if section.stored_len != expected_len {
        return Err(FnlpqWriteError::StoredIdentity {
            section: section.name.clone(),
            expected: format!("{expected_len} planned bytes"),
            actual: format!("{} declared bytes", section.stored_len),
        });
    }
    let mut observed_len = 0_u64;
    for index in 0..census.len() {
        let entry = &census[index];
        let route = &routes[index];
        let panel_plan = &panels[index];
        validate_generic_layout(entry, route, &generic.tensors[index]).map_err(|detail| {
            FnlpqWriteError::Tensor {
                tensor: entry.name.clone(),
                reason: detail,
            }
        })?;
        stream_routed_bf16_panels(
            std::slice::from_ref(entry),
            std::slice::from_ref(route),
            std::slice::from_ref(panel_plan),
            |source_entry, panel| prepared.read_verified_panel(&source_entry.name, panel),
            |source_entry, source_route, panel, source_bf16, decoded_f32| {
                let encoded = encode_streaming_panel(
                    source_entry,
                    source_route,
                    panel,
                    source_bf16,
                    decoded_f32,
                )?;
                let bytes = match target {
                    GenericSection::Payload => encoded.data,
                    GenericSection::Scales => encoded.scales,
                    GenericSection::RowSums => encoded.row_sums,
                };
                sink.write_all(&bytes).map_err(|error| ConverterError::Io {
                    path: source_dir.to_path_buf(),
                    operation: "write streaming generic section",
                    detail: error.to_string(),
                })?;
                let bytes_len =
                    u64::try_from(bytes.len()).map_err(|_| ConverterError::Arithmetic {
                        invariant: "streaming generic section bytes to u64",
                    })?;
                observed_len =
                    observed_len
                        .checked_add(bytes_len)
                        .ok_or(ConverterError::Arithmetic {
                            invariant: "streaming generic section observed bytes",
                        })?;
                Ok(())
            },
        )
        .map_err(|error| FnlpqWriteError::Io {
            operation: "replay verified generic section",
            detail: error.to_string(),
        })?;
        let completed = index + 1;
        if completed % EMISSION_PROGRESS_TENSOR_INTERVAL == 0 || completed == census.len() {
            eprintln!(
                "CONVERT STAGE=emission section={} tensors={}/{}",
                section.name,
                completed,
                census.len(),
            );
        }
    }
    if observed_len != expected_len {
        return Err(FnlpqWriteError::StoredIdentity {
            section: section.name.clone(),
            expected: format!("{expected_len} planned bytes"),
            actual: format!("{observed_len} replayed bytes"),
        });
    }
    Ok(())
}

fn emit_materialized_section(
    section: &StreamingSection,
    bytes: &[u8],
    sink: &mut dyn Write,
) -> Result<(), FnlpqWriteError> {
    let observed_len = u64::try_from(bytes.len()).map_err(|_| FnlpqWriteError::Arithmetic {
        field: "materialized streaming section bytes",
    })?;
    if section.stored_len != observed_len {
        return Err(FnlpqWriteError::StoredIdentity {
            section: section.name.clone(),
            expected: format!("{} declared bytes", section.stored_len),
            actual: format!("{observed_len} materialized bytes"),
        });
    }
    sink.write_all(bytes).map_err(|error| FnlpqWriteError::Io {
        operation: "write materialized streaming section",
        detail: error.to_string(),
    })
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
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
    use std::{
        ffi::OsString,
        io::Cursor,
        path::{Path, PathBuf},
        process::ExitCode,
    };

    use clap::{Parser, error::ErrorKind};

    use super::{
        Cli, Command, KvCacheQuantization, LogicalTensorFirstPass, ModelsSubcommand,
        ReleaseSubcommand, RobotSubcommand, cli_main_with_reader, confirm_convert,
        conversion_receipt_path, conversion_staging_path, validate_generic_tensor_authorities,
    };
    use crate::artifact::converter::{
        ConvertArch, ConvertRequest, GenericPayloadPlan, GenericTensorLayout, OutputRange,
        StorageStage, expected_nanbeige42_census,
    };
    use crate::artifact::format::validate_authority_identifier;
    use crate::artifact::safetensors::{SafetensorDtype, TensorCensusEntry};

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
    fn conversion_confirmation_distinguishes_tty_yes_robot_and_pipe_modes() {
        let request = |yes, robot| ConvertRequest {
            source_dir: "/models/source".into(),
            source_manifest: "/models/source.json".into(),
            recipe_id: "nanbeige42-int8-v1".to_owned(),
            arch: ConvertArch::Generic,
            output: "/models/output.fnlpq".into(),
            converter_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            yes,
            strict_source_dir: false,
            robot,
        };

        assert_eq!(
            confirm_convert(&request(true, false), &mut Cursor::new(b""), true),
            Ok("BYPASSED reason=--yes")
        );
        assert_eq!(
            confirm_convert(&request(false, true), &mut Cursor::new(b""), true),
            Ok("SKIPPED reason=robot-noninteractive")
        );
        assert_eq!(
            confirm_convert(&request(false, false), &mut Cursor::new(b""), false),
            Ok("SKIPPED reason=stdin-not-terminal")
        );
        assert_eq!(
            confirm_convert(&request(false, false), &mut Cursor::new(b"yes\n"), true),
            Ok("ACCEPTED reason=tty-y")
        );
        assert_eq!(
            confirm_convert(&request(false, false), &mut Cursor::new(b"n\n"), true),
            Err("tty-confirmation-declined".to_owned())
        );
    }

    #[test]
    fn conversion_receipt_sidecar_is_a_distinct_sibling_of_the_artifact() {
        assert_eq!(
            conversion_receipt_path(Path::new("/models/output.fnlpq")),
            Ok(PathBuf::from("/models/output.fnlpq.receipt.json"))
        );
        assert_ne!(
            conversion_receipt_path(Path::new("/models/output.fnlpq")),
            Ok(PathBuf::from("/models/output.fnlpq"))
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
    fn models_derive_requires_explicit_generic_arch_and_model_root() {
        let parsed = Cli::try_parse_from([
            "fnlp",
            "models",
            "derive",
            "--generic",
            "/models/generic.fnlpq",
            "--arch",
            "x86-avx2",
            "--model-dir",
            "/models",
        ])
        .expect("derive command accepts exact explicit paths");
        assert!(matches!(
            parsed.command,
            Some(Command::Models {
                command: ModelsSubcommand::Derive(_)
            })
        ));
        assert!(
            Cli::try_parse_from([
                "fnlp",
                "models",
                "derive",
                "--generic",
                "/models/generic.fnlpq",
                "--arch",
                "x86-avx2",
            ])
            .is_err()
        );
    }

    #[test]
    fn convert_reference_invocation_requires_every_named_authority() {
        let missing_converter_commit = match Cli::try_parse_from([
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
        ]) {
            Ok(_) => panic!("the receipt-producing invocation requires converter provenance"),
            Err(error) => error,
        };
        assert_eq!(
            missing_converter_commit.kind(),
            ErrorKind::MissingRequiredArgument
        );

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
            "--converter-commit",
            "0123456789abcdef0123456789abcdef01234567",
        ])
        .expect("fully authority-bound reference conversion invocation parses");
        assert!(matches!(convert.command, Some(Command::Convert(..))));
    }

    #[test]
    fn invalid_planned_tensor_name_refuses_before_source_traversal() {
        let invalid_plan = GenericPayloadPlan {
            tensors: vec![GenericTensorLayout {
                source_name: "model.layers[0].self_attn.k_proj.weight".to_owned(),
                internal_name: "layer[0].attn.k".to_owned(),
                stage: StorageStage::Int8Stage2B,
                shape: vec![1, 1],
                quantization: "portable-int8-row-v1".to_owned(),
                data: OutputRange {
                    name: "layer[0].attn.k.data".to_owned(),
                    offset: 0,
                    len: 1,
                },
                scale: OutputRange {
                    name: "layer[0].attn.k.scale".to_owned(),
                    offset: 0,
                    len: 4,
                },
                row_sum: OutputRange {
                    name: "layer[0].attn.k.row_sum".to_owned(),
                    offset: 0,
                    len: 4,
                },
            }],
            payload_bytes: 1,
            scale_bytes: 4,
            row_sum_bytes: 4,
        };

        let error = validate_generic_tensor_authorities(&invalid_plan)
            .expect_err("an invalid header authority must reject before a source pass");
        assert!(error.contains("invalid tensor.name authority identifier"));
    }

    #[test]
    fn internal_route_spelling_is_not_the_artifact_tensor_authority() {
        let plan = GenericPayloadPlan {
            tensors: vec![GenericTensorLayout {
                source_name: "model.layers.0.self_attn.k_proj.weight".to_owned(),
                internal_name: "layer[0].attn.k".to_owned(),
                stage: StorageStage::Int8Stage2B,
                shape: vec![1, 1],
                quantization: "portable-int8-row-v1".to_owned(),
                data: OutputRange {
                    name: "layer[0].attn.k.data".to_owned(),
                    offset: 0,
                    len: 1,
                },
                scale: OutputRange {
                    name: "layer[0].attn.k.scale".to_owned(),
                    offset: 0,
                    len: 4,
                },
                row_sum: OutputRange {
                    name: "layer[0].attn.k.row_sum".to_owned(),
                    offset: 0,
                    len: 4,
                },
            }],
            payload_bytes: 1,
            scale_bytes: 4,
            row_sum_bytes: 4,
        };

        validate_generic_tensor_authorities(&plan)
            .expect("the source census name, not a route nickname, reaches the artifact header");
    }

    #[test]
    fn logical_tensor_header_uses_the_source_census_authority() {
        let entry = TensorCensusEntry {
            name: "model.layers.0.self_attn.k_proj.weight".to_owned(),
            dtype: SafetensorDtype::Bf16,
            shape: vec![1, 1],
            len: 2,
        };
        let layout = GenericTensorLayout {
            source_name: entry.name.clone(),
            internal_name: "layer[0].attn.k".to_owned(),
            stage: StorageStage::Int8Stage2B,
            shape: entry.shape.clone(),
            quantization: "portable-int8-row-v1".to_owned(),
            data: OutputRange {
                name: "layer[0].attn.k.data".to_owned(),
                offset: 0,
                len: 1,
            },
            scale: OutputRange {
                name: "layer[0].attn.k.scale".to_owned(),
                offset: 0,
                len: 4,
            },
            row_sum: OutputRange {
                name: "layer[0].attn.k.row_sum".to_owned(),
                offset: 0,
                len: 4,
            },
        };

        let tensor = LogicalTensorFirstPass::new(&entry, &layout)
            .expect("source-census tensor declaration is structurally valid");
        assert_eq!(tensor.input.name, entry.name);
    }

    #[test]
    fn frozen_source_census_names_all_satisfy_the_header_authority_grammar() {
        for entry in expected_nanbeige42_census() {
            validate_authority_identifier("tensor.name", &entry.name)
                .expect("every frozen source-census name must be writer-valid");
        }
    }

    #[test]
    fn conversion_stage_is_a_sibling_never_the_final_destination() {
        let destination = Path::new("/models/nanbeige.fnlpq");
        let stage = conversion_staging_path(destination, 0).expect("named final destination");
        let later_stage =
            conversion_staging_path(destination, 1).expect("second retained staging name");
        assert_eq!(stage.parent(), destination.parent());
        assert_ne!(stage, destination);
        assert_ne!(stage, later_stage);
        assert_eq!(
            stage.file_name().and_then(|name| name.to_str()),
            Some(".nanbeige.fnlpq.fnlpq-stage.0")
        );
    }

    #[test]
    fn robot_plan_accepts_memory_budget_alias_and_preserves_explicit_terms() {
        let parsed = Cli::try_parse_from([
            "fnlp",
            "robot",
            "plan",
            "--ctx",
            "8192",
            "--batch",
            "64",
            "--quant",
            "int8",
            "--memory-budget",
            "999999999999",
            "--fixed-mapped-bytes",
            "1000",
            "--fixed-resident-bytes",
            "900",
            "--kv-page-metadata-per-token",
            "7",
        ])
        .expect("robot plan accepts the documented budget alias");
        let Some(Command::Robot {
            command: RobotSubcommand::Plan(command),
        }) = parsed.command
        else {
            panic!("robot plan must parse to its typed command");
        };
        let request = command
            .admission_request()
            .expect("complete plan values convert to admission request");
        assert_eq!(request.context_tokens, 8_192);
        assert_eq!(request.batch_rows, 64);
        assert_eq!(request.kv_quantization, KvCacheQuantization::Int8F32Scales);
        assert_eq!(request.local_memory_budget_bytes, Some(999_999_999_999));
        assert_eq!(
            request
                .fixed_residency
                .expect("fixed terms exist")
                .mapped_bytes,
            1_000
        );
        assert_eq!(
            request
                .fixed_residency
                .expect("fixed terms exist")
                .resident_bytes,
            900
        );
        assert_eq!(request.kv_page_metadata_bytes_per_token, Some(7));
    }
}
