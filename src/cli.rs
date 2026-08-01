use std::{
    ffi::OsString,
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{CommandFactory, Parser, Subcommand};
use sha2::{Digest, Sha256};

use crate::{
    artifact::converter::{
        plan_generic_payload, prepare_convert_request, stream_routed_bf16_panels, ConvertArch,
        ConvertRequest, ConverterError, GenericPayloadPlan, PreparedConversionInput,
        DEFAULT_PANEL_BYTES,
    },
    artifact::format::{
        framed_sha256, logical_model_sha256, validate_authority_identifier, write_streaming,
        ArchTarget, CanonicalDtype, FnlpqStreamingInput, FnlpqWriteError,
        LogicalTensorStreamingHasher, PackingSetInput, SectionKind, SectionRange, StreamingSection,
        StreamedFnlpq, StreamingSectionHasher, TensorInput,
    },
    artifact::fs_tx::open_ratified_model_root,
    artifact::package::{package_model, verify_model_package, PackageRequest},
    artifact::packing::{NativePackingTarget, TILE_TABLE_VERSION_V1},
    artifact::quantize::{encode_generic_panel, GenericPanelBytes},
    artifact::safetensors::{RowPanel, SafetensorsRangeIndex, TensorCensusEntry},
    error::ErrorCode,
    grammar::{compile_json_schema, CompileLimits, CompiledSchema, SchemaError},
    robot::{self, RobotCommand},
};

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
    /// Bypass the later interactive confirmation step.
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
        Some(Command::Convert(command)) => run_convert_command(command),
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

fn run_convert_command(command: ConvertCommand) -> ExitCode {
    let ConvertCommand {
        source,
        source_manifest,
        recipe,
        arch,
        output,
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
        Ok(prepared) => match plan_generic_payload(&prepared.census, &prepared.routes) {
            Ok(generic) => {
                eprintln!(
                    "CONVERT STAGE=census RESULT=END source-root-sha256={} census-sha256={} tensors={}",
                    prepared.source.source_root_sha256,
                    prepared.census_sha256,
                    prepared.census.len(),
                );
                run_streaming_convert(&request, &prepared, &generic)
            }
            Err(error) => emit_convert_refusal(error),
        },
        Err(error) => emit_convert_refusal(error),
    }
}

fn run_streaming_convert(
    request: &ConvertRequest,
    prepared: &PreparedConversionInput,
    generic: &GenericPayloadPlan,
) -> ExitCode {
    if let Err(error) = require_streaming_conversion_root(&request.output) {
        return emit_streaming_activation_blocked(&request.output, prepared, error);
    }
    eprintln!(
        "CONVERT STAGE=plan RESULT=START tensors={}",
        prepared.census.len()
    );
    let materialized = match read_materialized_sources(&request.source_dir, prepared) {
        Ok(value) => value,
        Err(error) => return emit_streaming_refusal("materialized-sources", error),
    };
    let plan = match build_streaming_envelope_plan(
        &request.source_dir,
        prepared,
        generic,
        &materialized,
    ) {
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
        prepared.census.len(),
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
    emit_unpublished_streaming_blocked(&request.output, &staging_output, prepared, &written)
}

/// Obtain the sole authority permitted to create a conversion stage.
///
/// Today the platform surface fails closed before touching the caller path.
/// Even after that surface is ratified, this deliberately refuses until its
/// opaque root exposes the complete staging, reload, receipt, and activation
/// transaction; an `Ok(RatifiedModelRoot)` alone is not permission to fall
/// back to raw filesystem calls.
fn require_streaming_conversion_root(destination: &Path) -> Result<(), String> {
    let output_root = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    open_ratified_model_root(output_root)
        .map_err(|error| format!("model-root {}: {error}", output_root.display()))?;
    Err(
        "ratified model root exposes no streaming conversion staging/reload/receipt transaction"
            .to_owned(),
    )
}

/// Report an authority refusal before source materialization, quantization, or
/// any filesystem output is attempted.
fn emit_streaming_activation_blocked(
    destination: &Path,
    prepared: &PreparedConversionInput,
    error: impl std::fmt::Display,
) -> ExitCode {
    eprintln!(
        "CONVERT RESULT=BLOCKED stage=activation destination={} source-root-sha256={} census-sha256={} tensors={} reason={error}; no-staging-output-created; no-final-output-created",
        destination.display(),
        prepared.source.source_root_sha256,
        prepared.census_sha256,
        prepared.census.len(),
    );
    ErrorCode::AdmissionOrResourceLimit.as_process_exit()
}

/// Fail closed after a streamed stage is fully durable.  `fs_tx` has not yet
/// supplied the non-replacing activation plus retained reload/receipt path
/// required to expose a canonical artifact, so this CLI must not turn a
/// staged file into a public output by using raw filesystem primitives.
fn emit_unpublished_streaming_blocked(
    destination: &Path,
    staging_output: &Path,
    prepared: &PreparedConversionInput,
    written: &StreamedFnlpq,
) -> ExitCode {
    eprintln!(
        "CONVERT RESULT=BLOCKED stage=activation staging-artifact={} destination={} source-root-sha256={} census-sha256={} tensors={} fnlpq-file-sha256={} staging-bytes={} license-bundle-sha256={} reason=ratified-fs-tx-activation-reload-and-receipt-path-unavailable; no-final-output-created",
        staging_output.display(),
        destination.display(),
        prepared.source.source_root_sha256,
        prepared.census_sha256,
        prepared.census.len(),
        hex_lower(&written.fnlpq_file_sha256),
        written.file_len,
        hex_lower(&written.license_bundle_sha256),
    );
    ErrorCode::AdmissionOrResourceLimit.as_process_exit()
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

/// A raw `create_new` stage would bypass the same root authority that blocks
/// activation.  Keep this fallback fail-closed until `fs_tx` provides a sink
/// through the ratified model-root capability.
fn create_conversion_stage(destination: &Path) -> Result<(PathBuf, fs::File), String> {
    let stage = conversion_staging_path(destination, 0)?;
    Err(format!(
        "ratified fs_tx streaming-stage sink unavailable for {} (would-stage={})",
        destination.display(),
        stage.display(),
    ))
}

fn emit_streaming_refusal(stage: &str, error: impl std::fmt::Display) -> ExitCode {
    eprintln!("CONVERT RESULT=FAIL stage={stage} reason={error}");
    ErrorCode::ArtifactIntegrityOrFormatOrVersion.as_process_exit()
}

struct MaterializedSources {
    model_config: Vec<u8>,
    tokenizer_model: Vec<u8>,
    tokenizer_config: Vec<u8>,
    chat_template: Vec<u8>,
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
    source_dir: &Path,
    prepared: &PreparedConversionInput,
) -> Result<MaterializedSources, String> {
    let model_config = read_verified_source_bytes(source_dir, prepared, "config.json")?;
    let tokenizer_model = read_verified_source_bytes(source_dir, prepared, "tokenizer.model")?;
    let tokenizer_config =
        read_verified_source_bytes(source_dir, prepared, "tokenizer_config.json")?;
    let parsed: serde_json::Value = serde_json::from_slice(&tokenizer_config).map_err(|error| {
        format!("parse verified tokenizer_config.json for chat_template: {error}")
    })?;
    let chat_template = parsed
        .get("chat_template")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "verified tokenizer_config.json lacks string chat_template".to_owned())?
        .as_bytes()
        .to_vec();
    Ok(MaterializedSources {
        model_config,
        tokenizer_model,
        tokenizer_config,
        chat_template,
        license_bundle: embedded_license_bundle()?,
    })
}

fn read_verified_source_bytes(
    source_dir: &Path,
    prepared: &PreparedConversionInput,
    name: &str,
) -> Result<Vec<u8>, String> {
    let expected = prepared
        .source
        .files
        .iter()
        .find(|file| file.name == name)
        .ok_or_else(|| format!("prepared source closure has no {name}"))?;
    let path = source_dir.join(name);
    let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let observed_len = u64::try_from(bytes.len())
        .map_err(|_| format!("{} length does not fit u64", path.display()))?;
    let observed_digest: [u8; 32] = Sha256::digest(&bytes).into();
    let observed_sha256 = hex_lower(&observed_digest);
    if observed_len != expected.bytes || observed_sha256 != expected.sha256 {
        return Err(format!(
            "verified source changed before streaming pass for {}: expected-bytes={} observed-bytes={observed_len} expected-sha256={} observed-sha256={observed_sha256}",
            path.display(),
            expected.bytes,
            expected.sha256
        ));
    }
    Ok(bytes)
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
    source_dir: &Path,
    prepared: &PreparedConversionInput,
    generic: &GenericPayloadPlan,
    materialized: &MaterializedSources,
) -> Result<StreamingEnvelopePlan, String> {
    validate_generic_tensor_authorities(generic)?;
    let (mut tensors, payload_sha256, scales_sha256, row_sums_sha256) =
        first_pass_generic_identities(source_dir, prepared, generic)?;
    tensors.sort_by(|left, right| left.0.name.as_bytes().cmp(right.0.name.as_bytes()));
    let tensor_digests = tensors
        .iter()
        .map(|(_, digest)| *digest)
        .collect::<Vec<_>>();
    let logical_model_sha256 = logical_model_sha256(
        &tensor_digests,
        &[
            ("model_config", materialized.model_config.as_slice()),
            ("tokenizer_model", materialized.tokenizer_model.as_slice()),
            ("tokenizer_config", materialized.tokenizer_config.as_slice()),
            ("chat_template", materialized.chat_template.as_slice()),
        ],
    )
    .map_err(|error| format!("logical model identity: {error}"))?;
    let license_bundle_sha256 = framed_sha256(
        "fnlpq-license-bundle-v1",
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
            &materialized.tokenizer_model,
        )?,
        streaming_small_section(
            MODEL_CONFIG_SECTION,
            SectionKind::ModelConfig,
            8,
            &materialized.model_config,
        )?,
        streaming_small_section(
            TOKENIZER_CONFIG_SECTION,
            SectionKind::TokenizerConfig,
            8,
            &materialized.tokenizer_config,
        )?,
        streaming_small_section(
            CHAT_TEMPLATE_SECTION,
            SectionKind::ChatTemplate,
            8,
            &materialized.chat_template,
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
            source_root_sha256: prepared.source.source_root_sha256.clone(),
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
    let stored_sha256 = framed_sha256("fnlpq-section-v1", &[name.as_bytes(), bytes])
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
    source_dir: &Path,
    prepared: &PreparedConversionInput,
    generic: &GenericPayloadPlan,
) -> Result<(Vec<(TensorInput, [u8; 32])>, [u8; 32], [u8; 32], [u8; 32]), String> {
    if prepared.census.len() != prepared.routes.len()
        || prepared.census.len() != prepared.panels.len()
        || prepared.census.len() != generic.tensors.len()
    {
        return Err(format!(
            "streaming plan count mismatch: census={} routes={} panels={} generic={}",
            prepared.census.len(),
            prepared.routes.len(),
            prepared.panels.len(),
            generic.tensors.len(),
        ));
    }
    let source = SafetensorsRangeIndex::open_pinned_nanbeige42(source_dir)
        .map_err(|error| format!("open verified safetensors source: {error}"))?;
    let mut payload = StreamingSectionHasher::new(GENERIC_PAYLOAD_SECTION, generic.payload_bytes)
        .map_err(|error| format!("start generic payload identity: {error}"))?;
    let mut scales = StreamingSectionHasher::new(GENERIC_SCALES_SECTION, generic.scale_bytes)
        .map_err(|error| format!("start generic scales identity: {error}"))?;
    let mut row_sums = StreamingSectionHasher::new(GENERIC_ROW_SUMS_SECTION, generic.row_sum_bytes)
        .map_err(|error| format!("start generic row-sums identity: {error}"))?;
    let mut tensors = Vec::with_capacity(generic.tensors.len());

    for index in 0..prepared.census.len() {
        let entry = &prepared.census[index];
        let route = &prepared.routes[index];
        let panel_plan = &prepared.panels[index];
        let layout = &generic.tensors[index];
        validate_generic_layout(entry, route, layout)?;
        let mut tensor = LogicalTensorFirstPass::new(entry, layout)?;
        stream_routed_bf16_panels(
            std::slice::from_ref(entry),
            std::slice::from_ref(route),
            std::slice::from_ref(panel_plan),
            |source_entry, panel| {
                source
                    .read_range(&source_entry.name, panel)
                    .map_err(ConverterError::Safetensors)
            },
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
    materialized: &MaterializedSources,
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
            emit_materialized_section(section, &materialized.tokenizer_model, sink)
        }
        (SectionKind::ModelConfig, MODEL_CONFIG_SECTION) => {
            emit_materialized_section(section, &materialized.model_config, sink)
        }
        (SectionKind::TokenizerConfig, TOKENIZER_CONFIG_SECTION) => {
            emit_materialized_section(section, &materialized.tokenizer_config, sink)
        }
        (SectionKind::ChatTemplate, CHAT_TEMPLATE_SECTION) => {
            emit_materialized_section(section, &materialized.chat_template, sink)
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
    if prepared.census.len() != prepared.routes.len()
        || prepared.census.len() != prepared.panels.len()
        || prepared.census.len() != generic.tensors.len()
    {
        return Err(FnlpqWriteError::StoredIdentity {
            section: section.name.clone(),
            expected: "prepared conversion arrays with equal lengths".to_owned(),
            actual: format!(
                "census={} routes={} panels={} generic={}",
                prepared.census.len(),
                prepared.routes.len(),
                prepared.panels.len(),
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
    let source = SafetensorsRangeIndex::open_pinned_nanbeige42(source_dir).map_err(|error| {
        FnlpqWriteError::Io {
            operation: "open verified safetensors source for section replay",
            detail: error.to_string(),
        }
    })?;
    let mut observed_len = 0_u64;
    for index in 0..prepared.census.len() {
        let entry = &prepared.census[index];
        let route = &prepared.routes[index];
        let panel_plan = &prepared.panels[index];
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
            |source_entry, panel| {
                source
                    .read_range(&source_entry.name, panel)
                    .map_err(ConverterError::Safetensors)
            },
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
        if completed % EMISSION_PROGRESS_TENSOR_INTERVAL == 0 || completed == prepared.census.len()
        {
            eprintln!(
                "CONVERT STAGE=emission section={} tensors={}/{}",
                section.name,
                completed,
                prepared.census.len(),
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
    use std::{ffi::OsString, io::Cursor, path::Path, process::ExitCode};

    use clap::Parser;

    use super::{
        cli_main_with_reader, conversion_staging_path, require_streaming_conversion_root,
        validate_generic_tensor_authorities, Cli, Command, LogicalTensorFirstPass,
        ModelsSubcommand, ReleaseSubcommand,
    };
    use crate::artifact::converter::{
        GenericPayloadPlan, GenericTensorLayout, OutputRange, StorageStage,
    };
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
        assert!(Cli::try_parse_from([
            "fnlp",
            "models",
            "derive",
            "--generic",
            "/models/generic.fnlpq",
            "--arch",
            "x86-avx2",
        ])
        .is_err());
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
    fn conversion_refuses_before_any_staging_without_a_ratified_root() {
        let error = require_streaming_conversion_root(Path::new("/models/nanbeige.fnlpq"))
            .expect_err("all current model-root targets must fail closed");
        assert!(error.contains("FS_TXN refused"));
    }
}
