//! Frozen exact-JSON gold evaluated through actual bounded native extraction.
use super::*;
use super::native::{file_path, mib, model_input};
use crate::{assay::native_extract,
    grammar::{CompileLimits, runtime::SourceRuntimeLimits, source_index::SourceLanguageLimits},
    native_engine::{int4::PackedInt4Kernel, portable_int8::ProjectionWork, simd::Int8KernelKind,
        strict_int4::{Int4MemoryRequirement, Int4Work}}, tasks::ir::TaskBudget,
    validation::grounded_fields::GroundingBudget};

mod schema;
#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)] mod tests;

const EXTRACT_PROTOCOL: &str = "fnlp.assay.native-extract-int4";

#[derive(Args)]
struct Statistics {
    /// Minimum held-out document count; smaller populations yield insufficient_data.
    #[arg(long, default_value_t = 20)] minimum_samples: usize,
    #[arg(long, default_value_t = 200)] bootstrap_resamples: usize,
    #[arg(long, default_value_t = 0.95)] confidence_level: f64,
    /// Explicit deterministic document-bootstrap seed, never model sampling.
    #[arg(long, default_value_t = 0)] bootstrap_seed: u64,
}
impl Statistics {
    fn options(&self) -> native_extract::EvalOptions {
        native_extract::EvalOptions { minimum_samples: self.minimum_samples,
            bootstrap: BootstrapOptions { resamples: self.bootstrap_resamples,
                confidence_level: self.confidence_level, seed: self.bootstrap_seed } }
    }
}

#[derive(Args)]
struct Arguments {
    #[command(flatten)] common: Common,
    #[command(flatten)] statistics: Statistics,
    /// Explicit candidate snapshot; there is no active-model lookup or fallback.
    #[arg(long, required_unless_present = "schema")] snapshot: Option<PathBuf>,
    /// Model-content digest retained independently from trusted conversion evidence.
    #[arg(long, required_unless_present = "schema")] expected_model_sha256: Option<String>,
    #[arg(long, default_value_t = 128)] group_width: usize,
    /// Process memory-ledger ceiling, including both models for qualification.
    #[arg(long, required_unless_present = "schema")] memory_mib: Option<u64>,
    #[arg(long, default_value_t = Int8KernelKind::Scalar)] kernel: Int8KernelKind,
    #[arg(long, default_value_t = 2048)] context_tokens: usize,
    /// Maximum complete constrained output tokens, including terminal EOS.
    #[arg(long, default_value_t = 512)] max_new_tokens: usize,
    /// Complete native result cap for each held-out document.
    #[arg(long, default_value_t = 1_048_576)] max_result_bytes: usize,
    /// Caller preparation headroom; compiler/source/vocabulary ownership is additional.
    #[arg(long, default_value_t = 512)] preparation_mib: u64,
    #[arg(long, default_value_t = 3600)] timeout_seconds: u64,
    #[arg(long, default_value_t = 1_000_000_000)] snapshot_checkpoints: u64,
    #[arg(long, default_value_t = 1_000_000_000)] max_checkpoints: u64,
    /// All native work ceilings cover held-out rows and both comparison arms.
    #[arg(long, default_value_t = 131_072)] max_forward_positions: u64,
    #[arg(long, default_value_t = 1_000_000_000)] max_projected_logits: u64,
    #[arg(long, default_value_t = 100_000_000_000)] max_attention_pairs: u64,
    #[arg(long, default_value_t = 10_000_000_000_000)] max_dot_products: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000)] max_multiply_accumulates: u64,
    /// Complete suite schema catalogue capacity, checked before typed schemas.
    #[arg(long, default_value_t = 16)] max_schemas: usize,
    /// Byte cap for each exact schema_json string, before native compilation.
    #[arg(long, default_value_t = 262_144)] max_schema_bytes: usize,
    /// Maximum terminal fields in each gold or generated JSON value.
    #[arg(long, default_value_t = 256)] max_fields_per_row: usize,
    #[arg(long, default_value_t = 16_384)] max_json_nodes_per_row: usize,
    #[arg(long, default_value_t = 64)] max_json_depth: usize,
    #[arg(long, default_value_t = 64)] max_languages: usize,
    #[arg(long, default_value_t = 16_384)] max_occurrences_per_row: usize,
    #[arg(long, default_value_t = 4096)] max_grammar_states: u32,
    /// Per-mask traversal cap, independent from the global replay allowance.
    #[arg(long, default_value_t = 2_000_000)] mask_step_node_visits: usize,
    #[arg(long, default_value_t = 1_000_000_000_000)] max_mask_node_visits: u64,
    /// Complete source verification allowance across held-out documents and both arms.
    #[arg(long, default_value_t = 67_108_864)] max_verification_steps: u64,
}

#[derive(Args)]
struct Comparison {
    #[arg(long, required_unless_present = "schema")] baseline_snapshot: Option<PathBuf>,
    #[arg(long, required_unless_present = "schema")] baseline_model_sha256: Option<String>,
    #[arg(long, default_value_t = 128)] baseline_group_width: usize,
    /// Complete explicit field-quality comparison gates.
    #[arg(long, required_unless_present = "schema")] gates: Option<PathBuf>,
    #[arg(long, required_unless_present = "schema")] host_scope: Option<String>,
}

pub(crate) struct Command { args: Arguments, comparison: Option<Comparison> }
pub(super) fn definition(operation: &str) -> clap::Command {
    let command = Arguments::augment_args(clap::Command::new("native-extract-int4")
        .about(if operation == "qualify" { "Compare pinned native extraction models against the same frozen exact-JSON gold" }
            else { "Evaluate actual native schema extraction against frozen exact-JSON gold" }))
        .mut_arg("suite", |arg| arg.help("Strict schema-extraction suite with raw schema_json strings, exact case pin and complete split membership"))
        .mut_arg("observations", |arg| arg.long("cases").help("Strict source/gold NDJSON; expected_json remains an exact JSON string; '-' reads stdin"))
        .after_help("Output is one complete canonical JSON report under fnlp.assay.native-extract-int4 v1. Only locked-test documents execute; every schema and gold value is validated before model loading. Schema validity and source membership do not establish semantic accuracy. Use --schema for exact contracts.");
    if operation == "qualify" { Comparison::augment_args(command) } else { command }
}
impl Command {
    pub(super) fn from_matches(operation: &str, matches: &ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { args: Arguments::from_arg_matches(matches)?,
            comparison: if operation == "qualify" { Some(Comparison::from_arg_matches(matches)?) } else { None } })
    }
    fn operation(&self) -> &'static str { if self.comparison.is_some() { "qualify" } else { "eval" } }
    fn validate(&self) -> Result<(Limits, native_extract::ExtractLimits), Failure> {
        let limits = self.args.common.limits()?; let args = &self.args;
        let extract = args.extract_limits(); extract.validate()?;
        args.statistics.options().validate(limits)?;
        model_input(args.snapshot.as_deref(), args.expected_model_sha256.as_deref(), args.group_width)?;
        mib(args.memory_mib.ok_or_else(invalid_options)?)?; mib(args.preparation_mib)?;
        if args.max_new_tokens == 0 || args.max_new_tokens >= args.context_tokens
            || !(1..=64 * 1024 * 1024).contains(&args.max_result_bytes) || args.preparation_mib < 512
            || !(1..=1024 * 1024).contains(&args.max_schema_bytes)
            || !(1..=86_400).contains(&args.timeout_seconds)
            || !(2..=1_000_000_000_000).contains(&args.snapshot_checkpoints)
            || !(2..=1_000_000_000_000).contains(&args.max_checkpoints)
            || !(1..=16 * 1024 * 1024).contains(&args.max_forward_positions)
            || args.max_projected_logits == 0 || args.max_attention_pairs == 0 || args.max_dot_products == 0
            || args.max_multiply_accumulates == 0 || !(1..=65_536).contains(&args.max_grammar_states)
            || !(1..=1_000_000_000).contains(&args.mask_step_node_visits)
            || !(args.mask_step_node_visits as u64..=1_000_000_000_000).contains(&args.max_mask_node_visits) {
            return Err(invalid_options());
        }
        Int4MemoryRequirement::for_context(args.context_tokens, args.group_width).map_err(|_| invalid_options())?;
        PackedInt4Kernel::detect(args.kernel).map_err(|_| invalid_options())?;
        if let Some(comparison) = &self.comparison {
            model_input(comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(),
                comparison.baseline_group_width)?;
            Int4MemoryRequirement::for_context(args.context_tokens, comparison.baseline_group_width)
                .map_err(|_| invalid_options())?;
            if comparison.gates.as_deref().is_none_or(|path| !file_path(path))
                || comparison.host_scope.as_deref().is_none_or(|scope| !identifier(scope)) { return Err(invalid_options()); }
        }
        Ok((limits, extract))
    }
    fn emit(&self, input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
        if self.args.common.schema {
            let bytes = assay::render(&schema::document(self.operation()), Limits::default())?;
            return output.write_all(&bytes).and_then(|()| output.flush()).map_err(|_| output_failure());
        }
        let (limits, extract) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(self, limits, extract, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (limits, extract, input, output); Err(Failure::new(ErrorCode::Generic, "native_runtime_unavailable")) }
    }
    pub(super) fn run(self, input: &mut impl Read, output: &mut impl Write, errors: &mut impl Write) -> ExitCode {
        match self.emit(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(mut error) => {
                error.protocol = EXTRACT_PROTOCOL;
                if let Ok(mut bytes) = canonjson::canonical_bytes(&error) {
                    bytes.push(b'\n'); let _ = errors.write_all(&bytes).and_then(|()| errors.flush());
                }
                error.code.as_process_exit()
            }
        }
    }
}
impl Arguments {
    fn extract_limits(&self) -> native_extract::ExtractLimits {
        native_extract::ExtractLimits { max_schemas: self.max_schemas, max_fields_per_row: self.max_fields_per_row,
            max_json_nodes_per_row: self.max_json_nodes_per_row, max_json_depth: self.max_json_depth,
            max_languages: self.max_languages, max_occurrences_per_row: self.max_occurrences_per_row,
            max_verification_steps: self.max_verification_steps }
    }
    fn compiler(&self) -> CompileLimits {
        CompileLimits { max_schema_bytes: self.max_schema_bytes, max_output_bytes: self.max_result_bytes,
            max_states: self.max_grammar_states as usize, ..CompileLimits::default() }
    }
    fn source(&self, limits: Limits) -> SourceRuntimeLimits {
        SourceRuntimeLimits { index: SourceLanguageLimits { max_source_bytes: limits.max_line_bytes,
                max_matches: self.max_occurrences_per_row, ..SourceLanguageLimits::default() },
            verification: GroundingBudget { max_fields: self.max_fields_per_row,
                max_matches: self.max_occurrences_per_row, max_scan_steps: self.max_verification_steps } }
    }
    fn work(&self) -> Int4Work {
        Int4Work { forward_positions: self.max_forward_positions, projected_logits: self.max_projected_logits,
            attention_pairs: self.max_attention_pairs, projections: ProjectionWork {
                dot_products: self.max_dot_products, multiply_accumulates: self.max_multiply_accumulates } }
    }
    fn task(&self) -> Result<TaskBudget, Failure> {
        let native = Int4MemoryRequirement::for_context(self.context_tokens, self.group_width).map_err(|_| invalid_options())?;
        Ok(TaskBudget { max_input_tokens: u32::try_from(self.context_tokens.checked_sub(self.max_new_tokens)
                .ok_or_else(invalid_options)?).map_err(|_| invalid_options())?,
            max_output_tokens: u32::try_from(self.max_new_tokens).map_err(|_| invalid_options())?,
            max_output_bytes: self.max_result_bytes as u64, max_grammar_states: self.max_grammar_states,
            max_kv_bytes: native.kv_bytes })
    }
}
