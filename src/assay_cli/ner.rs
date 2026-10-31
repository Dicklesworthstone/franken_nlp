//! Explicit native NER replay; gold spans never become model instructions.
use super::*;
use super::native::{file_path, mib, model_input};
use crate::{assay::native_ner,
    native_engine::{int4::PackedInt4Kernel, portable_int8::ProjectionWork, simd::Int8KernelKind,
        strict_int4::{Int4MemoryRequirement, Int4Work}}, tasks::ir::TaskBudget};

mod schema;
#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)] mod tests;

const NER_PROTOCOL: &str = "fnlp.assay.native-ner-int4";

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
    fn options(&self) -> native_ner::EvalOptions {
        native_ner::EvalOptions { minimum_samples: self.minimum_samples,
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
    /// Independently retained model-content digest from trusted conversion evidence.
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
    /// Caller preparation headroom; source/compiler/vocabulary ownership is additional.
    #[arg(long, default_value_t = 512)] preparation_mib: u64,
    #[arg(long, default_value_t = 3600)] timeout_seconds: u64,
    #[arg(long, default_value_t = 1_000_000_000)] snapshot_checkpoints: u64,
    #[arg(long, default_value_t = 1_000_000_000)] max_checkpoints: u64,
    /// All native work ceilings apply across held-out rows and both arms.
    #[arg(long, default_value_t = 131_072)] max_forward_positions: u64,
    #[arg(long, default_value_t = 1_000_000_000)] max_projected_logits: u64,
    #[arg(long, default_value_t = 100_000_000_000)] max_attention_pairs: u64,
    #[arg(long, default_value_t = 10_000_000_000_000)] max_dot_products: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000)] max_multiply_accumulates: u64,
    /// Cap each gold span list and the suite's predicted entity-count option before typed row parsing.
    #[arg(long, default_value_t = 64)] max_spans_per_row: usize,
    #[arg(long, default_value_t = 16_384)] max_occurrences_per_row: usize,
    #[arg(long, default_value_t = 64)] max_languages: usize,
    #[arg(long, default_value_t = 4096)] max_grammar_states: u32,
    /// Per-mask traversal ceiling; independent from the total replay allowance.
    #[arg(long, default_value_t = 2_000_000)] mask_step_node_visits: usize,
    #[arg(long, default_value_t = 1_000_000_000_000)] max_mask_node_visits: u64,
    /// Independent source verification steps shared by all held-out rows and both arms.
    #[arg(long, default_value_t = 67_108_864)] max_verification_steps: u64,
}

#[derive(Args)]
struct Comparison {
    #[arg(long, required_unless_present = "schema")] baseline_snapshot: Option<PathBuf>,
    #[arg(long, required_unless_present = "schema")] baseline_model_sha256: Option<String>,
    #[arg(long, default_value_t = 128)] baseline_group_width: usize,
    /// Explicit span-quality gates; never inferred from model scores.
    #[arg(long, required_unless_present = "schema")] gates: Option<PathBuf>,
    #[arg(long, required_unless_present = "schema")] host_scope: Option<String>,
}

pub(crate) struct Command { args: Arguments, comparison: Option<Comparison> }
pub(super) fn definition(operation: &str) -> clap::Command {
    let command = Arguments::augment_args(clap::Command::new("native-ner-int4")
        .about(if operation == "qualify" { "Compare pinned native NER models on exactly the same held-out source spans" }
            else { "Evaluate actual pinned native NER against a frozen source-span suite" }))
        .mut_arg("suite", |arg| arg.help("Strict named_entity_recognition suite with pinned cases, NER options and complete split membership"))
        .mut_arg("observations", |arg| arg.long("cases").help("Strict labeled-source NDJSON; '-' reads stdin; supplied predictions are rejected"))
        .after_help("Output is one complete canonical JSON report under fnlp.assay.native-ner-int4 v1. Only locked-test documents are replayed; every split is validated and bound. No mention confidence, probability calibration, or model activation is implied. Use --schema for exact contracts.");
    if operation == "qualify" { Comparison::augment_args(command) } else { command }
}
impl Command {
    pub(super) fn from_matches(operation: &str, matches: &ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { args: Arguments::from_arg_matches(matches)?,
            comparison: if operation == "qualify" { Some(Comparison::from_arg_matches(matches)?) } else { None } })
    }
    fn operation(&self) -> &'static str { if self.comparison.is_some() { "qualify" } else { "eval" } }
    fn validate(&self) -> Result<(Limits, native_ner::NerLimits), Failure> {
        let limits = self.args.common.limits()?; let args = &self.args;
        let ner = args.ner_limits(); ner.validate()?;
        args.statistics.options().validate(limits)?;
        model_input(args.snapshot.as_deref(), args.expected_model_sha256.as_deref(), args.group_width)?;
        mib(args.memory_mib.ok_or_else(invalid_options)?)?; mib(args.preparation_mib)?;
        if args.max_new_tokens == 0 || args.max_new_tokens >= args.context_tokens
            || !(1..=64 * 1024 * 1024).contains(&args.max_result_bytes) || args.preparation_mib < 512
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
        Ok((limits, ner))
    }
    fn emit(&self, input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
        if self.args.common.schema {
            let bytes = assay::render(&schema::document(self.operation()), Limits::default())?;
            return output.write_all(&bytes).and_then(|()| output.flush()).map_err(|_| output_failure());
        }
        let (limits, ner) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(self, limits, ner, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (limits, ner, input, output); Err(Failure::new(ErrorCode::Generic, "native_runtime_unavailable")) }
    }
    pub(super) fn run(self, input: &mut impl Read, output: &mut impl Write, errors: &mut impl Write) -> ExitCode {
        match self.emit(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(mut error) => {
                error.protocol = NER_PROTOCOL;
                if let Ok(mut bytes) = canonjson::canonical_bytes(&error) {
                    bytes.push(b'\n'); let _ = errors.write_all(&bytes).and_then(|()| errors.flush());
                }
                error.code.as_process_exit()
            }
        }
    }
}
impl Arguments {
    fn ner_limits(&self) -> native_ner::NerLimits {
        native_ner::NerLimits { max_spans_per_row: self.max_spans_per_row,
            max_occurrences_per_row: self.max_occurrences_per_row, max_languages: self.max_languages,
            max_verification_steps: self.max_verification_steps }
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
