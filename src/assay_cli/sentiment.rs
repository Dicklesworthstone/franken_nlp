//! Native affect ratings compared with a frozen, source-disjoint human suite.
use super::*;
use super::native::{file_path, mib, model_input};
use crate::{assay::native_sentiment,
    native_engine::{int4::PackedInt4Kernel, portable_int8::ProjectionWork, simd::Int8KernelKind,
        strict_int4::{Int4MemoryRequirement, Int4Work}}, tasks::ir::TaskBudget};
mod schema;
#[cfg(feature = "asupersync-runtime")] mod runtime;
#[cfg(test)] mod tests;
const SENTIMENT_PROTOCOL: &str = "fnlp.assay.native-sentiment-int4";

#[derive(Args)]
struct Statistics {
    /// Minimum distinct held-out documents, never the number of affect axes.
    #[arg(long, default_value_t = 20)] minimum_samples: usize,
    #[arg(long, default_value_t = 200)] bootstrap_resamples: usize,
    #[arg(long, default_value_t = 0.95)] confidence_level: f64,
    /// Deterministic document bootstrap seed; model inference remains greedy scoring.
    #[arg(long, default_value_t = 0)] bootstrap_seed: u64,
}
impl Statistics {
    fn options(&self) -> native_sentiment::EvalOptions {
        native_sentiment::EvalOptions { minimum_samples: self.minimum_samples,
            bootstrap: BootstrapOptions { resamples: self.bootstrap_resamples,
                confidence_level: self.confidence_level, seed: self.bootstrap_seed } }
    }
}
#[derive(Args)]
struct Arguments {
    #[command(flatten)] common: Common,
    #[command(flatten)] statistics: Statistics,
    /// Explicit candidate snapshot; no active-model lookup or fallback.
    #[arg(long, required_unless_present = "schema")] snapshot: Option<PathBuf>,
    /// Independently retained model-content digest from trusted conversion evidence.
    #[arg(long, required_unless_present = "schema")] expected_model_sha256: Option<String>,
    #[arg(long, default_value_t = 128)] group_width: usize,
    /// Process memory-ledger ceiling, including both models for qualification.
    #[arg(long, required_unless_present = "schema")] memory_mib: Option<u64>,
    #[arg(long, default_value_t = Int8KernelKind::Scalar)] kernel: Int8KernelKind,
    #[arg(long, default_value_t = 2048)] context_tokens: usize,
    /// Fixed suite axes must fit this bound; every requested axis executes.
    #[arg(long, default_value_t = 4)] max_axes: usize,
    #[arg(long, default_value_t = 1_048_576)] max_result_bytes: usize,
    #[arg(long, default_value_t = 512)] preparation_mib: u64,
    #[arg(long, default_value_t = 3600)] timeout_seconds: u64,
    #[arg(long, default_value_t = 1_000_000_000)] snapshot_checkpoints: u64,
    #[arg(long, default_value_t = 1_000_000_000)] max_checkpoints: u64,
    /// Every native work ceiling is shared by all documents, axes and both models.
    #[arg(long, default_value_t = 131_072)] max_forward_positions: u64,
    #[arg(long, default_value_t = 1_000_000_000)] max_projected_logits: u64,
    #[arg(long, default_value_t = 100_000_000_000)] max_attention_pairs: u64,
    #[arg(long, default_value_t = 10_000_000_000_000)] max_dot_products: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000)] max_multiply_accumulates: u64,
}
#[derive(Args)]
struct Comparison {
    #[arg(long, required_unless_present = "schema")] baseline_snapshot: Option<PathBuf>,
    #[arg(long, required_unless_present = "schema")] baseline_model_sha256: Option<String>,
    #[arg(long, default_value_t = 128)] baseline_group_width: usize,
    /// Per-axis error and coverage gates; optional rank/conditional-error gates.
    #[arg(long, required_unless_present = "schema")] gates: Option<PathBuf>,
    #[arg(long, required_unless_present = "schema")] host_scope: Option<String>,
}
pub(crate) struct Command { args: Arguments, comparison: Option<Comparison> }
pub(super) fn definition(operation: &str) -> clap::Command {
    let command = Arguments::augment_args(clap::Command::new("native-sentiment-int4")
        .about(if operation == "qualify" { "Compare pinned native affect estimates on one held-out human-rated suite" }
            else { "Evaluate actual pinned native affect estimates against held-out human ratings" }))
        .mut_arg("suite", |arg| arg.help("Strict sentiment suite with fixed axes/policy, pinned cases and complete disjoint splits"))
        .mut_arg("observations", |arg| arg.long("cases").help("Strict text/gold NDJSON; value_milli is -1000..1000; '-' reads stdin"))
        .mut_arg("max_bootstrap_draws", |arg| arg.help("Whole-report document draws across every requested axis and both model arms"))
        .after_help("One complete canonical JSON report under fnlp.assay.native-sentiment-int4 v1. Only locked-test documents execute. Gold value_milli/1000 uses sentiment-v1's normalized -1..1 coordinate, not -100..100. Every axis has MAE, RMSE, tie-aware Spearman, estimated coverage and estimated-only MAE. Bootstrap resamples whole documents with identical draws across axes and paired arms; rank is a point estimate only. Reports grant no calibration, psychological measurement or model activation authority. Use --schema for exact contracts.");
    if operation == "qualify" { Comparison::augment_args(command) } else { command }
}
impl Command {
    pub(super) fn from_matches(operation: &str, matches: &ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { args: Arguments::from_arg_matches(matches)?,
            comparison: if operation == "qualify" { Some(Comparison::from_arg_matches(matches)?) } else { None } })
    }
    fn operation(&self) -> &'static str { if self.comparison.is_some() { "qualify" } else { "eval" } }
    fn validate(&self) -> Result<Limits, Failure> {
        let limits = self.args.common.limits()?; let args = &self.args;
        args.statistics.options().validate(limits)?;
        model_input(args.snapshot.as_deref(), args.expected_model_sha256.as_deref(), args.group_width)?;
        mib(args.memory_mib.ok_or_else(invalid_options)?)?; mib(args.preparation_mib)?;
        if args.context_tokens < 5 || !(1..=native_sentiment::MAX_AXES).contains(&args.max_axes)
            || !(1..=64 * 1024 * 1024).contains(&args.max_result_bytes) || args.preparation_mib < 512
            || !(1..=86_400).contains(&args.timeout_seconds)
            || !(2..=1_000_000_000_000).contains(&args.snapshot_checkpoints)
            || !(2..=1_000_000_000_000).contains(&args.max_checkpoints)
            || !(1..=16 * 1024 * 1024).contains(&args.max_forward_positions)
            || args.max_projected_logits == 0 || args.max_attention_pairs == 0
            || args.max_dot_products == 0 || args.max_multiply_accumulates == 0 { return Err(invalid_options()); }
        Int4MemoryRequirement::for_context(args.context_tokens, args.group_width).map_err(|_| invalid_options())?;
        PackedInt4Kernel::detect(args.kernel).map_err(|_| invalid_options())?;
        if let Some(comparison) = &self.comparison {
            model_input(comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(),
                comparison.baseline_group_width)?;
            Int4MemoryRequirement::for_context(args.context_tokens, comparison.baseline_group_width).map_err(|_| invalid_options())?;
            if comparison.gates.as_deref().is_none_or(|path| !file_path(path))
                || comparison.host_scope.as_deref().is_none_or(|scope| !identifier(scope)) { return Err(invalid_options()); }
        }
        Ok(limits)
    }
    fn emit(&self, input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
        if self.args.common.schema {
            let bytes = assay::render(&schema::document(self.operation()), Limits::default())?;
            return output.write_all(&bytes).and_then(|()| output.flush()).map_err(|_| output_failure());
        }
        let limits = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (limits, input, output); Err(Failure::new(ErrorCode::Generic, "native_runtime_unavailable")) }
    }
    pub(super) fn run(self, input: &mut impl Read, output: &mut impl Write, errors: &mut impl Write) -> ExitCode {
        match self.emit(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(mut error) => {
                error.protocol = SENTIMENT_PROTOCOL;
                if let Ok(mut bytes) = canonjson::canonical_bytes(&error) {
                    bytes.push(b'\n'); let _ = errors.write_all(&bytes).and_then(|()| errors.flush());
                }
                error.code.as_process_exit()
            }
        }
    }
}
impl Arguments {
    fn work(&self) -> Int4Work {
        Int4Work { forward_positions: self.max_forward_positions, projected_logits: self.max_projected_logits,
            attention_pairs: self.max_attention_pairs, projections: ProjectionWork {
                dot_products: self.max_dot_products, multiply_accumulates: self.max_multiply_accumulates } }
    }
    fn task(&self) -> Result<TaskBudget, Failure> {
        let native = Int4MemoryRequirement::for_context(self.context_tokens, self.group_width).map_err(|_| invalid_options())?;
        Ok(TaskBudget { max_input_tokens: u32::try_from(self.context_tokens.checked_sub(4)
                .ok_or_else(invalid_options)?).map_err(|_| invalid_options())?,
            max_output_tokens: 5, max_output_bytes: self.max_result_bytes as u64,
            max_grammar_states: 17, max_kv_bytes: native.kv_bytes })
    }
}
