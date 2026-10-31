//! Ordered finite-score streams using one resident grouped-INT4 model.
//! Transport capacity never multiplies or renews native work authority.
use super::*;
use crate::{
    batch::{BatchLimits as TransportLimits, BatchWork,
        scored_int4::{Int4ScoredBatchLimits,
            memory::{Int4ScoredCohortMemoryLimits, MAX_COHORT_OUTPUT_BYTES}}},
    candidate_cli::{batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES},
        scored_batch::{self as shared, Defaults, ScoreTask}},
    tasks::ir::TaskBudget,
};

#[cfg(feature = "asupersync-runtime")]
pub(super) mod runtime;
#[cfg(test)]
mod tests;

const MAX_REQUESTS: u64 = 100_000;
const MAX_TRANSPORT_BYTES: u64 = 16 * 1024 * MIB;
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Eq, PartialEq)]
pub(super) struct BatchArgs {
    scoring: classify::ScoringArgs,
    stream: StreamArgs,
}
impl std::fmt::Debug for BatchArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("ScoredBatchArgs") }
}
#[derive(Args, Clone, Eq, PartialEq)]
struct StreamArgs {
    /// One fixed task for the entire stream; records cannot switch task families.
    #[arg(long, value_parser = ["classify", "sentiment", "judge"])]
    task: String,
    /// Private shared task data; no document, budget, model or executable identity.
    #[arg(long, value_name = "FILE")]
    defaults: Option<PathBuf>,
    /// Maximum buffered document/error records per native cohort; defaults to
    /// the head ceiling narrowed by the aggregate complete-output limit.
    #[arg(long, value_name = "N", requires = "cohort_heads")]
    cohort_documents: Option<usize>,
    /// Whole-stream nonempty records, including malformed input and flush commands.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    /// Entire transport, including oversized records, blank lines and CR/LF.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_input_bytes: u64,
    /// Every emitted byte, including candidate provenance and terminal frames.
    #[arg(long, default_value_t = 67_108_864)]
    max_total_output_bytes: u64,
    /// Maximum NDJSON bytes before LF, including JSON syntax and trailing CR.
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
}

pub(super) fn definition() -> clap::Command {
    StreamArgs::augment_args(classify::definition().name("score-batch")
        .about("Classify, score sentiment or judge ordered NDJSON with one resident INT4 model")
        .long_about("Run one fixed finite-scoring task over bounded ordered {id,text,task_args?} NDJSON, with exact {\"flush\":true} controls. --task selects classify, sentiment or judge. --defaults supplies shared task data without a document, budget, model or execution identity; a non-null task_args replaces the complete defaults under the same host ceiling. Source conversion and independently digest-pinned snapshot loading share one resident model. Execution is serial unless --cohort-heads N explicitly groups complete documents' scoring heads in one bounded native layer-major cohort. --cohort-documents caps buffered document/error records; otherwise the head ceiling and aggregate output cap determine it. Interactive producers must fill a window or send a flush; there is no timer-driven partial-window execution. Every candidate includes terminal EOS and full-vocabulary scoring; classification scores every label, sentiment retains independent axes and one fixed run policy, and judgment preserves all pairwise orders, rubric criteria or faithfulness evidence. Every admitted cohort preflights before its first native forward; a document with more heads than the ceiling is rejected without serial fallback. All five work flags bound the ENTIRE RUN and never renew per item, cohort or flush; accepting more records grants no additional compute. Complete results and errors remain ordered. IDs reset only after a drained flush. Every event carries non-authoritative candidate provenance. Require run_complete AND exit 0 for whole-run success; earlier complete results survive later item failure. No generated-label shortcut, sampling, thinking, tools, durable resume, calibrated confidence or semantic-accuracy claim. Blocking IO is cooperative, not preemptible.")
        .mut_arg("cohort_heads", |arg| arg.hide(false)
            .help("Explicit cross-document layer-major scoring, at most N simultaneous heads (1..=64); complete documents are never split"))
        .mut_arg("input", |arg| arg.help("NDJSON file, or '-' for stdin; one bounded record at a time"))
        .mut_arg("max_input_bytes", |arg| arg.help("Maximum decoded UTF-8 document bytes per record"))
        .mut_arg("max_result_bytes", |arg| arg.help("Maximum complete scored result per record; candidate framing is additional"))
        .mut_arg("max_checkpoints", |arg| arg.help("Nonrenewable cooperative checkpoint allowance for the ENTIRE stream"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-stream forward ceiling across every item, head and epoch"))
        .mut_arg("max_projected_logits", |arg| arg.help("Whole-stream full-vocabulary scoring ceiling"))
        .mut_arg("max_attention_pairs", |arg| arg.help("Whole-stream attention pair ceiling"))
        .mut_arg("max_dot_products", |arg| arg.help("Whole-stream actual grouped native dot-product ceiling"))
        .mut_arg("max_multiply_accumulates", |arg| arg.help("Whole-stream multiply-accumulate ceiling")))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BatchLimits {
    pub common: Int4Limits,
    pub transport: TransportLimits,
    pub native: Int4ScoredBatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
    pub preparation_reserve_bytes: u64,
    pub cohort: Option<CohortWindow>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CohortWindow { pub max_documents: usize, pub max_heads: usize }
impl BatchArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: classify::ScoringArgs::from_batch_matches(matches)?,
            stream: StreamArgs::from_arg_matches(matches)? })
    }
    fn kind(&self) -> Result<ScoreTask, CandidateError> {
        ScoreTask::named(&self.stream.task).ok_or(CandidateError::Arguments)
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<BatchLimits, CandidateError> {
        self.kind()?;
        let stream = &self.stream;
        if !(1..=MAX_REQUESTS).contains(&stream.max_requests)
            || stream.max_line_bytes < args.max_input_bytes
            || !(1..=MAX_LINE_BYTES).contains(&stream.max_line_bytes)
            || !(1..=MAX_TRANSPORT_BYTES).contains(&stream.max_total_input_bytes)
            || !(1..=MAX_TRANSPORT_BYTES).contains(&stream.max_total_output_bytes)
            || stream.defaults.as_ref().is_some_and(|path| path.as_os_str().is_empty() || path.as_os_str() == "-") {
            return Err(CandidateError::Arguments);
        }
        // Keep hardware selection, source/snapshot exclusivity and scored-only
        // policy refusal ahead of defaults, document or model IO.
        let mut common = self.scoring.validate_batch(args)?;
        let work = self.scoring.work_ceiling();
        let native = Int4ScoredBatchLimits { max_item_work: work, max_model_work: work };
        native.validate().map_err(|_| CandidateError::Arguments)?;
        let frames = stream.max_requests.checked_add(4).ok_or(CandidateError::Arguments)?;
        let framing = times(frames, FRAME_ALLOWANCE)?;
        let transport = TransportLimits {
            max_line_bytes: stream.max_line_bytes, max_document_bytes: args.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: stream.max_total_input_bytes, max_requests: stream.max_requests,
            max_output_line_bytes: common.result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: stream.max_total_output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_bytes = sum(&[transport.max_output_line_bytes as u64, FRAME_ALLOWANCE,
            2 * IO_BUFFER_BYTES as u64])?;
        let cohort = match self.scoring.cohort_head_limit() {
            None => {
                if stream.cohort_documents.is_some() { return Err(CandidateError::Arguments); }
                let corpus = sum(&[times(transport.max_line_bytes as u64, 8)?,
                    times(transport.max_output_line_bytes as u64, 4)?,
                    times(transport.max_epoch_ids as u64, 128)?, transport.max_epoch_id_bytes as u64,
                    preparation_reserve_bytes, io_bytes])?;
                // CLI preparation stays owned while the serial host reserves
                // compiler/defaults, one plan, transport and staged output.
                common.minimum_memory_bytes = sum(&[common.minimum_memory_bytes, corpus])?;
                None
            }
            Some(max_heads) => {
                let output_documents = usize::try_from(MAX_COHORT_OUTPUT_BYTES
                    / transport.max_output_line_bytes as u64).map_err(|_| CandidateError::Arguments)?;
                let max_documents = stream.cohort_documents.unwrap_or(max_heads.min(output_documents));
                if max_documents == 0 || max_documents > max_heads { return Err(CandidateError::Arguments); }
                let quote = Int4ScoredCohortMemoryLimits {
                    max_documents, max_heads, context_tokens: args.context_tokens,
                    preparation_reserve_bytes, io_reserve_bytes: io_bytes,
                    allocator_reserve_bytes: ALLOCATOR_BYTES,
                }.requirements(args.group_width, native, transport).map_err(|_| CandidateError::Arguments)?;
                let execution = sum(&[quote.native.kv_bytes, quote.workspace_bytes,
                    quote.input_bytes, quote.output_reservation_bytes])?;
                common.native = quote.native;
                common.minimum_memory_bytes = grouped_memory_floor(args, common, execution)?;
                Some(CohortWindow { max_documents, max_heads })
            }
        };
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(BatchLimits { common, transport, native, output_bytes: stream.max_total_output_bytes,
            io_bytes, preparation_reserve_bytes, cohort })
    }
    fn load_defaults(&self, text: Option<&str>, budget: TaskBudget, args: &Int4Args)
        -> Result<Defaults, CandidateError> {
        shared::parse_defaults_with_limits(self.kind()?, self.scoring.classification_limits(args),
            args.max_input_bytes, text, budget)
    }
    pub(super) fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(
        self, args: Int4Args, input: R, output: W) -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
fn grouped_memory_floor(args: &Int4Args, common: Int4Limits, execution: u64) -> Result<u64, CandidateError> {
    let resident = sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES])?;
    let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
        else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
    let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
    // The shared host quote already includes native memory, all buffered plans
    // plus lookahead, task workspace, IO and complete output reservations.
    // Retain outer CLI preparation and resident weights across every phase.
    sum(&[common.preparation_bytes, resident, loading.max(saving).max(execution), CLEANUP_BYTES])
}
fn times(value: u64, count: u64) -> Result<u64, CandidateError> {
    value.checked_mul(count).ok_or(CandidateError::Arguments)
}
