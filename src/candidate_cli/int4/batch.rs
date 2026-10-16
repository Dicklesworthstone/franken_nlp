//! Finite NDJSON preparation and complete-cohort admission for source INT4.
//! This is one fixed group, not a daemon window, refill schedule or token stream.
use super::*;
use std::io::BufRead;
use crate::candidate_cli::text_batch::{self as wire, Record};
use crate::native_engine::{
    generation::grouped_int4::batch::Int4BatchGenerationBudget,
    strict_int4::{Int4RunBudget, Int4Work, cohort::MAX_INT4_SEQUENCES},
    lmhead::NANBEIGE_VOCAB_SIZE,
};

#[cfg(feature = "asupersync-runtime")]
mod runtime;

pub(super) const PROTOCOL: &str = "fnlp-source-int4-text-batch-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub(super) enum BatchTask { Generate, Chat }
#[derive(Args, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BatchArgs {
    /// Generate records contain id/prompt; chat records contain id/messages.
    #[arg(long, value_enum, default_value = "generate")]
    task: BatchTask,
    /// Complete fixed cohort ceiling, 1..=64; not a per-window quota.
    #[arg(long, default_value_t = 8)]
    max_records: usize,
    /// Total input transport bytes, including all JSON and CR/LF framing.
    #[arg(long, default_value_t = 1_048_576)]
    max_total_input_bytes: u64,
}

pub(super) fn definition() -> clap::Command {
    BatchArgs::augment_args(Int4Args::augment_args(clap::Command::new("text-batch")
        .about("Execute one bounded NDJSON generate/chat cohort on a shared INT4 model")))
        .mut_arg("input", |arg| arg.help("NDJSON file, or '-' for stdin; the complete nonempty cohort is read before model loading"))
        .mut_arg("prefill_rows", |arg| arg.help("Explicit packed prompt/decode schedule: at most this many TOTAL tokens per pass, 1..=64; extra workspace is admitted"))
        .long_about("Explicit non-certified fixed-cohort INT4 execution. Input is 1..=max-records NDJSON records: {\"id\":\"unique\",\"prompt\":\"...\"} for generate; chat uses a messages array. Optional sample_index is a u64. Shared CLI generation options apply independently to each row; IDs, not row order, address seeded draws. --context-tokens is the allocated per-row capacity and --max-input-bytes includes one line's JSON/newlines. All rows, samplers and retained results are admitted together. --max-checkpoints covers the entire native cohort, never each row or pack. By default advance one prompt/feedback token per unfinished sequence per tick. --prefill-rows N instead bounds TOTAL prompt/decode tokens per layer-major pack; it rotates fairly even when N is smaller than the cohort and never batches unsampled feedback. Read, validate and plan the entire input before one shared-model execution; emit one completed JSON batch only after every task finalizes. No provisional token events, successful-prefix result, automatic refill or retry. Sources/snapshot pins and optional save behavior match the single-request source-int4 commands. Blocking IO is cooperative, not preemptible.")
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BatchLimits {
    pub common: Int4Limits,
    pub generation: Int4BatchGenerationBudget,
    pub result_bytes: u64,
    pub publication_bytes: usize,
}
fn times(a: u64, b: u64) -> Result<u64, CandidateError> {
    a.checked_mul(b).ok_or(CandidateError::Arguments)
}
impl BatchArgs {
    pub(super) fn task(self) -> Task {
        match self.task { BatchTask::Generate => Task::Generate, BatchTask::Chat => Task::Chat }
    }
    pub(super) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (args, limits, input, output); Err(CandidateError::Unavailable) }
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<BatchLimits, CandidateError> {
        if self.max_records == 0 || self.max_records > MAX_INT4_SEQUENCES
            || self.max_total_input_bytes == 0 || self.max_total_input_bytes > 64 * MIB {
            return Err(CandidateError::Arguments);
        }
        // Retain source/snapshot exclusivity, real CPU checks and policy bounds.
        let mut common = args.validate()?;
        let rows = self.max_records as u64;
        let contexts = [args.context_tokens; MAX_INT4_SEQUENCES];
        let native = Int4MemoryRequirement::for_cohort(&contexts[..self.max_records], args.group_width)
            .map_err(|_| CandidateError::Arguments)?;
        // TOTAL token width is independent of document count. These packed
        // rails coexist with the ordinary cohort, not one extra copy per row.
        let extra = common.prefill.map(Int4PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        let native_payload = sum(&[native.kv_bytes, native.rope_bytes, native.scratch_payload_bound, extra])?;
        let sampler = times(rows, SAMPLER_BYTES)?;
        let tokens = times(rows, args.max_new_tokens as u64)?;
        let raw_per_row = sum(&[times(args.max_output_bytes as u64, 3)?,
            times(args.max_new_tokens as u64, if args.logprobs { 12 } else { 8 })?, 4096])?;
        let raw_output = times(rows, raw_per_row)?;
        // Same task wrapper rule as Int4ChatBatchLimits, plus CLI ID/source framing.
        let result_bytes = sum(&[times(rows, common.result_bytes as u64)?, 16_384])?;
        let publication_bytes = usize::try_from(sum(&[result_bytes, times(rows, 2048)?, 8192])?)
            .map_err(|_| CandidateError::Arguments)?;
        let preparation_floor = sum(&[128 * MIB,
            times(times(rows, args.max_input_bytes as u64)?, 32)?,
            times(sum(&[publication_bytes as u64, 4096])?, 2)?, 65_536])?;
        if common.preparation_bytes < preparation_floor { return Err(CandidateError::Arguments); }
        // Count the actual complete phase occupancies, never sum mutually exclusive
        // load/save/execute scratch and never price a batch as a single sequence.
        let resident = sum(&[common.load.weight_payload_bytes, ALLOCATOR_BYTES])?;
        let loading = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        let saving = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
        let output_charge = sum(&[times(result_bytes, 4)?, times(tokens, 8)?, 4096])?;
        let execution = sum(&[native_payload, sampler, raw_output, ALLOCATOR_BYTES, output_charge])?;
        common.minimum_memory_bytes = sum(&[common.preparation_bytes, resident,
            loading.max(saving).max(execution), CLEANUP_BYTES])?;
        if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Arguments); }
        // Finite structural upper bound: each row may fill its own context and
        // request at most max_new_tokens full heads. Native plans tighten it.
        let one = Int4Work::for_sequence(args.group_width, 0, args.context_tokens,
            args.max_new_tokens.checked_mul(NANBEIGE_VOCAB_SIZE).ok_or(CandidateError::Arguments)?)
            .map_err(|_| CandidateError::Arguments)?;
        let mut work = Int4Work::default();
        for _ in 0..self.max_records { work = work.checked_add(one).map_err(|_| CandidateError::Arguments)?; }
        Ok(BatchLimits { common, result_bytes, publication_bytes,
            generation: Int4BatchGenerationBudget { native: Int4RunBudget::exact(work),
                max_native_payload_bytes: native_payload, max_sampler_payload_bytes: sampler,
                max_output_payload_bytes: raw_output } })
    }
    pub(super) fn read(self, input: &mut impl BufRead, line_cap: usize,
        mut checkpoint: impl FnMut() -> Result<(), CandidateError>) -> Result<Vec<Record>, CandidateError> {
        // Recheck local bounds because this helper is callable independently of CLI validation.
        if self.max_records == 0 || self.max_records > MAX_INT4_SEQUENCES
            || line_cap == 0 || line_cap > MAX_INPUT_BYTES || self.max_total_input_bytes == 0
            || self.max_total_input_bytes > 64 * MIB { return Err(CandidateError::Arguments); }
        let mut records: Vec<Record> = Vec::new();
        records.try_reserve_exact(self.max_records).map_err(|_| CandidateError::Memory)?;
        let mut consumed = 0;
        loop {
            checkpoint()?;
            if input.fill_buf().map_err(|_| CandidateError::Input)?.is_empty() { break; }
            if records.len() == self.max_records { return Err(CandidateError::Input); }
            let line = wire::read_line(input, line_cap, &mut consumed, self.max_total_input_bytes)?
                .ok_or(CandidateError::Input)?;
            let task = match self.task { BatchTask::Generate => wire::TextTask::Generate, BatchTask::Chat => wire::TextTask::Chat };
            let record = wire::parse_record(task, &line, line_cap)?;
            if records.iter().any(|prior| prior.id == record.id) { return Err(CandidateError::Input); }
            records.push(record);
        }
        if records.is_empty() { return Err(CandidateError::Input); }
        checkpoint()?;
        Ok(records)
    }
}

#[cfg(test)] mod tests;
