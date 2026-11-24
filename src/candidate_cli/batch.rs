//! Owned, bounded source-task corpus commands. No per-document model reload.
use super::*;
use crate::{
    batch::{BatchLimits, BatchSummary, BatchWork, source::{SourceBatchArgs, SourceMaskBudget,
        quantized::{Int8SourceBatchLimits, MAX_SOURCE_ARGUMENT_BYTES}}},
    native_engine::{constrained_int8, strict_int8::{Int8Work, prefill::Int8PrefillLimits}},
    tasks::{BuiltInTask, keyphrases::KeyphraseOptions, ner::NerOptions,
        summarize::SummaryOptions, ir::TaskBudget},
};
use source::{Kind, SourceHostArgs};
mod writer;
mod extraction;
mod document;
pub(super) use writer::CandidateWriter;
#[cfg(test)] mod tests;

/// Upper bound on the additional JSON provenance bytes for each emitted event.
/// The private writer checks the actual prefix against this bound before IO.
pub(super) const FRAME_ALLOWANCE: u64 = 4096;
pub(super) const IO_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Args)]
pub(crate) struct BatchCommand {
    /// NDJSON {id,text,task_args?} records; '-' reads stdin. No whole-file buffering.
    #[arg(default_value = "-")]
    pub(super) input: PathBuf,
    #[arg(long, value_parser = ["ner", "keyphrases", "summarize", "answer", "extract"])]
    pub(super) task: String,
    #[command(flatten)]
    pub(super) host: SourceHostArgs,
    #[command(flatten)]
    pub(super) document: document::DocumentArgs,
    /// Optional bounded local SourceBatchArgs or ExtractionBatchArgs JSON.
    /// Without defaults, QA/extract records require complete typed task_args.
    #[arg(long, value_name = "FILE")]
    pub(super) defaults: Option<PathBuf>,
    /// Extract only: a shared exact local schema, using the CLI task budget.
    /// Otherwise supply --defaults or a complete schema in every task_args.
    #[arg(long, value_name = "FILE", conflicts_with = "defaults")]
    pub(super) schema: Option<PathBuf>,
    /// Extract only: bind the shared schema's verbatim fields to EACH document.
    #[arg(long, requires = "schema")]
    pub(super) source_membership: bool,
    /// Nonempty records, including malformed documents and flush commands.
    #[arg(long, default_value_t = 1000)]
    pub(super) max_requests: u64,
    /// Whole-stream bytes, including oversized records, delimiters and blank lines.
    #[arg(long, default_value_t = 1024)]
    pub(super) max_input_mib: u64,
    /// All emitted bytes, including per-event candidate provenance and terminal frames.
    #[arg(long, default_value_t = 1024)]
    pub(super) max_output_mib: u64,
    /// Maximum NDJSON bytes before LF, including JSON syntax and any trailing CR.
    #[arg(long, default_value_t = 1_048_576)]
    pub(super) max_line_bytes: usize,
    #[arg(skip)]
    pub(super) encoding: input_encoding::InputEncodingArgs,
    #[arg(skip)]
    pub(super) compute: compute::Args,
    #[command(flatten)]
    pub(super) prefill: source::prefill::SourcePrefillArgs,
}

#[derive(Clone, Copy)]
pub(super) struct CorpusEnvelope {
    pub transport: BatchLimits,
    pub native: Int8SourceBatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
}

pub(super) fn definition() -> clap::Command {
    compute::definition(input_encoding::definition(BatchCommand::augment_args(clap::Command::new("batch")
        .about("Process bounded schema/source-task NDJSON using one resident candidate model")
        .long_about("Run one fixed task over bounded ordered NDJSON. Each event carries non-authoritative candidate provenance. Weights and the native engine are reused. --compute-threads reuses one persistent INT8 row team across every record and chunk, and --prefill-rows selects bounded prompt morsels on that same team. Documents remain ordered; work, mask and tokenizer limits retain their original scope. Terminal run_complete follows physical worker join under the original invocation control. Earlier completed records remain valid if later records fail. Any document failure produces a nonzero process exit. Add --task extract --chunked --schema FILE for complete documents larger than one context. Each successful record retains independent exact-schema JSON strings for every chunk and original-coordinate verbatim evidence when --source-membership is selected. Required fields must be satisfiable per chunk; objects are never merged or repaired. Chunked records cannot override the shared schema or budgets in task_args. No partial document success is emitted."))))
        .mut_arg("prefill_rows", |arg| arg.help("Bounded layer-major INT8 prompt morsels, 1..=64 rows, for all source/extraction records and every chunk of complete-document extraction. Composes with --compute-threads; both coordinator and worker buffers are admitted before private IO"))
        .after_long_help("--input-encoding safe-bpe-v1 uses one bounded tokenizer allowance per complete record: schema plus source for extract, or all untrusted source segments including QA passages and IDs. Extraction's aggregate byte ceiling is max-input-bytes plus the 64-KiB schema allowance, capped at 1 MiB; separate schema and document limits remain enforced. The allowance is not renewed between a record's segments. The maximum record count bounds total preparation; native work and mask quotas remain shared across the whole invocation and flushes. One transient encoder workspace is admitted before private IO. Exact schema numbers, source bytes and citation coordinates are preserved; the policy binds prompt identity and is reported in every compact frame. This option changes prompt tokens and makes no parity, quality or speed claim. Chunked extraction currently requires byte-fallback.")
}
impl BatchCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        command.encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        command.compute = compute::Args::from_matches(matches)?;
        Ok(command)
    }
    pub(super) fn kind(&self) -> Result<Kind, CandidateError> {
        Kind::named(&self.task).ok_or(CandidateError::Arguments)
    }
    pub(super) fn task(&self) -> Result<BuiltInTask, CandidateError> {
        if self.task == "extract" { return Ok(BuiltInTask::Extract); }
        Ok(match self.kind()? { Kind::Ner => BuiltInTask::Ner, Kind::Keyphrases => BuiltInTask::Keyphrases,
            Kind::Summarize => BuiltInTask::Summarize, Kind::Answer => BuiltInTask::Answer })
    }
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits, CorpusEnvelope), CandidateError> {
        self.task()?;
        self.validate_extraction_flags()?;
        if self.document.chunked && !self.encoding.is_default() {
            return Err(CandidateError::Arguments);
        }
        let prefill = self.prefill.limits()?;
        let compute = self.compute.admit_prefill(prefill)?;
        let (mut args, mut limits) = self.host.common_with_encoding(self.input.clone(), &self.encoding)?;
        args.compute = self.compute;
        limits.compute = compute;
        if self.task == "extract" {
            extract::validate_preparation(&self.host, &self.encoding_policy(limits)?, limits)?;
        }
        if self.max_requests == 0 || self.max_requests > 100_000
            || self.max_line_bytes < self.host.max_input_bytes || self.max_line_bytes > 4 * 1024 * 1024
            || self.max_input_mib == 0 || self.max_input_mib > 1024 * 1024
            || self.max_output_mib == 0 || self.max_output_mib > 1024 * 1024
            || self.defaults.as_ref().is_some_and(|p| p.as_os_str().is_empty() || p.as_os_str() == "-")
        { return Err(CandidateError::Arguments); }
        let output_bytes = self.max_output_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
        // The existing runner emits one start, at most one event per request,
        // one EOF flush, and one terminal event. Reserve N+4 framing allowances,
        // including a framing/read failure at the next nonempty record. Blank
        // lines emit nothing; explicit flushes count toward max_requests.
        let framing = self.max_requests.checked_add(4).and_then(|n| n.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let inner_bytes = output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?;
        let per_item = constrained_int8::planned_work(self.host.context_tokens - self.host.max_new_tokens,
            self.host.max_new_tokens).map_err(|_| CandidateError::Arguments)?;
        // Scale independent item-local executions, NOT one fictitious long
        // context whose attention triangle would incorrectly square corpus size.
        let work = scale_work(per_item, self.max_requests)?;
        let native = Int8SourceBatchLimits { max_model_work: work, masks: SourceMaskBudget {
            per_mask: self.host.masks(), max_visits_per_item: self.host.max_mask_node_visits,
            max_visits_per_run: self.host.max_mask_node_visits.checked_mul(self.max_requests)
                .ok_or(CandidateError::Arguments)?,
        } };
        let transport = BatchLimits {
            max_line_bytes: self.max_line_bytes, max_document_bytes: self.host.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: self.max_input_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?,
            max_requests: self.max_requests,
            max_output_line_bytes: self.host.max_result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: inner_bytes,
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        crate::batch::source::quantized::validate_limits(native).map_err(|_| CandidateError::Arguments)?;
        // The writer retains one inner event; no collecting stdout or corpus
        // cursor is secretly assumed free. The host prices its own NDJSON
        // copies, compiler state and output separately.
        let io_bytes = (transport.max_output_line_bytes as u64).checked_add(FRAME_ALLOWANCE)
            .and_then(|n| n.checked_add(IO_BUFFER_BYTES as u64 * 2)).ok_or(CandidateError::Arguments)?;
        let envelope = self.document_envelope(limits, CorpusEnvelope { transport, native, output_bytes, io_bytes })?;
        self.check_scheduling_memory(limits, envelope, prefill)?;
        Ok((args, limits, envelope))
    }
    /// Price the simultaneous resident model, caller preparation, transferred
    /// compiler/transport, encoder, native, prompt, worker and one item output.
    /// A longer stream spends more work without accumulating result memory.
    pub(super) fn schedule_peak_bytes(&self, limits: Limits, envelope: CorpusEnvelope,
        prefill: Option<Int8PrefillLimits>) -> Result<u64, CandidateError> {
        use crate::tokenizer::untrusted::UntrustedEncoding;
        let native = Int8MemoryRequirement::for_context(self.host.context_tokens)
            .map_err(|_| CandidateError::Arguments)?;
        let encoding = match self.encoding_policy(limits)? {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(policy) => policy.requirements()
                .map_err(|_| CandidateError::Arguments)?.workspace_bytes,
        };
        let (document, output_tokens) = self.schedule_document_reserve()?;
        let t = envelope.transport;
        let transferred = (t.max_line_bytes as u64).checked_mul(8)
            .and_then(|n| n.checked_add((t.max_output_line_bytes as u64).checked_mul(4)?))
            .and_then(|n| n.checked_add((t.max_epoch_ids as u64).checked_mul(128)?))
            .and_then(|n| n.checked_add(t.max_epoch_id_bytes as u64))
            .and_then(|n| n.checked_add(limits.preparation_bytes))
            .and_then(|n| n.checked_add(envelope.io_bytes))
            .and_then(|n| n.checked_add(encoding))
            .and_then(|n| n.checked_add(document)).ok_or(CandidateError::Arguments)?;
        let extra = prefill.map(Int8PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        compute::request_peak_bytes(limits, native, t.max_output_line_bytes as u64,
            output_tokens, 0, transferred)?.checked_add(extra).ok_or(CandidateError::Arguments)
    }
    pub(super) fn check_scheduling_memory(&self, limits: Limits, envelope: CorpusEnvelope,
        prefill: Option<Int8PrefillLimits>) -> Result<(), CandidateError> {
        if limits.compute != self.compute.admit_prefill(prefill)? {
            return Err(CandidateError::Arguments);
        }
        if (limits.compute.threads > 1 || prefill.is_some())
            && self.schedule_peak_bytes(limits, envelope, prefill)? > limits.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
    pub(super) fn encoding_policy(&self, limits: Limits)
        -> Result<crate::tokenizer::untrusted::UntrustedEncoding, CandidateError> {
        if self.task == "extract" { return extract::encoding_policy(&self.host, &self.encoding, limits); }
        self.encoding.policy(self.host.max_input_bytes, limits.max_prompt_tokens)
    }
    pub(super) fn load_defaults(&self, text: Option<&str>, ceiling: TaskBudget)
        -> Result<Option<SourceBatchArgs>, CandidateError> {
        let defaults = match text {
            Some(text) => {
                if text.len() > MAX_SOURCE_ARGUMENT_BYTES { return Err(CandidateError::Input); }
                let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
                    max_depth: 16, max_string_bytes: MAX_SOURCE_ARGUMENT_BYTES,
                }).map_err(|_| CandidateError::Input)?;
                Some(serde_json::from_value::<SourceBatchArgs>(value).map_err(|_| CandidateError::Input)?)
            }
            None => match self.kind()? {
                Kind::Ner => Some(SourceBatchArgs::Ner { options: NerOptions::default(), budget: ceiling }),
                Kind::Keyphrases => Some(SourceBatchArgs::Keyphrases { options: KeyphraseOptions::default(), budget: ceiling }),
                Kind::Summarize => Some(SourceBatchArgs::Summarize { options: SummaryOptions::default(), budget: ceiling }),
                Kind::Answer => None, // Never invent or recycle another document's QA evidence.
            },
        };
        if let Some(defaults) = &defaults { self.validate_defaults(defaults, ceiling)?; }
        Ok(defaults)
    }
    fn validate_defaults(&self, defaults: &SourceBatchArgs, ceiling: TaskBudget) -> Result<(), CandidateError> {
        let (kind, budget) = match defaults {
            SourceBatchArgs::Ner { options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?; (Kind::Ner, budget)
            }
            SourceBatchArgs::Keyphrases { options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?; (Kind::Keyphrases, budget)
            }
            SourceBatchArgs::Summarize { options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?; (Kind::Summarize, budget)
            }
            SourceBatchArgs::Answer { passages, options, budget } => {
                options.schema_source().map_err(|_| CandidateError::Planning)?;
                if passages.is_empty() || passages.len() > source::MAX_PASSAGES { return Err(CandidateError::Input); }
                let mut ids = std::collections::BTreeSet::new();
                let mut bytes = 0_usize;
                for passage in passages {
                    if passage.id.is_empty() || passage.id.len() > 128 || passage.id.chars().any(char::is_control)
                        || !ids.insert(passage.id.as_str()) { return Err(CandidateError::Input); }
                    bytes = bytes.checked_add(passage.id.len()).and_then(|n| n.checked_add(passage.text.len()))
                        .filter(|&n| n <= self.host.max_input_bytes).ok_or(CandidateError::Input)?;
                }
                (Kind::Answer, budget)
            }
        };
        budget.validate().map_err(|_| CandidateError::Planning)?;
        // Native KV is allocated for the fixed whole context. Smaller default
        // KV authority cannot cover that storage even for a short document.
        if kind != self.kind()? || budget.max_input_tokens > ceiling.max_input_tokens
            || budget.max_output_tokens > ceiling.max_output_tokens || budget.max_output_bytes > ceiling.max_output_bytes
            || budget.max_grammar_states > ceiling.max_grammar_states || budget.max_kv_bytes != ceiling.max_kv_bytes {
            return Err(CandidateError::Planning);
        }
        Ok(())
    }
    pub(super) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(self,
        input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(diagnostics, "fnlp candidate batch: {}", error.message());
                error.exit_code()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(self,
        input: R, output: W) -> Result<(), CandidateError> {
        let (args, limits, envelope) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::batch_tasks::execute(self, args, limits, envelope, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, envelope, input, output); Err(CandidateError::Unavailable) }
    }
}

fn scale_work(mut work: Int8Work, count: u64) -> Result<Int8Work, CandidateError> {
    if count == 0 { return Err(CandidateError::Arguments); }
    for value in [&mut work.forward_positions, &mut work.projected_logits, &mut work.attention_pairs,
        &mut work.projections.dot_products, &mut work.projections.multiply_accumulates] {
        *value = value.checked_mul(count).ok_or(CandidateError::Arguments)?;
    }
    Ok(work)
}
pub(super) fn completed(summary: BatchSummary) -> Result<(), CandidateError> {
    if summary.failed != 0 { Err(CandidateError::Batch) } else { Ok(()) }
}
