//! Resident finite-score corpora; no free-generation or grammar-mask switches.
use super::*;
use serde::Deserialize;
use crate::{batch::{BatchLimits, BatchWork, classify::ClassificationBatchArgs},
    tasks::{classify::{ClassificationLabel, ClassificationLimits, ClassificationMode, ClassificationPolicy},
        sentiment::{SentimentAxis, SentimentPolicy, batch::SentimentBatchArgs}, ir::TaskBudget}};
use super::{scored::{ScoredArgs, prefill::ScoringPrefillArgs}, batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES}};
use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;
use crate::tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding};

mod judge;

/// Bulk-task selection is distinct from the single classify/sentiment parser.
/// Adding a bulk task must not make the single-request parser accept it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ScoreTask { Classify, Sentiment, Judge }
impl ScoreTask {
    pub(super) fn named(name: &str) -> Option<Self> {
        match name { "classify" => Some(Self::Classify), "sentiment" => Some(Self::Sentiment),
            "judge" => Some(Self::Judge), _ => None }
    }
}

pub(super) const DEFAULTS_BYTES: usize = 1024 * 1024;

#[derive(Args)]
pub(crate) struct ScoreBatchCommand {
    #[arg(long, value_parser = ["classify", "sentiment", "judge"])]
    pub(super) task: String,
    #[command(flatten)]
    pub(super) host: ScoredArgs,
    #[command(flatten)]
    pub(super) prefill: ScoringPrefillArgs,
    #[arg(skip)]
    pub(super) encoding: input_encoding::InputEncodingArgs,
    #[arg(skip)]
    pub(super) judge_encoding: super::judge::EncodingArgs,
    #[command(flatten)]
    pub(super) encoding_work: super::jobs::EncodingWorkArgs,
    #[arg(skip)]
    pub(super) compute: compute::Args,
    /// Local task settings, without document, budget, model or execution identity.
    #[arg(long, value_name = "FILE")]
    pub(super) defaults: Option<PathBuf>,
    /// All nonempty records, including malformed records and flush commands.
    #[arg(long, default_value_t = 1000)]
    max_requests: u64,
    #[arg(long, default_value_t = 1024)]
    max_input_mib: u64,
    #[arg(long, default_value_t = 1024)]
    max_output_mib: u64,
    #[arg(long, default_value_t = 1_048_576)]
    max_line_bytes: usize,
}
#[derive(Clone, Copy)]
pub(super) struct ScoreEnvelope {
    pub transport: BatchLimits,
    pub output_bytes: u64,
    pub io_bytes: u64,
}
pub(super) fn definition() -> clap::Command {
    compute::definition(super::judge::encoding_definition(ScoreBatchCommand::augment_args(clap::Command::new("score-batch")
        .about("Classify, score sentiment or judge with one resident candidate INT8 model")
        .long_about("Run a fixed finite-scoring task over bounded ordered NDJSON. All five model-work limits are WHOLE-RUN ceilings, not renewed per document or flush. Full-vocabulary candidate/EOS scoring, no generated-label shortcut, no network and no calibrated-confidence claim. Judge mode supports pairwise, rubric and whole-source faithfulness; text supplies A, the document or the complete source. All three tasks support --prefill-rows for bounded per-head prompt morsels and --compute-threads for one persistent INT8 row team across the entire stream. Documents remain ordered; both schedules admit their full simultaneous memory before IO. Omitted options preserve serial execution. A threaded run_complete is emitted only after every worker physically joins, under the original invocation deadline. Every event retains non-authoritative candidate provenance; any failed document causes a nonzero exit."))
        .mut_arg("input", |arg| arg.help("NDJSON {id,text,task_args?} records; '-' reads stdin without whole-file buffering"))))
        .after_long_help("All tasks accept --input-encoding safe-bpe-v1. Every record gets one bounded tokenizer allowance shared by all its untrusted document, label, criterion, candidate and claim segments. Judgment additionally accepts --max-tokenizer-input-bytes and --max-tokenizer-output-tokens for complete-record planning work, independently of original line size and native head context; faithfulness includes the full source and every original-byte evidence window. Compact judgment shares one additional --max-total-tokenizer-* invocation allowance across all records, failed preparation and flush boundaries. These four total flags are judgment-only; classification and sentiment retain their per-record tokenizer contract. Native work and transport ceilings remain whole-run limits for every task. One transient encoder workspace and any invocation ledger are reserved beside retained plans and stream buffers before private IO. Every compact event reports its exact policy; judgment events also report invocation caps, including flush and run_complete. Byte-fallback remains the default. Compact input changes prompt tokens without a quality, speed or identical-output claim. Retained score-job commands keep their separate scheduling contract.")
}
impl ScoreBatchCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        command.encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        command.judge_encoding = super::judge::EncodingArgs::from_arg_matches(matches)?;
        command.compute = compute::Args::from_matches(matches)?;
        Ok(command)
    }
    pub(super) fn kind(&self) -> Result<ScoreTask, CandidateError> {
        ScoreTask::named(&self.task).ok_or(CandidateError::Arguments)
    }
    pub(super) fn prefill_limits(&self) -> Result<Option<Int8PrefillLimits>, CandidateError> {
        self.kind()?;
        self.prefill.limits()
    }
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits, ScoreEnvelope), CandidateError> {
        let prefill = self.prefill_limits()?;
        let compute = self.compute.admit_prefill(prefill)?;
        self.encoding_policy()?;
        self.encoding_work_limits()?;
        // Judgment tokenization happens inside the transferred corpus owner.
        // Its complete-record policy is independent of the wire/head ceilings,
        // and peak_bytes prices that actual workspace beside retained plans.
        let selected = if self.kind()? == ScoreTask::Judge {
            input_encoding::InputEncodingArgs::default()
        } else { self.encoding.clone() };
        let (mut args, mut limits) = self.host.common_with_encoding(&selected)?;
        args.compute = self.compute;
        limits.compute = compute;
        if !(1..=100_000).contains(&self.max_requests)
            || self.max_line_bytes < self.host.max_input_bytes || self.max_line_bytes > 4 * 1024 * 1024
            || !(1..=1024 * 1024).contains(&self.max_input_mib)
            || !(1..=1024 * 1024).contains(&self.max_output_mib)
            || self.defaults.as_ref().is_some_and(|p| p.as_os_str().is_empty() || p.as_os_str() == "-") {
            return Err(CandidateError::Arguments);
        }
        let output_bytes = self.max_output_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?;
        let framing = self.max_requests.checked_add(4).and_then(|n| n.checked_mul(FRAME_ALLOWANCE))
            .ok_or(CandidateError::Arguments)?;
        let work = self.host.work_ceiling();
        let transport = BatchLimits {
            max_line_bytes: self.max_line_bytes, max_document_bytes: self.host.max_input_bytes,
            max_id_bytes: 128, max_epoch_ids: 4096, max_epoch_id_bytes: 256 * 1024, max_json_depth: 16,
            max_input_bytes: self.max_input_mib.checked_mul(MIB).ok_or(CandidateError::Arguments)?,
            max_requests: self.max_requests,
            max_output_line_bytes: self.host.max_result_bytes.checked_add(4096).ok_or(CandidateError::Arguments)?,
            max_output_bytes: output_bytes.checked_sub(framing).ok_or(CandidateError::Arguments)?,
            // Never multiply compute authority merely because the transport
            // accepts more records. Every native counter has one finite total.
            max_work: BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits },
        };
        transport.validate().map_err(|_| CandidateError::Arguments)?;
        let io_bytes = (transport.max_output_line_bytes as u64).checked_add(FRAME_ALLOWANCE)
            .and_then(|n| n.checked_add(IO_BUFFER_BYTES as u64 * 2)).ok_or(CandidateError::Arguments)?;
        let envelope = ScoreEnvelope { transport, output_bytes, io_bytes };
        self.check_scheduling_memory(limits, envelope, prefill)?;
        Ok((args, limits, envelope))
    }

    pub(super) fn encoding_policy(&self) -> Result<UntrustedEncoding, CandidateError> {
        if self.kind()? == ScoreTask::Judge {
            self.judge_encoding.policy(&self.encoding)
        } else {
            if !self.judge_encoding.is_default() { return Err(CandidateError::Arguments); }
            self.host.encoding_policy(&self.encoding)
        }
    }

    pub(super) fn encoding_work_limits(&self) -> Result<Option<SafeBpeWorkLimits>, CandidateError> {
        if self.kind()? == ScoreTask::Judge {
            self.encoding_work.limits(&self.encoding).map_err(|_| CandidateError::Arguments)
        } else if self.encoding_work == super::jobs::EncodingWorkArgs::default() {
            Ok(None)
        } else { Err(CandidateError::Arguments) }
    }

    pub(super) fn encoding_workspace_bytes(&self) -> Result<u64, CandidateError> {
        if self.kind()? != ScoreTask::Judge {
            let tokens = self.host.context_tokens.checked_sub(self.host.max_candidate_tokens)
                .and_then(|tokens| tokens.checked_add(1)).ok_or(CandidateError::Arguments)?;
            return self.encoding.workspace_bytes(self.host.max_input_bytes, tokens);
        }
        match self.encoding_policy()? {
            UntrustedEncoding::ByteFallback => Ok(0),
            UntrustedEncoding::SafeBpeV1(policy) => policy.requirements()
                .map_err(|_| CandidateError::Arguments)?.workspace_bytes
                .checked_add(1024).ok_or(CandidateError::Arguments),
        }
    }

    /// Complete simultaneous stream commitment. The caller's preparation,
    /// transferred compiler/IO owner, temporary encoder, coordinator prompt
    /// scratch, fixed worker buffers and one output guard all coexist.
    pub(super) fn peak_bytes(&self, limits: Limits, envelope: ScoreEnvelope,
        prefill: Option<Int8PrefillLimits>) -> Result<u64, CandidateError> {
        let native = Int8MemoryRequirement::for_context(self.host.context_tokens)
            .map_err(|_| CandidateError::Arguments)?;
        let t = envelope.transport;
        let transferred = (t.max_line_bytes as u64).checked_mul(8)
            .and_then(|n| n.checked_add((t.max_output_line_bytes as u64).checked_mul(4)?))
            .and_then(|n| n.checked_add((t.max_epoch_ids as u64).checked_mul(128)?))
            .and_then(|n| n.checked_add(t.max_epoch_id_bytes as u64))
            .and_then(|n| n.checked_add(limits.preparation_bytes))
            .and_then(|n| n.checked_add(envelope.io_bytes)).ok_or(CandidateError::Arguments)?;
        let encoding_bytes = self.encoding_workspace_bytes()?;
        let transferred = transferred.checked_add(encoding_bytes).ok_or(CandidateError::Arguments)?;
        let tokens = self.host.work_ceiling().projected_logits
            / crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64;
        let extra = prefill.map(Int8PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        compute::request_peak_bytes(limits, native, t.max_output_line_bytes as u64,
            tokens, 0, transferred)?.checked_add(extra).ok_or(CandidateError::Arguments)
    }

    pub(super) fn check_scheduling_memory(&self, limits: Limits, envelope: ScoreEnvelope,
        prefill: Option<Int8PrefillLimits>) -> Result<(), CandidateError> {
        if limits.compute != self.compute.admit_prefill(prefill)? {
            return Err(CandidateError::Arguments);
        }
        if (limits.compute.threads > 1 || prefill.is_some()
            || (self.kind()? == ScoreTask::Judge && !self.encoding.is_default()))
            && self.peak_bytes(limits, envelope, prefill)? > limits.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }

    pub(super) fn run_owned<R: Read + Send + 'static, W: Write + Send + 'static>(self,
        input: R, output: W, diagnostics: &mut impl Write) -> ExitCode {
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(diagnostics, "fnlp candidate score-batch: {}", error.message());
                error.exit_code()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static, W: Write + Send + 'static>(self,
        input: R, output: W) -> Result<(), CandidateError> {
        let (args, limits, envelope) = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::scored_tasks::batch::execute(self, args, limits, envelope, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, envelope, input, output); Err(CandidateError::Unavailable) }
    }
    /// Parse run settings before opening model metadata. Defaults cannot
    /// supply a document, budget or executable identity; the CLI owns ceilings.
    pub(super) fn parse_defaults(&self, json: Option<&str>, budget: TaskBudget) -> Result<Defaults, CandidateError> {
        parse_defaults(self.kind()?, &self.host, json, budget)
    }
}
/// Every retained scoring adapter implements bounded prompt scheduling.
/// Keep capability selection exhaustive when another task family is added.
pub(super) fn corpus_prefill(kind: ScoreTask, args: &ScoringPrefillArgs)
    -> Result<Option<Int8PrefillLimits>, CandidateError> {
    match kind { ScoreTask::Classify | ScoreTask::Sentiment | ScoreTask::Judge => args.limits() }
}

/// Shared by live and durable scoring. Retention cannot change option parsing,
/// candidate policies, fallback axes or the caller's exact label spellings.
pub(super) fn parse_defaults(kind: ScoreTask, host: &ScoredArgs, json: Option<&str>, budget: TaskBudget)
    -> Result<Defaults, CandidateError> {
    parse_defaults_with_limits(kind, host.classification_limits(), host.max_input_bytes, json, budget)
}

/// Native INT8 and INT4 callers share one rejecting task-data parser. Only
/// typed task limits and the caller-owned budget cross this boundary.
pub(super) fn parse_defaults_with_limits(kind: ScoreTask, classification: ClassificationLimits,
    max_input_bytes: usize, json: Option<&str>, budget: TaskBudget) -> Result<Defaults, CandidateError> {
    let value = json.map(|text| {
        if text.len() > DEFAULTS_BYTES { return Err(CandidateError::Input); }
        canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
            max_depth: 8, max_string_bytes: DEFAULTS_BYTES,
        }).map_err(|_| CandidateError::Input)
    }).transpose()?;
    budget.validate().map_err(|_| CandidateError::Arguments)?;
    Ok(match kind {
        ScoreTask::Judge => return judge::defaults(value, max_input_bytes, budget),
        ScoreTask::Classify => {
            let Some(value) = value else { return Ok(Defaults::Classify(None)); };
            let input: ClassificationDefaults = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
            let l = classification;
            if input.labels.is_empty() || input.labels.len() > l.max_labels
                || (input.mode == ClassificationMode::Exclusive && input.labels.len() < 2)
                || input.policy.minimum_candidate_weight_ppm > 1_000_000 || input.policy.minimum_margin_ppm > 1_000_000 {
                return Err(CandidateError::Input);
            }
            let mut ids = std::collections::BTreeSet::new(); let mut bytes = 0_usize;
            for label in &input.labels {
                bytes = bytes.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
                    .ok_or(CandidateError::Input)?;
                if label.id.trim().is_empty() || label.id.len() > l.max_label_id_bytes
                    || label.description.len() > l.max_label_description_bytes || bytes > l.max_total_label_bytes
                    || label.id.chars().any(char::is_control) || !ids.insert(label.id.as_str()) {
                    return Err(CandidateError::Input);
                }
            }
            drop(ids);
            Defaults::Classify(Some(ClassificationBatchArgs { labels: input.labels,
                mode: input.mode, policy: input.policy, budget }))
        }
        ScoreTask::Sentiment => {
            let input = match value { Some(value) => serde_json::from_value::<SentimentDefaults>(value)
                .map_err(|_| CandidateError::Input)?, None => SentimentDefaults::default() };
            if input.axes.is_empty() || input.axes.len() > 4
                || input.axes.iter().enumerate().any(|(i, axis)| input.axes[..i].contains(axis))
                || input.policy.minimum_peak_weight_ppm > 1_000_000
                || input.policy.maximum_normalized_entropy_ppm > 1_000_000 {
                return Err(CandidateError::Input);
            }
            Defaults::Sentiment { args: SentimentBatchArgs { axes: input.axes, budget }, policy: input.policy }
        }
    })
}
pub(super) enum Defaults {
    Judge(Option<crate::batch::judge::JudgeBatchArgs>),
    Classify(Option<ClassificationBatchArgs>),
    Sentiment { args: SentimentBatchArgs, policy: SentimentPolicy },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassificationDefaults {
    labels: Vec<ClassificationLabel>,
    #[serde(default = "exclusive")]
    mode: ClassificationMode,
    #[serde(default)]
    policy: ClassificationPolicy,
}
fn exclusive() -> ClassificationMode { ClassificationMode::Exclusive }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SentimentDefaults {
    #[serde(default = "axes")]
    axes: Vec<SentimentAxis>,
    #[serde(default = "policy")]
    policy: SentimentPolicy,
}
fn axes() -> Vec<SentimentAxis> { SentimentAxis::ALL.to_vec() }
fn policy() -> SentimentPolicy {
    SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 }
}
impl Default for SentimentDefaults { fn default() -> Self { Self { axes: axes(), policy: policy() } } }

#[cfg(test)] pub(super) mod tests;
