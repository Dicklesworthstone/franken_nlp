//! Resident finite-score corpora; no free-generation or grammar-mask switches.
use super::*;
use serde::Deserialize;
use crate::{batch::{BatchLimits, BatchWork, classify::ClassificationBatchArgs},
    tasks::{classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy},
        sentiment::{SentimentAxis, SentimentPolicy, batch::SentimentBatchArgs}, ir::TaskBudget}};
use super::{scored::{Kind, ScoredArgs}, batch::{FRAME_ALLOWANCE, IO_BUFFER_BYTES}};

pub(super) const DEFAULTS_BYTES: usize = 1024 * 1024;

#[derive(Args)]
pub(crate) struct ScoreBatchCommand {
    #[arg(long, value_parser = ["classify", "sentiment"])]
    pub(super) task: String,
    #[command(flatten)]
    pub(super) host: ScoredArgs,
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
    ScoreBatchCommand::augment_args(clap::Command::new("score-batch")
        .about("Classify or score sentiment with one resident candidate INT8 model")
        .long_about("Run a fixed finite-scoring task over bounded ordered NDJSON. All five model-work limits are WHOLE-RUN ceilings, not renewed per document or flush. Full-vocabulary candidate/EOS scoring, no generated-label shortcut, no network and no calibrated-confidence claim. Every event retains non-authoritative candidate provenance; any failed document causes a nonzero exit."))
        .mut_arg("input", |arg| arg.help("NDJSON {id,text,task_args?} records; '-' reads stdin without whole-file buffering"))
}
impl ScoreBatchCommand {
    pub(super) fn kind(&self) -> Result<Kind, CandidateError> {
        Kind::named(&self.task).ok_or(CandidateError::Arguments)
    }
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits, ScoreEnvelope), CandidateError> {
        self.kind()?;
        let (args, limits) = self.host.common()?;
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
        Ok((args, limits, ScoreEnvelope { transport, output_bytes, io_bytes }))
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
        let value = json.map(|text| {
            if text.len() > DEFAULTS_BYTES { return Err(CandidateError::Input); }
            canonjson::parse_str_with_limits(text, canonjson::ParseLimits {
                max_depth: 8, max_string_bytes: DEFAULTS_BYTES,
            }).map_err(|_| CandidateError::Input)
        }).transpose()?;
        Ok(match self.kind()? {
            Kind::Classify => {
                let Some(value) = value else { return Ok(Defaults::Classify(None)); };
                let input: ClassificationDefaults = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
                let l = self.host.classification_limits();
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
            Kind::Sentiment => {
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
}
pub(super) enum Defaults {
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
