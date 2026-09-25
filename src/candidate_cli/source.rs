//! Raw source-task commands. Options are data, never prompt instructions.
use super::*;
use serde::{Deserialize, de::DeserializeOwned};
use crate::{
    grammar::{CompileLimits, runtime::SourceRuntimeLimits, mask::MaskWorkLimits},
    tasks::{answer::{AnswerOptions, AnswerPassage}, keyphrases::KeyphraseOptions,
        ner::NerOptions, summarize::SummaryOptions, ir::TaskBudget,
        source_planning::{SourcePlanningLimits, SourceTaskRequest}},
};

pub(super) const OPTIONS_BYTES: usize = 16 * 1024;
pub(super) const MAX_PASSAGES: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Kind { Ner, Keyphrases, Summarize, Answer }
impl Kind {
    pub(super) fn named(name: &str) -> Option<Self> {
        match name { "ner" => Some(Self::Ner), "keyphrases" => Some(Self::Keyphrases),
            "summarize" => Some(Self::Summarize), "answer" => Some(Self::Answer), _ => None }
    }
    pub(super) fn name(self) -> &'static str {
        match self { Self::Ner => "ner", Self::Keyphrases => "keyphrases",
            Self::Summarize => "summarize", Self::Answer => "answer" }
    }
}

/// Structured-task resource options exclude every free-text sampler switch.
/// Validation delegates shared host arithmetic to CandidateArgs, so the two
/// command families cannot disagree on weight/context/memory ceilings.
#[derive(Args)]
pub(super) struct SourceHostArgs {
    /// Explicit local current-candidate INT8 .fnlpq. No discovery or download.
    #[arg(long, value_name = "FILE")]
    pub model: PathBuf,
    /// Process ledger ceiling in MiB, not an operating-system RSS limit.
    #[arg(long)]
    pub memory_mib: u64,
    #[arg(long, default_value_t = 2048)]
    pub context_tokens: usize,
    /// Maximum constrained tokens, including the terminal EOS.
    #[arg(long, default_value_t = 512)]
    pub max_new_tokens: usize,
    /// Whole typed task-result byte ceiling; candidate provenance adds 4096.
    #[arg(long, default_value_t = 1_048_576)]
    pub max_result_bytes: usize,
    /// Raw UTF-8 input ceiling, including JSON syntax for passage QA.
    #[arg(long, default_value_t = 65_536)]
    pub max_input_bytes: usize,
    #[arg(long, default_value_t = 6144)]
    pub max_weight_mib: u64,
    /// Modeled preparation reserve retained through delivery. Hosted source
    /// execution separately reserves this amount for its transferred inputs.
    #[arg(long, default_value_t = 512)]
    pub preparation_mib: u64,
    /// Cooperative deadline for the entire invocation; blocking IO is not preempted.
    #[arg(long, default_value_t = 3600)]
    pub timeout_seconds: u64,
    #[arg(long, default_value_t = 1_000_000_000)]
    pub max_checkpoints: u64,
    #[arg(long, default_value_t = 4096)]
    pub max_grammar_states: u32,
    /// Total grammar-mask node visits for one request.
    #[arg(long, default_value_t = 1_000_000_000)]
    pub max_mask_node_visits: u64,
    /// Per-mask traversal ceiling, separate from the aggregate visit budget.
    #[arg(long, default_value_t = 2_000_000)]
    pub mask_step_node_visits: u64,
}
impl SourceHostArgs {
    pub(super) fn common(&self, input: PathBuf) -> Result<(CandidateArgs, Limits), CandidateError> {
        if self.max_new_tokens >= self.context_tokens || self.preparation_mib < 512
            || self.max_grammar_states == 0 || self.max_grammar_states > 65_536
            || self.mask_step_node_visits == 0 || self.mask_step_node_visits > 1_000_000_000
            || self.max_mask_node_visits < self.mask_step_node_visits
            || self.max_mask_node_visits > 1_000_000_000_000
        { return Err(CandidateError::Arguments); }
        let common = CandidateArgs { input, model: self.model.clone(), memory_mib: self.memory_mib,
            context_tokens: self.context_tokens, max_new_tokens: self.max_new_tokens,
            max_output_bytes: self.max_result_bytes, max_input_bytes: self.max_input_bytes,
            max_weight_mib: self.max_weight_mib, preparation_mib: self.preparation_mib,
            timeout_seconds: self.timeout_seconds, max_checkpoints: self.max_checkpoints,
            seed: None, temperature_milli: None, top_k: None, top_p_ppm: None, logprobs: false };
        let limits = common.validate()?;
        Ok((common, limits))
    }
    pub(super) fn task_budget(&self, limits: Limits) -> TaskBudget {
        TaskBudget { max_input_tokens: (self.context_tokens - self.max_new_tokens) as u32,
            max_output_tokens: self.max_new_tokens as u32, max_output_bytes: self.max_result_bytes as u64,
            max_grammar_states: self.max_grammar_states, max_kv_bytes: limits.kv_bytes }
    }
    pub(super) fn planning(&self) -> SourcePlanningLimits {
        SourcePlanningLimits { max_input_bytes: self.max_input_bytes,
            max_context_tokens: self.context_tokens, max_passages: MAX_PASSAGES,
            compiler: CompileLimits { max_states: self.max_grammar_states as usize,
                max_output_bytes: self.max_result_bytes, ..CompileLimits::default() },
            source: SourceRuntimeLimits::default() }
    }
    pub(super) fn masks(&self) -> MaskWorkLimits {
        MaskWorkLimits { max_trie_node_visits: self.mask_step_node_visits as usize, checkpoint_interval_nodes: 256 }
    }
}

#[derive(Args)]
pub(super) struct SourceArgs {
    /// Exact source text; answer instead accepts {"question":...,"passages":[{"id":...,"text":...}]}.
    /// '-' reads stdin. Whitespace and Unicode are preserved.
    #[arg(default_value = "-")]
    pub input: PathBuf,
    #[command(flatten)]
    pub host: SourceHostArgs,
    /// Optional local JSON file containing the selected task's complete options.
    /// Defaults are used only when omitted; unknown/partial options are rejected.
    #[arg(long, value_name = "FILE")]
    pub options: Option<PathBuf>,
}

pub(crate) struct SourceCommand { pub(super) kind: Kind, pub(super) args: SourceArgs }

pub(super) fn definitions() -> Vec<clap::Command> {
    [(Kind::Ner, "Find source-bound named entities with exact occurrence spans"),
        (Kind::Keyphrases, "Rank exact source keyphrases"),
        (Kind::Summarize, "Produce a summary with exact source citations"),
        (Kind::Answer, "Answer a question using only explicit evidence passages")]
        .into_iter().map(|(kind, about)| SourceArgs::augment_args(clap::Command::new(kind.name()).about(about))).collect()
}
impl SourceCommand {
    pub(super) fn from_matches(kind: Kind, matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { kind, args: SourceArgs::from_arg_matches(matches)? })
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (common, limits) = self.args.host.common(self.args.input.clone())?;
        if self.args.options.as_ref().is_some_and(|path| path.as_os_str().is_empty() || path.as_os_str() == "-") {
            return Err(CandidateError::Arguments);
        }
        #[cfg(feature = "asupersync-runtime")]
        { runtime::source_tasks::execute(self, common, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, common, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

/// Parse the full typed options through the rejecting JSON boundary. A file
/// cannot supply a task selector, instruction, budget or replacement identity.
fn options<T: DeserializeOwned + Default>(json: Option<&str>) -> Result<T, CandidateError> {
    match json {
        None => Ok(T::default()),
        Some(json) => {
            if json.len() > OPTIONS_BYTES { return Err(CandidateError::Input); }
            let value = canonjson::parse_str_with_limits(json, canonjson::ParseLimits {
                max_depth: 8, max_string_bytes: OPTIONS_BYTES,
            }).map_err(|_| CandidateError::Input)?;
            serde_json::from_value(value).map_err(|_| CandidateError::Input)
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnswerInput { question: String, passages: Vec<AnswerPassage> }

pub(super) fn request(kind: Kind, text: String, option_json: Option<&str>,
    budget: TaskBudget, max_input_bytes: usize) -> Result<SourceTaskRequest, CandidateError> {
    if text.len() > max_input_bytes { return Err(CandidateError::Input); }
    // Validate options and produce the same code-owned schema the actual
    // planner consumes BEFORE a model file is opened or weights are loaded.
    Ok(match kind {
        Kind::Ner => {
            let options = options::<NerOptions>(option_json)?;
            options.schema_source().map_err(|_| CandidateError::Planning)?;
            SourceTaskRequest::Ner { document: text, options, budget }
        }
        Kind::Keyphrases => {
            let options = options::<KeyphraseOptions>(option_json)?;
            options.schema_source().map_err(|_| CandidateError::Planning)?;
            SourceTaskRequest::Keyphrases { document: text, options, budget }
        }
        Kind::Summarize => {
            let options = options::<SummaryOptions>(option_json)?;
            options.schema_source().map_err(|_| CandidateError::Planning)?;
            SourceTaskRequest::Summarize { document: text, options, budget }
        }
        Kind::Answer => {
            let options = options::<AnswerOptions>(option_json)?;
            options.schema_source().map_err(|_| CandidateError::Planning)?;
            let value = canonjson::parse_str_with_limits(&text, canonjson::ParseLimits {
                max_depth: 4, max_string_bytes: max_input_bytes,
            }).map_err(|_| CandidateError::Input)?;
            let input: AnswerInput = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
            if input.question.trim().is_empty() || input.passages.is_empty() || input.passages.len() > MAX_PASSAGES {
                return Err(CandidateError::Input);
            }
            let mut ids = std::collections::BTreeSet::new();
            if input.passages.iter().any(|p| p.id.is_empty() || !ids.insert(p.id.as_str())) {
                return Err(CandidateError::Input);
            }
            drop(ids);
            SourceTaskRequest::Answer { question: input.question, passages: input.passages, options, budget }
        }
    })
}

#[cfg(test)] pub(super) mod tests;
