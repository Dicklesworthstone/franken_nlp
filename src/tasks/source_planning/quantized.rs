//! Source-backed NER, keyphrases, cited summaries and passage QA on INT8.
//!
//! Shares the pinned prompt/source compiler and existing semantic finalizers.
//! No BF16 executable is constructed or relabeled. Source grammar, task result,
//! complete envelope and late cancellation all finish inside the native session.

use super::*;
use crate::{
    native_engine::{
        constrained_int8::{self, Int8JsonBudget, Int8JsonError},
        decode::DecodeCancellationKind,
        strict_int8::{Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE},
    },
    tasks::{
        answer::{Int8AnswerFinalizer, ANSWER_TASK_VERSION},
        extract::{ExtractResult, quantized::{self as extract_int8, Int8ExtractPlan, Int8ExtractRun,
            Int8ExtractError, INT8_EXTRACT_VERSION}},
        keyphrases::{self, KEYPHRASES_TASK_VERSION},
        ner::{self, NER_TASK_VERSION},
        summarize::{self, SUMMARIZE_TASK_VERSION},
    },
};

pub mod long;
pub mod capacity;

pub const INT8_SOURCE_EXECUTION: &str = "portable-int8-source-portfolio-v1";

/// Debug/Display redact nested source/compiler content. Typed causes remain
/// available for explicit diagnostic handling, not default prompt logging.
pub enum Int8SourceError {
    Planning(SourcePlanningError),
    Extraction(Int8ExtractError),
    Cancelled(DecodeCancellationKind),
    InvalidResult,
}
impl fmt::Display for Int8SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "int8 source task planning or finalization refused",
            Self::Extraction(_) => "int8 source task constrained execution failed",
            Self::Cancelled(_) => "int8 source task cancelled",
            Self::InvalidResult => "int8 source task complete-result contract diverged",
        })
    }
}
impl fmt::Debug for Int8SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8SourceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Extraction(e) => Some(e), _ => None }
    }
}
impl From<SourcePlanningError> for Int8SourceError { fn from(e: SourcePlanningError) -> Self { Self::Planning(e) } }
impl From<Int8ExtractError> for Int8SourceError { fn from(e: Int8ExtractError) -> Self { Self::Extraction(e) } }
impl From<Int8JsonError> for Int8SourceError { fn from(e: Int8JsonError) -> Self { Self::Extraction(e.into()) } }
impl From<ExtractError> for Int8SourceError { fn from(e: ExtractError) -> Self { Self::Extraction(e.into()) } }
impl From<NerError> for Int8SourceError { fn from(e: NerError) -> Self { Self::Planning(e.into()) } }
impl From<KeyphraseError> for Int8SourceError { fn from(e: KeyphraseError) -> Self { Self::Planning(e.into()) } }
impl From<SummaryError> for Int8SourceError { fn from(e: SummaryError) -> Self { Self::Planning(e.into()) } }
impl From<AnswerError> for Int8SourceError { fn from(e: AnswerError) -> Self { Self::Planning(e.into()) } }
impl Int8SourceError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) => Some(*c), Self::Extraction(e) => e.cancellation(), _ => None }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8SourceTaskRun {
    pub schema_version: u32,
    pub execution: String,
    pub result: SourceTaskResult,
    pub model_work: Int8Work,
}

enum Finalizer {
    Ner(NerOptions),
    Keyphrases(KeyphraseOptions),
    Summarize(SummaryOptions),
    Answer(Int8AnswerFinalizer, AnswerOptions),
}
impl Finalizer {
    fn task_spec(&self) -> &'static str {
        match self { Self::Ner(_) => NER_TASK_VERSION, Self::Keyphrases(_) => KEYPHRASES_TASK_VERSION,
            Self::Summarize(_) => SUMMARIZE_TASK_VERSION, Self::Answer(..) => ANSWER_TASK_VERSION }
    }
    fn finish(&self, raw: ExtractResult, cap: u64) -> Result<SourceTaskResult, Int8SourceError> {
        if raw.output.schema_version != 1 || raw.output.numerics_profile != STRICT_INT8_PROFILE
            || raw.task_spec_version != self.task_spec() { return Err(Int8SourceError::InvalidResult); }
        Ok(match self {
            Self::Ner(options) => SourceTaskResult::Ner(ner::finalize(raw, options, cap)?),
            Self::Keyphrases(options) => SourceTaskResult::Keyphrases(keyphrases::finalize(raw, *options, cap)?),
            Self::Summarize(options) => SourceTaskResult::Summarize(summarize::finalize(raw, *options, cap)?),
            Self::Answer(passages, options) => SourceTaskResult::Answer(passages.finish(raw, *options, cap)?),
        })
    }
}

/// Exact, non-deserializable executable; no Deref or conversion exposes the
/// shared extraction core or a plan that could enter the eager backend.
pub struct PreparedInt8SourceTask {
    extraction: Int8ExtractPlan,
    finalizer: Finalizer,
    budget: TaskBudget,
}
impl SourceTaskPlanner {
    /// The supplied context must ALREADY name the strict INT8 profile/backend.
    /// Existing bounded tokenization/grammar operations are checkpointed around
    /// their calls; this does not promise mid-operation or OS-thread preemption.
    pub fn plan_int8_with_control<C: DecodeStepControl>(&self, request: &SourceTaskRequest,
        context: &PlanContext<'_>, limits: SourcePlanningLimits, control: &mut C)
        -> Result<PreparedInt8SourceTask, Int8SourceError> {
        checkpoint(control)?;
        constrained_int8::check_profile(context.execution_identity())?;
        let kind = request.task(); let budget = request.budget();
        self.check_context_for_profile(kind, context, budget, limits, NumericsProfile::StrictQuantized { version: 1 })?;
        let task_spec = match kind {
            BuiltInTask::Ner => NER_TASK_VERSION, BuiltInTask::Keyphrases => KEYPHRASES_TASK_VERSION,
            BuiltInTask::Summarize => SUMMARIZE_TASK_VERSION, BuiltInTask::Answer => ANSWER_TASK_VERSION,
            _ => return Err(SourcePlanningError::Contract("unsupported int8 source task").into()),
        };
        let encode = |text: &str| self.encoder.encode(text, limits.max_input_bytes, budget.max_input_tokens as usize);
        let mut compile = |document: &SourceDocument, schema: &str| -> Result<Int8ExtractPlan, Int8SourceError> {
            checkpoint(control)?;
            let task = self.task(kind, document, schema, context, budget, limits)?;
            let decode = JsonDecodeOptions { max_new_tokens: budget.max_output_tokens as usize,
                eos_token_id: self.eos, excluded_token_ids: Default::default() };
            let extraction = Int8ExtractPlan::from_builtin(&task, schema, decode, limits.compiler, &self.controls,
                Some((document, limits.source)), task_spec, context.execution_identity().clone())?
                .with_finalizer_version(INT8_SOURCE_EXECUTION)?;
            checkpoint(control)?;
            Ok(extraction)
        };
        let (extraction, finalizer) = match request {
            SourceTaskRequest::Ner { document, options, .. } => {
                let schema = options.schema_source()?; let source = encode(document)?;
                (compile(&source, &schema)?, Finalizer::Ner(options.clone()))
            }
            SourceTaskRequest::Keyphrases { document, options, .. } => {
                let schema = options.schema_source()?; let source = encode(document)?;
                (compile(&source, &schema)?, Finalizer::Keyphrases(*options))
            }
            SourceTaskRequest::Summarize { document, options, .. } => {
                let schema = options.schema_source()?; let source = encode(document)?;
                (compile(&source, &schema)?, Finalizer::Summarize(*options))
            }
            SourceTaskRequest::Answer { question, passages, options, .. } => {
                let schema = options.schema_source()?;
                let answer = AnswerContext::encode(&self.encoder, question, passages, AnswerInputLimits {
                    max_passages: limits.max_passages, max_input_bytes: limits.max_input_bytes,
                    max_input_tokens: budget.max_input_tokens as usize,
                })?;
                let extraction = compile(answer.document(), &schema)?;
                (extraction, Finalizer::Answer(answer.into_int8_finalizer(), *options))
            }
        };
        checkpoint(control)?;
        Ok(PreparedInt8SourceTask { extraction, finalizer, budget })
    }
}
impl PreparedInt8SourceTask {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.extraction.execution_identity() }
    pub fn planned_work(&self) -> Int8Work { self.extraction.planned_work() }
    pub fn prompt_tokens(&self) -> usize { self.extraction.prompt_tokens() }
    pub fn max_result_bytes(&self) -> u64 { self.extraction.max_result_bytes() }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8SourceError> {
        self.extraction.verify_identity(admitted).map_err(Into::into)
    }
    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary,
        work: Int8JsonBudget, control: &mut C) -> Result<Int8SourceTaskRun, Int8SourceError> {
        checkpoint(control)?;
        self.extraction.execute_with(engine, admitted, vocabulary, work, control, |raw| self.finish(raw))
    }
    fn finish(&self, raw: Int8ExtractRun) -> Result<Int8SourceTaskRun, Int8SourceError> {
        if raw.schema_version != 1 || raw.execution != INT8_EXTRACT_VERSION {
            return Err(Int8SourceError::InvalidResult);
        }
        let result = self.finalizer.finish(raw.result, self.max_result_bytes())?;
        let run = Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
            result, model_work: raw.model_work };
        extract_int8::check_size(&run, self.max_result_bytes())?;
        Ok(run)
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int8SourceError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(Int8SourceError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)] mod tests;
