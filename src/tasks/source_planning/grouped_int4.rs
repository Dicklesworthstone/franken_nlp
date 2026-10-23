//! Source-backed NER, keyphrases, cited summaries and passage QA on INT4.
//!
//! The pinned source compiler and independent semantic validators are shared
//! with the other native profiles. This sealed adapter binds the actual INT4
//! model, group partition and task policy; no INT8 plan is relabeled. Complete
//! result validation remains inside the native session's cancellation/drain
//! boundary. Citation membership does not certify factual support or quality.

use super::*;
use crate::{
    native_engine::{
        constrained_int4::{self, Int4JsonBudget, Int4JsonError},
        decode::DecodeCancellationKind,
        strict_int4::{Int4Work, StrictInt4Engine, STRICT_INT4_PROFILE},
    },
    tasks::{
        answer::{Int4AnswerFinalizer, ANSWER_TASK_VERSION},
        extract::{ExtractResult, grouped_int4::{Int4ExtractError, Int4ExtractPlan,
            Int4ExtractRun, INT4_EXTRACT_VERSION}, quantized::check_size},
        keyphrases::{self, KEYPHRASES_TASK_VERSION},
        ner::{self, NER_TASK_VERSION},
        summarize::{self, SUMMARIZE_TASK_VERSION},
    },
};

pub const INT4_SOURCE_EXECUTION: &str = "grouped-int4-source-portfolio-v1";

/// Default diagnostics redact source/compiler content. Explicit diagnostic
/// handling can inspect typed causes without logging a private document.
pub enum Int4SourceError {
    Planning(SourcePlanningError),
    Extraction(Int4ExtractError),
    Cancelled(DecodeCancellationKind),
    InvalidResult,
}
impl fmt::Display for Int4SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "INT4 source task planning or finalization refused",
            Self::Extraction(_) => "INT4 source task constrained execution failed",
            Self::Cancelled(_) => "INT4 source task cancelled",
            Self::InvalidResult => "INT4 source task complete-result contract diverged",
        })
    }
}
impl fmt::Debug for Int4SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int4SourceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Extraction(e) => Some(e), _ => None }
    }
}
impl From<SourcePlanningError> for Int4SourceError { fn from(e: SourcePlanningError) -> Self { Self::Planning(e) } }
impl From<Int4ExtractError> for Int4SourceError { fn from(e: Int4ExtractError) -> Self { Self::Extraction(e) } }
impl From<Int4JsonError> for Int4SourceError { fn from(e: Int4JsonError) -> Self { Self::Extraction(e.into()) } }
impl From<ExtractError> for Int4SourceError { fn from(e: ExtractError) -> Self { Self::Extraction(e.into()) } }
impl From<NerError> for Int4SourceError { fn from(e: NerError) -> Self { Self::Planning(e.into()) } }
impl From<KeyphraseError> for Int4SourceError { fn from(e: KeyphraseError) -> Self { Self::Planning(e.into()) } }
impl From<SummaryError> for Int4SourceError { fn from(e: SummaryError) -> Self { Self::Planning(e.into()) } }
impl From<AnswerError> for Int4SourceError { fn from(e: AnswerError) -> Self { Self::Planning(e.into()) } }
impl Int4SourceError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(c) => Some(*c), Self::Extraction(e) => e.cancellation(), _ => None }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4SourceTaskRun {
    pub schema_version: u32,
    pub execution: String,
    pub group_width: usize,
    pub result: SourceTaskResult,
    pub model_work: Int4Work,
}

enum Finalizer {
    Ner(NerOptions),
    Keyphrases(KeyphraseOptions),
    Summarize(SummaryOptions),
    Answer(Int4AnswerFinalizer, AnswerOptions),
}
impl Finalizer {
    fn task_spec(&self) -> &'static str {
        match self { Self::Ner(_) => NER_TASK_VERSION, Self::Keyphrases(_) => KEYPHRASES_TASK_VERSION,
            Self::Summarize(_) => SUMMARIZE_TASK_VERSION, Self::Answer(..) => ANSWER_TASK_VERSION }
    }
    fn finish(&self, raw: ExtractResult, cap: u64) -> Result<SourceTaskResult, Int4SourceError> {
        if raw.output.schema_version != 1 || raw.output.numerics_profile != STRICT_INT4_PROFILE
            || raw.task_spec_version != self.task_spec() { return Err(Int4SourceError::InvalidResult); }
        Ok(match self {
            Self::Ner(options) => SourceTaskResult::Ner(ner::finalize(raw, options, cap)?),
            Self::Keyphrases(options) => SourceTaskResult::Keyphrases(keyphrases::finalize(raw, *options, cap)?),
            Self::Summarize(options) => SourceTaskResult::Summarize(summarize::finalize(raw, *options, cap)?),
            Self::Answer(passages, options) => SourceTaskResult::Answer(passages.finish(raw, *options, cap)?),
        })
    }
}

/// Exact immutable executable, with no wire constructor, Debug, Deref or
/// extraction escape. The host admits this complete identity unchanged.
pub struct PreparedInt4SourceTask {
    extraction: Int4ExtractPlan,
    finalizer: Finalizer,
    budget: TaskBudget,
}
impl SourceTaskPlanner {
    /// The context must already identify the strict-v2 candidate and exact
    /// group partition. The existing bounded compiler/tokenizer operations
    /// are checkpointed around calls; no mid-operation preemption is claimed.
    pub fn plan_int4_with_control<C: DecodeStepControl>(&self, request: &SourceTaskRequest,
        context: &PlanContext<'_>, limits: SourcePlanningLimits, group_width: usize, control: &mut C)
        -> Result<PreparedInt4SourceTask, Int4SourceError> {
        checkpoint(control)?;
        constrained_int4::check_profile(context.execution_identity(), group_width)?;
        let kind = request.task(); let budget = request.budget();
        self.check_context_for_profile(kind, context, budget, limits, NumericsProfile::StrictQuantized { version: 2 })?;
        let task_spec = match kind {
            BuiltInTask::Ner => NER_TASK_VERSION, BuiltInTask::Keyphrases => KEYPHRASES_TASK_VERSION,
            BuiltInTask::Summarize => SUMMARIZE_TASK_VERSION, BuiltInTask::Answer => ANSWER_TASK_VERSION,
            _ => return Err(SourcePlanningError::Contract("unsupported INT4 source task").into()),
        };
        let encode = |text: &str| self.encoder.encode(text, limits.max_input_bytes, budget.max_input_tokens as usize);
        let mut compile = |document: &SourceDocument, schema: &str| -> Result<Int4ExtractPlan, Int4SourceError> {
            checkpoint(control)?;
            let task = self.task(kind, document, schema, context, budget, limits)?;
            let decode = JsonDecodeOptions { max_new_tokens: budget.max_output_tokens as usize,
                eos_token_id: self.eos, excluded_token_ids: Default::default() };
            let extraction = Int4ExtractPlan::from_builtin(&task, schema, decode, limits.compiler, &self.controls,
                Some((document, limits.source)), task_spec, context.execution_identity().clone(), group_width)?
                .with_finalizer_version(INT4_SOURCE_EXECUTION)?;
            checkpoint(control)?;
            Ok(extraction)
        };
        let (extraction, finalizer) = match request {
            SourceTaskRequest::Ner { document, options, .. } => {
                let schema = options.schema_source()?; let document = encode(document)?;
                (compile(&document, &schema)?, Finalizer::Ner(options.clone()))
            }
            SourceTaskRequest::Keyphrases { document, options, .. } => {
                let schema = options.schema_source()?; let document = encode(document)?;
                (compile(&document, &schema)?, Finalizer::Keyphrases(*options))
            }
            SourceTaskRequest::Summarize { document, options, .. } => {
                let schema = options.schema_source()?; let document = encode(document)?;
                (compile(&document, &schema)?, Finalizer::Summarize(*options))
            }
            SourceTaskRequest::Answer { question, passages, options, .. } => {
                let schema = options.schema_source()?;
                let answer = AnswerContext::encode(&self.encoder, question, passages, AnswerInputLimits {
                    max_passages: limits.max_passages, max_input_bytes: limits.max_input_bytes,
                    max_input_tokens: budget.max_input_tokens as usize,
                })?;
                let extraction = compile(answer.document(), &schema)?;
                (extraction, Finalizer::Answer(answer.into_int4_finalizer(), *options))
            }
        };
        checkpoint(control)?;
        Ok(PreparedInt4SourceTask { extraction, finalizer, budget })
    }
}
impl PreparedInt4SourceTask {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.extraction.execution_identity() }
    pub fn group_width(&self) -> usize { self.extraction.group_width() }
    pub fn planned_work(&self) -> Int4Work { self.extraction.planned_work() }
    pub fn prompt_tokens(&self) -> usize { self.extraction.prompt_tokens() }
    pub fn max_result_bytes(&self) -> u64 { self.extraction.max_result_bytes() }
    pub fn max_kv_bytes(&self) -> u64 { self.extraction.max_kv_bytes() }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int4SourceError> {
        Ok(self.extraction.verify_identity(admitted)?)
    }

    pub fn execute_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, work: Int4JsonBudget,
        control: &mut C) -> Result<Int4SourceTaskRun, Int4SourceError> {
        checkpoint(control)?;
        self.extraction.execute_with(engine, admitted, vocabulary, work, control, |raw| self.finish(raw))
    }

    fn finish(&self, raw: Int4ExtractRun) -> Result<Int4SourceTaskRun, Int4SourceError> {
        if raw.schema_version != 1 || raw.execution != INT4_EXTRACT_VERSION || raw.group_width != self.group_width() {
            return Err(Int4SourceError::InvalidResult);
        }
        let result = self.finalizer.finish(raw.result, self.max_result_bytes())?;
        let run = Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(),
            group_width: self.group_width(), result, model_work: raw.model_work };
        check_size(&run, self.max_result_bytes())?;
        Ok(run)
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), Int4SourceError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(Int4SourceError::Cancelled(cause)), None => Ok(()) }
}

#[cfg(test)] mod tests;
