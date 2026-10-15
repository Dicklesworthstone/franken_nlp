//! Pinned text/chat planning for the complete grouped-INT4 source candidate.
//!
//! Reuses segmented prompt compilation, control exclusion, tokenizer and task
//! finalization. The native plan remains a separate type and never falls back
//! to INT8/eager. Model loading, runtime admission and publication are external.

use super::*;
use crate::native_engine::{
    decode::DecodeCancellationKind,
    generation::grouped_int4::{Int4GenerationBudget, Int4GenerationError, Int4GenerationPlan, Int4GenerationRun},
    quant_algebra::MAX_MODEL_K,
    strict_int4::{Int4Work, StrictInt4Engine},
};

/// The host supplies an explicitly INT4-bound candidate identity. Construction
/// pins prompt assets; each compiled request checks the complete source-format,
/// recipe and partition contract, and execution checks the actual model too.
pub struct Int4ChatPlanner { compiler: ChatPlanner, group_width: usize }

/// Exact private transcript and executable request. No public conversion to
/// an eager/INT8 plan, no deserialization, and no prompt-bearing Debug output.
pub struct PreparedInt4Chat {
    task: TaskPlan, native: Int4GenerationPlan,
    tokenizer: Arc<EmbeddedTokenizer>, sample_index: u64,
}

#[derive(Clone, PartialEq, Serialize)]
pub struct Int4ChatResult {
    pub schema_version: u32,
    pub result: ChatResult,
    pub model_work: Int4Work,
}
#[derive(Debug)]
pub enum Int4ChatError { Chat(ChatError), Native(Int4GenerationError), WorkMismatch }
impl Int4ChatError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Native(error) => error.cancellation(),
            Self::Chat(ChatError::Native(GenerationError::Cancelled(cause))) => Some(*cause),
            _ => None,
        }
    }
}
impl fmt::Display for Int4ChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chat(error) => write!(f, "INT4 chat task refused: {error}"),
            Self::Native(error) => write!(f, "INT4 chat native execution refused: {error}"),
            Self::WorkMismatch => f.write_str("INT4 chat complete group/model work mismatch"),
        }
    }
}
impl Error for Int4ChatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chat(error) => Some(error), Self::Native(error) => Some(error), Self::WorkMismatch => None }
    }
}
impl From<ChatError> for Int4ChatError { fn from(error: ChatError) -> Self { Self::Chat(error) } }
impl From<Int4GenerationError> for Int4ChatError { fn from(error: Int4GenerationError) -> Self { Self::Native(error) } }

impl Int4ChatPlanner {
    pub fn pinned(controls: &TemplateControlIds, eos: u32, identity: ExecutionIdentity,
        group_width: usize, ceiling: TaskBudget, limits: ChatLimits) -> Result<Self, Int4ChatError> {
        if group_width == 0 || group_width > MAX_MODEL_K { return Err(ChatError::Contract("INT4 group width").into()); }
        let compiler = ChatPlanner::pinned_for_backend(controls, eos, identity, ceiling, limits, ChatBackend::GroupedInt4)?;
        Ok(Self { compiler, group_width })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.compiler.identity }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn plan_chat(&self, request: &ChatRequest) -> Result<PreparedInt4Chat, Int4ChatError> {
        self.compile(BuiltInTask::Chat, &request.item_id, request.sample_index,
            &request.messages, &request.generation, request.budget)
    }
    /// As in existing generate, a single untrusted user message passes through
    /// the pinned template. This is not a raw-control-token escape hatch.
    pub fn plan_generate(&self, request: &GenerateRequest) -> Result<PreparedInt4Chat, Int4ChatError> {
        if request.prompt.len() > self.compiler.limits.max_message_bytes
            || request.prompt.len() > self.compiler.limits.max_total_message_bytes {
            return Err(ChatError::Limit("prompt bytes").into());
        }
        self.compile(BuiltInTask::Generate, &request.item_id, request.sample_index,
            &[ChatMessage { role: ChatRole::User, content: request.prompt.clone() }], &request.generation, request.budget)
    }
    fn compile(&self, kind: BuiltInTask, item: &str, sample: u64, messages: &[ChatMessage],
        options: &GenerationOptions, budget: TaskBudget) -> Result<PreparedInt4Chat, Int4ChatError> {
        let input = self.compiler.compile_input(kind, messages, options, budget)?;
        let native = Int4GenerationPlan::compile(input.prompt, input.options, input.identity,
            self.group_width, item, sample, self.compiler.limits.generation)?;
        Ok(PreparedInt4Chat { task: input.task, native,
            tokenizer: Arc::clone(&self.compiler.tokenizer), sample_index: sample })
    }
}
impl PreparedInt4Chat {
    pub fn task_plan(&self) -> &TaskPlan { &self.task }
    pub fn native_plan(&self) -> &Int4GenerationPlan { &self.native }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.native.execution_identity() }
    pub fn planned_work(&self) -> Int4Work { self.native.planned_work() }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int4ChatError> {
        self.native.verify_identity(admitted).map_err(Into::into)
    }
    fn task_budget(&self, mut budget: Int4GenerationBudget) -> Int4GenerationBudget {
        budget.max_kv_bytes = budget.max_kv_bytes.min(self.task.ir().budget().max_kv_bytes);
        budget
    }
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt4Engine<'_>,
        budget: Int4GenerationBudget) -> Result<(), Int4ChatError> {
        self.native.preflight(admitted, engine, self.task_budget(budget)).map_err(Into::into)
    }
    pub fn execute<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt4Engine<'_>, request_seq: u64, budget: Int4GenerationBudget,
        control: &mut C) -> Result<Int4ChatResult, Int4ChatError> {
        let raw = self.native.execute(admitted, engine, self.tokenizer.tokenizer(),
            request_seq, self.task_budget(budget), control)?;
        self.finish(raw)
    }
    /// Token events remain provisional until the complete task result passes
    /// independent decoding, UTF-8, termination, work and wrapper-size checks.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_sink<S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt4Engine<'_>, request_seq: u64,
        budget: Int4GenerationBudget, sink: &mut S, control: &mut C) -> Result<Int4ChatResult, Int4ChatError> {
        let raw = self.native.execute_with_sink(admitted, engine, self.tokenizer.tokenizer(),
            request_seq, self.task_budget(budget), sink, control)?;
        self.finish(raw)
    }
    fn finish(&self, raw: Int4GenerationRun) -> Result<Int4ChatResult, Int4ChatError> {
        if raw.schema_version != 1 { return Err(Int4ChatError::WorkMismatch); }
        let work = raw.model_work;
        let expected = Int4Work::for_sequence(self.native.group_width(), 0,
            usize::try_from(raw.sequence.native_work.forward_positions).map_err(|_| Int4ChatError::WorkMismatch)?,
            usize::try_from(raw.sequence.native_work.projected_logits).map_err(|_| Int4ChatError::WorkMismatch)?)
            .map_err(|_| Int4ChatError::WorkMismatch)?;
        if work != expected { return Err(Int4ChatError::WorkMismatch); }
        let result = ChatCompletion { task: &self.task, tokenizer: &self.tokenizer, options: self.native.options(),
            sample_index: self.sample_index, prompt_tokens: self.native.prompt_tokens(),
            max_forward_positions: self.native.planned_work().forward_positions, backend: ChatBackend::GroupedInt4 }
            .finish(raw.sequence)?;
        let output = Int4ChatResult { schema_version: 1, result, model_work: work };
        bounds::result(&output, self.task.ir().budget().max_output_bytes)?;
        Ok(output)
    }
}

#[cfg(test)] mod tests;

mod model;
mod prefill;
pub mod batch;
