//! Raw-text generation and multi-turn chat on the native strict-INT8 driver.
//!
//! Shares the pinned, control-excluding prompt compiler and independent text
//! finalizer with eager chat, NOT its native plan or its numerics identity.
//! The embedding host owns model activation, runtime admission and delivery.
//! This adapter does not turn a current-candidate artifact into a certified one.

use super::*;
use crate::native_engine::{
    decode::DecodeCancellationKind,
    generation::quantized::{Int8GenerationBudget, Int8GenerationError, Int8GenerationPlan, Int8GenerationRun},
    strict_int8::{Int8Work, StrictInt8Engine},
};

/// Separate static type: neither Deref nor a fallback to the eager planner is
/// exposed. The supplied identity must already name the strict-INT8 backend.
pub struct Int8ChatPlanner { compiler: ChatPlanner }

/// Exact private prompt and admitted plan, never serialized or debug-logged.
pub struct PreparedInt8Chat {
    task: TaskPlan,
    native: Int8GenerationPlan,
    tokenizer: Arc<EmbeddedTokenizer>,
    sample_index: u64,
}

/// Both the task result and complete model work survive publication. Decoder
/// projection MACs and attention pairs must not disappear into head-only counts.
#[derive(Clone, PartialEq, Serialize)]
pub struct Int8ChatResult {
    pub schema_version: u32,
    pub result: ChatResult,
    pub model_work: Int8Work,
}

#[derive(Debug)]
pub enum Int8ChatError { Chat(ChatError), Native(Int8GenerationError), WorkMismatch }
impl Int8ChatError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Native(error) => error.cancellation(),
            Self::Chat(ChatError::Native(GenerationError::Cancelled(cause))) => Some(*cause),
            _ => None,
        }
    }
}
impl fmt::Display for Int8ChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chat(error) => write!(f, "INT8 chat task refused: {error}"),
            Self::Native(error) => write!(f, "INT8 chat native execution refused: {error}"),
            Self::WorkMismatch => f.write_str("INT8 chat complete model work mismatch"),
        }
    }
}
impl Error for Int8ChatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Chat(error) => Some(error), Self::Native(error) => Some(error), Self::WorkMismatch => None }
    }
}
impl From<ChatError> for Int8ChatError { fn from(error: ChatError) -> Self { Self::Chat(error) } }
impl From<Int8GenerationError> for Int8ChatError { fn from(error: Int8GenerationError) -> Self { Self::Native(error) } }

impl Int8ChatPlanner {
    pub fn pinned(controls: &TemplateControlIds, eos: u32, identity: ExecutionIdentity,
        ceiling: TaskBudget, limits: ChatLimits) -> Result<Self, Int8ChatError> {
        let compiler = ChatPlanner::pinned_for_backend(controls, eos, identity, ceiling, limits, ChatBackend::Int8)?;
        Ok(Self { compiler })
    }
    pub fn plan_chat(&self, request: &ChatRequest) -> Result<PreparedInt8Chat, Int8ChatError> {
        self.compile(BuiltInTask::Chat, &request.item_id, request.sample_index,
            &request.messages, &request.generation, request.budget)
    }
    pub fn plan_generate(&self, request: &GenerateRequest) -> Result<PreparedInt8Chat, Int8ChatError> {
        if request.prompt.len() > self.compiler.limits.max_message_bytes
            || request.prompt.len() > self.compiler.limits.max_total_message_bytes {
            return Err(ChatError::Limit("prompt bytes").into());
        }
        self.compile(BuiltInTask::Generate, &request.item_id, request.sample_index,
            &[ChatMessage { role: ChatRole::User, content: request.prompt.clone() }], &request.generation, request.budget)
    }
    fn compile(&self, kind: BuiltInTask, item: &str, sample: u64, messages: &[ChatMessage],
        options: &GenerationOptions, budget: TaskBudget) -> Result<PreparedInt8Chat, Int8ChatError> {
        let input = self.compiler.compile_input(kind, messages, options, budget)?;
        let native = Int8GenerationPlan::compile(input.prompt, input.options, input.identity,
            item, sample, self.compiler.limits.generation)?;
        Ok(PreparedInt8Chat { task: input.task, native,
            tokenizer: Arc::clone(&self.compiler.tokenizer), sample_index: sample })
    }
}
impl PreparedInt8Chat {
    pub fn task_plan(&self) -> &TaskPlan { &self.task }
    pub fn native_plan(&self) -> &Int8GenerationPlan { &self.native }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.native.execution_identity() }
    pub fn planned_work(&self) -> Int8Work { self.native.planned_work() }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8ChatError> {
        self.native.verify_identity(admitted).map_err(Into::into)
    }
    fn task_budget(&self, mut budget: Int8GenerationBudget) -> Int8GenerationBudget {
        budget.max_kv_bytes = budget.max_kv_bytes.min(self.task.ir().budget().max_kv_bytes);
        budget
    }
    pub fn preflight(&self, admitted: &ExecutionIdentity, engine: &StrictInt8Engine<'_>,
        budget: Int8GenerationBudget) -> Result<(), Int8ChatError> {
        self.native.preflight(admitted, engine, self.task_budget(budget)).map_err(Into::into)
    }
    /// The native session clears all 44 logical KV slots on every exit. Its
    /// poison state and typed cancellation are preserved, not hidden by retry.
    pub fn execute<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut StrictInt8Engine<'_>, request_seq: u64, budget: Int8GenerationBudget,
        control: &mut C) -> Result<Int8ChatResult, Int8ChatError> {
        let raw = self.native.execute(admitted, engine, self.tokenizer.tokenizer(),
            request_seq, self.task_budget(budget), control)?;
        self.finish(raw)
    }
    /// Stream events are provisional until the completed task result validates.
    /// Incomplete UTF-8, delivery failure or cancellation cannot mint success.
    /// The caller keeps its output/admission guards through final delivery.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_sink<S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut StrictInt8Engine<'_>, request_seq: u64,
        budget: Int8GenerationBudget, sink: &mut S, control: &mut C) -> Result<Int8ChatResult, Int8ChatError> {
        let raw = self.native.execute_with_sink(admitted, engine, self.tokenizer.tokenizer(),
            request_seq, self.task_budget(budget), sink, control)?;
        self.finish(raw)
    }
    // Only a native execution (or private unit fixture) reaches this method.
    // Deserializing an Int8GenerationRun never becomes public task authority.
    fn finish(&self, raw: Int8GenerationRun) -> Result<Int8ChatResult, Int8ChatError> {
        if raw.schema_version != 1 { return Err(Int8ChatError::WorkMismatch); }
        let work = raw.model_work;
        let expected = Int8Work::for_sequence(0,
            usize::try_from(raw.sequence.native_work.forward_positions).map_err(|_| Int8ChatError::WorkMismatch)?,
            usize::try_from(raw.sequence.native_work.projected_logits).map_err(|_| Int8ChatError::WorkMismatch)?)
            .map_err(|_| Int8ChatError::WorkMismatch)?;
        if work != expected { return Err(Int8ChatError::WorkMismatch); }
        let result = ChatCompletion { task: &self.task, tokenizer: &self.tokenizer, options: self.native.options(),
            sample_index: self.sample_index, prompt_tokens: self.native.prompt_tokens(),
            max_forward_positions: self.native.planned_work().forward_positions, backend: ChatBackend::Int8 }
            .finish(raw.sequence)?;
        let output = Int8ChatResult { schema_version: 1, result, model_work: work };
        // Charge the WHOLE wrapper, not just ChatResult or assistant text.
        bounds::result(&output, self.task.ir().budget().max_output_bytes)?;
        Ok(output)
    }
}

#[cfg(test)] pub(crate) mod tests;
