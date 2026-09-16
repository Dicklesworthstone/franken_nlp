//! Resident-engine generation and multi-turn chat on the ordered NDJSON stream.
//! The caller's stable ID addresses the semantic request; the engine-assigned
//! sequence only correlates delivery. Neither epochs nor read order reseed the
//! sampler. One pinned planner and engine survive between fully drained items.

use crate::{
    execution_identity::ExecutionIdentity,
    native_engine::{generation::{GenerationBudget, GenerationError, GenerationOptions},
        hf_bf16_eager::HfBf16EagerEngine, kv::KV_BYTES_PER_TOKEN},
    tasks::{chat::{ChatError, ChatMessage, ChatPlanner, ChatRequest, ChatResult, ChatRole,
        GenerateRequest, PreparedChat}, ir::TaskBudget},
};
use super::*;
pub use super::output::GuardedOutput;

/// Additional typed-argument cap, also applied to trusted embedding defaults
/// BEFORE they can be cloned into each document. The transport has its own
/// independent complete-line/depth/document limits.
pub const MAX_GENERATION_ARGUMENT_BYTES: usize = 1024 * 1024;
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum GenerationBatchArgs {
    Generate { generation: GenerationOptions, budget: TaskBudget, sample_index: u64 },
    /// `text` becomes the new final User message. History must already obey
    /// the pinned role contract; it is neither truncated nor repaired here.
    Chat { history: Vec<ChatMessage>, generation: GenerationOptions, budget: TaskBudget, sample_index: u64 },
}

pub struct GenerationBatchPlanner<'a> {
    planner: &'a ChatPlanner,
    defaults: Option<GenerationBatchArgs>,
}
impl<'a> GenerationBatchPlanner<'a> {
    pub fn new(planner: &'a ChatPlanner, defaults: Option<GenerationBatchArgs>) -> Result<Self, BatchFault> {
        if let Some(args) = &defaults { check_argument_size(args).map_err(|_| BatchCode::InvalidLimits)?; }
        Ok(Self { planner, defaults })
    }
    pub fn prepare(&self, document: BatchDocument<GenerationBatchArgs>) -> Result<PreparedChat, BatchItemFailure> {
        let args = document.task_args.or_else(|| self.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        check_argument_size(&args).map_err(|_| BatchItemFailure::reject(BatchCode::Planning))?;
        let result = match args {
            GenerationBatchArgs::Generate { generation, budget, sample_index } => self.planner.plan_generate(&GenerateRequest {
                item_id: document.id, prompt: document.text, sample_index, generation, budget,
            }),
            GenerationBatchArgs::Chat { mut history, generation, budget, sample_index } => {
                if history.len() >= 128 { return Err(BatchItemFailure::reject(BatchCode::Planning)); }
                history.try_reserve_exact(1).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
                history.push(ChatMessage { role: ChatRole::User, content: document.text });
                self.planner.plan_chat(&ChatRequest { item_id: document.id, sample_index, messages: history, generation, budget })
            }
        };
        result.map_err(planning_failure)
    }
}

/// Private identity plus the real request quantities exposed to the embedding
/// host's existing admission path. This is NOT an activation certificate or a
/// replacement PermitBroker. The host additionally owns the engine's weights,
/// raw-logit/activation storage, allocator margin and batch-envelope staging.
pub struct GenerationAdmission<'a> {
    pub identity: &'a ExecutionIdentity,
    pub work: BatchWork,
    /// Bounded sampler payload, distinct from raw model logits and RSS.
    pub sampler_bytes: u64,
    /// The engine's complete reserved KV capacity, not only occupied positions.
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait GenerationBatchAdmission {
    type Guard;
    /// Return the identity ACTUALLY admitted and its real resource guard. There
    /// is deliberately no default implementation fabricating either authority.
    fn admit(&mut self, request: GenerationAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

pub struct NativeGenerationBatch<'p, 'e, A: GenerationBatchAdmission> {
    compiler: GenerationBatchPlanner<'p>,
    engine: &'e mut HfBf16EagerEngine,
    admission: A,
    max_sampler_bytes: u64,
}
impl<'p, 'e, A: GenerationBatchAdmission> NativeGenerationBatch<'p, 'e, A> {
    pub fn new(compiler: GenerationBatchPlanner<'p>, engine: &'e mut HfBf16EagerEngine,
        admission: A, max_sampler_bytes: u64) -> Result<Self, BatchFault> {
        if !engine.kv_cache().all_slots_have_len(0) { return Err(BatchCode::InvalidExecution.into()); }
        if max_sampler_bytes == 0 { return Err(BatchCode::InvalidLimits.into()); }
        Ok(Self { compiler, engine, admission, max_sampler_bytes })
    }
    fn budget(&self, prepared: &PreparedChat) -> GenerationBudget {
        let work = prepared.planned_work();
        GenerationBudget { max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
            max_kv_bytes: prepared.task_plan().ir().budget().max_kv_bytes, max_sampler_bytes: self.max_sampler_bytes }
    }
}
impl<A: GenerationBatchAdmission> BatchProcessor for NativeGenerationBatch<'_, '_, A> {
    type Args = GenerationBatchArgs;
    type Prepared = PreparedChat;
    type Output = GuardedOutput<ChatResult, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        let prepared = self.compiler.prepare(document)?;
        prepared.preflight_eager(prepared.execution_identity(), self.engine, self.budget(&prepared)).map_err(execution_failure)?;
        Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    /// The native batch adapter needs the runner's real sequence. Standalone
    /// callers use PreparedChat::execute_eager with their explicit sequence;
    /// this method must not invent an ID or silently reuse sequence zero.
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        if context.request_seq == 0 || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let work = self.planned_work(&prepared);
        let kv_reservation_bytes = (self.engine.kv_cache().capacity_positions() as u64)
            .checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
        let budget = self.budget(&prepared);
        let (identity, guard) = self.admission.admit(GenerationAdmission {
            identity: prepared.execution_identity(), work, sampler_bytes: prepared.native_plan().sampler_bytes(),
            kv_reservation_bytes, max_result_bytes: prepared.task_plan().ir().budget().max_output_bytes,
        })?;
        prepared.preflight_eager(&identity, self.engine, budget).map_err(execution_failure)?;
        // The runner charged aggregate forward/projection work BEFORE entering
        // this method. An early finish or failure never refunds that allowance.
        let result = prepared.execute_eager(&identity, self.engine, context.request_seq, budget, control);
        if !self.engine.kv_cache().all_slots_have_len(0) { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        let result = result.map_err(execution_failure)?;
        if result.request_seq != context.request_seq || result.native_work.forward_positions > work.forward_positions
            || result.native_work.projected_logits > work.projected_logits {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        // Result storage drops before the host's guard, after the canonical
        // event has been written AND flushed, or after failed delivery drains.
        Ok(GuardedOutput::new(result, guard))
    }
}
fn planning_failure(error: ChatError) -> BatchItemFailure {
    match error {
        ChatError::Allocation | ChatError::Native(GenerationError::Allocation) => BatchItemFailure::fatal(BatchCode::Allocation),
        ChatError::Identity | ChatError::Native(GenerationError::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::reject(BatchCode::Planning),
    }
}
fn execution_failure(error: ChatError) -> BatchItemFailure {
    match error {
        ChatError::Native(GenerationError::Cancelled(cause)) => BatchItemFailure::fatal(BatchFault::cancelled(cause)),
        ChatError::Native(GenerationError::Limit("context" | "complete KV reservation")) => BatchItemFailure::reject(BatchCode::Admission),
        ChatError::Native(GenerationError::Limit(_)) => BatchItemFailure::reject(BatchCode::WorkLimit),
        ChatError::Limit("complete result bytes") => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        ChatError::NoResult("incomplete UTF-8") | ChatError::Native(GenerationError::NoLegalToken) => BatchItemFailure::reject(BatchCode::Execution),
        ChatError::Allocation | ChatError::Native(GenerationError::Allocation) => BatchItemFailure::fatal(BatchCode::Allocation),
        ChatError::Identity | ChatError::Native(GenerationError::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        ChatError::Native(GenerationError::Engine(_)) => BatchItemFailure::fatal(BatchCode::Execution),
        ChatError::Native(GenerationError::Stream) => BatchItemFailure::fatal(BatchCode::OutputIo),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}
fn check_argument_size(args: &GenerationBatchArgs) -> Result<(), ()> {
    struct Counter(usize);
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_sub(bytes.len()).ok_or_else(|| std::io::Error::other("argument bound"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter(MAX_GENERATION_ARGUMENT_BYTES), args).map_err(|_| ())
}

#[cfg(test)] mod tests;
