//! Resident shared-weight generation for the grouped NDJSON runner.
//!
//! Compile with the existing pinned planner, price the ENTIRE input window,
//! acquire one real host admission guard, verify every admitted identity, then
//! execute compatible waves through tasks::chat::batched. No scalar generation
//! fallback, alternate sampler/tokenizer, model loader or private runtime exists.
//! Slot reuse happens only after the preceding native wave has fully quiesced.

use super::*;
use crate::{
    batch::generation::{GenerationBatchArgs, GenerationBatchPlanner},
    execution_identity::{ExecutionIdentity, NumericsProfile},
    native_engine::{
        batchsched::{EagerBatchEngine, MAX_BATCH_ROWS},
        generation::{GenerationError, GenerationWork, batched::BatchGenerationRequirements},
        kv::KV_BYTES_PER_TOKEN,
        lmhead::NANBEIGE_VOCAB_SIZE,
    },
    tasks::chat::{ChatError, ChatResult, PreparedChat,
        batched::{self as chat, BatchChatItem, BatchChatNoResult, BatchChatRequest}},
};
pub use crate::tasks::chat::batched::BatchChatBudget;

/// Private semantic identity plus complete request quantities. Repeated slots
/// in different waves share the already-reserved arena; do not sum those slots
/// as distinct simultaneous caches. The aggregate native price counts the arena
/// once and includes unrelated slots, shared RoPE, scratch and raw logits.
pub struct GenerationCohortItem<'a> {
    pub identity: &'a ExecutionIdentity,
    pub context: BatchRequestContext,
    pub wave: usize,
    pub slot: usize,
    pub work: BatchWork,
    pub sampler_bytes: u64,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}

/// All waves are priced BEFORE the first forward. Sampler/output payload is
/// conservatively summed across the window, including retained earlier results.
/// The host separately owns weights, bounded prepared plans, allocator/metadata
/// overhead, decoder internals and NDJSON/canonical serialization staging.
/// max_result_bytes bounds the SUM of complete canonical native wave envelopes,
/// not just text. It does not replace the transport's per-line/whole-run limits.
pub struct GenerationCohortAdmission<'a> {
    pub items: &'a [GenerationCohortItem<'a>],
    pub requirements: BatchGenerationRequirements,
    pub max_result_bytes: u64,
    pub waves: usize,
}

/// Existing process/model/request admission authority supplied by the embedder.
/// There is no default implementation. Return every ACTUALLY admitted identity
/// in original request order and one reservation covering the complete window.
/// The returned guard is retained until ALL results are delivered or discarded.
pub trait GenerationCohortHost {
    type Guard;
    fn admit(&mut self, request: GenerationCohortAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchFault>;
}

pub struct NativeGenerationGroups<'planner, 'engine, 'weights, H: GenerationCohortHost> {
    compiler: GenerationBatchPlanner<'planner>,
    engine: &'engine mut EagerBatchEngine<'weights>,
    loaded_model: ExecutionIdentity,
    slots: Vec<usize>,
    host: H,
    budget: BatchChatBudget,
}
impl<'p, 'e, 'w, H: GenerationCohortHost> NativeGenerationGroups<'p, 'e, 'w, H> {
    /// Borrow an existing engine and explicitly selected empty slots. Other
    /// active slots are never reset. This neither loads nor activates a model.
    pub fn new(compiler: GenerationBatchPlanner<'p>, engine: &'e mut EagerBatchEngine<'w>,
        loaded_model: ExecutionIdentity, slots: Vec<usize>, host: H, budget: BatchChatBudget) -> Result<Self, BatchFault> {
        loaded_model.validate().map_err(|_| BatchCode::Admission)?;
        if loaded_model.numerics_profile != NumericsProfile::HfBf16Eager || loaded_model.kv_dtype != "bf16"
            || budget.max_result_bytes == 0 || budget.generation.max_forward_positions == 0
            || budget.generation.max_projected_logits == 0 || budget.generation.max_sampler_payload_bytes == 0
            || budget.generation.max_output_payload_bytes == 0
            || engine.envelope().payload().total_bytes > budget.generation.max_native_payload_bytes {
            return Err(BatchCode::InvalidLimits.into());
        }
        validate_slots(engine.envelope().capacities(), &slots)?;
        for &slot in &slots { engine.preflight_sequence_slot(slot, 1).map_err(|_| BatchCode::Admission)?; }
        Ok(Self { compiler, engine, loaded_model, slots, host, budget })
    }
}
impl<H: GenerationCohortHost> GroupedBatchProcessor for NativeGenerationGroups<'_, '_, '_, H> {
    type Args = GenerationBatchArgs;
    type Prepared = PreparedChat;
    type Output = ChatResult;
    type Guard = H::Guard;
    fn max_group_size(&self) -> usize { MAX_GROUP_RECORDS }
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<PreparedChat, BatchItemFailure> {
        let prepared = self.compiler.prepare(document)?;
        // A caller's inadequate per-task KV limit is an ordinary admission
        // refusal, not a broken host that should abort unrelated documents.
        let shape = shape(&prepared).map_err(|_| BatchItemFailure::reject(BatchCode::Admission))?;
        if !self.slots.iter().any(|&slot| shape.fits(self.engine.envelope().capacities()[slot])) {
            return Err(BatchItemFailure::reject(BatchCode::Admission));
        }
        Ok(prepared)
    }
    fn planned_work(&self, prepared: &PreparedChat) -> Result<BatchWork, BatchItemFailure> {
        work(prepared).map_err(BatchItemFailure::fatal)
    }
    fn execute_group<C: DecodeStepControl>(&mut self, requests: Vec<GroupedRequest<PreparedChat>>, control: &mut C)
        -> Result<GroupedOutput<ChatResult, H::Guard>, BatchFault> {
        validate_contexts(&requests)?;
        checkpoint(control)?;
        let mut shapes = reserve(requests.len())?;
        for request in &requests { shapes.push(shape(&request.prepared)?); }
        let waves = plan_waves(self.engine.envelope().capacities(), &self.slots, &shapes)?;
        let mut wave_prices = reserve(waves.len())?;
        let mut required = BatchGenerationRequirements { planned_work: GenerationWork::default(),
            native_payload_bytes: self.engine.envelope().payload().total_bytes,
            sampler_payload_bytes: 0, output_payload_upper_bytes: 0 };
        // These are non-authoritative structural/sizing checks against compiled
        // identities, NOT an admission decision or fabricated host receipt.
        for wave in &waves {
            let rows = native_requests(&requests, wave, None)?;
            let price = chat::preflight(self.engine, &self.loaded_model, &rows, self.budget).map_err(failure)?;
            add_price(&mut required, price)?; wave_prices.push(price);
        }
        check_budget(required, self.budget)?;
        let mut assigned = reserve(requests.len())?; assigned.resize(requests.len(), (usize::MAX, usize::MAX));
        for (index, wave) in waves.iter().enumerate() {
            for row in wave { assigned[row.row] = (index, row.slot); }
        }
        let mut items = reserve(requests.len())?;
        let mut charged = BatchWork::default();
        for (index, request) in requests.iter().enumerate() {
            let (wave, slot) = assigned[index];
            let work = work(&request.prepared)?; charged = charged.add(work)?;
            let kv_reservation_bytes = (self.engine.envelope().capacities()[slot] as u64)
                .checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(BatchCode::Admission)?;
            items.push(GenerationCohortItem { identity: request.prepared.execution_identity(), context: request.context,
                wave, slot, work, sampler_bytes: request.prepared.native_plan().sampler_bytes(), kv_reservation_bytes,
                max_result_bytes: request.prepared.task_plan().ir().budget().max_output_bytes });
        }
        if charged.forward_positions != required.planned_work.forward_positions
            || charged.projected_logits != required.planned_work.projected_logits { return Err(BatchCode::InvalidExecution.into()); }
        checkpoint(control)?;
        let (identities, guard) = self.host.admit(GenerationCohortAdmission { items: &items,
            requirements: required, max_result_bytes: self.budget.max_result_bytes, waves: waves.len() })?;
        // Check the LAST wave's identities too, before the FIRST wave can run.
        verify_identities(&requests, &identities)?;
        for wave in &waves {
            let rows = native_requests(&requests, wave, Some(&identities))?;
            chat::preflight(self.engine, &self.loaded_model, &rows, self.budget).map_err(failure)?;
        }
        // Declared AFTER guard: on errors/unwind retained results drop first.
        let mut completed = reserve(requests.len())?; completed.resize_with(requests.len(), || None);
        let mut actual = GenerationWork::default();
        let mut remaining_result_bytes = self.budget.max_result_bytes;
        for (wave, price) in waves.iter().zip(&wave_prices) {
            checkpoint(control)?;
            let rows = native_requests(&requests, wave, Some(&identities))?;
            let wave_budget = BatchChatBudget { max_result_bytes: remaining_result_bytes, ..self.budget };
            // This is the concrete shared-weight path, never N scalar forwards.
            let raw = chat::execute(self.engine, &self.loaded_model, &rows, wave_budget, control).map_err(failure)?;
            if raw.results.len() != wave.len() || raw.planned_work != price.planned_work {
                return Err(BatchCode::InvalidExecution.into());
            }
            actual = sum_work(actual, raw.actual_work)?;
            if !within(actual, required.planned_work) { return Err(BatchCode::InvalidExecution.into()); }
            // Task finalization already did a no-allocation bound check before
            // canonical serialization. Count the same complete envelope against
            // the window-wide cap; this temporary is dropped before retaining rows.
            let size = canonjson::canonical_bytes(&raw).map_err(|_| BatchCode::Serialization)?.len() as u64;
            remaining_result_bytes = remaining_result_bytes.checked_sub(size).ok_or(BatchCode::OutputLimit)?;
            for assignment in wave {
                self.engine.preflight_sequence_slot(assignment.slot, 1).map_err(|_| BatchCode::InvalidExecution)?;
            }
            for (assignment, item) in wave.iter().zip(raw.results) {
                let sequence = requests[assignment.row].context.request_seq;
                let result = translate(item, sequence)?;
                if completed[assignment.row].is_some() { return Err(BatchCode::InvalidExecution.into()); }
                completed[assignment.row] = Some(GroupedRow { request_seq: sequence, result });
            }
        }
        let mut rows = reserve(requests.len())?;
        for row in completed { rows.push(row.ok_or(BatchCode::InvalidExecution)?); }
        Ok(GroupedOutput::new(rows, guard))
    }
}

#[derive(Clone, Copy, Debug)]
struct RowShape { minimum: u64, maximum: u64 }
impl RowShape { fn fits(self, capacity: usize) -> bool { self.minimum <= capacity as u64 && capacity as u64 <= self.maximum } }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Assignment { row: usize, slot: usize }
fn shape(prepared: &PreparedChat) -> Result<RowShape, BatchFault> {
    let minimum = prepared.planned_work().forward_positions;
    let maximum = prepared.task_plan().ir().budget().max_kv_bytes / KV_BYTES_PER_TOKEN as u64;
    if minimum == 0 || minimum > maximum { return Err(BatchCode::Admission.into()); }
    Ok(RowShape { minimum, maximum })
}
fn work(prepared: &PreparedChat) -> Result<BatchWork, BatchFault> {
    Ok(BatchWork { forward_positions: prepared.planned_work().forward_positions,
        projected_logits: (prepared.native_plan().options().max_new_tokens as u64)
            .checked_mul(NANBEIGE_VOCAB_SIZE as u64).ok_or(BatchCode::WorkLimit)? })
}
fn validate_slots(capacities: &[usize], slots: &[usize]) -> Result<(), BatchFault> {
    if capacities.is_empty() || capacities.len() > MAX_BATCH_ROWS || slots.is_empty() || slots.len() > MAX_BATCH_ROWS {
        return Err(BatchCode::InvalidLimits.into());
    }
    let mut seen = [false; MAX_BATCH_ROWS];
    for &slot in slots {
        if slot >= capacities.len() || seen[slot] || capacities[slot] == 0 { return Err(BatchCode::InvalidLimits.into()); }
        seen[slot] = true;
    }
    Ok(())
}
/// Earliest upper bound first, smallest fitting free slot. This preserves a
/// feasible single-wave interval matching. Requests left over use later waves,
/// so multiple long documents do not fail merely because only one slot is big.
/// No minimum-wave-count or throughput claim is made. Original row order is
/// restored inside each wave and at final delivery; slot order never reseeds.
fn plan_waves(capacities: &[usize], slots: &[usize], shapes: &[RowShape]) -> Result<Vec<Vec<Assignment>>, BatchFault> {
    validate_slots(capacities, slots)?;
    if shapes.is_empty() || shapes.len() > MAX_GROUP_RECORDS
        || shapes.iter().any(|shape| shape.minimum == 0 || !slots.iter().any(|&slot| shape.fits(capacities[slot]))) {
        return Err(BatchCode::Admission.into());
    }
    let mut candidates = reserve(slots.len())?; candidates.extend_from_slice(slots);
    candidates.sort_unstable_by_key(|&slot| (capacities[slot], slot));
    let mut pending = reserve(shapes.len())?; pending.extend(0..shapes.len());
    pending.sort_unstable_by_key(|&row| (shapes[row].maximum, shapes[row].minimum, row));
    let mut waves = reserve(shapes.len())?;
    while !pending.is_empty() {
        let mut used = reserve(candidates.len())?; used.resize(candidates.len(), false);
        let mut wave = reserve(candidates.len().min(pending.len()))?;
        let mut next = reserve(pending.len())?;
        for row in pending {
            let chosen = (0..candidates.len()).find(|&at| !used[at] && shapes[row].fits(capacities[candidates[at]]));
            if let Some(at) = chosen { used[at] = true; wave.push(Assignment { row, slot: candidates[at] }); }
            else { next.push(row); }
        }
        if wave.is_empty() { return Err(BatchCode::Admission.into()); }
        wave.sort_unstable_by_key(|assignment| assignment.row);
        waves.push(wave); pending = next;
    }
    Ok(waves)
}
fn validate_contexts<T>(requests: &[GroupedRequest<T>]) -> Result<(), BatchFault> {
    if requests.is_empty() || requests.len() > MAX_GROUP_RECORDS { return Err(BatchCode::InvalidExecution.into()); }
    let epoch = requests[0].context.epoch;
    for (index, request) in requests.iter().enumerate() {
        let c = request.context;
        if c.epoch == 0 || c.epoch != epoch || c.request_seq == 0 || c.input_line == 0
            || (index > 0 && (c.request_seq <= requests[index - 1].context.request_seq
                || c.input_line <= requests[index - 1].context.input_line
                || c.byte_offset <= requests[index - 1].context.byte_offset)) {
            return Err(BatchCode::InvalidExecution.into());
        }
    }
    Ok(())
}
fn verify_identities(requests: &[GroupedRequest<PreparedChat>], identities: &[ExecutionIdentity]) -> Result<(), BatchFault> {
    if requests.len() != identities.len() { return Err(BatchCode::Admission.into()); }
    for (request, identity) in requests.iter().zip(identities) {
        request.prepared.native_plan().verify_identity(identity).map_err(|_| BatchCode::Admission)?;
    }
    Ok(())
}
fn native_requests<'a>(requests: &'a [GroupedRequest<PreparedChat>], wave: &[Assignment],
    identities: Option<&'a [ExecutionIdentity]>) -> Result<Vec<BatchChatRequest<'a>>, BatchFault> {
    let mut rows = reserve(wave.len())?;
    for assignment in wave {
        let request = requests.get(assignment.row).ok_or(BatchCode::InvalidExecution)?;
        let admitted_identity = match identities {
            Some(identities) => identities.get(assignment.row).ok_or(BatchCode::Admission)?,
            None => request.prepared.execution_identity(),
        };
        rows.push(BatchChatRequest { prepared: &request.prepared, admitted_identity,
            slot: assignment.slot, request_seq: request.context.request_seq });
    }
    Ok(rows)
}
fn add_price(total: &mut BatchGenerationRequirements, wave: BatchGenerationRequirements) -> Result<(), BatchFault> {
    if total.native_payload_bytes != wave.native_payload_bytes { return Err(BatchCode::InvalidExecution.into()); }
    total.planned_work = sum_work(total.planned_work, wave.planned_work)?;
    total.sampler_payload_bytes = total.sampler_payload_bytes.checked_add(wave.sampler_payload_bytes).ok_or(BatchCode::Admission)?;
    total.output_payload_upper_bytes = total.output_payload_upper_bytes.checked_add(wave.output_payload_upper_bytes).ok_or(BatchCode::Admission)?;
    Ok(())
}
fn sum_work(a: GenerationWork, b: GenerationWork) -> Result<GenerationWork, BatchFault> {
    Ok(GenerationWork {
        forward_positions: a.forward_positions.checked_add(b.forward_positions).ok_or(BatchCode::WorkLimit)?,
        projected_logits: a.projected_logits.checked_add(b.projected_logits).ok_or(BatchCode::WorkLimit)?,
        sampled_steps: a.sampled_steps.checked_add(b.sampled_steps).ok_or(BatchCode::WorkLimit)?,
    })
}
fn within(actual: GenerationWork, planned: GenerationWork) -> bool {
    actual.forward_positions <= planned.forward_positions && actual.projected_logits <= planned.projected_logits
        && actual.sampled_steps <= planned.sampled_steps
}
fn check_budget(r: BatchGenerationRequirements, budget: BatchChatBudget) -> Result<(), BatchFault> {
    let b = budget.generation;
    if r.planned_work.forward_positions > b.max_forward_positions || r.planned_work.projected_logits > b.max_projected_logits {
        return Err(BatchCode::WorkLimit.into());
    }
    if r.native_payload_bytes > b.max_native_payload_bytes || r.sampler_payload_bytes > b.max_sampler_payload_bytes
        || r.output_payload_upper_bytes > b.max_output_payload_bytes { return Err(BatchCode::Admission.into()); }
    Ok(())
}
fn translate(item: BatchChatItem, sequence: u64) -> Result<Result<ChatResult, BatchItemFailure>, BatchFault> {
    match item {
        BatchChatItem::Completed { result } if result.request_seq == sequence => Ok(Ok(result)),
        BatchChatItem::NoResult { request_seq, reason, .. } if request_seq == sequence => {
            Ok(Err(BatchItemFailure::reject(match reason {
                BatchChatNoResult::IncompleteUtf8 => BatchCode::Execution,
                BatchChatNoResult::ResultByteLimit => BatchCode::OutputLineLimit,
            })))
        }
        _ => Err(BatchCode::InvalidExecution.into()),
    }
}
fn failure(error: ChatError) -> BatchFault {
    // A failed native wave invalidates the cohort. Never turn cancellation,
    // broken engine state or a lost wave into N apparently independent rejects.
    match error {
        ChatError::Native(GenerationError::Cancelled(cause)) => BatchFault::cancelled(cause),
        ChatError::Identity | ChatError::Native(GenerationError::Identity) => BatchCode::Admission.into(),
        ChatError::Allocation | ChatError::Native(GenerationError::Allocation) => BatchCode::Allocation.into(),
        ChatError::Limit("complete result bytes" | "cohort result bytes") => BatchCode::OutputLimit.into(),
        ChatError::Limit(_) | ChatError::Native(GenerationError::Limit(_)) => BatchCode::WorkLimit.into(),
        ChatError::Native(GenerationError::Engine(_) | GenerationError::NoLegalToken) => BatchCode::Execution.into(),
        ChatError::Native(GenerationError::Stream) => BatchCode::OutputIo.into(),
        ChatError::Serialization => BatchCode::Serialization.into(),
        _ => BatchCode::InvalidExecution.into(),
    }
}

#[cfg(test)] mod tests;
