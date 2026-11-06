//! Complete-document classification on one resident native INT4 stream.
//! The original request is owned between preparation and execution. Replanning
//! verifies every frozen chunk identity; it never borrows through its own owner.
use std::sync::Arc;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput},
    execution_identity::ExecutionIdentity,
    native_engine::{constrained_int4, decode::DecodeStepControl, portable_int8::ProjectionWork,
        strict_int4::{Int4MemoryRequirement, Int4RunBudget, Int4Work, StrictInt4Engine,
            StrictInt4Error, scoring::{Int4ScoringBudget, Int4ScoringError}}},
    tasks::{classify::{ClassificationLabel, ClassificationLimits, ClassificationMode,
        ClassificationRequest,
        grouped_int4::{Int4ClassificationError, Int4ClassificationPlanner,
            long::{Int4ClassificationMapError, Int4ClassificationMapLimits,
                Int4ClassificationMapPreflight, Int4ClassificationMapRun}}},
        ir::TaskBudget},
};
use super::ClassificationBatchArgs;

/// Frozen identities, admission copies and document metadata retained while
/// the original request is recompiled. This is additional to the map quote.
pub const CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;

/// Owns the actual model-bound compiler and optional complete task defaults.
/// `mapping.max_model_work` is a per-document cap; `max_run_work` is one
/// nonrenewable invocation cap across all items and flush boundaries.
pub struct Int4ClassificationDocumentBatchPlanner {
    planner: Int4ClassificationPlanner,
    defaults: Option<ClassificationBatchArgs>,
    mapping: Int4ClassificationMapLimits,
    max_run_work: Int4Work,
    seal: Arc<()>,
}
impl Int4ClassificationDocumentBatchPlanner {
    pub fn new(planner: Int4ClassificationPlanner, defaults: Option<ClassificationBatchArgs>,
        mapping: Int4ClassificationMapLimits, max_run_work: Int4Work) -> Result<Self, BatchFault> {
        validate_limits(planner.task_budget(), planner.planning_limits(), mapping, max_run_work)?;
        constrained_int4::check_profile(planner.execution_identity(), planner.group_width())
            .map_err(|_| BatchCode::Admission)?;
        if planner.execution_identity().task_spec != "classify-v1" { return Err(BatchCode::Admission.into()); }
        if let Some(args) = &defaults {
            validate_args(args, planner.task_budget(), planner.planning_limits()).map_err(|e| e.fault)?;
            // Refuse fixed defaults that cannot leave any source capacity.
            // No document content or model forward is involved in this check.
            let request = args.clone().into_request("x".to_owned());
            planner.max_document_bytes(&request, mapping.chunks.context_tokens)
                .map_err(|e| map_failure(e.into(), false).fault)?;
        }
        Ok(Self { planner, defaults, mapping, max_run_work, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    pub fn group_width(&self) -> usize { self.planner.group_width() }
    pub fn task_budget(&self) -> TaskBudget { self.planner.task_budget() }
    pub fn planning_limits(&self) -> ClassificationLimits { self.planner.planning_limits() }
    pub fn mapping_limits(&self) -> Int4ClassificationMapLimits { self.mapping }
    pub fn max_run_work(&self) -> Int4Work { self.max_run_work }
    /// Actual transferred capacities, so an embedding caller cannot hide a
    /// greatly overallocated defaults owner inside a small logical label set.
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        let Some(args) = &self.defaults else { return Ok(0); };
        let base = (args.labels.capacity() as u64).checked_mul(std::mem::size_of::<ClassificationLabel>() as u64)
            .ok_or(BatchCode::InvalidLimits)?;
        args.labels.iter().try_fold(base, |bytes, label| bytes.checked_add(label.id.capacity() as u64)
            .and_then(|n| n.checked_add(label.description.capacity() as u64)).ok_or_else(|| BatchCode::InvalidLimits.into()))
    }
    fn request(&self, document: BatchDocument<ClassificationBatchArgs>) -> Result<ClassificationRequest, BatchItemFailure> {
        build_request(document, &self.defaults, self.mapping, self.task_budget(), self.planning_limits())
    }
}

/// An opaque original request and a complete frozen preparation snapshot.
/// There is no deserializer, public source replacement or native-score input.
pub struct PreparedInt4ClassificationDocument {
    request: ClassificationRequest,
    expected: Int4ClassificationMapPreflight,
    identities: Vec<ExecutionIdentity>,
    required_context: usize,
    seal: Arc<()>,
}
impl PreparedInt4ClassificationDocument {
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn planned_work(&self) -> Int4Work { self.expected.planned_work() }
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
}

/// Admission covers every exact chunk identity and the complete document
/// result. Summed forwards do not describe the reused cache's capacity.
pub struct Int4ClassificationDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    pub group_width: usize,
    pub required_context: usize,
    pub task_budget: TaskBudget,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait Int4ClassificationDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4ClassificationDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt4ClassificationDocumentBatch<'e, 'w, A: Int4ClassificationDocumentBatchAdmission> {
    compiler: Int4ClassificationDocumentBatchPlanner,
    engine: &'e mut StrictInt4Engine<'w>,
    admission: A,
    run: RunState,
}
impl<'e, 'w, A: Int4ClassificationDocumentBatchAdmission> NativeInt4ClassificationDocumentBatch<'e, 'w, A> {
    pub fn new(compiler: Int4ClassificationDocumentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        constrained_int4::check_model(compiler.base_execution_identity(), engine.model_identity())
            .map_err(|_| BatchCode::Admission)?;
        capacity(&compiler, compiler.task_budget(), engine).map_err(|e| e.fault)?;
        let run = RunState::new(compiler.max_run_work);
        Ok(Self { compiler, engine, admission, run })
    }
    pub fn remaining_work(&self) -> Int4Work { self.run.remaining }
    pub fn is_poisoned(&self) -> bool { self.run.state != State::Ready || !clean(self.engine) }
}
impl<A: Int4ClassificationDocumentBatchAdmission> BatchProcessor for NativeInt4ClassificationDocumentBatch<'_, '_, A> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedInt4ClassificationDocument;
    type Output = GuardedOutput<Int4ClassificationMapRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            checkpoint(control)?;
            let request = self.compiler.request(document)?;
            let kv = capacity(&self.compiler, request.budget, self.engine)?;
            let plan = self.compiler.planner.plan_document_with_control(&request, self.compiler.mapping, control)
                .map_err(|e| map_failure(e, false))?;
            subtract(self.run.remaining, plan.planned_work())?;
            let identities = plan.identities().map_err(|e| map_failure(e, false))?;
            plan.preflight_with_control(&identities, self.engine, budget(plan.planned_work(), kv), control)
                .map_err(|e| map_failure(e, false))?;
            let expected = plan.preflight_metadata(); let required_context = plan.required_context();
            drop(plan);
            checkpoint(control)?;
            Ok(PreparedInt4ClassificationDocument { request, expected, identities, required_context,
                seal: Arc::clone(&self.compiler.seal) })
        })();
        self.run.finish_preparation(result, clean(self.engine))
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            if !Arc::ptr_eq(&prepared.seal, &self.compiler.seal) {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            // Charge before replanning, output admission or a native call.
            // All later failures keep this debit and close the processor.
            self.run.charge(prepared.planned_work(), context)?;
            checkpoint(control)?;
            let kv = capacity(&self.compiler, prepared.request.budget, self.engine)?;
            let plan = self.compiler.planner.plan_document_with_control(&prepared.request, self.compiler.mapping, control)
                .map_err(|e| map_failure(e, true))?;
            let actual_ids = plan.identities().map_err(|e| map_failure(e, true))?;
            if plan.preflight_metadata() != prepared.expected || actual_ids != prepared.identities
                || plan.required_context() != prepared.required_context {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            drop(actual_ids);
            let native_budget = budget(prepared.planned_work(), kv);
            plan.preflight_with_control(&prepared.identities, self.engine, native_budget, control)
                .map_err(|e| map_failure(e, true))?;
            let (admitted, guard) = self.admission.admit(Int4ClassificationDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.planned_work(),
                group_width: plan.group_width(), required_context: prepared.required_context,
                task_budget: prepared.request.budget, kv_reservation_bytes: kv,
                max_result_bytes: plan.max_result_bytes(),
            })?;
            // The embedding host must authorize exactly the complete frozen
            // request, not just its first chunk or another compiler identity.
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            checkpoint(control)?;
            let output = plan.execute_with_control(&admitted, self.engine, native_budget, control)
                .map_err(|e| map_failure(e, true))?;
            prepared.expected.verify_completed(&output).map_err(|e| map_failure(e, true))?;
            if !clean(self.engine) { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
            checkpoint(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        self.run.finish_execution(result, clean(self.engine))
    }
}

pub(crate) fn validate_limits(task: TaskBudget, planning: ClassificationLimits,
    mapping: Int4ClassificationMapLimits, max_run_work: Int4Work) -> Result<(), BatchFault> {
    mapping.validate(task).map_err(|_| BatchCode::InvalidLimits)?;
    planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if !nonzero(max_run_work) || !fits(mapping.max_model_work, max_run_work)
        || mapping.chunks.max_input_bytes > planning.max_input_bytes
        || mapping.chunks.context_tokens > planning.max_context_tokens {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn build_request(document: BatchDocument<ClassificationBatchArgs>, defaults: &Option<ClassificationBatchArgs>,
    mapping: Int4ClassificationMapLimits, task: TaskBudget, planning: ClassificationLimits)
    -> Result<ClassificationRequest, BatchItemFailure> {
    if document.text.is_empty() || document.text.len() > mapping.chunks.max_input_bytes {
        return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
    }
    // A supplied object replaces the entire default. Missing/null uses the
    // frozen default; neither case merges label, policy or budget data.
    let args = document.task_args.or_else(|| defaults.clone())
        .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
    validate_args(&args, task, planning)?;
    Ok(args.into_request(document.text))
}
fn validate_args(args: &ClassificationBatchArgs, ceiling: TaskBudget, limits: ClassificationLimits)
    -> Result<(), BatchItemFailure> {
    let budget = args.budget;
    if budget.validate().is_err() || budget.max_input_tokens > ceiling.max_input_tokens
        || budget.max_output_tokens > ceiling.max_output_tokens || budget.max_output_bytes > ceiling.max_output_bytes
        || budget.max_grammar_states > ceiling.max_grammar_states || budget.max_kv_bytes > ceiling.max_kv_bytes
        || args.labels.is_empty() || args.labels.len() > limits.max_labels
        || (args.mode == ClassificationMode::Exclusive && args.labels.len() < 2)
        || args.policy.minimum_candidate_weight_ppm > 1_000_000 || args.policy.minimum_margin_ppm > 1_000_000 {
        return Err(BatchItemFailure::reject(BatchCode::Planning));
    }
    let mut total = 0_usize;
    for (index, label) in args.labels.iter().enumerate() {
        total = total.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        if label.id.trim().is_empty() || label.id.chars().any(char::is_control)
            || label.id.len() > limits.max_label_id_bytes || label.description.len() > limits.max_label_description_bytes
            || total > limits.max_total_label_bytes || args.labels[..index].iter().any(|old| old.id == label.id) {
            return Err(BatchItemFailure::reject(BatchCode::Planning));
        }
    }
    Ok(())
}
fn capacity(compiler: &Int4ClassificationDocumentBatchPlanner, task: TaskBudget, engine: &StrictInt4Engine<'_>)
    -> Result<u64, BatchItemFailure> {
    if !clean(engine) || engine.model_identity().group_width() != compiler.group_width()
        || engine.kv_cache().capacity_positions() < compiler.mapping.chunks.context_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(engine.kv_cache().capacity_positions(), compiler.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if required.kv_bytes > task.max_kv_bytes { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
    Ok(required.kv_bytes)
}
fn clean(engine: &StrictInt4Engine<'_>) -> bool { !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0) }
fn budget(work: Int4Work, max_kv_bytes: u64) -> Int4ScoringBudget {
    Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes }
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4ClassificationMapError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use Int4ClassificationMapError as M;
    use Int4ClassificationError as C;
    let mut failure = match error {
        M::EmptySource => BatchItemFailure::reject(BatchCode::DocumentLimit),
        M::Limits | M::Chunk(_) => BatchItemFailure::reject(BatchCode::Planning),
        M::Work | M::Classification(C::WorkBudget | C::Native(StrictInt4Error::Work)
            | C::Scoring(Int4ScoringError::Native(StrictInt4Error::Work)))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        M::Allocation | M::Classification(C::Scoring(Int4ScoringError::Allocation))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        M::Output | M::Classification(C::Scoring(Int4ScoringError::OutputBudget))
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        M::Serialization | M::Classification(C::Scoring(Int4ScoringError::Serialization))
            => BatchItemFailure::fatal(BatchCode::Serialization),
        M::Classification(C::Planning(e)) => super::planning_failure(e),
        M::Classification(C::Native(StrictInt4Error::Context | StrictInt4Error::Memory)
            | C::Scoring(Int4ScoringError::Native(StrictInt4Error::Context | StrictInt4Error::Memory)))
            => BatchItemFailure::reject(BatchCode::Admission),
        M::Admission | M::Classification(C::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    if executing { failure.stop = true; }
    failure
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct RunState { remaining: Int4Work, last_sequence: u64, state: State }
impl RunState {
    fn new(remaining: Int4Work) -> Self { Self { remaining, last_sequence: 0, state: State::Ready } }
    fn enter(&mut self) -> Result<(), BatchItemFailure> {
        if self.state != State::Ready { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.state = State::Running; Ok(())
    }
    fn charge(&mut self, work: Int4Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        if self.state != State::Running || context.request_seq <= self.last_sequence
            || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let remaining = subtract(self.remaining, work).map_err(|e| BatchItemFailure::fatal(e.fault))?;
        self.remaining = remaining; self.last_sequence = context.request_seq; Ok(())
    }
    fn finish_preparation<T>(&mut self, result: Result<T, BatchItemFailure>, clean: bool) -> Result<T, BatchItemFailure> {
        match result {
            Ok(value) if clean && self.state == State::Running => { self.state = State::Ready; Ok(value) }
            Ok(_) => { self.state = State::Failed; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
            Err(mut error) => {
                // An aborted native session is intentionally poisoned. Keep
                // its original typed cancellation/failure while refusing reuse.
                if !clean || self.state != State::Running || error.fault.cancellation.is_some() {
                    error.stop = true;
                }
                self.state = if error.stop { State::Failed } else { State::Ready };
                Err(error)
            }
        }
    }
    fn finish_execution<T>(&mut self, result: Result<T, BatchItemFailure>, clean: bool) -> Result<T, BatchItemFailure> {
        // Even an admission refusal happens after the full attempt is charged.
        // Continuing after any execution error would renew a failed attempt.
        self.finish_preparation(result.map_err(|mut e| { e.stop = true; e }), clean)
    }
}
pub(crate) fn nonzero(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
pub(crate) fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn subtract(a: Int4Work, b: Int4Work) -> Result<Int4Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int4Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}

#[cfg(test)] mod tests;
