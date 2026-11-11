//! Complete source/claim judgments on one resident native invocation.
//! Every original source retains all evidence and counterevidence. Fixed
//! policy, native budgets and schedule cannot be replaced by a stream record.
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput,
        int4_document::{RunState, fits, nonzero, subtract}},
    execution_identity::ExecutionIdentity,
    native_engine::{constrained_int4, decode::DecodeStepControl, int4::PackedInt4Kernel,
        lmhead::scoring::ScoringError,
        strict_int4::{Int4MemoryRequirement, Int4ModelWeights, Int4RunBudget, Int4Work,
            StrictInt4Engine, StrictInt4Error, cohort::MAX_INT4_SEQUENCES,
            scoring::{Int4ScoringBudget, Int4ScoringError}}},
    tasks::{ir::TaskBudget, mapreduce::MapReduceError,
        judge::{FaithfulnessPolicy, FaithfulnessAggregationError, JudgeError, JudgeLimits, JudgeRequest,
            grouped_int4::{Int4JudgeError, Int4JudgePlanner,
                faithfulness_map::{Int4FaithfulnessMapError, Int4FaithfulnessMapLimits,
                    Int4FaithfulnessMapPreflight, Int4FaithfulnessMapRun, Int4FaithfulnessMapCohortRun,
                    PreparedInt4FaithfulnessMap, MAX_FAITHFULNESS_MAP_HEADS,
                    memory::Int4FaithfulnessMapMemoryLimits}}}},
};
mod backend;
use backend::Backend;

/// Original identities remain owned while the complete source is recompiled
/// and admitted. This is additional to the one-shot map's memory quote.
pub const FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;
const COHORT_WRAPPER_BYTES: u64 = 4096;

/// `text` is the complete source; its exact claim belongs to the same item.
/// Records cannot inject task budgets, policies, models or executable plans.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4FaithfulnessDocumentArgs { pub claim: String }

pub struct Int4FaithfulnessDocumentBatchPlanner {
    compiler: Int4JudgePlanner,
    policy: FaithfulnessPolicy,
    mapping: Int4FaithfulnessMapLimits,
    max_run_work: Int4Work,
    max_heads: usize,
    cohort_chunks: Option<usize>,
    seal: Arc<()>,
}
impl Int4FaithfulnessDocumentBatchPlanner {
    /// Derive all task and planning authority from the actual model-bound
    /// compiler. Only the source and claim vary between records.
    pub fn new(compiler: Int4JudgePlanner, policy: FaithfulnessPolicy,
        mapping: Int4FaithfulnessMapLimits, max_run_work: Int4Work) -> Result<Self, BatchFault> {
        let (task, planning) = compiler.semantic_configuration();
        validate_limits(task, planning, mapping, max_run_work)?;
        policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
        constrained_int4::check_profile(compiler.execution_identity(), compiler.group_width())
            .map_err(|_| BatchCode::Admission)?;
        if compiler.execution_identity().task_spec != "judge-v1" { return Err(BatchCode::Admission.into()); }
        Ok(Self { compiler, policy, mapping, max_run_work, max_heads: MAX_FAITHFULNESS_MAP_HEADS,
            cohort_chunks: None, seal: Arc::new(()) })
    }
    /// Explicitly bound every source chunk's complete evidence-head set.
    /// The fixed policy must fit in full: this never truncates evidence,
    /// changes its partition or widens the factory's task budget. Omitting
    /// this opt-in preserves the original 32-head resource declaration.
    pub fn with_max_heads(mut self, max_heads: usize) -> Result<Self, BatchFault> {
        validate_head_cap(self.policy, max_heads)?;
        self.max_heads = max_heads;
        self.seal = Arc::new(());
        Ok(self)
    }
    /// Fixed chunk windows retain every original evidence head. Head counts
    /// vary by source; every actual ragged window is checked after planning,
    /// without silently reducing the requested width or falling back to serial.
    pub fn with_chunk_cohorts(mut self, chunks: usize) -> Result<Self, BatchFault> {
        validate_cohort_chunks(chunks, self.mapping)?;
        self.mapping.max_result_bytes.checked_add(COHORT_WRAPPER_BYTES).ok_or(BatchCode::InvalidLimits)?;
        self.cohort_chunks = Some(chunks);
        self.seal = Arc::new(());
        Ok(self)
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.compiler.execution_identity() }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    pub fn task_budget(&self) -> TaskBudget { self.compiler.semantic_configuration().0 }
    pub fn planning_limits(&self) -> JudgeLimits { self.compiler.semantic_configuration().1 }
    pub fn mapping_limits(&self) -> Int4FaithfulnessMapLimits { self.mapping }
    pub fn group_width(&self) -> usize { self.compiler.group_width() }
    pub fn policy(&self) -> FaithfulnessPolicy { self.policy }
    pub fn max_run_work(&self) -> Int4Work { self.max_run_work }
    pub fn max_heads(&self) -> usize { self.max_heads }
    pub fn cohort_chunks(&self) -> Option<usize> { self.cohort_chunks }
    pub fn max_result_bytes(&self) -> u64 {
        self.mapping.max_result_bytes + if self.cohort_chunks.is_some() { COHORT_WRAPPER_BYTES } else { 0 }
    }
    /// The fixed policy is inline data; no source, claim or caller-owned
    /// defaults allocation is retained by this factory.
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> { Ok(0) }
    fn request(&self, document: BatchDocument<Int4FaithfulnessDocumentArgs>) -> Result<JudgeRequest, BatchItemFailure> {
        build_request(document, self.policy, self.task_budget(), self.mapping)
    }
}

/// Sealed whole-source preparation, never a wire object or external scorer.
pub struct PreparedInt4FaithfulnessDocument {
    request: JudgeRequest,
    expected: Int4FaithfulnessMapPreflight,
    identities: Vec<ExecutionIdentity>,
    required_context: usize,
    seal: Arc<()>,
}
impl PreparedInt4FaithfulnessDocument {
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn planned_work(&self) -> Int4Work { self.expected.planned_work() }
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    /// Total native heads across every source chunk, not the physical peak.
    pub fn head_count(&self) -> usize { self.expected.head_count() }
    pub fn max_head_count(&self) -> usize { self.expected.max_head_count() }
}

/// Preserve the standalone complete-source map wire values verbatim.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Int4FaithfulnessDocumentRun {
    Serial(Int4FaithfulnessMapRun),
    ChunkCohorts(Int4FaithfulnessMapCohortRun),
}
pub struct Int4FaithfulnessDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    pub group_width: usize,
    pub required_context: usize,
    /// All native heads across the original source.
    pub head_count: usize,
    /// Largest individual source chunk's head count.
    pub max_head_count: usize,
    pub task_budget: TaskBudget,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub cohort_chunks: Option<usize>,
}
pub trait Int4FaithfulnessDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4FaithfulnessDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt4FaithfulnessDocumentBatch<'e, 'w, A: Int4FaithfulnessDocumentBatchAdmission> {
    compiler: Int4FaithfulnessDocumentBatchPlanner,
    backend: Backend<'e, 'w>,
    admission: A,
    run: RunState,
}
impl<'e, 'w, A: Int4FaithfulnessDocumentBatchAdmission> NativeInt4FaithfulnessDocumentBatch<'e, 'w, A> {
    pub fn new(compiler: Int4FaithfulnessDocumentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        if compiler.cohort_chunks.is_some() { return Err(BatchCode::InvalidLimits.into()); }
        let backend = Backend::Serial(engine);
        backend.check(&compiler).map_err(|error| error.fault)?;
        let run = RunState::new(compiler.max_run_work);
        Ok(Self { compiler, backend, admission, run })
    }
    pub fn new_cohorts(compiler: Int4FaithfulnessDocumentBatchPlanner, weights: &'w Int4ModelWeights,
        kernel: PackedInt4Kernel, memory: Int4FaithfulnessMapMemoryLimits, admission: A)
        -> Result<Self, BatchFault> {
        let backend = Backend::cohorts(&compiler, weights, kernel, memory)?;
        let run = RunState::new(compiler.max_run_work);
        Ok(Self { compiler, backend, admission, run })
    }
    pub fn remaining_work(&self) -> Int4Work { self.run.remaining() }
    pub fn is_poisoned(&self) -> bool { self.run.is_unavailable() || !self.backend.clean() }
}
impl<A: Int4FaithfulnessDocumentBatchAdmission> BatchProcessor for NativeInt4FaithfulnessDocumentBatch<'_, '_, A> {
    type Args = Int4FaithfulnessDocumentArgs;
    type Prepared = PreparedInt4FaithfulnessDocument;
    type Output = GuardedOutput<Int4FaithfulnessDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.fail(); Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            checkpoint(control)?;
            let request = self.compiler.request(document)?;
            self.backend.check(&self.compiler)?;
            let plan = self.compiler.compiler.plan_faithfulness_document_with_control(&request,
                self.compiler.mapping, control).map_err(|error| map_failure(error, false))?;
            validate_plan(&plan, &self.compiler)?;
            subtract(self.run.remaining(), plan.planned_work())?;
            let identities = plan.identities().map_err(|error| map_failure(error, false))?;
            self.backend.preflight(&plan, &identities, control).map_err(|error| map_failure(error, false))?;
            let expected = plan.preflight_metadata(); let required_context = plan.required_context();
            drop(plan); checkpoint(control)?;
            Ok(PreparedInt4FaithfulnessDocument { request, expected, identities, required_context,
                seal: Arc::clone(&self.compiler.seal) })
        })();
        self.run.finish_preparation(result, self.backend.clean())
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.fail(); Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            if !Arc::ptr_eq(&prepared.seal, &self.compiler.seal) {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            // Charge all five axes before replan, admission or native work.
            // Every subsequent failure terminates without refunding a debit.
            self.run.charge(prepared.planned_work(), context)?;
            checkpoint(control)?;
            let kv = self.backend.check(&self.compiler)?;
            let plan = self.compiler.compiler.plan_faithfulness_document_with_control(&prepared.request,
                self.compiler.mapping, control).map_err(|error| map_failure(error, true))?;
            validate_plan(&plan, &self.compiler)?;
            let actual = plan.identities().map_err(|error| map_failure(error, true))?;
            if plan.preflight_metadata() != prepared.expected || actual != prepared.identities
                || plan.required_context() != prepared.required_context {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            self.backend.preflight(&plan, &actual, control).map_err(|error| map_failure(error, true))?;
            let (admitted, guard) = self.admission.admit(Int4FaithfulnessDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.planned_work(),
                group_width: plan.group_width(), required_context: prepared.required_context,
                head_count: prepared.head_count(), max_head_count: prepared.max_head_count(),
                task_budget: self.compiler.task_budget(), kv_reservation_bytes: kv,
                max_result_bytes: self.compiler.max_result_bytes(), cohort_chunks: self.compiler.cohort_chunks,
            })?;
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            checkpoint(control)?;
            let output = self.backend.execute(&plan, &admitted, control).map_err(|error| map_failure(error, true))?;
            backend::verify_completed(&prepared.expected, self.compiler.cohort_chunks, &output, control)
                .map_err(|error| map_failure(error, true))?;
            checkpoint(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        self.run.finish_execution(result, self.backend.clean())
    }
}

pub(crate) fn validate_limits(task: TaskBudget, planning: JudgeLimits,
    mapping: Int4FaithfulnessMapLimits, run: Int4Work) -> Result<(), BatchFault> {
    mapping.validate(task).map_err(|_| BatchCode::InvalidLimits)?;
    let head = planning.per_head;
    if mapping.chunks.max_input_bytes == 0 || !nonzero(run) || !fits(mapping.max_model_work, run)
        || head.max_candidates == 0 || head.max_total_tokens == 0 || head.max_nodes == 0
        || head.max_depth == 0 || head.max_candidate_id_bytes == 0 || head.max_projected_logits == 0
        || planning.max_total_prompt_tokens == 0 || planning.max_total_candidate_tokens == 0
        || planning.max_total_projected_logits == 0 || planning.max_output_bytes == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn validate_cohort_chunks(chunks: usize, mapping: Int4FaithfulnessMapLimits) -> Result<(), BatchFault> {
    if chunks == 0 || chunks > MAX_INT4_SEQUENCES || chunks > mapping.chunks.max_chunks {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn validate_head_cap(policy: FaithfulnessPolicy, max_heads: usize) -> Result<(), BatchFault> {
    policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
    // A single complete evidence window reuses the full-source head. Any
    // larger partition retains its own full-source head plus every window.
    let policy_heads = if policy.max_evidence_windows == 1 { 1 }
        else { policy.max_evidence_windows as usize + 1 };
    if !(1..=MAX_FAITHFULNESS_MAP_HEADS).contains(&max_heads) || policy_heads > max_heads {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn build_request(document: BatchDocument<Int4FaithfulnessDocumentArgs>, policy: FaithfulnessPolicy,
    task: TaskBudget, mapping: Int4FaithfulnessMapLimits) -> Result<JudgeRequest, BatchItemFailure> {
    let args = document.task_args.ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
    if document.text.is_empty() || args.claim.is_empty()
        || document.text.len().checked_add(args.claim.len()).is_none_or(|bytes| bytes > mapping.chunks.max_input_bytes) {
        return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
    }
    Ok(JudgeRequest::Faithfulness { source: document.text, claim: args.claim, policy, budget: task })
}
fn validate_plan(plan: &PreparedInt4FaithfulnessMap<'_>, compiler: &Int4FaithfulnessDocumentBatchPlanner)
    -> Result<(), BatchItemFailure> {
    if plan.chunk_count() == 0 || plan.chunk_count() > compiler.mapping.chunks.max_chunks
        || plan.head_count() < plan.chunk_count()
        || !(1..=compiler.max_heads).contains(&plan.max_head_count())
        || plan.head_count() > plan.chunk_count().checked_mul(compiler.max_heads)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?
        || plan.group_width() != compiler.group_width() || plan.task_budget() != compiler.task_budget()
        || plan.required_context() == 0 || plan.required_context() > compiler.mapping.chunks.context_tokens
        || plan.max_result_bytes() != compiler.mapping.max_result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4FaithfulnessMapError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use Int4FaithfulnessMapError as M;
    use Int4JudgeError as J;
    let mut failure = match error {
        M::EmptySource => BatchItemFailure::reject(BatchCode::DocumentLimit),
        M::Allocation | M::Chunk(MapReduceError::AllocationRefused)
            | M::Aggregation(FaithfulnessAggregationError::Allocation)
            | M::Judge(J::Task(JudgeError::AllocationRefused | JudgeError::Scoring(ScoringError::AllocationRefused))
                | J::Native(StrictInt4Error::Allocation)
                | J::Scoring(Int4ScoringError::Allocation | Int4ScoringError::Native(StrictInt4Error::Allocation)))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        M::Work | M::Judge(J::WorkBudget | J::Native(StrictInt4Error::Work)
            | J::Scoring(Int4ScoringError::Native(StrictInt4Error::Work)))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        M::Limits | M::Chunk(_) | M::Judge(J::Task(JudgeError::Contract(_) | JudgeError::Limit(_)))
            => BatchItemFailure::reject(BatchCode::Planning),
        M::Output | M::Judge(J::Scoring(Int4ScoringError::OutputBudget))
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        M::Serialization | M::Judge(J::Task(JudgeError::Serialization) | J::Scoring(Int4ScoringError::Serialization))
            => BatchItemFailure::fatal(BatchCode::Serialization),
        M::Judge(J::Native(StrictInt4Error::Context | StrictInt4Error::Memory)
            | J::Scoring(Int4ScoringError::Native(StrictInt4Error::Context | StrictInt4Error::Memory)))
            => BatchItemFailure::reject(BatchCode::Admission),
        M::Admission | M::Judge(J::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    if executing { failure.stop = true; }
    failure
}

#[cfg(test)] mod tests;
