//! Fixed native schedules with one complete-result guard through durable sync.
use super::*;

pub struct Int4ClassificationDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    /// The declared physical peak, independently of each chunk's task ceiling.
    pub kv_reservation_bytes: u64,
    /// Complete stored wire object, including the explicit cohort wrapper.
    pub max_result_bytes: u64,
    pub chunk_count: usize,
    /// Independent classifier heads per chunk, not the number of documents.
    pub head_count: usize,
}
pub trait Int4ClassificationDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4ClassificationDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

/// Preserve the corresponding native wire object without an extra wrapper.
/// The original chunk results and explicitly named aggregate remain intact.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Int4ClassificationDocumentRun {
    Serial(Int4ClassificationMapRun),
    Cohorts(Int4ClassificationMapCohortRun),
}

pub struct Int4ClassificationDocumentJobProcessor<'p, 'e, 'w, A: Int4ClassificationDocumentBatchAdmission> {
    planner: &'p Int4ClassificationDocumentJobPlanner,
    backend: Backend<'e, 'w>,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, A: Int4ClassificationDocumentBatchAdmission> Int4ClassificationDocumentJobProcessor<'p, 'e, 'w, A> {
    pub fn new(planner: &'p Int4ClassificationDocumentJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        Self::configured(planner, Backend::Serial(engine), admission)
    }
    pub fn new_cohorts(planner: &'p Int4ClassificationDocumentJobPlanner, weights: &'w Int4ModelWeights,
        budget: Int4ClassificationMapCohortBudget, kernel: PackedInt4Kernel, admission: A)
        -> Result<Self, BatchFault> {
        Self::configured(planner, Backend::Cohorts { weights, budget, kernel }, admission)
    }
    fn configured(planner: &'p Int4ClassificationDocumentJobPlanner, backend: Backend<'e, 'w>, admission: A)
        -> Result<Self, BatchFault> {
        planner.check_job_limits(planner.limits)?;
        backend.check_capacity(planner).map_err(|failure| failure.fault)?;
        Ok(Self { planner, backend, admission, ledger: Ledger::default() })
    }
}
impl<A: Int4ClassificationDocumentBatchAdmission> BatchProcessor for Int4ClassificationDocumentJobProcessor<'_, '_, '_, A> {
    type Args = Int4ClassificationDocumentArgs;
    type Prepared = PreparedInt4ClassificationDocument;
    type Output = GuardedOutput<Int4ClassificationDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        self.ledger.failed = true;
        self.backend.check_capacity(self.planner)?;
        let prepared = self.planner.prepare_with_control(document, control)?;
        self.ledger.failed = false;
        Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        BatchWork { forward_positions: prepared.work.model.forward_positions,
            projected_logits: prepared.work.model.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, mut prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        // JobRunner persisted the exact whole-document debit already. Poison
        // before any callback so failure/unwinding can never silently retry.
        self.ledger.begin(prepared.work, context, self.planner.limits)?;
        self.planner.owns(&prepared)?;
        poll(control)?;
        let kv = self.backend.check_capacity(self.planner)?;
        let plan = self.planner.plan_with_encoding_work(&prepared.request, control, &mut prepared.encoding_work)?;
        self.planner.compare(&plan, &prepared, control)?;
        self.backend.preflight(self.planner, &plan, &prepared.identities, control)?;
        let (admitted, guard) = self.admission.admit(Int4ClassificationDocumentAdmission {
            identities: &prepared.identities, model_work: prepared.work.int4_model_work(),
            kv_reservation_bytes: kv, max_result_bytes: self.planner.max_result_bytes(),
            chunk_count: prepared.chunk_count(), head_count: self.planner.head_count(),
        })?;
        poll(control)?;
        if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        // The native map preflights every future chunk/head/window before its
        // first forward, using the same immutable original identities.
        let output = self.backend.execute(self.planner, &plan, &admitted, control)?;
        match &output {
            Int4ClassificationDocumentRun::Serial(run) if !self.planner.uses_chunk_cohorts() =>
                prepared.expected.verify_completed_with_control(run, control).map_err(map_failure)?,
            Int4ClassificationDocumentRun::Cohorts(run) => {
                let chunks = self.planner.cohort_chunks()
                    .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
                prepared.expected.verify_completed_cohorts_with_control(run, chunks, control).map_err(map_failure)?;
            }
            _ => return Err(BatchItemFailure::fatal(BatchCode::Admission)),
        }
        self.backend.check_capacity(self.planner)?;
        poll(control)?;
        self.ledger.failed = false;
        // This guard travels through canonical encoding, spool sync and the
        // authenticated journal acknowledgement of the original source item.
        Ok(GuardedOutput::new(output, guard))
    }
}
impl<A: Int4ClassificationDocumentBatchAdmission> DurableBatchProcessor for Int4ClassificationDocumentJobProcessor<'_, '_, '_, A> {
    type Recipe = Int4ClassificationDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.ledger.ready()?;
        self.planner.owns(prepared)?;
        self.backend.check_capacity(self.planner)?;
        if JobWork::from_int4(prepared.expected.planned_work(), 0) != prepared.work {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(prepared.work)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.max_result_bytes() }
}

enum Backend<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, budget: Int4ClassificationMapCohortBudget,
        kernel: PackedInt4Kernel },
}
impl Backend<'_, '_> {
    fn model_identity(&self) -> &Int4ModelIdentity {
        match self { Self::Serial(engine) => engine.model_identity(), Self::Cohorts { weights, .. } => weights.identity() }
    }
    fn check_capacity(&self, planner: &Int4ClassificationDocumentJobPlanner) -> Result<u64, BatchItemFailure> {
        constrained_int4::check_model(planner.execution_identity(), self.model_identity())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if self.model_identity().group_width() != planner.group_width() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        match self {
            Self::Serial(engine) if !planner.uses_chunk_cohorts() => {
                let context = planner.mapping.chunks.context_tokens;
                let required = Int4MemoryRequirement::for_context(context, planner.group_width())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
                    || engine.kv_cache().capacity_positions() != context || required.kv_bytes == 0
                    || required.kv_bytes > planner.document_budget().max_kv_bytes
                    || required.kv_bytes > planner.task_ceiling().max_kv_bytes {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                Ok(required.kv_bytes)
            }
            Self::Cohorts { budget, .. } if planner.uses_chunk_cohorts() => {
                let required = planner.cohort_requirements().map_err(BatchItemFailure::fatal)?;
                check_cohort_budget(required, planner.mapping.max_model_work, *budget)?;
                Ok(required.memory.native.kv_bytes)
            }
            _ => Err(BatchItemFailure::fatal(BatchCode::Admission)),
        }
    }
    fn preflight<C: DecodeStepControl>(&self, planner: &Int4ClassificationDocumentJobPlanner,
        plan: &PreparedInt4ClassificationMap<'_>, identities: &[ExecutionIdentity], control: &mut C)
        -> Result<(), BatchItemFailure> {
        for identity in identities {
            poll(control)?;
            constrained_int4::check_model(identity, self.model_identity())
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        if let Self::Cohorts { budget, .. } = self {
            let chunks = planner.cohort_chunks().ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
            plan.preflight_cohorts_with_control(identities, self.model_identity(), chunks,
                Int4ClassificationMapCohortBudget { native: Int4RunBudget::exact(plan.planned_work()), ..*budget },
                control).map_err(map_failure)?;
        }
        poll(control)
    }
    fn execute<C: DecodeStepControl>(&mut self, planner: &Int4ClassificationDocumentJobPlanner,
        plan: &PreparedInt4ClassificationMap<'_>, identities: &[ExecutionIdentity], control: &mut C)
        -> Result<Int4ClassificationDocumentRun, BatchItemFailure> {
        match self {
            Self::Serial(engine) => plan.execute_with_control(identities, engine,
                Int4ScoringBudget { native: Int4RunBudget::exact(plan.planned_work()),
                    max_kv_bytes: planner.document_budget().max_kv_bytes }, control)
                .map(Int4ClassificationDocumentRun::Serial).map_err(map_failure),
            Self::Cohorts { weights, budget, kernel } => {
                let chunks = planner.cohort_chunks().ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
                plan.execute_cohorts_with_control(identities, weights, *kernel, chunks,
                    Int4ClassificationMapCohortBudget { native: Int4RunBudget::exact(plan.planned_work()), ..*budget },
                    control).map(Int4ClassificationDocumentRun::Cohorts).map_err(map_failure)
            }
        }
    }
}

fn check_cohort_budget(required: Int4ClassificationMapCohortMemoryRequirement, work: Int4Work,
    budget: Int4ClassificationMapCohortBudget) -> Result<(), BatchItemFailure> {
    let native = required.memory.native;
    if native.kv_bytes > budget.memory.max_kv_bytes || native.rope_bytes > budget.memory.max_rope_bytes
        || native.scratch_payload_bound > budget.memory.max_scratch_payload_bytes
        || required.task_workspace_bytes > budget.max_task_workspace_bytes
        || required.max_cohort_output_bytes != budget.max_output_bytes
        || work.forward_positions > budget.native.max_forward_positions
        || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}

/// Invocation defence only. Persisted nonrefundable work and attempts remain
/// the owned journal's authority, including failures in earlier invocations.
#[derive(Default)]
struct Ledger { work: JobWork, attempts: u64, sequence: u64, failed: bool }
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, work: JobWork, context: BatchRequestContext, limits: JobLimits) -> Result<(), BatchItemFailure> {
        self.ready()?;
        self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0
            || work.mask_node_visits != 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let attempts = self.attempts.checked_add(1).filter(|&n| n <= limits.max_attempts)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let next = self.work.checked_add(work).ok().filter(|work| work.fits(limits.max_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.work = next; self.attempts = attempts; self.sequence = context.request_seq;
        Ok(())
    }
}

#[cfg(test)] mod tests;
