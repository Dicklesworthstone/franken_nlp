//! One resident engine, complete guarded result and nonrefundable work debit.
use super::*;

pub struct Int4GenerationAdmission<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int4Work,
    pub sampler_bytes: u64,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub max_output_tokens: u64,
}
pub trait Int4GenerationBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4GenerationAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

pub struct Int4GenerationJobProcessor<'p, 'e, 'w, A: Int4GenerationBatchAdmission> {
    planner: &'p Int4GenerationJobPlanner,
    engine: &'e mut StrictInt4Engine<'w>,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, A: Int4GenerationBatchAdmission> Int4GenerationJobProcessor<'p, 'e, 'w, A> {
    pub fn new(planner: &'p Int4GenerationJobPlanner, engine: &'e mut StrictInt4Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        check_capacity(planner, engine).map_err(|error| error.fault)?;
        Ok(Self { planner, engine, admission, ledger: Ledger::default() })
    }
}
impl<A: Int4GenerationBatchAdmission> BatchProcessor for Int4GenerationJobProcessor<'_, '_, '_, A> {
    type Args = GenerationJobArgs;
    type Prepared = PreparedInt4GenerationJob;
    type Output = GuardedOutput<Int4ChatResult, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        self.ledger.failed = true;
        check_capacity(self.planner, self.engine)?;
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
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        // JobRunner has already synced the full durable attempt charge. This
        // additional invocation ledger is poisoned before any host callback.
        self.ledger.begin(prepared.work, context, self.planner.job_limits())?;
        self.planner.check(&prepared)?;
        poll(control)?;
        let kv = check_capacity(self.planner, self.engine)?;
        prepared.native.preflight(prepared.execution_identity(), self.engine, self.planner.native_limits())
            .map_err(|error| failure(error, BatchCode::Admission))?;
        let (admitted, guard) = self.admission.admit(Int4GenerationAdmission {
            identity: prepared.execution_identity(), model_work: prepared.model_work(), sampler_bytes: prepared.sampler_bytes(),
            kv_reservation_bytes: kv, max_result_bytes: self.planner.max_result_bytes(),
            max_output_tokens: self.planner.output_token_ceiling(),
        })?;
        poll(control)?;
        prepared.native.verify_identity(&admitted).map_err(|error| failure(error, BatchCode::Admission))?;
        // Delivery uses the ORIGINAL job ordinal. Native addressed draws use
        // the unchanged original item ID, sample index and fixed effective seed.
        let output = prepared.native.execute(&admitted, self.engine, context.request_seq,
            self.planner.native_limits(), control).map_err(|error| failure(error, BatchCode::Execution))?;
        if output.schema_version != 1 || output.result.schema_version != 1
            || output.result.task != self.planner.recipe.task.identity()
            || output.result.request_seq != context.request_seq || output.result.sample_index != prepared.sample_index
            || output.result.execution != INT4_GENERATION_VERSION || output.result.numerics_profile != STRICT_INT4_PROFILE
            || !JobWork::from_int4(output.model_work, 0).fits(prepared.work)
            || output.result.native_work.forward_positions != output.model_work.forward_positions
            || output.result.native_work.projected_logits != output.model_work.projected_logits {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        check_capacity(self.planner, self.engine)?;
        poll(control)?;
        self.ledger.failed = false;
        // JobRunner borrows the actual native wire object through spool sync
        // and journal acknowledgement, retaining this exact output admission.
        Ok(GuardedOutput::new(output, guard))
    }
}
impl<A: Int4GenerationBatchAdmission> DurableBatchProcessor for Int4GenerationJobProcessor<'_, '_, '_, A> {
    type Recipe = Int4GenerationJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.ledger.ready()?;
        self.planner.check(prepared)?;
        check_capacity(self.planner, self.engine)?;
        Ok(prepared.work)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.max_result_bytes() }
}

#[derive(Default)]
struct Ledger { work: JobWork, attempts: u64, sequence: u64, failed: bool }
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, work: JobWork, context: BatchRequestContext, limits: JobLimits) -> Result<(), BatchItemFailure> {
        self.ready()?;
        self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let attempts = self.attempts.checked_add(1).filter(|&n| n <= limits.max_attempts)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let next = self.work.checked_add(work).ok().filter(|w| w.fits(limits.max_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.work = next; self.attempts = attempts; self.sequence = context.request_seq;
        Ok(())
    }
}

#[cfg(test)] mod tests;
