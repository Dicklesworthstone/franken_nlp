//! Whole-corpus pairwise/rubric/faithfulness execution with real host admission.
//! One resident native engine; complete per-document judgments, no retries.

use super::*;
use crate::{
    batch::{BatchDocument, BatchFault, BatchProcessor, BatchWork, judge::JudgeBatchArgs},
    native_engine::{constrained_int8::check_profile,
        decode::{DecodeCancellationKind, DecodeStepControl},
        lmhead::scoring::ScoringError,
        strict_int8::{StrictInt8Engine, scoring::{Int8ScoringBudget, Int8ScoringError}}},
    tasks::{ir::{PlanContext, TaskBudget}, judge::{JudgeError, JudgeLimits, JudgePlanner,
        quantized::{Int8JudgeError, Int8JudgeRun, PreparedInt8Judge}}},
};

/// Caller-owned planning settings, not deserialized executable task authority.
/// No Debug because defaults may contain private source, criteria or claims.
pub struct JudgeCorpusConfig {
    pub identity: ExecutionIdentity,
    pub task_ceiling: TaskBudget,
    pub planning: JudgeLimits,
    pub defaults: Option<JudgeBatchArgs>,
    /// Nonrenewable complete-head work for the whole stream, including failures.
    pub max_model_work: Int8Work,
}

impl NlpEngine {
    /// Execute bounded ordered NDJSON judge requests using the common argument
    /// schema. `text` means A for pairwise, the rubric document, or the full
    /// faithfulness source. No model text is parsed into a judgment; every
    /// decision comes from complete full-vocabulary/EOS candidate scoring.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int8_judge<R, W>(&self, model: &ResidentInt8,
        planner: Arc<JudgePlanner>, config: JudgeCorpusConfig, limits: CorpusLimits,
        reader: R, writer: W, cancellation: CancellationToken) -> Result<BatchSummary, HostedError>
    where R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        check_binding(&config.identity, planner.template_digest(), planner.tokenizer_digest())?;
        config.task_ceiling.validate().map_err(|_| HostedError::Limits("judge corpus task ceiling"))?;
        validate_work(config.max_model_work)?;
        let required = requirements(limits.native)?;
        if required.kv_bytes > config.task_ceiling.max_kv_bytes {
            return Err(HostedError::Limits("judge corpus whole KV allocation"));
        }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.reservation_bytes()?)?,
            || Ok(StreamInput { planner: Some((planner, config)), reader, writer }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Preserve the entire storage-before-charge package if queued work
            // is discarded; capturing individual fields would lose that order.
            let mut input = input;
            let (planner, config) = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let admission = CorpusAdmission { lease: &lease, model: model.artifact_identity(),
                kv_bytes: required.kv_bytes, sampler_bytes: 0,
                output_bytes: limits.transport.max_output_line_bytes as u64 };
            let result = {
                let ledger = WorkLedger::new(config.max_model_work);
                let mut processor = JudgeProcessor { planner: &planner, config,
                    engine: &mut engine.value, admission, ledger };
                batch::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                    &mut processor, limits.transport, control).map_err(HostedError::Batch)
            };
            drop(engine);
            drop(planner);
            drop(input);
            drop(lease);
            result
        })
    }
}

fn check_binding(identity: &ExecutionIdentity, template: &Sha256Digest, tokenizer: Sha256Digest)
    -> Result<(), HostedError> {
    check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
    if identity.task_spec != "judge-v1" || identity.template_digest != *template
        || identity.tokenizer_digest != tokenizer { return Err(HostedError::ModelIdentity); }
    Ok(())
}

struct JudgeProcessor<'p, 'e, 'w, 'a> {
    planner: &'p JudgePlanner,
    config: JudgeCorpusConfig,
    engine: &'e mut StrictInt8Engine<'w>,
    admission: CorpusAdmission<'a>,
    ledger: WorkLedger,
}
impl BatchProcessor for JudgeProcessor<'_, '_, '_, '_> {
    type Args = JudgeBatchArgs;
    type Prepared = PreparedInt8Judge;
    type Output = GuardedOutput<Int8JudgeRun, Pending>;

    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        let args = document.task_args.or_else(|| self.config.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        let context = PlanContext::new(&self.config.identity, self.config.task_ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        // The common runner checks cancellation around prepare. Its current
        // trait does not lend control during planning; do not invent a renewed
        // run budget or claim preemption within the bounded legacy tokenizer.
        let prepared = self.planner.plan_int8_with_control(&args.into_request(document.text),
            &context, self.config.planning, &mut PlanningControl).map_err(planning_failure)?;
        self.ledger.preview(prepared.planned_work())?;
        if prepared.max_result_bytes() > self.admission.output_bytes {
            return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
        }
        prepared.preflight(prepared.execution_identity(), self.engine,
            scoring_budget(&prepared, self.admission.kv_bytes)).map_err(planning_failure)?;
        Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        // Charge the full attempt BEFORE admission or native callbacks. An
        // unwind leaves Running and cannot turn into a reusable partial task.
        self.ledger.begin(prepared.planned_work())?;
        let result = (|| {
            checkpoint(control)?;
            let (admitted, guard) = self.admission.admit_output(prepared.execution_identity(),
                prepared.planned_work(), self.admission.kv_bytes, 0, prepared.max_result_bytes())?;
            checkpoint(control)?;
            let run = prepared.execute_with_control(&admitted, self.engine,
                scoring_budget(&prepared, self.admission.kv_bytes), control);
            let healthy = !self.engine.is_poisoned() && self.engine.kv_cache().all_slots_have_len(0);
            let run = run.map_err(|error| execution_outcome_failure(error, healthy))?;
            if !healthy || run.model_work != prepared.planned_work() || run.head_count != prepared.head_count() {
                return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
            }
            checkpoint(control)?;
            Ok(GuardedOutput::new(run, guard))
        })();
        self.ledger.finish(result.as_ref().err());
        result
    }
}
fn scoring_budget(plan: &PreparedInt8Judge, allocated_kv: u64) -> Int8ScoringBudget {
    Int8ScoringBudget { native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: allocated_kv }
}
struct PlanningControl;
impl DecodeStepControl for PlanningControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkPhase { Ready, Running, Failed }
struct WorkLedger { spent: Int8Work, ceiling: Int8Work, phase: WorkPhase }
impl WorkLedger {
    fn new(ceiling: Int8Work) -> Self { Self { spent: Int8Work::default(), ceiling, phase: WorkPhase::Ready } }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.phase == WorkPhase::Ready { Ok(()) } else { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
    }
    fn preview(&self, work: Int8Work) -> Result<Int8Work, BatchItemFailure> {
        self.ready()?;
        let next = self.spent.checked_add(work).map_err(|_| BatchItemFailure::reject(BatchCode::WorkLimit))?;
        if next.forward_positions > self.ceiling.forward_positions || next.projected_logits > self.ceiling.projected_logits
            || next.attention_pairs > self.ceiling.attention_pairs || !next.projections.fits(self.ceiling.projections) {
            return Err(BatchItemFailure::reject(BatchCode::WorkLimit));
        }
        Ok(next)
    }
    fn begin(&mut self, work: Int8Work) -> Result<(), BatchItemFailure> {
        self.spent = self.preview(work)?;
        self.phase = WorkPhase::Running;
        Ok(())
    }
    fn finish(&mut self, failure: Option<&BatchItemFailure>) {
        self.phase = if self.phase != WorkPhase::Running || failure.is_some_and(|e| e.stop) {
            WorkPhase::Failed
        } else { WorkPhase::Ready };
    }
}

fn planning_failure(error: Int8JudgeError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8JudgeError::Task(JudgeError::Contract(_) | JudgeError::Limit(_))
            | Int8JudgeError::Scoring(Int8ScoringError::Input | Int8ScoringError::Scoring(ScoringError::LimitExceeded(_)))
            => BatchItemFailure::reject(BatchCode::Planning),
        Int8JudgeError::WorkBudget | Int8JudgeError::Native(StrictInt8Error::Work)
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int8JudgeError::Native(StrictInt8Error::Context | StrictInt8Error::Memory)
            | Int8JudgeError::Scoring(Int8ScoringError::Native(StrictInt8Error::Context | StrictInt8Error::Memory))
            => BatchItemFailure::reject(BatchCode::Admission),
        other => execution_failure(other),
    }
}
fn execution_outcome_failure(error: Int8JudgeError, healthy: bool) -> BatchItemFailure {
    let mut failure = execution_failure(error);
    // Preserve the original cancellation/error cause even when failure also
    // poisoned the native session. Never repair or silently reuse that engine.
    if !healthy { failure.stop = true; }
    failure
}

fn execution_failure(error: Int8JudgeError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8JudgeError::Task(JudgeError::Limit("complete_output_bytes"))
            | Int8JudgeError::Scoring(Int8ScoringError::OutputBudget) => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        Int8JudgeError::Identity => BatchItemFailure::fatal(BatchCode::Admission),
        Int8JudgeError::Task(JudgeError::AllocationRefused) | Int8JudgeError::Native(StrictInt8Error::Allocation)
            | Int8JudgeError::Scoring(Int8ScoringError::Allocation | Int8ScoringError::Scoring(ScoringError::AllocationRefused))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        Int8JudgeError::Task(JudgeError::Serialization) | Int8JudgeError::Scoring(Int8ScoringError::Serialization)
            => BatchItemFailure::fatal(BatchCode::Serialization),
        Int8JudgeError::Accounting | Int8JudgeError::Scoring(Int8ScoringError::Accounting | Int8ScoringError::Traversal)
            => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}

#[cfg(test)] mod tests;
