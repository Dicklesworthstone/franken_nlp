//! Resident schema extraction followed by explicit correlated field reviews.
//!
//! Each complete document owns its source, grammar and dynamic review plans.
//! The immutable extraction/judge compilers and native engine span the stream.
//! Data-dependent review work is admitted conservatively: every attempt spends
//! the full declared document ceiling before either stage can run.

use std::sync::Arc;
use super::{ExtractionBatchArgs, ExtractionMaskBudget, GuardedOutput,
    grouped_int4::Int4ExtractionBatchPlanner};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork},
    execution_identity::ExecutionIdentity,
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int4, decode::DecodeStepControl,
        strict_int4::{Int4MemoryRequirement, Int4Work, StrictInt4Engine, STRICT_INT4_PROFILE}},
    tasks::{ir::TaskBudget, judge::{grouped_int4::Int4JudgePlanner, JudgeLimits},
        extract::{ExtractionVocabulary, semantic::{SemanticLimits,
            grouped_int4::{Int4SemanticEncoding, Int4SemanticError, Int4SemanticExtractionRun,
                Int4SemanticOptions, PreparedInt4SemanticExtraction, INT4_SEMANTIC_EXECUTION}}}},
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding},
};

/// Six invocation-wide ceilings, never renewed by a flush or failed attempt.
/// Int4SemanticOptions supplies the independent whole-document native ceiling.
#[derive(Clone, Copy, Debug)]
pub struct Int4SemanticExtractionBatchLimits {
    pub max_model_work: Int4Work,
    pub masks: ExtractionMaskBudget,
}
impl Int4SemanticExtractionBatchLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        let masks = self.masks;
        if !nonzero(self.max_model_work) || masks.per_mask.max_trie_node_visits == 0
            || masks.per_mask.checkpoint_interval_nodes == 0 || masks.max_visits_per_item == 0
            || masks.max_visits_per_run < masks.max_visits_per_item {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    pub fn validate_item(self, item: Int4Work) -> Result<(), BatchFault> {
        self.validate()?;
        if !nonzero(item) || !fits(item, self.max_model_work) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// Fixed exact schema, grounding, task budgets and experimental review policy.
/// Per-record overrides are rejected before document encoding or compilation.
pub struct Int4SemanticExtractionBatchPlanner {
    extraction: Int4ExtractionBatchPlanner,
    judge: Arc<Int4JudgePlanner>,
    options: Int4SemanticOptions,
    limits: Int4SemanticExtractionBatchLimits,
    extraction_budget: TaskBudget,
    seal: Arc<()>,
}
impl Int4SemanticExtractionBatchPlanner {
    pub fn new(extraction: Int4ExtractionBatchPlanner, judge: Arc<Int4JudgePlanner>,
        options: Int4SemanticOptions, limits: Int4SemanticExtractionBatchLimits)
        -> Result<Self, BatchFault> {
        options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        limits.validate_item(options.max_model_work)?;
        if extraction.selected_rows().is_some() || extraction.group_width() != judge.group_width()
            || extraction.document_encoding() != judge.document_encoding()
            || extraction.base_execution_identity().task_spec != "extract-v1"
            || judge.execution_identity().task_spec != "judge-v1" {
            return Err(BatchCode::InvalidLimits.into());
        }
        let extraction_budget = extraction.shared_default_budget()?;
        let (schema, compiler_limits, grounding) = extraction.shared_default_schema()?;
        let schema = crate::grammar::runtime::JsonValidationSchema::compile(schema, compiler_limits)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if schema.requires_verbatim_source() != (grounding == super::ExtractionBatchGrounding::SourceMembership) {
            return Err(BatchCode::InvalidLimits.into());
        }
        options.validate_schema(schema.root()).map_err(|_| BatchCode::InvalidLimits)?;
        let extraction = extraction.require_shared_defaults()?;
        Ok(Self { extraction, judge, options, limits, extraction_budget, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.extraction.base_execution_identity() }
    pub fn judge_identity(&self) -> &ExecutionIdentity { self.judge.execution_identity() }
    pub fn group_width(&self) -> usize { self.extraction.group_width() }
    pub fn extraction_budget(&self) -> TaskBudget { self.extraction_budget }
    pub fn judge_budget(&self) -> TaskBudget { self.options.specification.judge_budget }
    pub fn judge_planning_limits(&self) -> JudgeLimits { self.judge.semantic_configuration().1 }
    pub fn semantic_limits(&self) -> SemanticLimits { self.options.limits }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.extraction.document_encoding() }
    pub fn options(&self) -> &Int4SemanticOptions { &self.options }
    pub fn limits(&self) -> Int4SemanticExtractionBatchLimits { self.limits }
    pub fn max_result_bytes(&self) -> u64 { self.options.limits.max_output_bytes }

    fn prepare<C: DecodeStepControl>(&self, document: BatchDocument<ExtractionBatchArgs>, control: &mut C)
        -> Result<PreparedInt4SemanticExtraction, BatchItemFailure> {
        checkpoint(control)?;
        if document.task_args.is_some() { return Err(BatchItemFailure::reject(BatchCode::InvalidEnvelope)); }
        let extraction = self.extraction.prepare_with_control(document, control)?;
        PreparedInt4SemanticExtraction::new_shared(extraction, Arc::clone(&self.judge),
            self.options.clone(), control).map_err(preparation_failure)
    }
}

/// Complete simultaneous task facts, including the full-source second reader.
/// Host admission owns preparation/native/transport storage separately, then
/// keeps its returned output guard through serialization and the actual flush.
pub struct Int4SemanticExtractionAdmission<'a> {
    pub identity: &'a ExecutionIdentity,
    pub judge_identity: &'a ExecutionIdentity,
    pub group_width: usize,
    pub model_work: Int4Work,
    pub extraction_reserved_work: Int4Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub extraction_budget: TaskBudget,
    pub judge_budget: TaskBudget,
    pub semantic_limits: SemanticLimits,
    pub judge_planning_limits: JudgeLimits,
    pub source_bytes: usize,
    pub input_encoding: &'a UntrustedEncoding,
}
pub trait Int4SemanticExtractionBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4SemanticExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

/// Sealed to the exact resident compiler; it cannot be deserialized or reused
/// under another schema/specification factory with compatible-looking counts.
pub struct PreparedInt4SemanticBatchExtraction {
    plan: PreparedInt4SemanticExtraction,
    seal: Arc<()>,
}
impl PreparedInt4SemanticBatchExtraction {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn model_work(&self) -> Int4Work { self.plan.max_model_work() }
    pub fn max_result_bytes(&self) -> u64 { self.plan.max_result_bytes() }
}

pub struct NativeInt4SemanticExtractionBatch<'e, 'w, 'v, A: Int4SemanticExtractionBatchAdmission> {
    compiler: Int4SemanticExtractionBatchPlanner,
    driver: NativeDriver<'e, 'w, 'v>,
    admission: A,
    run: RunState,
}
impl<'e, 'w, 'v, A: Int4SemanticExtractionBatchAdmission> NativeInt4SemanticExtractionBatch<'e, 'w, 'v, A> {
    pub fn new(compiler: Int4SemanticExtractionBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        compiler.limits.validate_item(compiler.options.max_model_work)?;
        if !clean(engine) || compiler.group_width() != engine.model_identity().group_width() {
            return Err(BatchCode::InvalidExecution.into());
        }
        for identity in [compiler.base_execution_identity(), compiler.judge_identity()] {
            constrained_int4::check_model(identity, engine.model_identity()).map_err(|_| BatchCode::Admission)?;
        }
        let run = RunState::new(compiler.limits);
        Ok(Self { compiler, driver: NativeDriver { engine, vocabulary }, admission, run })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.run.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.run.masks }
    pub fn is_poisoned(&self) -> bool { self.run.failed || !self.driver.clean() }
    /// Durable adapters validate the private factory seal before reserving an
    /// attempt. A plan from another schema/review factory cannot spend work
    /// under this invocation's authenticated recipe, even if counts coincide.
    pub(crate) fn verify_prepared(&self, prepared: &PreparedInt4SemanticBatchExtraction)
        -> Result<(), BatchItemFailure> {
        self.run.ready()?;
        if !self.driver.clean() || !Arc::ptr_eq(&prepared.seal, &self.compiler.seal) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
}
impl<A: Int4SemanticExtractionBatchAdmission> BatchProcessor for NativeInt4SemanticExtractionBatch<'_, '_, '_, A> {
    type Args = ExtractionBatchArgs;
    type Prepared = PreparedInt4SemanticBatchExtraction;
    type Output = GuardedOutput<Int4SemanticExtractionRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.run.ready()?;
        self.run.failed = true;
        let result = (|| {
            let plan = self.compiler.prepare(document, control)?;
            preflight(&plan, &self.driver, self.run.limits)?;
            self.run.check_available(plan.max_model_work())?;
            checkpoint(control)?;
            Ok(PreparedInt4SemanticBatchExtraction { plan, seal: Arc::clone(&self.compiler.seal) })
        })();
        if !self.driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        if !result.as_ref().is_err_and(|failure: &BatchItemFailure| failure.stop) {
            self.run.failed = false;
        }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.model_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute(prepared.plan, prepared.seal, &self.compiler.seal, &mut self.driver,
            &mut self.admission, &mut self.run, context, control)
    }
}

struct RunState {
    limits: Int4SemanticExtractionBatchLimits,
    reserved: Int4Work,
    masks: u64,
    last_sequence: u64,
    failed: bool,
}
impl RunState {
    fn new(limits: Int4SemanticExtractionBatchLimits) -> Self {
        Self { limits, reserved: Int4Work::default(), masks: 0, last_sequence: 0, failed: false }
    }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn totals(&self, work: Int4Work) -> Result<(Int4Work, u64), BatchItemFailure> {
        let total = self.reserved.checked_add(work).ok().filter(|total| fits(*total, self.limits.max_model_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.masks.checked_add(self.limits.masks.max_visits_per_item)
            .filter(|&total| total <= self.limits.masks.max_visits_per_run)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        Ok((total, masks))
    }
    fn check_available(&self, work: Int4Work) -> Result<(), BatchItemFailure> { self.totals(work).map(|_| ()) }
    fn begin(&mut self, work: Int4Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.last_sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let (total, masks) = self.totals(work)?;
        self.reserved = total; self.masks = masks; self.last_sequence = context.request_seq;
        Ok(())
    }
}

// Private static seams test state transitions without accepting a caller's
// replacement native driver, synthetic task result or admission certificate.
trait Plan {
    type Output;
    type Completion;
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget) -> Int4SemanticExtractionAdmission<'_>;
    fn completion(&self) -> Self::Completion;
    fn verify_completion(expected: Self::Completion, output: &Self::Output, masks: u64)
        -> Result<(), BatchItemFailure>;
}
struct Completion {
    width: usize, extraction: Int4Work, ceiling: Int4Work,
    encoding: UntrustedEncoding, initial_work: SafeBpeWork,
}
impl Plan for PreparedInt4SemanticExtraction {
    type Output = Int4SemanticExtractionRun;
    type Completion = Completion;
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget) -> Int4SemanticExtractionAdmission<'_> {
        Int4SemanticExtractionAdmission { identity: self.execution_identity(), judge_identity: self.judge_identity(),
            group_width: self.group_width(), model_work: self.max_model_work(),
            extraction_reserved_work: self.extraction_reserved_work(), mask_node_visits: masks.max_visits_per_item,
            mask_limits: masks.per_mask, kv_reservation_bytes: kv, max_result_bytes: self.max_result_bytes(),
            extraction_budget: self.extraction_budget(), judge_budget: self.judge_budget(),
            semantic_limits: self.semantic_limits(), judge_planning_limits: self.judge_planning_limits(),
            source_bytes: self.source_bytes(), input_encoding: self.input_encoding() }
    }
    fn completion(&self) -> Completion {
        Completion { width: self.group_width(), extraction: self.extraction_reserved_work(), ceiling: self.max_model_work(),
            encoding: self.input_encoding().clone(), initial_work: self.input_encoding_work().unwrap_or_default() }
    }
    fn verify_completion(expected: Completion, run: &Int4SemanticExtractionRun, masks: u64)
        -> Result<(), BatchItemFailure> {
        let invalid = || BatchItemFailure::fatal(BatchCode::InvalidExecution);
        let encoding_work = run.input_encoding.as_ref().map(|report| report.work).unwrap_or_default();
        let encoding = Int4SemanticEncoding::from_work(&expected.encoding, encoding_work).map_err(|_| invalid())?;
        let reserved = expected.extraction.checked_add(run.semantic_model_work).map_err(|_| invalid())?;
        let actual = run.extraction.model_work.checked_add(run.semantic_model_work).map_err(|_| invalid())?;
        if run.schema_version != 1 || run.execution != INT4_SEMANTIC_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != expected.width
            || run.extraction.group_width != expected.width || !fits(run.extraction.model_work, expected.extraction)
            || run.extraction_reserved_work != expected.extraction || run.reserved_model_work != reserved
            || run.model_work != actual || !fits(actual, reserved) || !fits(reserved, expected.ceiling)
            || run.extraction.result.output.mask_node_visit_charge > masks
            || run.input_encoding != encoding || encoding_work.checked_sub(expected.initial_work).is_none()
            || !run.semantic.same_model_correlated
            || run.semantic.calibration != "experimental_uncalibrated_diagnostic_only" {
            return Err(invalid());
        }
        Ok(())
    }
}
trait Driver<P: Plan> {
    fn clean(&self) -> bool;
    fn capacity(&self) -> usize;
    fn group_width(&self) -> usize;
    fn execute<C: DecodeStepControl>(&mut self, prepared: P, masks: ExtractionMaskBudget, control: &mut C)
        -> Result<P::Output, BatchItemFailure>;
}
struct NativeDriver<'e, 'w, 'v> { engine: &'e mut StrictInt4Engine<'w>, vocabulary: &'v ExtractionVocabulary }
impl Driver<PreparedInt4SemanticExtraction> for NativeDriver<'_, '_, '_> {
    fn clean(&self) -> bool { clean(self.engine) }
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn group_width(&self) -> usize { self.engine.model_identity().group_width() }
    fn execute<C: DecodeStepControl>(&mut self, prepared: PreparedInt4SemanticExtraction,
        masks: ExtractionMaskBudget, control: &mut C) -> Result<Int4SemanticExtractionRun, BatchItemFailure> {
        prepared.execute_with_control(self.engine, self.vocabulary, masks.per_mask, masks.max_visits_per_item, control)
            .map_err(execution_failure)
    }
}
fn preflight<P: Plan, D: Driver<P>>(plan: &P, driver: &D, limits: Int4SemanticExtractionBatchLimits)
    -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let facts = plan.admission(0, limits.masks);
    if facts.group_width != driver.group_width() || facts.identity.task_spec != "extract-v1"
        || facts.judge_identity.task_spec != "judge-v1" || !nonzero(facts.model_work)
        || !nonzero(facts.extraction_reserved_work) || !fits(facts.extraction_reserved_work, facts.model_work)
        || !fits(facts.model_work, limits.max_model_work) || facts.source_bytes == 0 || facts.max_result_bytes == 0
        || facts.extraction_budget.validate().is_err() || facts.judge_budget.validate().is_err() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let memory = Int4MemoryRequirement::for_context(driver.capacity(), driver.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if facts.extraction_reserved_work.forward_positions > driver.capacity() as u64
        || memory.kv_bytes > facts.extraction_budget.max_kv_bytes
        || memory.kv_bytes > facts.judge_budget.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(memory.kv_bytes)
}
#[allow(clippy::too_many_arguments)]
fn execute<P: Plan, D: Driver<P>, A: Int4SemanticExtractionBatchAdmission, C: DecodeStepControl>(
    plan: P, seal: Arc<()>, expected_seal: &Arc<()>, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<P::Output, A::Guard>, BatchItemFailure> {
    run.ready()?;
    if !Arc::ptr_eq(&seal, expected_seal) {
        run.failed = true; return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let work = plan.admission(0, run.limits.masks).model_work;
    run.begin(work, context)?;
    let result = (|| {
        checkpoint(control)?;
        let kv = preflight(&plan, driver, run.limits)?;
        let facts = plan.admission(kv, run.limits.masks);
        let expected_identity = facts.identity.clone();
        let (identity, guard) = admission.admit(facts)?;
        if identity != expected_identity { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        checkpoint(control)?;
        let expected = plan.completion();
        let output = driver.execute(plan, run.limits.masks, control)?;
        P::verify_completion(expected, &output, run.limits.masks.max_visits_per_item)?;
        checkpoint(control)?;
        if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(GuardedOutput::new(output, guard))
    })();
    match result {
        Ok(output) => { run.failed = false; Ok(output) }
        Err(mut failure) => { failure.stop = true; Err(failure) }
    }
}
fn clean(engine: &StrictInt4Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0)
}
fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn nonzero(work: Int4Work) -> bool {
    work.forward_positions > 0 && work.projected_logits > 0 && work.attention_pairs > 0
        && work.projections.dot_products > 0 && work.projections.multiply_accumulates > 0
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn preparation_failure(error: Int4SemanticError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use crate::tasks::{extract::semantic::SemanticError, judge::JudgeError};
    match error {
        Int4SemanticError::Contract
            | Int4SemanticError::Semantic(SemanticError::Judge(JudgeError::Contract(_))) =>
            BatchItemFailure::reject(BatchCode::Planning),
        Int4SemanticError::WorkBudget | Int4SemanticError::Semantic(SemanticError::Limit(_)
            | SemanticError::Judge(JudgeError::Limit(_))) => BatchItemFailure::reject(BatchCode::WorkLimit),
        _ => execution_failure(error),
    }
}
fn execution_failure(error: Int4SemanticError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(match error {
        Int4SemanticError::WorkBudget => BatchCode::WorkLimit,
        Int4SemanticError::Contract | Int4SemanticError::Accounting => BatchCode::InvalidExecution,
        _ => BatchCode::Execution,
    })
}

#[cfg(test)] mod tests;
