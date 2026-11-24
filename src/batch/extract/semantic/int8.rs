//! Resident INT8 extraction followed by explicit complete-source field review.
//! The genuine consuming task owns both stages. This adapter owns only bounded
//! population admission: fixed compilers, one invocation ledger and guarded
//! complete results. No replacement native driver or saved-result entry exists.

use std::sync::Arc;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork,
        extract::{ExtractionBatchArgs, ExtractionBatchGrounding, ExtractionMaskBudget, GuardedOutput,
            quantized::Int8ExtractionBatchPlanner}},
    execution_identity::ExecutionIdentity,
    grammar::{mask::MaskWorkLimits, runtime::JsonValidationSchema},
    native_engine::{constrained_int8, decode::DecodeStepControl, kv::KV_BYTES_PER_TOKEN,
        strict_int8::{Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE, prefill::Int8PrefillLimits}},
    tasks::{ir::TaskBudget, judge::{JudgeLimits, JudgePlanner},
        extract::{ExtractionVocabulary, semantic::{SemanticError, SemanticLimits,
            int8::{Int8SemanticError, Int8SemanticExtractionRun, Int8SemanticOptions,
                PreparedInt8SemanticExtraction, INT8_SEMANTIC_EXECUTION, check_judge_binding}}}},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
};

/// Five native work axes and mask work for the complete invocation. An attempt
/// reserves the whole document ceiling before either native stage can run.
#[derive(Clone, Copy, Debug)]
pub struct Int8SemanticExtractionBatchLimits {
    pub max_model_work: Int8Work,
    pub masks: ExtractionMaskBudget,
}
impl Int8SemanticExtractionBatchLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        let masks = self.masks;
        if !nonzero(self.max_model_work) || masks.per_mask.max_trie_node_visits == 0
            || masks.per_mask.checkpoint_interval_nodes == 0 || masks.max_visits_per_item == 0
            || masks.max_visits_per_run < masks.max_visits_per_item {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    pub fn validate_item(self, item: Int8Work) -> Result<(), BatchFault> {
        self.validate()?;
        if !nonzero(item) || !fits(item, self.max_model_work) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// Fixed schema, grounding, native/review budgets and encoding declarations.
/// Sharing this actual owner keeps both pinned compilers alive through physical
/// worker join; mutable invocation counters belong to the native processor.
pub struct Int8SemanticExtractionBatchPlanner {
    extraction: Int8ExtractionBatchPlanner,
    judge: Arc<JudgePlanner>,
    judge_identity: ExecutionIdentity,
    judge_planning: JudgeLimits,
    options: Int8SemanticOptions,
    limits: Int8SemanticExtractionBatchLimits,
    extraction_budget: TaskBudget,
    encoding_work_limits: Option<SafeBpeWorkLimits>,
}
impl Int8SemanticExtractionBatchPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(extraction: Int8ExtractionBatchPlanner, judge: Arc<JudgePlanner>,
        judge_identity: ExecutionIdentity, judge_planning: JudgeLimits,
        options: Int8SemanticOptions, limits: Int8SemanticExtractionBatchLimits,
        encoding_work_limits: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        limits.validate_item(options.max_model_work)?;
        check_judge_limits(judge_planning)?;
        encoding_workspace(extraction.document_encoding(), encoding_work_limits)?;
        if extraction.map_document_encoding_limits().is_some()
            || extraction.document_encoding() != judge.document_encoding() {
            return Err(BatchCode::InvalidLimits.into());
        }
        check_judge_binding(extraction.base_execution_identity(), &judge, &judge_identity)
            .map_err(|_| BatchCode::Admission)?;
        let extraction_budget = extraction.shared_default_budget()?;
        let (schema, compiler_limits, grounding) = extraction.shared_default_schema()?;
        let schema = JsonValidationSchema::compile(schema, compiler_limits)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if schema.requires_verbatim_source() != (grounding == ExtractionBatchGrounding::SourceMembership) {
            return Err(BatchCode::InvalidLimits.into());
        }
        options.validate_schema(schema.root()).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { extraction, judge, judge_identity, judge_planning, options, limits,
            extraction_budget, encoding_work_limits })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.extraction.base_execution_identity() }
    pub fn judge_identity(&self) -> &ExecutionIdentity { &self.judge_identity }
    pub fn extraction_budget(&self) -> TaskBudget { self.extraction_budget }
    pub fn judge_budget(&self) -> TaskBudget { self.options.specification.judge_budget }
    pub fn judge_planning_limits(&self) -> JudgeLimits { self.judge_planning }
    pub fn semantic_limits(&self) -> SemanticLimits { self.options.limits }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.extraction.document_encoding() }
    pub fn encoding_work_limits(&self) -> Option<SafeBpeWorkLimits> { self.encoding_work_limits }
    pub fn options(&self) -> &Int8SemanticOptions { &self.options }
    pub fn limits(&self) -> Int8SemanticExtractionBatchLimits { self.limits }
    pub fn max_result_bytes(&self) -> u64 { self.options.limits.max_output_bytes }
    /// One transient encoder plus the small invocation ledger, beside all
    /// simultaneous original-source, grammar, claim, judge and result owners.
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> {
        encoding_workspace(self.document_encoding(), self.encoding_work_limits)
    }
    fn prepare<C: DecodeStepControl>(&self, document: BatchDocument<ExtractionBatchArgs>,
        used: &mut SafeBpeWork, control: &mut C) -> Result<PreparedInt8SemanticExtraction, BatchItemFailure> {
        checkpoint(control)?;
        if document.task_args.is_some() { return Err(BatchItemFailure::reject(BatchCode::InvalidEnvelope)); }
        let extraction = self.extraction.prepare_with_invocation_encoding_work(document,
            used, self.encoding_work_limits, control)?;
        PreparedInt8SemanticExtraction::new(extraction, Arc::clone(&self.judge),
            self.judge_identity.clone(), self.judge_planning, self.options.clone(), control)
            .map_err(preparation_failure)
    }
}

pub(crate) fn encoding_workspace(policy: &UntrustedEncoding, total: Option<SafeBpeWorkLimits>)
    -> Result<u64, BatchFault> {
    policy.validate().map_err(|_| BatchCode::InvalidLimits)?;
    match (policy, total) {
        (UntrustedEncoding::ByteFallback, None) => Ok(0),
        (UntrustedEncoding::SafeBpeV1(call), Some(total)) => {
            total.validate().map_err(|_| BatchCode::InvalidLimits)?;
            call.requirements().map_err(|_| BatchCode::InvalidLimits)?.workspace_bytes
                .checked_add(1024).ok_or_else(|| BatchCode::InvalidLimits.into())
        }
        _ => Err(BatchCode::InvalidLimits.into()),
    }
}
pub(crate) fn check_judge_limits(limits: JudgeLimits) -> Result<(), BatchFault> {
    if limits.max_total_prompt_tokens == 0 || limits.max_total_candidate_tokens == 0
        || limits.max_total_projected_logits == 0 || limits.max_output_bytes == 0
        || limits.per_head.max_candidates == 0 || limits.per_head.max_total_tokens == 0
        || limits.per_head.max_nodes == 0 || limits.per_head.max_depth == 0
        || limits.per_head.max_candidate_id_bytes == 0 || limits.per_head.max_projected_logits == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

/// Both genuine model bindings and the complete two-stage envelope. The host
/// retains preparation/native claims separately and returns an output owner
/// held through canonical serialization and the actual flush or durable commit.
pub struct Int8SemanticExtractionAdmission<'a> {
    pub identity: &'a ExecutionIdentity,
    pub judge_identity: &'a ExecutionIdentity,
    pub model_work: Int8Work,
    pub extraction_reserved_work: Int8Work,
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
    pub encoding_work_limits: Option<SafeBpeWorkLimits>,
}
pub trait Int8SemanticExtractionBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int8SemanticExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

/// A plan belongs to exactly one processor invocation, even if another
/// processor shares the same immutable factory. Transferring a prepared item
/// cannot evade the invocation that paid its original tokenizer work.
pub struct PreparedInt8SemanticBatchExtraction {
    plan: PreparedInt8SemanticExtraction,
    seal: Arc<()>,
}
impl PreparedInt8SemanticBatchExtraction {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn model_work(&self) -> Int8Work { self.plan.max_model_work() }
    pub fn max_result_bytes(&self) -> u64 { self.plan.max_result_bytes() }
}

pub struct NativeInt8SemanticExtractionBatch<'e, 'w, 'v, A: Int8SemanticExtractionBatchAdmission> {
    compiler: Arc<Int8SemanticExtractionBatchPlanner>,
    driver: NativeDriver<'e, 'w, 'v>,
    admission: A,
    run: RunState,
}
impl<'e, 'w, 'v, A: Int8SemanticExtractionBatchAdmission> NativeInt8SemanticExtractionBatch<'e, 'w, 'v, A> {
    pub fn new(compiler: Int8SemanticExtractionBatchPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::new_shared(Arc::new(compiler), engine, vocabulary, admission)
    }
    pub fn new_shared(compiler: Arc<Int8SemanticExtractionBatchPlanner>, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::scheduled(compiler, engine, vocabulary, admission, None)
    }
    pub fn new_shared_layer_major(compiler: Arc<Int8SemanticExtractionBatchPlanner>,
        engine: &'e mut StrictInt8Engine<'w>, vocabulary: &'v ExtractionVocabulary, admission: A,
        prefill: Int8PrefillLimits) -> Result<Self, BatchFault> {
        Self::scheduled(compiler, engine, vocabulary, admission, Some(prefill))
    }
    fn scheduled(compiler: Arc<Int8SemanticExtractionBatchPlanner>, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A, prefill: Option<Int8PrefillLimits>)
        -> Result<Self, BatchFault> {
        compiler.limits.validate_item(compiler.options.max_model_work)?;
        compiler.encoding_workspace_bytes()?;
        if let Some(prefill) = prefill { prefill.validate().map_err(|_| BatchCode::InvalidLimits)?; }
        if !clean(engine) { return Err(BatchCode::InvalidExecution.into()); }
        for identity in [compiler.base_execution_identity(), compiler.judge_identity()] {
            constrained_int8::check_model(identity, engine.artifact_identity()).map_err(|_| BatchCode::Admission)?;
        }
        let run = RunState::new(compiler.limits, compiler.encoding_work_limits);
        Ok(Self { compiler, driver: NativeDriver { engine, vocabulary, prefill }, admission, run })
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.run.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.run.masks }
    pub fn encoding_work(&self) -> Option<SafeBpeWork> {
        self.compiler.encoding_work_limits.map(|_| self.run.encoding)
    }
    pub fn is_poisoned(&self) -> bool { self.run.failed || !self.driver.clean() }
    pub(crate) fn verify_prepared(&self, prepared: &PreparedInt8SemanticBatchExtraction)
        -> Result<(), BatchItemFailure> {
        self.run.ready()?;
        if !self.driver.clean() || !Arc::ptr_eq(&prepared.seal, &self.run.seal) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
}
impl<A: Int8SemanticExtractionBatchAdmission> BatchProcessor for NativeInt8SemanticExtractionBatch<'_, '_, '_, A> {
    type Args = ExtractionBatchArgs;
    type Prepared = PreparedInt8SemanticBatchExtraction;
    type Output = GuardedOutput<Int8SemanticExtractionRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.run.ready()?;
        self.run.failed = true;
        let result = (|| {
            self.run.check_available(self.compiler.options.max_model_work)?;
            let plan = self.compiler.prepare(document, &mut self.run.encoding, control)?;
            preflight(&plan, &self.driver, self.run.limits, self.run.encoding_limits)?;
            checkpoint(control)?;
            Ok(PreparedInt8SemanticBatchExtraction { plan, seal: Arc::clone(&self.run.seal) })
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
        execute(prepared.plan, prepared.seal, &mut self.driver, &mut self.admission, &mut self.run, context, control)
    }
}

struct RunState {
    limits: Int8SemanticExtractionBatchLimits,
    reserved: Int8Work,
    masks: u64,
    encoding: SafeBpeWork,
    encoding_limits: Option<SafeBpeWorkLimits>,
    seal: Arc<()>,
    last_sequence: u64,
    failed: bool,
}
impl RunState {
    fn new(limits: Int8SemanticExtractionBatchLimits, encoding_limits: Option<SafeBpeWorkLimits>) -> Self {
        Self { limits, reserved: Int8Work::default(), masks: 0, encoding: SafeBpeWork::default(),
            encoding_limits, seal: Arc::new(()), last_sequence: 0, failed: false }
    }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn totals(&self, work: Int8Work) -> Result<(Int8Work, u64), BatchItemFailure> {
        let total = self.reserved.checked_add(work).ok().filter(|total| fits(*total, self.limits.max_model_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.masks.checked_add(self.limits.masks.max_visits_per_item)
            .filter(|&total| total <= self.limits.masks.max_visits_per_run)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        Ok((total, masks))
    }
    fn check_available(&self, work: Int8Work) -> Result<(), BatchItemFailure> { self.totals(work).map(|_| ()) }
    fn begin(&mut self, work: Int8Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.last_sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let (total, masks) = self.totals(work)?;
        self.reserved = total; self.masks = masks; self.last_sequence = context.request_seq;
        Ok(())
    }
}

// Closed static seams exercise lifecycle failures without exposing a caller's
// fake model, saved semantic result, or replaceable native implementation.
trait Plan {
    type Output;
    type Completion;
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget, total: Option<SafeBpeWorkLimits>)
        -> Int8SemanticExtractionAdmission<'_>;
    fn completion(&self) -> Self::Completion;
    fn verify_completion(expected: Self::Completion, output: &Self::Output, masks: u64,
        before: SafeBpeWork, after: SafeBpeWork) -> Result<(), BatchItemFailure>;
}
struct Completion {
    extraction: Int8Work, ceiling: Int8Work, encoding: UntrustedEncoding, initial_work: SafeBpeWork,
}
impl Plan for PreparedInt8SemanticExtraction {
    type Output = Int8SemanticExtractionRun;
    type Completion = Completion;
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget, total: Option<SafeBpeWorkLimits>)
        -> Int8SemanticExtractionAdmission<'_> {
        Int8SemanticExtractionAdmission { identity: self.execution_identity(), judge_identity: self.judge_identity(),
            model_work: self.max_model_work(), extraction_reserved_work: self.extraction_reserved_work(),
            mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
            kv_reservation_bytes: kv, max_result_bytes: self.max_result_bytes(),
            extraction_budget: self.extraction_budget(), judge_budget: self.judge_budget(),
            semantic_limits: self.semantic_limits(), judge_planning_limits: self.judge_planning_limits(),
            source_bytes: self.source_bytes(), input_encoding: self.input_encoding(), encoding_work_limits: total }
    }
    fn completion(&self) -> Completion {
        Completion { extraction: self.extraction_reserved_work(), ceiling: self.max_model_work(),
            encoding: self.input_encoding().clone(), initial_work: self.input_encoding_work().unwrap_or_default() }
    }
    fn verify_completion(expected: Completion, run: &Int8SemanticExtractionRun, masks: u64,
        before: SafeBpeWork, after: SafeBpeWork) -> Result<(), BatchItemFailure> {
        let invalid = || BatchItemFailure::fatal(BatchCode::InvalidExecution);
        run.verify_input_encoding(&expected.encoding, expected.initial_work).map_err(|_| invalid())?;
        let work = run.input_encoding.as_ref().map(|report| report.work).unwrap_or_default();
        let delta = work.checked_sub(expected.initial_work).ok_or_else(invalid)?;
        let reserved = expected.extraction.checked_add(run.semantic_model_work).map_err(|_| invalid())?;
        let actual = run.extraction.model_work.checked_add(run.semantic_model_work).map_err(|_| invalid())?;
        if run.schema_version != 1 || run.execution != INT8_SEMANTIC_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || !fits(run.extraction.model_work, expected.extraction)
            || run.extraction_reserved_work != expected.extraction || run.reserved_model_work != reserved
            || run.model_work != actual || !fits(actual, reserved) || !fits(reserved, expected.ceiling)
            || run.extraction.result.output.mask_node_visit_charge > masks
            || before.checked_add(delta) != Some(after) || !run.semantic.same_model_correlated
            || run.semantic.calibration != "experimental_uncalibrated_diagnostic_only" {
            return Err(invalid());
        }
        Ok(())
    }
}
trait Driver<P: Plan> {
    fn clean(&self) -> bool;
    fn capacity(&self) -> usize;
    fn execute<C: DecodeStepControl>(&mut self, prepared: P, masks: ExtractionMaskBudget,
        used: &mut SafeBpeWork, total: Option<SafeBpeWorkLimits>, control: &mut C)
        -> Result<P::Output, BatchItemFailure>;
}
struct NativeDriver<'e, 'w, 'v> {
    engine: &'e mut StrictInt8Engine<'w>,
    vocabulary: &'v ExtractionVocabulary,
    prefill: Option<Int8PrefillLimits>,
}
impl Driver<PreparedInt8SemanticExtraction> for NativeDriver<'_, '_, '_> {
    fn clean(&self) -> bool { clean(self.engine) }
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn execute<C: DecodeStepControl>(&mut self, prepared: PreparedInt8SemanticExtraction,
        masks: ExtractionMaskBudget, used: &mut SafeBpeWork, total: Option<SafeBpeWorkLimits>, control: &mut C)
        -> Result<Int8SemanticExtractionRun, BatchItemFailure> {
        prepared.execute_with_invocation_encoding_work(self.engine, self.vocabulary, masks.per_mask,
            masks.max_visits_per_item, self.prefill, used, total, control).map_err(execution_failure)
    }
}
fn preflight<P: Plan, D: Driver<P>>(plan: &P, driver: &D,
    limits: Int8SemanticExtractionBatchLimits, total: Option<SafeBpeWorkLimits>) -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let facts = plan.admission(0, limits.masks, total);
    if facts.identity.task_spec != "extract-v1" || facts.judge_identity.task_spec != "judge-v1"
        || !nonzero(facts.model_work) || !nonzero(facts.extraction_reserved_work)
        || !fits(facts.extraction_reserved_work, facts.model_work) || !fits(facts.model_work, limits.max_model_work)
        || facts.source_bytes == 0 || facts.max_result_bytes == 0
        || facts.extraction_budget.validate().is_err() || facts.judge_budget.validate().is_err()
        || encoding_workspace(facts.input_encoding, total).is_err() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let bytes = (driver.capacity() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .filter(|&bytes| bytes != 0).ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if facts.extraction_reserved_work.forward_positions > driver.capacity() as u64
        || bytes > facts.extraction_budget.max_kv_bytes || bytes > facts.judge_budget.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(bytes)
}
fn execute<P: Plan, D: Driver<P>, A: Int8SemanticExtractionBatchAdmission, C: DecodeStepControl>(
    plan: P, seal: Arc<()>, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<P::Output, A::Guard>, BatchItemFailure> {
    run.ready()?;
    if !Arc::ptr_eq(&seal, &run.seal) {
        run.failed = true; return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let work = plan.admission(0, run.limits.masks, run.encoding_limits).model_work;
    run.begin(work, context)?;
    let result = (|| {
        checkpoint(control)?;
        let kv = preflight(&plan, driver, run.limits, run.encoding_limits)?;
        let facts = plan.admission(kv, run.limits.masks, run.encoding_limits);
        let expected_identity = facts.identity.clone();
        let (identity, guard) = admission.admit(facts)?;
        if identity != expected_identity { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        checkpoint(control)?;
        let expected = plan.completion();
        let before = run.encoding;
        let output = driver.execute(plan, run.limits.masks, &mut run.encoding, run.encoding_limits, control)?;
        P::verify_completion(expected, &output, run.limits.masks.max_visits_per_item, before, run.encoding)?;
        checkpoint(control)?;
        if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(GuardedOutput::new(output, guard))
    })();
    match result {
        Ok(output) => { run.failed = false; Ok(output) }
        Err(mut failure) => { failure.stop = true; Err(failure) }
    }
}
fn clean(engine: &StrictInt8Engine<'_>) -> bool {
    !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0)
}
fn fits(work: Int8Work, ceiling: Int8Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn nonzero(work: Int8Work) -> bool {
    work.forward_positions > 0 && work.projected_logits > 0 && work.attention_pairs > 0
        && work.projections.dot_products > 0 && work.projections.multiply_accumulates > 0
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn preparation_failure(error: Int8SemanticError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use crate::tasks::judge::JudgeError;
    match error {
        Int8SemanticError::Contract
            | Int8SemanticError::Semantic(SemanticError::Judge(JudgeError::Contract(_))) =>
            BatchItemFailure::reject(BatchCode::Planning),
        Int8SemanticError::WorkBudget | Int8SemanticError::Semantic(SemanticError::Limit(_)
            | SemanticError::Judge(JudgeError::Limit(_))) => BatchItemFailure::reject(BatchCode::WorkLimit),
        _ => execution_failure(error),
    }
}
fn execution_failure(error: Int8SemanticError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(match error {
        Int8SemanticError::WorkBudget => BatchCode::WorkLimit,
        Int8SemanticError::Contract | Int8SemanticError::Accounting => BatchCode::InvalidExecution,
        _ => BatchCode::Execution,
    })
}

#[cfg(test)] mod tests;
