//! Resident cited summaries and passage answers with complete-source review.
//!
//! A stream fixes the task, generation options, budgets and experimental review
//! policy. Only original documents or questions/passages vary. Every attempt
//! spends its full declared native and mask ceiling before execution because
//! complete-statement review plans depend on the generated source result.

use std::{ops::Deref, sync::Arc};
use serde::{Deserialize, Serialize};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, extract::{ExtractionMaskBudget, GuardedOutput}},
    execution_identity::{ExecutionIdentity, Sha256Digest},
    grammar::{mask::MaskWorkLimits, runtime::JsonValidationSchema},
    native_engine::{constrained_int4, decode::DecodeStepControl,
        strict_int4::{Int4MemoryRequirement, Int4Work, StrictInt4Engine, STRICT_INT4_PROFILE}},
    tasks::{BuiltInTask, answer::{AnswerOptions, AnswerPassage},
        extract::{ExtractionVocabulary, semantic::grouped_int4::Int4SemanticEncoding},
        ir::{PlanContext, TaskBudget}, judge::{JudgeError, JudgeLimits,
            grouped_int4::{Int4JudgeError, Int4JudgePlanner}},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskRequest, SourceTaskResult,
            grouped_int4::{Int4SourceError, INT4_SOURCE_EXECUTION, semantic::{self as semantic_task,
                Int4SourceSemanticError, Int4SourceSemanticOptions, Int4SourceSemanticRun,
                PreparedInt4SourceSemantic, SourceSemanticInputFacts, SourceSemanticLimits,
                SourceSemanticStatus, INT4_SOURCE_SEMANTIC_EXECUTION, SOURCE_SEMANTIC_VERSION,
                SOURCE_SEMANTIC_CLAIM_VERSION, SOURCE_SEMANTIC_SCOPE, SOURCE_SEMANTIC_CALIBRATION}}},
        summarize::SummaryOptions},
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding},
};

/// The same six nonrenewable invocation ceilings as semantic extraction.
pub use crate::batch::extract::semantic::Int4SemanticExtractionBatchLimits as Int4SemanticSourceBatchLimits;

/// Fixed public task settings; private record arguments cannot replace these.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub enum Int4SemanticSourceBatchTask {
    Summarize { options: SummaryOptions, budget: TaskBudget },
    Answer { options: AnswerOptions, budget: TaskBudget },
}
impl Int4SemanticSourceBatchTask {
    pub fn built_in(self) -> BuiltInTask {
        match self { Self::Summarize { .. } => BuiltInTask::Summarize, Self::Answer { .. } => BuiltInTask::Answer }
    }
    pub fn budget(self) -> TaskBudget {
        match self { Self::Summarize { budget, .. } | Self::Answer { budget, .. } => budget }
    }
    pub fn validate(self) -> Result<(), BatchFault> {
        self.budget().validate().map_err(|_| BatchCode::InvalidLimits)?;
        match self {
            Self::Summarize { options, .. } => options.validate().map_err(|_| BatchCode::InvalidLimits)?,
            Self::Answer { options, .. } => options.validate().map_err(|_| BatchCode::InvalidLimits)?,
        }
        Ok(())
    }
    /// Fixed-policy validation needs no invented source or placeholder answer.
    pub fn validate_semantic(self, options: &Int4SourceSemanticOptions) -> Result<(), BatchFault> {
        self.validate()?;
        options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if self.budget().max_output_bytes > options.limits.max_output_bytes
            || matches!(self, Self::Summarize { options: summary, .. }
                if summary.max_bullets > options.limits.max_statements) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    fn schema_source(self) -> Result<String, BatchFault> {
        match self {
            Self::Summarize { options, .. } => options.schema_source().map_err(|_| BatchCode::InvalidLimits.into()),
            Self::Answer { options, .. } => options.schema_source().map_err(|_| BatchCode::InvalidLimits.into()),
        }
    }
    fn request(self, document: BatchDocument<Int4SemanticSourceBatchArgs>) -> Result<SourceTaskRequest, BatchItemFailure> {
        match (self, document.task_args) {
            (Self::Summarize { options, budget }, None) =>
                Ok(SourceTaskRequest::Summarize { document: document.text, options, budget }),
            (Self::Answer { options, budget }, Some(args)) =>
                Ok(SourceTaskRequest::Answer { question: document.text, passages: args.passages, options, budget }),
            _ => Err(BatchItemFailure::reject(BatchCode::InvalidEnvelope)),
        }
    }
}

/// Answer records put the question in text and the complete original passage
/// set here. Summaries forbid non-null task_args. No source data implements Debug.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4SemanticSourceBatchArgs { pub passages: Vec<AnswerPassage> }

enum SourceCompiler<'p> {
    Borrowed(&'p SourceTaskPlanner),
    Owned(Arc<SourceTaskPlanner>),
}
impl Deref for SourceCompiler<'_> {
    type Target = SourceTaskPlanner;
    fn deref(&self) -> &Self::Target {
        match self { Self::Borrowed(planner) => planner, Self::Owned(planner) => planner }
    }
}

pub struct Int4SemanticSourceBatchPlanner<'p> {
    planner: SourceCompiler<'p>,
    identity: ExecutionIdentity,
    ceiling: TaskBudget,
    planning: SourcePlanningLimits,
    task: Int4SemanticSourceBatchTask,
    judge: Arc<Int4JudgePlanner>,
    options: Int4SourceSemanticOptions,
    limits: Int4SemanticSourceBatchLimits,
    seal: Arc<()>,
}
impl Int4SemanticSourceBatchPlanner<'static> {
    /// Own the genuine pinned source compiler with the shared judge. A host
    /// may retain this immutable factory beyond a worker's processor lifetime
    /// without duplicating tokenizers or constructing a self reference.
    #[allow(clippy::too_many_arguments)]
    pub fn new_owned(planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity, ceiling: TaskBudget,
        planning: SourcePlanningLimits, task: Int4SemanticSourceBatchTask, judge: Arc<Int4JudgePlanner>,
        options: Int4SourceSemanticOptions, limits: Int4SemanticSourceBatchLimits) -> Result<Self, BatchFault> {
        Self::configured(SourceCompiler::Owned(planner), identity, ceiling, planning, task, judge, options, limits)
    }
}
impl<'p> Int4SemanticSourceBatchPlanner<'p> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: &'p SourceTaskPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        planning: SourcePlanningLimits, task: Int4SemanticSourceBatchTask, judge: Arc<Int4JudgePlanner>,
        options: Int4SourceSemanticOptions, limits: Int4SemanticSourceBatchLimits) -> Result<Self, BatchFault> {
        Self::configured(SourceCompiler::Borrowed(planner), identity, ceiling, planning, task, judge, options, limits)
    }
    #[allow(clippy::too_many_arguments)]
    fn configured(planner: SourceCompiler<'p>, identity: ExecutionIdentity, ceiling: TaskBudget,
        planning: SourcePlanningLimits, task: Int4SemanticSourceBatchTask, judge: Arc<Int4JudgePlanner>,
        options: Int4SourceSemanticOptions, limits: Int4SemanticSourceBatchLimits) -> Result<Self, BatchFault> {
        task.validate_semantic(&options)?;
        limits.validate_item(options.max_model_work)?;
        super::grouped_int4::check_configuration_with_document_encoding(&planner, &identity,
            ceiling, planning, None, judge.group_width())?;
        if identity.task_spec != task.built_in().spec().identity()
            || judge.execution_identity().task_spec != "judge-v1"
            || planner.document_encoding() != judge.document_encoding()
            || !budget_fits(task.budget(), ceiling) {
            return Err(BatchCode::InvalidLimits.into());
        }
        semantic_task::same_model(&identity, judge.execution_identity()).map_err(|_| BatchCode::Admission)?;
        let (judge_ceiling, judge_limits) = judge.semantic_configuration();
        if !budget_fits(options.judge_budget, judge_ceiling)
            || judge_limits.max_total_prompt_tokens == 0 || judge_limits.max_total_candidate_tokens == 0
            || judge_limits.max_total_projected_logits == 0 || judge_limits.max_output_bytes == 0
            || judge_limits.per_head.max_candidates == 0 || judge_limits.per_head.max_total_tokens == 0
            || judge_limits.per_head.max_nodes == 0 || judge_limits.per_head.max_depth == 0
            || judge_limits.per_head.max_candidate_id_bytes == 0 || judge_limits.per_head.max_projected_logits == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        // Native declaration semantics include Unicode scalar lengths and
        // verbatim strings without a fabricated source-language authority.
        JsonValidationSchema::compile(&task.schema_source()?, planning.compiler)
            .map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { planner, identity, ceiling, planning, task, judge, options, limits, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn judge_identity(&self) -> &ExecutionIdentity { self.judge.execution_identity() }
    pub fn group_width(&self) -> usize { self.judge.group_width() }
    pub fn task(&self) -> &Int4SemanticSourceBatchTask { &self.task }
    pub fn source_budget(&self) -> TaskBudget { self.task.budget() }
    pub fn task_ceiling(&self) -> TaskBudget { self.ceiling }
    pub fn planning_limits(&self) -> SourcePlanningLimits { self.planning }
    pub fn judge_budget(&self) -> TaskBudget { self.options.judge_budget }
    pub fn judge_planning_limits(&self) -> JudgeLimits { self.judge.semantic_configuration().1 }
    pub fn semantic_limits(&self) -> SourceSemanticLimits { self.options.limits }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.document_encoding() }
    pub fn options(&self) -> &Int4SourceSemanticOptions { &self.options }
    pub fn limits(&self) -> Int4SemanticSourceBatchLimits { self.limits }
    pub fn max_result_bytes(&self) -> u64 { self.options.limits.max_output_bytes }

    fn prepare<C: DecodeStepControl>(&self, document: BatchDocument<Int4SemanticSourceBatchArgs>, control: &mut C)
        -> Result<PreparedInt4SourceSemantic, BatchItemFailure> {
        checkpoint(control)?;
        let request = self.task.request(document)?;
        self.options.input_facts(&request).map_err(preparation_failure)?;
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        // The native planner applies AnswerInputLimits to question + joined
        // passages + canonical manifest BEFORE tokenization. Checking only the
        // transport text field would leave the larger QA payload unbounded.
        let source = self.planner.plan_int4_with_control(&request, &context, self.planning,
            self.group_width(), control).map_err(source_preparation_failure)?;
        PreparedInt4SourceSemantic::new_shared(source, request, Arc::clone(&self.judge),
            self.options.clone(), control).map_err(preparation_failure)
    }
}

pub struct Int4SemanticSourceAdmission<'a> {
    pub identity: &'a ExecutionIdentity,
    pub judge_identity: &'a ExecutionIdentity,
    pub group_width: usize,
    pub model_work: Int4Work,
    pub source_reserved_work: Int4Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub source_budget: TaskBudget,
    pub judge_budget: TaskBudget,
    pub semantic_limits: SourceSemanticLimits,
    pub judge_planning_limits: JudgeLimits,
    pub input_facts: SourceSemanticInputFacts,
    pub input_encoding: &'a UntrustedEncoding,
}
pub trait Int4SemanticSourceBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4SemanticSourceAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

/// An exact compiler owner seal accompanies every private, consumed task plan.
pub struct PreparedInt4SemanticBatchSource { plan: PreparedInt4SourceSemantic, seal: Arc<()> }
impl PreparedInt4SemanticBatchSource {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn model_work(&self) -> Int4Work { self.plan.max_model_work() }
    pub fn max_result_bytes(&self) -> u64 { self.plan.max_result_bytes() }
}

pub struct NativeInt4SemanticSourceBatch<'p, 'e, 'w, 'v, A: Int4SemanticSourceBatchAdmission> {
    compiler: Arc<Int4SemanticSourceBatchPlanner<'p>>,
    driver: NativeDriver<'e, 'w, 'v>,
    admission: A,
    run: RunState,
}
impl<'p, 'e, 'w, 'v, A: Int4SemanticSourceBatchAdmission> NativeInt4SemanticSourceBatch<'p, 'e, 'w, 'v, A> {
    pub fn new(compiler: Int4SemanticSourceBatchPlanner<'p>, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::new_shared(Arc::new(compiler), engine, vocabulary, admission)
    }
    /// Retain the actual compiler assets until the embedding invocation has
    /// physically joined its workers; each processor still owns fresh native
    /// state, work counters and sealed prepared records.
    pub fn new_shared(compiler: Arc<Int4SemanticSourceBatchPlanner<'p>>, engine: &'e mut StrictInt4Engine<'w>,
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
    /// Authenticate the exact factory before a durable attempt is charged.
    /// Equal visible limits never authorize a foreign source/review plan.
    pub(crate) fn verify_prepared(&self, prepared: &PreparedInt4SemanticBatchSource)
        -> Result<(), BatchItemFailure> {
        self.run.ready()?;
        if !self.driver.clean() || !Arc::ptr_eq(&prepared.seal, &self.compiler.seal) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
}
impl<A: Int4SemanticSourceBatchAdmission> BatchProcessor for NativeInt4SemanticSourceBatch<'_, '_, '_, '_, A> {
    type Args = Int4SemanticSourceBatchArgs;
    type Prepared = PreparedInt4SemanticBatchSource;
    type Output = GuardedOutput<Int4SourceSemanticRun, A::Guard>;
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
            Ok(PreparedInt4SemanticBatchSource { plan, seal: Arc::clone(&self.compiler.seal) })
        })();
        if !self.driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        if !result.as_ref().is_err_and(|failure: &BatchItemFailure| failure.stop) { self.run.failed = false; }
        result
    }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork {
        let work = plan.model_work();
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
    limits: Int4SemanticSourceBatchLimits,
    reserved: Int4Work,
    masks: u64,
    last_sequence: u64,
    failed: bool,
}
impl RunState {
    fn new(limits: Int4SemanticSourceBatchLimits) -> Self {
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

// Static private seams permit accounting and lifecycle fixtures without an
// externally replaceable logits provider, prepared plan or completion proof.
trait Plan {
    type Output;
    type Completion;
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget) -> Int4SemanticSourceAdmission<'_>;
    fn completion(&self) -> Self::Completion;
    fn verify_completion(expected: Self::Completion, output: &Self::Output, masks: u64)
        -> Result<(), BatchItemFailure>;
}
struct Completion {
    width: usize, task: String, source: Int4Work, ceiling: Int4Work, policy: Sha256Digest,
    encoding: UntrustedEncoding, initial_work: Option<SafeBpeWork>,
}
impl Plan for PreparedInt4SourceSemantic {
    type Output = Int4SourceSemanticRun;
    type Completion = Completion;
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget) -> Int4SemanticSourceAdmission<'_> {
        Int4SemanticSourceAdmission { identity: self.execution_identity(), judge_identity: self.judge_identity(),
            group_width: self.group_width(), model_work: self.max_model_work(), source_reserved_work: self.source_reserved_work(),
            mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask, kv_reservation_bytes: kv,
            max_result_bytes: self.max_result_bytes(), source_budget: self.source_budget(), judge_budget: self.judge_budget(),
            semantic_limits: self.semantic_limits(), judge_planning_limits: self.judge_planning_limits(),
            input_facts: self.input_facts(), input_encoding: self.input_encoding() }
    }
    fn completion(&self) -> Completion {
        Completion { width: self.group_width(), task: self.execution_identity().task_spec.clone(),
            source: self.source_reserved_work(), ceiling: self.max_model_work(), policy: self.policy_digest(),
            encoding: self.input_encoding().clone(), initial_work: self.input_encoding_work() }
    }
    fn verify_completion(expected: Completion, run: &Int4SourceSemanticRun, masks: u64)
        -> Result<(), BatchItemFailure> {
        let invalid = || BatchItemFailure::fatal(BatchCode::InvalidExecution);
        let encoding_work = run.input_encoding.as_ref().map(|report| report.work).unwrap_or_default();
        let encoding = Int4SemanticEncoding::from_work(&expected.encoding, encoding_work).map_err(|_| invalid())?;
        let reserved = expected.source.checked_add(run.semantic_model_work).map_err(|_| invalid())?;
        let actual = run.source_task.model_work.checked_add(run.semantic_model_work).map_err(|_| invalid())?;
        let task_matches = matches!((&run.source_task.result, expected.task.as_str()),
            (SourceTaskResult::Summarize(_), "summarize-v1") | (SourceTaskResult::Answer(_), "answer-v1"));
        let semantic = &run.semantic;
        if run.schema_version != 1 || run.execution != INT4_SOURCE_SEMANTIC_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != expected.width
            || run.source_task.schema_version != 1 || run.source_task.execution != INT4_SOURCE_EXECUTION
            || run.source_task.group_width != expected.width || !task_matches
            || run.source_reserved_work != expected.source || run.reserved_model_work != reserved
            || run.model_work != actual || !fits(run.source_task.model_work, expected.source)
            || !fits(actual, reserved) || !fits(reserved, expected.ceiling)
            || super::observed(&run.source_task.result).mask_visits > masks
            || run.input_encoding != encoding || expected.initial_work.is_some() != encoding.is_some()
            || encoding_work.checked_sub(expected.initial_work.unwrap_or_default()).is_none()
            || !semantic.same_model_correlated || semantic.policy_digest != expected.policy
            || semantic.schema_version != 1 || semantic.algorithm != SOURCE_SEMANTIC_VERSION
            || semantic.claim_rendering != SOURCE_SEMANTIC_CLAIM_VERSION
            || semantic.evidence_scope != SOURCE_SEMANTIC_SCOPE || semantic.calibration != SOURCE_SEMANTIC_CALIBRATION
            || semantic.work.projected_logits != run.semantic_model_work.projected_logits
            || semantic.decided_statements.checked_add(semantic.abstained_statements) != Some(semantic.attempted_statements)
            || semantic.attempted_statements.checked_add(semantic.not_checked_statements) != Some(semantic.statements.len())
            || semantic.statements.iter().filter(|statement| statement.status == SourceSemanticStatus::NotChecked).count()
                != semantic.not_checked_statements {
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
impl Driver<PreparedInt4SourceSemantic> for NativeDriver<'_, '_, '_> {
    fn clean(&self) -> bool { clean(self.engine) }
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn group_width(&self) -> usize { self.engine.model_identity().group_width() }
    fn execute<C: DecodeStepControl>(&mut self, prepared: PreparedInt4SourceSemantic,
        masks: ExtractionMaskBudget, control: &mut C) -> Result<Int4SourceSemanticRun, BatchItemFailure> {
        prepared.execute_with_control(self.engine, self.vocabulary, masks.per_mask, masks.max_visits_per_item, control)
            .map_err(execution_failure)
    }
}
fn preflight<P: Plan, D: Driver<P>>(plan: &P, driver: &D, limits: Int4SemanticSourceBatchLimits)
    -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let facts = plan.admission(0, limits.masks);
    if facts.group_width != driver.group_width() || !matches!(facts.identity.task_spec.as_str(), "summarize-v1" | "answer-v1")
        || facts.judge_identity.task_spec != "judge-v1" || !nonzero(facts.model_work)
        || !nonzero(facts.source_reserved_work) || !fits(facts.source_reserved_work, facts.model_work)
        || !fits(facts.model_work, limits.max_model_work) || facts.input_facts.source_bytes == 0 || facts.max_result_bytes == 0
        || facts.source_budget.validate().is_err() || facts.judge_budget.validate().is_err() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let memory = Int4MemoryRequirement::for_context(driver.capacity(), driver.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if facts.source_reserved_work.forward_positions > driver.capacity() as u64
        || memory.kv_bytes > facts.source_budget.max_kv_bytes || memory.kv_bytes > facts.judge_budget.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(memory.kv_bytes)
}
#[allow(clippy::too_many_arguments)]
fn execute<P: Plan, D: Driver<P>, A: Int4SemanticSourceBatchAdmission, C: DecodeStepControl>(
    plan: P, seal: Arc<()>, expected_seal: &Arc<()>, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<P::Output, A::Guard>, BatchItemFailure> {
    run.ready()?;
    if !Arc::ptr_eq(&seal, expected_seal) {
        run.failed = true; return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    run.begin(plan.admission(0, run.limits.masks).model_work, context)?;
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
fn clean(engine: &StrictInt4Engine<'_>) -> bool { !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0) }
fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn nonzero(work: Int4Work) -> bool {
    work.forward_positions > 0 && work.projected_logits > 0 && work.attention_pairs > 0
        && work.projections.dot_products > 0 && work.projections.multiply_accumulates > 0
}
fn budget_fits(budget: TaskBudget, ceiling: TaskBudget) -> bool {
    budget.max_input_tokens <= ceiling.max_input_tokens && budget.max_output_tokens <= ceiling.max_output_tokens
        && budget.max_output_bytes <= ceiling.max_output_bytes && budget.max_grammar_states <= ceiling.max_grammar_states
        && budget.max_kv_bytes <= ceiling.max_kv_bytes
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn source_preparation_failure(error: Int4SourceError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int4SourceError::Planning(error) => super::preparation_failure(error),
        Int4SourceError::Extraction(crate::tasks::extract::grouped_int4::Int4ExtractError::Extraction(error)) =>
            super::preparation_failure(error.into()),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}
fn preparation_failure(error: Int4SourceSemanticError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int4SourceSemanticError::Source(error) => source_preparation_failure(error),
        Int4SourceSemanticError::Contract | Int4SourceSemanticError::Judge(Int4JudgeError::Task(JudgeError::Contract(_))) =>
            BatchItemFailure::reject(BatchCode::Planning),
        Int4SourceSemanticError::WorkBudget | Int4SourceSemanticError::ClaimBudget
            | Int4SourceSemanticError::Judge(Int4JudgeError::WorkBudget | Int4JudgeError::Task(JudgeError::Limit(_))) =>
            BatchItemFailure::reject(BatchCode::WorkLimit),
        Int4SourceSemanticError::Allocation
            | Int4SourceSemanticError::Judge(Int4JudgeError::Task(JudgeError::AllocationRefused)) =>
            BatchItemFailure::fatal(BatchCode::Allocation),
        _ => execution_failure(error),
    }
}
fn execution_failure(error: Int4SourceSemanticError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(match error {
        Int4SourceSemanticError::WorkBudget | Int4SourceSemanticError::ClaimBudget => BatchCode::WorkLimit,
        Int4SourceSemanticError::Contract | Int4SourceSemanticError::InputMismatch | Int4SourceSemanticError::Accounting =>
            BatchCode::InvalidExecution,
        Int4SourceSemanticError::Allocation => BatchCode::Allocation,
        _ => BatchCode::Execution,
    })
}

#[cfg(test)] mod tests;
