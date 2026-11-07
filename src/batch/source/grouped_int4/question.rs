//! Complete original documents and private per-record questions on one native
//! population. Discovery and optional evidence-only synthesis are one attempt;
//! no chunk, failed item, synthesis pass or flush replenishes its authority.
use std::{fmt, sync::Arc};
use serde::{Deserialize, Serialize};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, generation::GuardedOutput},
    execution_identity::ExecutionIdentity,
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int4, decode::DecodeStepControl, int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryRequirement, Int4ModelIdentity, Int4ModelWeights, Int4Work, StrictInt4Engine}},
    tasks::{answer::{AnswerOptions, ANSWER_TASK_VERSION}, extract::ExtractionVocabulary,
        ir::{PlanContext, TaskBudget}, mapreduce::MapReduceError,
        source_planning::{SourcePlanningError, SourcePlanningLimits, SourceTaskPlanner,
            grouped_int4::{Int4SourceError, long::{Int4SourceMapError, Int4SourceMapLimits,
                cohort::{Int4SourceMapCohortBudget, Int4SourceMapCohortMemoryLimits, Int4SourceMapCohortRequirement,
                    hierarchy::Int4HierarchyCohortSchedule},
                question::{Int4QuestionPreflight, Int4QuestionRun, PreparedInt4Question, SourceQuestion,
                    synthesis::{Int4QuestionSynthesisPreflight, Int4QuestionSynthesisRun,
                        PreparedInt4QuestionSynthesis, QuestionSynthesisLimits, SourceQuestionSynthesis,
                        hierarchy::{Int4QuestionHierarchyPreflight, Int4QuestionHierarchyRun,
                            PreparedInt4QuestionHierarchy, QuestionHierarchyLimits}}}}}}},
    validation::grounded_fields::GroundingBudget,
};
mod planning;
mod backend;
pub mod memory;
use planning::{Expected, NativePlan};
use backend::Backend;

/// Frozen, reconstructed and admitted identities for every possible native row.
pub const QUESTION_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;
// Four simultaneously retained/reconstructed/admitted projections fit the
// public metadata reservation; counting performs no canonical allocation.
const MAX_QUESTION_IDENTITY_BYTES: usize = 8192;

/// Strict object-only arguments. The record may supply exactly one question;
/// its options, evidence policy, source, budget and schedule cannot be replaced.
pub struct QuestionDocumentArgs { pub question: String }
impl<'de> Deserialize<'de> for QuestionDocumentArgs {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Arguments;
        impl<'de> serde::de::Visitor<'de> for Arguments {
            type Value = QuestionDocumentArgs;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("an object containing one question") }
            fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut question = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key != "question" { return Err(serde::de::Error::unknown_field(&key, &["question"])); }
                    if question.is_some() { return Err(serde::de::Error::duplicate_field("question")); }
                    question = Some(map.next_value::<String>()?);
                }
                Ok(QuestionDocumentArgs { question: question.ok_or_else(|| serde::de::Error::missing_field("question"))? })
            }
        }
        deserializer.deserialize_map(Arguments)
    }
}

/// Fixed policy for every source/question pair, with independently finite run
/// allowances. Each admitted attempt reserves BOTH complete verification pools.
#[derive(Clone, Copy)]
pub struct Int4QuestionDocumentBatchConfig {
    pub options: AnswerOptions,
    pub budget: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub mapping: Int4SourceMapLimits,
    pub verification: GroundingBudget,
    pub synthesis: Option<QuestionSynthesisLimits>,
    /// Explicit lossy quotation selection. Requires the fixed synthesis policy;
    /// all permitted passes reserve work before the first discovery forward.
    pub hierarchy: Option<QuestionHierarchyLimits>,
    pub max_question_bytes: usize,
    pub max_model_work: Int4Work,
    pub max_mask_visits: u64,
    pub max_map_verification: GroundingBudget,
    pub max_verification: GroundingBudget,
}
impl Int4QuestionDocumentBatchConfig {
    pub(crate) fn validate(&self) -> Result<(), BatchFault> {
        self.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        self.planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
        self.options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let schema = self.options.schema_source().map_err(|_| BatchCode::InvalidLimits)?;
        if schema.len() > self.planning.compiler.max_schema_bytes { return Err(BatchCode::InvalidLimits.into()); }
        self.mapping.validate(self.budget, self.planning).map_err(|_| BatchCode::InvalidLimits)?;
        if self.mapping.chunks.context_tokens != self.planning.max_context_tokens
            || !(1..=64 * 1024 * 1024).contains(&self.mapping.chunks.max_input_bytes)
            || !(1..=64 * 1024 * 1024).contains(&self.max_question_bytes)
            || self.max_question_bytes > self.planning.max_input_bytes
            || !positive(self.mapping.max_model_work) || !fits(self.mapping.max_model_work, self.max_model_work)
            || self.mapping.max_mask_visits > self.max_mask_visits
            || !valid_verification(self.verification)
            || !verification_fits(self.verification, self.max_verification)
            || !verification_fits(self.mapping.verification, self.max_map_verification) {
            return Err(BatchCode::InvalidLimits.into());
        }
        if let Some(limits) = self.synthesis { limits.validate(self.planning).map_err(|_| BatchCode::InvalidLimits)?; }
        if let Some(limits) = self.hierarchy {
            if self.synthesis.is_none() { return Err(BatchCode::InvalidLimits.into()); }
            limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        }
        Ok(())
    }
    pub fn continuation_passes(&self) -> usize {
        self.hierarchy.map_or(usize::from(self.synthesis.is_some()), |limits| limits.max_passes)
    }
    /// Hierarchy receipts retain every native continuation transcript. Plain
    /// discovery and flat synthesis retain only semantic text and citations.
    pub(crate) fn retained_output_tokens(&self) -> Result<u64, BatchFault> {
        (self.hierarchy.map_or(0, |limits| limits.max_passes) as u64)
            .checked_mul(u64::from(self.budget.max_output_tokens)).ok_or_else(|| BatchCode::InvalidLimits.into())
    }
}

/// Immutable actual-model binding; no question or source is retained as a
/// default. Changing the native schedule invalidates prior prepared ownership.
pub struct Int4QuestionDocumentBatchPlanner {
    planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity, group_width: usize,
    config: Int4QuestionDocumentBatchConfig, chunk_cohorts: bool,
    hierarchy_groups: Option<usize>, owner: Arc<()>,
}
impl Int4QuestionDocumentBatchPlanner {
    pub fn for_model(model: &Int4ModelIdentity, planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity,
        config: Int4QuestionDocumentBatchConfig) -> Result<Self, BatchFault> {
        constrained_int4::check_model(&identity, model).map_err(|_| BatchCode::Admission)?;
        Self::configured(planner, identity, model.group_width(), config)
    }
    #[cfg(test)]
    pub(crate) fn for_planning_fixture(planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity,
        group_width: usize, config: Int4QuestionDocumentBatchConfig) -> Result<Self, BatchFault> {
        Self::configured(planner, identity, group_width, config)
    }
    // Model-free planning fixtures stay private and cannot manufacture a
    // native model identity or replace a production execution driver.
    fn configured(planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity, group_width: usize,
        config: Int4QuestionDocumentBatchConfig) -> Result<Self, BatchFault> {
        config.validate()?;
        check_identity_size(&identity)?;
        super::check_configuration(&planner, &identity, config.budget, config.planning, None, group_width)?;
        let row = Int4MemoryRequirement::for_context(config.planning.max_context_tokens, group_width)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if identity.task_spec != ANSWER_TASK_VERSION || row.kv_bytes != config.budget.max_kv_bytes {
            return Err(BatchCode::Admission.into());
        }
        check_synthesis_reserve(&config, group_width)?;
        Ok(Self { planner, identity, group_width, config, chunk_cohorts: false,
            hierarchy_groups: None, owner: Arc::new(()) })
    }
    pub fn config(&self) -> &Int4QuestionDocumentBatchConfig { &self.config }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn uses_chunk_cohorts(&self) -> bool { self.chunk_cohorts }
    pub fn uses_native_cohorts(&self) -> bool { self.chunk_cohorts || self.hierarchy_groups.is_some() }
    pub fn hierarchy_cohort_schedule(&self) -> Option<Int4HierarchyCohortSchedule> {
        self.hierarchy_groups.map(|max_groups| Int4HierarchyCohortSchedule {
            max_groups, discovery_cohorts: self.chunk_cohorts })
    }
    pub fn max_result_bytes(&self) -> u64 { self.config.mapping.reduction.max_result_bytes as u64 }
    pub fn max_output_tokens(&self) -> u64 {
        // Immutable construction validated max_passes <= 256 and a u32 token
        // allowance; their product therefore fits u64.
        self.config.hierarchy.map_or(0, |limits| limits.max_passes as u64 * u64::from(self.config.budget.max_output_tokens))
    }
    pub fn with_chunk_cohorts(mut self) -> Result<Self, BatchFault> {
        let changed = !self.chunk_cohorts;
        self.chunk_cohorts = true;
        self.cohort_requirements()?;
        if changed { self.owner = Arc::new(()); }
        Ok(self)
    }
    pub fn with_hierarchy_cohorts(mut self, max_groups: usize) -> Result<Self, BatchFault> {
        let limits = self.config.hierarchy.ok_or(BatchCode::InvalidLimits)?;
        Int4HierarchyCohortSchedule { max_groups, discovery_cohorts: self.chunk_cohorts }
            .validate(limits.max_passes).map_err(|_| BatchCode::InvalidLimits)?;
        let changed = self.hierarchy_groups != Some(max_groups);
        self.hierarchy_groups = Some(max_groups);
        self.cohort_requirements()?;
        if changed { self.owner = Arc::new(()); }
        Ok(self)
    }
    pub fn cohort_requirements(&self) -> Result<Int4SourceMapCohortRequirement, BatchFault> {
        let c = self.config; let chunks = c.mapping.reduction.map_batch_chunks;
        let discovery = if self.chunk_cohorts || self.hierarchy_groups.is_none() {
            if chunks > c.mapping.chunks.max_chunks { return Err(BatchCode::InvalidLimits.into()); }
            Some(Int4SourceMapCohortMemoryLimits { max_chunks: chunks,
            context_tokens: c.planning.max_context_tokens,
            max_output_tokens_per_chunk: u64::from(c.budget.max_output_tokens),
            max_result_bytes_per_chunk: c.budget.max_output_bytes }.requirements(self.group_width)
            .map_err(|_| BatchCode::InvalidLimits)?)
        } else { None };
        if let Some(schedule) = self.hierarchy_cohort_schedule() {
            return schedule.requirements(discovery, c.planning.max_context_tokens, self.group_width, c.budget)
                .map_err(|_| BatchCode::Admission.into());
        }
        let quote = discovery.ok_or(BatchCode::InvalidLimits)?;
        if c.synthesis.is_some() {
            quote.with_synthesis_context(c.planning.max_context_tokens, self.group_width, c.budget.max_kv_bytes)
                .map_err(|_| BatchCode::Admission.into())
        } else { Ok(quote) }
    }
    pub(crate) fn owns(&self, prepared: &PreparedInt4QuestionDocument) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
    /// Seal a complete source/question pair without allocating native state.
    /// Executing processors still preflight the actual backend and reconstruct
    /// every exact native identity against this factory's private owner.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, source: String,
        args: QuestionDocumentArgs, control: &mut C) -> Result<PreparedInt4QuestionDocument, BatchItemFailure> {
        let plan = self.plan(&source, &args.question, control, false)?;
        let (expected, identities) = self.snapshot(&plan, control)?;
        drop(plan);
        poll(control)?;
        Ok(PreparedInt4QuestionDocument { source, question: args.question,
            expected, identities, owner: Arc::clone(&self.owner) })
    }
}

pub struct PreparedInt4QuestionDocument {
    source: String, question: String, expected: Expected, identities: Vec<ExecutionIdentity>, owner: Arc<()>,
}
impl PreparedInt4QuestionDocument {
    pub fn chunk_count(&self) -> usize { self.expected.discovery().chunk_count() }
    pub fn native_chunks(&self) -> usize { self.expected.discovery().native_chunks() }
    pub fn planned_work(&self) -> Int4Work { self.expected.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.expected.masks }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
}
#[derive(Serialize)]
#[serde(untagged)]
pub enum Int4QuestionDocumentRun {
    Independent(Int4QuestionRun), Synthesis(Int4QuestionSynthesisRun), Hierarchy(Int4QuestionHierarchyRun),
}
pub struct Int4QuestionDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity], pub model_work: Int4Work, pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits, pub kv_reservation_bytes: u64, pub max_result_bytes: u64,
    pub max_output_tokens: u64,
    pub chunk_count: usize, pub native_chunks: usize,
}
pub trait Int4QuestionDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4QuestionDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}
pub struct NativeInt4QuestionDocumentBatch<'p, 'e, 'w, 'v, A: Int4QuestionDocumentBatchAdmission> {
    planner: &'p Int4QuestionDocumentBatchPlanner, backend: Backend<'e, 'w>,
    vocabulary: &'v ExtractionVocabulary, admission: A, ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: Int4QuestionDocumentBatchAdmission> NativeInt4QuestionDocumentBatch<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int4QuestionDocumentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let backend = Backend::Serial(engine); backend.capacity(planner).map_err(|e| e.fault)?;
        Ok(Self { planner, backend, vocabulary, admission, ledger: Ledger::default() })
    }
    pub fn new_cohorts(planner: &'p Int4QuestionDocumentBatchPlanner, weights: &'w Int4ModelWeights,
        vocabulary: &'v ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        let backend = Backend::Cohorts { weights, budget, kernel }; backend.capacity(planner).map_err(|e| e.fault)?;
        Ok(Self { planner, backend, vocabulary, admission, ledger: Ledger::default() })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.ledger.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.ledger.masks }
    pub fn reserved_map_verification(&self) -> GroundingBudget { self.ledger.map_verification }
    pub fn reserved_verification(&self) -> GroundingBudget { self.ledger.verification }
    pub fn is_poisoned(&self) -> bool { self.ledger.failed || !self.backend.clean() }
}
impl<A: Int4QuestionDocumentBatchAdmission> BatchProcessor for NativeInt4QuestionDocumentBatch<'_, '_, '_, '_, A> {
    type Args = QuestionDocumentArgs;
    type Prepared = PreparedInt4QuestionDocument;
    type Output = GuardedOutput<Int4QuestionDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?; self.ledger.failed = true;
        let result = (|| {
            poll(control)?;
            self.backend.capacity(self.planner)?;
            let args = document.task_args.ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
            let plan = self.planner.plan(&document.text, &args.question, control, false)?;
            let (expected, identities) = self.planner.snapshot(&plan, control)?;
            self.backend.preflight(self.planner, &plan, &identities, control)?;
            drop(plan); poll(control)?;
            Ok(PreparedInt4QuestionDocument { source: document.text, question: args.question,
                expected, identities, owner: Arc::clone(&self.planner.owner) })
        })();
        if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) && self.backend.clean() { self.ledger.failed = false; }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work(); BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        if let Err(error) = self.planner.owns(&prepared) { self.ledger.failed = true; return Err(error); }
        self.ledger.begin(&self.planner.config, prepared.expected, context)?;
        let result = (|| {
            poll(control)?;
            let kv = self.backend.capacity(self.planner)?;
            let plan = self.planner.plan(&prepared.source, &prepared.question, control, true)?;
            self.planner.compare(&plan, &prepared, control)?;
            self.backend.preflight(self.planner, &plan, &prepared.identities, control)?;
            let c = &self.planner.config;
            let (admitted, guard) = self.admission.admit(Int4QuestionDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.planned_work(),
                mask_node_visits: prepared.reserved_mask_visits(), mask_limits: c.mapping.mask_limits,
                kv_reservation_bytes: kv, max_result_bytes: self.planner.max_result_bytes(),
                max_output_tokens: self.planner.max_output_tokens(),
                chunk_count: prepared.chunk_count(), native_chunks: prepared.native_chunks(),
            })?;
            poll(control)?;
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            let output = self.backend.execute(self.planner, plan, &admitted, self.vocabulary, control)?;
            prepared.expected.verify(&output)?;
            self.backend.capacity(self.planner)?; poll(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        match result { Ok(output) => { self.ledger.failed = false; Ok(output) }, Err(mut failure) => { failure.stop = true; Err(failure) } }
    }
}

fn positive(w: Int4Work) -> bool {
    w.forward_positions > 0 && w.projected_logits > 0 && w.attention_pairs > 0
        && w.projections.dot_products > 0 && w.projections.multiply_accumulates > 0
}
fn check_identity_size(identity: &ExecutionIdentity) -> Result<(), BatchFault> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_sub(bytes.len()).ok_or_else(|| std::io::Error::other("question identity bound"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter(MAX_QUESTION_IDENTITY_BYTES), identity).map_err(|_| BatchCode::InvalidLimits.into())
}
fn fits(a: Int4Work, b: Int4Work) -> bool {
    a.forward_positions <= b.forward_positions && a.projected_logits <= b.projected_logits
        && a.attention_pairs <= b.attention_pairs && a.projections.fits(b.projections)
}
fn valid_verification(v: GroundingBudget) -> bool {
    (1..=1_000_000).contains(&v.max_fields) && (1..=1_000_000).contains(&v.max_matches) && v.max_scan_steps > 0
}
fn verification_fits(a: GroundingBudget, b: GroundingBudget) -> bool {
    a.max_fields <= b.max_fields && a.max_matches <= b.max_matches && a.max_scan_steps <= b.max_scan_steps
}
fn add_verification(a: GroundingBudget, b: GroundingBudget) -> Option<GroundingBudget> {
    Some(GroundingBudget { max_fields: a.max_fields.checked_add(b.max_fields)?,
        max_matches: a.max_matches.checked_add(b.max_matches)?, max_scan_steps: a.max_scan_steps.checked_add(b.max_scan_steps)? })
}
struct Ledger { work: Int4Work, masks: u64, map_verification: GroundingBudget, verification: GroundingBudget,
    sequence: u64, epoch: u64, line: u64, failed: bool }
impl Default for Ledger {
    fn default() -> Self {
        let zero = GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 };
        Self { work: Int4Work::default(), masks: 0, map_verification: zero, verification: zero,
            sequence: 0, epoch: 0, line: 0, failed: false }
    }
}
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, c: &Int4QuestionDocumentBatchConfig, expected: Expected, context: BatchRequestContext)
        -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.epoch < self.epoch || context.input_line <= self.line {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let exhausted = || BatchItemFailure::fatal(BatchCode::WorkLimit);
        let work = self.work.checked_add(expected.work).ok().filter(|&w| fits(w, c.max_model_work)).ok_or_else(exhausted)?;
        let masks = self.masks.checked_add(expected.masks).filter(|&n| n <= c.max_mask_visits).ok_or_else(exhausted)?;
        let map_verification = add_verification(self.map_verification, c.mapping.verification)
            .filter(|&v| verification_fits(v, c.max_map_verification)).ok_or_else(exhausted)?;
        let verification = add_verification(self.verification, c.verification)
            .filter(|&v| verification_fits(v, c.max_verification)).ok_or_else(exhausted)?;
        self.work = work; self.masks = masks; self.map_verification = map_verification; self.verification = verification;
        self.sequence = context.request_seq; self.epoch = context.epoch; self.line = context.input_line;
        Ok(())
    }
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()) }
}
fn map_failure(error: Int4SourceMapError, fatal: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    let (code, stop) = match error {
        Int4SourceMapError::InvalidLimits => (BatchCode::InvalidLimits, true),
        Int4SourceMapError::EmptySource => (BatchCode::DocumentLimit, fatal),
        Int4SourceMapError::WorkLimit => (BatchCode::WorkLimit, fatal),
        Int4SourceMapError::Admission => (BatchCode::Admission, true),
        Int4SourceMapError::Allocation | Int4SourceMapError::Chunk(MapReduceError::AllocationRefused)
            => (BatchCode::Allocation, true),
        Int4SourceMapError::Chunk(MapReduceError::InvalidLimits) => (BatchCode::InvalidLimits, true),
        Int4SourceMapError::Chunk(_) => (BatchCode::Planning, fatal),
        Int4SourceMapError::Source(Int4SourceError::Planning(SourcePlanningError::InputBudget)) => (BatchCode::DocumentLimit, fatal),
        Int4SourceMapError::Source(Int4SourceError::Planning(SourcePlanningError::ContextBudget)) => (BatchCode::WorkLimit, fatal),
        Int4SourceMapError::Source(_) | Int4SourceMapError::Reduction(_) => (BatchCode::InvalidExecution, true),
    };
    if stop { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) }
}
fn check_synthesis_reserve(c: &Int4QuestionDocumentBatchConfig, width: usize) -> Result<(), BatchFault> {
    if c.synthesis.is_none() { return Ok(()); }
    let output = c.budget.max_output_tokens as usize;
    let prompt = c.planning.max_context_tokens.checked_sub(output)
        .map(|n| n.min(c.budget.max_input_tokens as usize)).filter(|&n| n > 0).ok_or(BatchCode::InvalidLimits)?;
    let pass = constrained_int4::planned_work(width, prompt, output).map_err(|_| BatchCode::InvalidLimits)?;
    let mut work = Int4Work::default();
    for _ in 0..c.continuation_passes() {
        work = work.checked_add(pass).map_err(|_| BatchCode::InvalidLimits)?;
    }
    let masks = c.mapping.mask_visits_per_chunk.checked_mul(c.continuation_passes() as u64)
        .ok_or(BatchCode::InvalidLimits)?;
    if !fits(work, c.mapping.max_model_work) || c.mapping.max_mask_visits <= masks {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

#[cfg(test)] mod tests;
