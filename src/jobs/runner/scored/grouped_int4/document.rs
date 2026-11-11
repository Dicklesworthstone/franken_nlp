//! Resumable complete-document classification on the existing native map.
//! Every source is one durable item. Preparation freezes every chunk identity
//! and all five native work counters before the journal records the attempt.
//! Neither chunks nor cohort windows are partial results or retry checkpoints.
use super::*;
use std::sync::Arc;
use serde::{Deserialize, Deserializer, Serialize, de::{self, MapAccess, Visitor}};
use crate::{
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding},
    batch::classify::ClassificationBatchArgs,
    jobs::manifest::bounded_json,
    native_engine::{constrained_int4, int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryRequirement, Int4ModelIdentity, Int4ModelWeights, Int4RunBudget,
            Int4Work, scoring::Int4ScoringBudget}},
    tasks::{classify::{ClassificationAggregationPolicy, ClassificationLimits, ClassificationMode,
            ClassificationRequest, CLASSIFICATION_AGGREGATION_PROFILE,
            grouped_int4::{Int4ClassificationPlanner,
                long::{Int4ClassificationMapError, Int4ClassificationMapLimits,
                    Int4ClassificationMapPreflight, Int4ClassificationMapRun,
                    Int4ClassificationMapCohortBudget, Int4ClassificationMapCohortRun,
                    PreparedInt4ClassificationMap, INT4_CLASSIFICATION_MAP_EXECUTION,
                    INT4_CLASSIFICATION_MAP_COHORT_EXECUTION,
                    memory::{Int4ClassificationMapMemoryLimits, Int4ClassificationMapMemoryRequirement,
                        Int4ClassificationMapCohortMemoryRequirement}}}},
        mapreduce::{ChunkLimits, CHUNK_PROFILE}},
};

/// Per-record task arguments are an empty JSON object. All classification
/// settings belong to the fixed authenticated job recipe. BatchDocument's
/// optional field also accepts an omitted value or null, never a sequence.
#[derive(Default)]
pub struct Int4ClassificationDocumentArgs {}
impl<'de> Deserialize<'de> for Int4ClassificationDocumentArgs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EmptyArguments;
        impl<'de> Visitor<'de> for EmptyArguments {
            type Value = Int4ClassificationDocumentArgs;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an empty classification document argument object")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                if map.next_key::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("classification document arguments must be empty"));
                }
                Ok(Int4ClassificationDocumentArgs {})
            }
        }
        deserializer.deserialize_map(EmptyArguments)
    }
}
mod execution;
pub use execution::{Int4ClassificationDocumentAdmission, Int4ClassificationDocumentBatchAdmission,
    Int4ClassificationDocumentJobProcessor, Int4ClassificationDocumentRun};

const RECIPE_BYTES: usize = 1024 * 1024 - 1024;

/// Private settings are serialized only into the owned job's keyed recipe
/// commitment. There is no public recipe constructor, deserializer or raw
/// content digest that could substitute for the actual model-bound compiler.
#[derive(Serialize)]
pub struct Int4ClassificationDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    aggregation_profile: &'static str,
    reservation_policy: &'static str,
    classification: Int4ScoredJobRecipe,
    chunks: ChunkLimits,
    per_document_model_work: Int4Work,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
    max_stored_result_bytes: u64,
    aggregation: ClassificationAggregationPolicy,
    cohort_chunks: Option<usize>,
    /// One operation includes initial preparation and execution reconstruction.
    /// Omission preserves every existing byte-mode recipe commitment.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<UntrustedEncoding>,
}
impl Int4ClassificationDocumentJobRecipe {
    fn new(classification: Int4ScoredJobRecipe, mapping: Int4ClassificationMapLimits)
        -> Result<Self, BatchFault> {
        // Exhaustive destructuring makes any new map setting a deliberate
        // addition to the authenticated resume contract.
        let Int4ClassificationMapLimits { chunks, max_model_work, max_retained_result_bytes,
            max_result_bytes, aggregation } = mapping;
        let recipe = Self { version: 1, dependency_scope: "item-local-complete-document",
            input_policy: "fixed-classification-policy-empty-item-args-v1",
            execution: INT4_CLASSIFICATION_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            aggregation_profile: CLASSIFICATION_AGGREGATION_PROFILE,
            reservation_policy: "exact-all-chunk-five-axis-work-before-durable-attempt-v1",
            classification, chunks, per_document_model_work: max_model_work,
            max_retained_result_bytes, max_result_bytes, max_stored_result_bytes: max_result_bytes,
            aggregation, cohort_chunks: None, input_encoding: None };
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

#[derive(Clone, Copy)]
struct CohortSchedule {
    chunks: usize,
    required: Int4ClassificationMapCohortMemoryRequirement,
}

/// Owns one immutable model-bound planner and fixed classification policy.
/// The same labels, mode and per-chunk policy apply to every original source.
/// No per-record argument can weaken the job's frozen policy or work limits.
pub struct Int4ClassificationDocumentJobPlanner {
    compiler: Int4ClassificationPlanner,
    settings: ClassificationBatchArgs,
    mapping: Int4ClassificationMapLimits,
    limits: JobLimits,
    recipe: Int4ClassificationDocumentJobRecipe,
    cohorts: Option<CohortSchedule>,
    owner: Arc<()>,
}
impl Int4ClassificationDocumentJobPlanner {
    pub fn new(compiler: Int4ClassificationPlanner, settings: ClassificationBatchArgs,
        mapping: Int4ClassificationMapLimits, limits: JobLimits) -> Result<Self, BatchFault> {
        let (ceiling, planning) = compiler.durable_configuration();
        validate_limits(compiler.group_width(), ceiling, settings.budget, planning, mapping, limits)?;
        // The existing private recipe checks every fixed label and policy and
        // every actual compiler limit BEFORE cloning the private defaults.
        // Retain the caller's owner so host accounting sees its spare capacity.
        let native = Int4ScoredBatchLimits { max_item_work: mapping.max_model_work,
            max_model_work: limits.max_work.int4_model_work() };
        let (classification, settings) = Int4ScoredJobRecipe::classification_retaining_defaults(
            compiler.group_width(), ceiling, planning, settings, native)?;
        compiler.document_encoding().validate().map_err(|_| BatchCode::InvalidLimits)?;
        let mut recipe = Int4ClassificationDocumentJobRecipe::new(classification, mapping)?;
        if compiler.document_encoding() != &UntrustedEncoding::ByteFallback {
            recipe.input_encoding = Some(compiler.document_encoding().clone());
            bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        }
        Ok(Self { compiler, settings, mapping, limits, recipe, cohorts: None, owner: Arc::new(()) })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.compiler.execution_identity() }
    pub fn task_ceiling(&self) -> TaskBudget { self.compiler.durable_configuration().0 }
    pub fn document_budget(&self) -> TaskBudget { self.settings.budget }
    pub fn planning_limits(&self) -> ClassificationLimits { self.compiler.durable_configuration().1 }
    pub fn document_limits(&self) -> Int4ClassificationMapLimits { self.mapping }
    pub fn job_limits(&self) -> JobLimits { self.limits }
    pub fn request(&self) -> &ClassificationBatchArgs { &self.settings }
    pub fn group_width(&self) -> usize { self.compiler.group_width() }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn head_count(&self) -> usize {
        if self.settings.mode == ClassificationMode::Exclusive { 1 } else { self.settings.labels.len() }
    }
    pub fn max_result_bytes(&self) -> u64 { self.recipe.max_stored_result_bytes }
    pub fn job_recipe(&self) -> &Int4ClassificationDocumentJobRecipe { &self.recipe }
    pub fn cohort_chunks(&self) -> Option<usize> { self.cohorts.map(|schedule| schedule.chunks) }
    pub fn uses_chunk_cohorts(&self) -> bool { self.cohorts.is_some() }

    /// Opt into complete chunk-by-head windows. Oversized fixed geometry is
    /// refused; it is never clamped, resized or silently executed serially.
    /// Host preparation/allocator reservations may change across invocations;
    /// the native schedule and result format remain part of resume authority.
    pub fn with_chunk_cohorts(mut self, chunks: usize, memory: Int4ClassificationMapMemoryLimits)
        -> Result<Self, BatchFault> {
        let schedule = cohort_schedule(self.group_width(), self.document_budget(), self.planning_limits(),
            self.mapping, self.head_count(), chunks, memory, self.limits)?;
        self.recipe.cohort_chunks = Some(chunks);
        self.recipe.execution = INT4_CLASSIFICATION_MAP_COHORT_EXECUTION;
        self.recipe.max_stored_result_bytes = schedule.required.memory.max_output_bytes;
        bounded_json(&self.recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        self.cohorts = Some(schedule);
        // A prepared serial attempt cannot be reused under a different native
        // schedule, even if the fixed task policy itself is unchanged.
        self.owner = Arc::new(());
        Ok(self)
    }
    pub fn cohort_requirements(&self) -> Result<Int4ClassificationMapCohortMemoryRequirement, BatchFault> {
        self.cohorts.map(|schedule| schedule.required).ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    /// Complete simultaneous ownership for the fixed policy and schedule.
    /// The host additionally owns the durable population, journal and spool
    /// encoding buffers, as required by JobRunner's separate contract.
    pub fn memory_requirements(&self, memory: Int4ClassificationMapMemoryLimits)
        -> Result<Int4ClassificationMapMemoryRequirement, BatchFault> {
        if memory.context_tokens != self.mapping.chunks.context_tokens {
            return Err(BatchCode::InvalidLimits.into());
        }
        let required = if let Some(chunks) = self.cohort_chunks() {
            memory.cohort_requirements(self.group_width(), self.document_budget(), self.planning_limits(),
                self.mapping, self.head_count(), chunks).map(|required| required.memory)
        } else {
            memory.requirements(self.group_width(), self.document_budget(), self.planning_limits(), self.mapping)
        };
        required.and_then(|required| required.with_input_encoding(self.document_encoding()))
            .map_err(|error| map_failure(error).fault)
    }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits || self.max_result_bytes() > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    fn plan_with_encoding_work<'a, C: DecodeStepControl>(&self, request: &'a ClassificationRequest,
        control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedInt4ClassificationMap<'a>, BatchItemFailure> {
        poll(control)?;
        if request.document.is_empty() || request.document.len() > self.mapping.chunks.max_input_bytes
            || request.document.len() > self.limits.max_input_bytes_per_item {
            return Err(BatchItemFailure::fatal(BatchCode::DocumentLimit));
        }
        self.compiler.plan_document_with_encoding_work(request, self.mapping, control, work).map_err(map_failure)
    }
    /// Performs no native inference or filesystem work. The exact source is
    /// moved into the preparation; every original chunk is compiled first.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<Int4ClassificationDocumentArgs>,
        control: &mut C) -> Result<PreparedInt4ClassificationDocument, BatchItemFailure> {
        poll(control)?;
        if document.text.is_empty() || document.text.len() > self.mapping.chunks.max_input_bytes
            || document.text.len() > self.limits.max_input_bytes_per_item {
            return Err(BatchItemFailure::fatal(BatchCode::DocumentLimit));
        }
        let request = ClassificationRequest { document: document.text, labels: self.settings.labels.clone(),
            mode: self.settings.mode, policy: self.settings.policy, budget: self.settings.budget };
        let mut encoding_work = SafeBpeWork::default();
        let plan = self.plan_with_encoding_work(&request, control, &mut encoding_work)?;
        let expected = plan.preflight_metadata();
        let identities = plan.identities().map_err(map_failure)?;
        let work = JobWork::from_int4(plan.planned_work(), 0);
        if identities.len() != expected.chunk_count() || expected.chunk_count() == 0
            || plan.group_width() != self.group_width() || plan.head_count() != self.head_count()
            || plan.task_budget() != self.document_budget()
            || plan.required_context() > self.mapping.chunks.context_tokens
            || plan.max_result_bytes() != self.mapping.max_result_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        if !work.fits(self.limits.max_work) { return Err(BatchItemFailure::fatal(BatchCode::WorkLimit)); }
        drop(plan);
        poll(control)?;
        Ok(PreparedInt4ClassificationDocument { request, identities, expected, work, encoding_work,
            owner: Arc::clone(&self.owner) })
    }
    fn owns(&self, prepared: &PreparedInt4ClassificationDocument) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
    fn compare<C: DecodeStepControl>(&self, plan: &PreparedInt4ClassificationMap<'_>,
        prepared: &PreparedInt4ClassificationDocument, control: &mut C) -> Result<(), BatchItemFailure> {
        self.owns(prepared)?;
        poll(control)?;
        if plan.preflight_metadata() != prepared.expected
            || JobWork::from_int4(plan.planned_work(), 0) != prepared.work
            || plan.group_width() != self.group_width() || plan.head_count() != self.head_count()
            || plan.task_budget() != self.document_budget()
            || plan.max_result_bytes() != self.mapping.max_result_bytes
            || plan.identities().map_err(map_failure)? != prepared.identities {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        poll(control)
    }
}

/// Owned exact request plus private native planning facts. No caller can
/// deserialize, mutate or substitute a partially completed document result.
pub struct PreparedInt4ClassificationDocument {
    request: ClassificationRequest,
    identities: Vec<ExecutionIdentity>,
    expected: Int4ClassificationMapPreflight,
    work: JobWork,
    encoding_work: SafeBpeWork,
    owner: Arc<()>,
}
impl PreparedInt4ClassificationDocument {
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn planned_work(&self) -> JobWork { self.work }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn preflight_metadata(&self) -> Int4ClassificationMapPreflight { self.expected }
    /// Preparation and reconstruction consume this one document allowance;
    /// native work and attempts separately retain their durable lifetime scope.
    pub fn encoding_work(&self) -> SafeBpeWork { self.encoding_work }
}

fn validate_limits(width: usize, ceiling: TaskBudget, task: TaskBudget, planning: ClassificationLimits,
    mapping: Int4ClassificationMapLimits, job: JobLimits) -> Result<(), BatchFault> {
    job.validate().map_err(|_| BatchCode::InvalidLimits)?;
    ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
    super::super::check_budget(task, ceiling)?;
    planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
    mapping.validate(task).map_err(|error| map_failure(error).fault)?;
    let native = Int4MemoryRequirement::for_context(mapping.chunks.context_tokens, width)
        .map_err(|_| BatchCode::InvalidLimits)?;
    if mapping.chunks.max_input_bytes == 0 || mapping.chunks.max_input_bytes > planning.max_input_bytes
        || mapping.chunks.context_tokens > planning.max_context_tokens || native.kv_bytes > task.max_kv_bytes
        || !JobWork::from_int4(mapping.max_model_work, 0).fits(job.max_work)
        || job.max_work.mask_node_visits != 0 || mapping.max_result_bytes > job.max_result_bytes as u64 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn cohort_schedule(width: usize, task: TaskBudget, planning: ClassificationLimits,
    mapping: Int4ClassificationMapLimits, heads: usize, chunks: usize,
    memory: Int4ClassificationMapMemoryLimits, job: JobLimits) -> Result<CohortSchedule, BatchFault> {
    if memory.context_tokens != mapping.chunks.context_tokens {
        return Err(BatchCode::InvalidLimits.into());
    }
    let required = memory.cohort_requirements(width, task, planning, mapping, heads, chunks)
        .map_err(|error| map_failure(error).fault)?;
    if required.memory.max_output_bytes > job.max_result_bytes as u64 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(CohortSchedule { chunks, required })
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4ClassificationMapError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(match error {
        Int4ClassificationMapError::Limits => BatchCode::InvalidLimits,
        Int4ClassificationMapError::EmptySource => BatchCode::DocumentLimit,
        Int4ClassificationMapError::Work => BatchCode::WorkLimit,
        Int4ClassificationMapError::Admission => BatchCode::Admission,
        Int4ClassificationMapError::Allocation => BatchCode::Allocation,
        Int4ClassificationMapError::Output => BatchCode::OutputLineLimit,
        Int4ClassificationMapError::Serialization => BatchCode::Serialization,
        Int4ClassificationMapError::Chunk(_) => BatchCode::Planning,
        Int4ClassificationMapError::Classification(_) | Int4ClassificationMapError::Aggregation(_)
            => BatchCode::InvalidExecution,
    })
}

#[cfg(test)] mod tests;
