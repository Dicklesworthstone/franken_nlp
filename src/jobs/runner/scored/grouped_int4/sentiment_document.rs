//! Fixed-policy complete sentiment documents on the existing owned-job runner.
//! Native maps, chunk cohorts, identity checks and result guards are delegated
//! to the same resident processor used by the ordered sentiment corpus.
use super::*;
use serde::{Deserialize, Deserializer, Serialize, de::{self, MapAccess, Visitor}};
use crate::{
    batch::sentiment::grouped_int4::{Int4SentimentDocumentBatchAdmission,
        Int4SentimentDocumentBatchPlanner, Int4SentimentDocumentRun,
        NativeInt4SentimentDocumentBatch, PreparedInt4SentimentDocument},
    jobs::manifest::bounded_json,
    native_engine::{int4::PackedInt4Kernel, strict_int4::{Int4ModelWeights, Int4Work}},
    tasks::{mapreduce::{CHUNK_PROFILE, ChunkLimits}, sentiment::{SentimentAggregationPolicy,
        SentimentLimits, SENTIMENT_AGGREGATION_PROFILE, batch::SentimentBatchArgs,
        grouped_int4::{Int4SentimentPlanner, long::{Int4SentimentMapLimits,
            INT4_SENTIMENT_MAP_EXECUTION, INT4_SENTIMENT_MAP_COHORT_EXECUTION,
            memory::Int4SentimentMapMemoryLimits}}}},
    tokenizer::untrusted::UntrustedEncoding,
};

const RECIPE_BYTES: usize = 1024 * 1024 - 1024;

/// Per-item policy cannot override the authenticated job's dimensions, budget,
/// aggregation or chunk schedule. Omitted/null task_args also use fixed policy.
#[derive(Default)]
pub struct Int4SentimentDocumentArgs {}
impl<'de> Deserialize<'de> for Int4SentimentDocumentArgs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EmptyArguments;
        impl<'de> Visitor<'de> for EmptyArguments {
            type Value = Int4SentimentDocumentArgs;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an empty sentiment document argument object")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                if map.next_key::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("sentiment document arguments must be empty"));
                }
                Ok(Int4SentimentDocumentArgs {})
            }
        }
        deserializer.deserialize_map(EmptyArguments)
    }
}

/// Only the real model-bound planner constructs this private projection. The
/// existing runner stores a keyed commitment, never the private recipe itself.
#[derive(Serialize)]
pub struct Int4SentimentDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    aggregation_profile: &'static str,
    reservation_policy: &'static str,
    sentiment: Int4ScoredJobRecipe,
    chunks: ChunkLimits,
    per_document_model_work: Int4Work,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
    max_stored_result_bytes: u64,
    aggregation: SentimentAggregationPolicy,
    cohort_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<UntrustedEncoding>,
}
impl Int4SentimentDocumentJobRecipe {
    fn new(sentiment: Int4ScoredJobRecipe, mapping: Int4SentimentMapLimits) -> Result<Self, BatchFault> {
        // New map fields must be deliberately added to authenticated resume.
        let Int4SentimentMapLimits { chunks, max_model_work, max_retained_result_bytes,
            max_result_bytes, aggregation } = mapping;
        let recipe = Self { version: 1, dependency_scope: "item-local-complete-document",
            input_policy: "fixed-sentiment-policy-empty-item-args-v1",
            execution: INT4_SENTIMENT_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            aggregation_profile: SENTIMENT_AGGREGATION_PROFILE,
            reservation_policy: "exact-all-chunk-five-axis-work-before-durable-attempt-v1",
            sentiment, chunks, per_document_model_work: max_model_work,
            max_retained_result_bytes, max_result_bytes, max_stored_result_bytes: max_result_bytes,
            aggregation, cohort_chunks: None, input_encoding: None };
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

pub struct Int4SentimentDocumentJobPlanner {
    compiler: Int4SentimentDocumentBatchPlanner,
    identity: ExecutionIdentity,
    recipe: Int4SentimentDocumentJobRecipe,
    limits: JobLimits,
}
impl Int4SentimentDocumentJobPlanner {
    pub fn new(planner: Int4SentimentPlanner, settings: SentimentBatchArgs,
        mapping: Int4SentimentMapLimits, limits: JobLimits) -> Result<Self, BatchFault> {
        check_limits(mapping, limits)?;
        let identity = planner.execution_identity().clone();
        let (ceiling, planning) = planner.durable_configuration();
        let native = Int4ScoredBatchLimits { max_item_work: mapping.max_model_work,
            max_model_work: limits.max_work.int4_model_work() };
        let (sentiment, settings) = Int4ScoredJobRecipe::sentiment_retaining_defaults(
            planner.group_width(), ceiling, planning, settings, native)?;
        let max_axes = settings.axes.len();
        let mut recipe = Int4SentimentDocumentJobRecipe::new(sentiment, mapping)?;
        if planner.document_encoding() != &UntrustedEncoding::ByteFallback {
            recipe.input_encoding = Some(planner.document_encoding().clone());
            bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        }
        let compiler = Int4SentimentDocumentBatchPlanner::new(planner, Some(settings), mapping,
            native.max_model_work)?.with_max_axes(max_axes)?;
        Ok(Self { compiler, identity, recipe, limits })
    }
    pub fn with_chunk_cohorts(mut self, chunks: usize) -> Result<Self, BatchFault> {
        self.compiler = self.compiler.with_chunk_cohorts(chunks)?;
        if self.compiler.max_result_bytes() > self.limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        self.recipe.execution = INT4_SENTIMENT_MAP_COHORT_EXECUTION;
        self.recipe.cohort_chunks = Some(chunks);
        self.recipe.max_stored_result_bytes = self.compiler.max_result_bytes();
        bounded_json(&self.recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn task_budget(&self) -> TaskBudget { self.compiler.task_budget() }
    pub fn planning_limits(&self) -> SentimentLimits { self.compiler.planning_limits() }
    pub fn document_limits(&self) -> Int4SentimentMapLimits { self.compiler.mapping_limits() }
    pub fn job_limits(&self) -> JobLimits { self.limits }
    pub fn group_width(&self) -> usize { self.compiler.group_width() }
    pub fn max_axes(&self) -> usize { self.compiler.max_axes() }
    pub fn cohort_chunks(&self) -> Option<usize> { self.compiler.cohort_chunks() }
    pub fn max_result_bytes(&self) -> u64 { self.compiler.max_result_bytes() }
    pub fn job_recipe(&self) -> &Int4SentimentDocumentJobRecipe { &self.recipe }
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> { self.compiler.retained_defaults_bytes() }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits || self.max_result_bytes() > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// All execution remains in the existing native processor. Its guarded output
/// stays owned by JobRunner through complete encoding, spool sync and journal
/// acknowledgement; neither chunks nor cohort windows are durable results.
pub struct Int4SentimentDocumentJobProcessor<'e, 'w, A: Int4SentimentDocumentBatchAdmission> {
    native: NativeInt4SentimentDocumentBatch<'e, 'w, A>,
    identity: ExecutionIdentity,
    recipe: Int4SentimentDocumentJobRecipe,
    limits: JobLimits,
}
impl<'e, 'w, A: Int4SentimentDocumentBatchAdmission> Int4SentimentDocumentJobProcessor<'e, 'w, A> {
    pub fn new(planner: Int4SentimentDocumentJobPlanner, engine: &'e mut StrictInt4Engine<'w>, admission: A)
        -> Result<Self, BatchFault> {
        let Int4SentimentDocumentJobPlanner { compiler, identity, recipe, limits } = planner;
        let native = NativeInt4SentimentDocumentBatch::new(compiler, engine, admission)?;
        Ok(Self { native, identity, recipe, limits })
    }
    pub fn new_cohorts(planner: Int4SentimentDocumentJobPlanner, weights: &'w Int4ModelWeights,
        kernel: PackedInt4Kernel, memory: Int4SentimentMapMemoryLimits, admission: A)
        -> Result<Self, BatchFault> {
        let Int4SentimentDocumentJobPlanner { compiler, identity, recipe, limits } = planner;
        let native = NativeInt4SentimentDocumentBatch::new_cohorts(compiler, weights, kernel, memory, admission)?;
        Ok(Self { native, identity, recipe, limits })
    }
}
impl<A: Int4SentimentDocumentBatchAdmission> BatchProcessor for Int4SentimentDocumentJobProcessor<'_, '_, A> {
    type Args = Int4SentimentDocumentArgs;
    type Prepared = PreparedInt4SentimentDocument;
    type Output = GuardedOutput<Int4SentimentDocumentRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(fixed_document(document))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_control(fixed_document(document), control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int4SentimentDocumentBatchAdmission> DurableBatchProcessor for Int4SentimentDocumentJobProcessor<'_, '_, A> {
    type Recipe = Int4SentimentDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits { return Err(BatchCode::InvalidLimits.into()); }
        Ok(())
    }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork::from_int4(prepared.planned_work(), 0))
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.max_stored_result_bytes }
}
fn fixed_document(document: BatchDocument<Int4SentimentDocumentArgs>) -> BatchDocument<SentimentBatchArgs> {
    BatchDocument { id: document.id, text: document.text, task_args: None }
}
fn check_limits(mapping: Int4SentimentMapLimits, limits: JobLimits) -> Result<(), BatchFault> {
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if limits.max_work.mask_node_visits != 0
        || !JobWork::from_int4(mapping.max_model_work, 0).fits(limits.max_work)
        || mapping.chunks.max_input_bytes > limits.max_input_bytes_per_item
        || mapping.max_result_bytes > limits.max_result_bytes as u64 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

#[cfg(test)] mod tests;
