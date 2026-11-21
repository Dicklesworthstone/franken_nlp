//! Authenticated retained populations of reconciled, reviewed root objects.
//! One attempt covers every extraction chunk and complete-source field review.
//! The resident native processor remains the only execution path.

use super::{CompilerRecipe, JudgeRecipe, check_judge_limits, encoding_workspace};
use std::sync::Arc;
use serde::{Serialize, Serializer};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork,
        extract::{ExtractionBatchArgs, GuardedOutput,
            grouped_int4::{Int4ExtractionBatchPlanner, long::{Int4ExtractionMapLimits,
                Int4ExtractionMappingLimits, object::{Int4ObjectExtractionMapLimits,
                    Int4ObjectExtractionPlanner, ROOT_OBJECT_MAP_PROMPT}}},
            semantic::document::{Int4SemanticObjectDocumentBatchAdmission,
                Int4SemanticObjectDocumentBatchLimits, Int4SemanticObjectDocumentBatchPlanner,
                NativeInt4SemanticObjectDocumentBatch, PreparedInt4SemanticObjectBatchDocument}}},
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, mask::MaskWorkLimits,
        runtime::{SourceRuntimeLimits, JSON_RUNTIME_VERSION, SOURCE_JSON_RUNTIME_VERSION}},
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl,
        strict_int4::{Int4ModelIdentity, Int4Work, StrictInt4Engine}},
    tasks::{extract::{ExtractionVocabulary, object_reconcile::{RootObjectLimits,
            ROOT_OBJECT_DERIVATION, ROOT_OBJECT_RECONCILIATION},
            semantic::{CLAIM_RENDER_VERSION, SEMANTIC_VERIFICATION_VERSION,
                grouped_int4::{Int4SemanticOptions, document::{Int4SemanticObjectDocumentRun,
                    INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION, SEMANTIC_OBJECT_DOCUMENT_SCOPE}}}},
        judge::{JudgeLimits, FaithfulnessAggregationPolicy, FAITHFULNESS_VERSION,
            JUDGE_PROMPT_VERSION, FAITHFULNESS_AGGREGATION_PROFILE,
            grouped_int4::{Int4JudgePlanner, INT4_JUDGE_EXECUTION,
                faithfulness_map::{Int4FaithfulnessMapLimits, INT4_FAITHFULNESS_MAP_EXECUTION}}},
        mapreduce::{ChunkLimits, ExecutionLimits, CHUNK_PROFILE}},
    tokenizer::untrusted::UntrustedEncoding,
    validation::grounded_fields::GroundingBudget,
};

pub(crate) const SEMANTIC_OBJECT_JOB_RECIPE_BYTES: usize = 1024 * 1024 - 1024;
const INPUT_POLICY: &str = "fixed-original-object-schema-and-complete-source-review-v1";
const RESERVATION: &str = "full-combined-declaration-per-durable-attempt-v1";
const ENCODING_SCOPE: &str = "one-original-document-all-chunks-claims-and-review-heads-v1";

/// Private fixed settings. Exact schema bytes, claim rules and refusal ceilings
/// enter a keyed commitment, never a public document fingerprint.
#[derive(Clone)]
pub struct Int4SemanticObjectDocumentJobConfig {
    pub extraction: ExtractionBatchArgs,
    pub compiler_limits: CompileLimits,
    pub source_limits: SourceRuntimeLimits,
    pub options: Int4SemanticOptions,
    pub judge_planning: JudgeLimits,
    pub encoding: UntrustedEncoding,
    pub extraction_limits: Int4ObjectExtractionMapLimits,
    pub review_limits: Int4FaithfulnessMapLimits,
    pub work: Int4SemanticObjectDocumentBatchLimits,
}

/// Grounding is not a journal work axis. Fund its full declaration for EVERY
/// permitted attempt, including retries and interrupted attempts, so reopening
/// cannot renew the authenticated lifetime grounding allowance.
pub(crate) fn check_job_limits(config: &Int4SemanticObjectDocumentJobConfig, limits: JobLimits)
    -> Result<(), BatchFault> {
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    config.options.validate().map_err(|_| BatchCode::InvalidLimits)?;
    config.work.validate_item(config.options.max_model_work,
        config.extraction_limits.mapping.mapping.max_mask_visits,
        config.extraction_limits.mapping.verification)?;
    let lifetime = JobWork::from_int4(config.work.max_model_work, config.work.max_mask_node_visits);
    if limits.max_work != lifetime || config.options.limits.max_output_bytes > limits.max_result_bytes as u64 {
        return Err(BatchCode::InvalidLimits.into());
    }
    let per = config.extraction_limits.mapping.verification;
    let cap = config.work.max_verification;
    let product = |n: u64| n.checked_mul(limits.max_attempts).ok_or(BatchCode::InvalidLimits);
    if product(per.max_fields as u64)? > cap.max_fields as u64
        || product(per.max_matches as u64)? > cap.max_matches as u64
        || product(per.max_scan_steps)? > cap.max_scan_steps {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

#[derive(Serialize)]
struct MappingRecipe {
    chunks: ChunkLimits,
    reduction: ExecutionLimits,
    max_model_work: Int4Work,
    max_trie_node_visits: usize,
    checkpoint_interval_nodes: usize,
    mask_visits_per_chunk: u64,
    max_mask_visits: u64,
    verification: GroundingBudget,
    reconciliation: RootObjectLimits,
}
impl TryFrom<Int4ObjectExtractionMapLimits> for MappingRecipe {
    type Error = BatchFault;
    fn try_from(limits: Int4ObjectExtractionMapLimits) -> Result<Self, Self::Error> {
        let Int4ObjectExtractionMapLimits { mapping, reconciliation } = limits;
        let Int4ExtractionMapLimits { mapping, verification, root_array_union } = mapping;
        if root_array_union.is_some() { return Err(BatchCode::InvalidLimits.into()); }
        let Int4ExtractionMappingLimits { chunks, reduction, max_model_work, mask_limits,
            mask_visits_per_chunk, max_mask_visits } = mapping;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = mask_limits;
        Ok(Self { chunks, reduction, max_model_work, max_trie_node_visits,
            checkpoint_interval_nodes, mask_visits_per_chunk, max_mask_visits,
            verification, reconciliation })
    }
}
#[derive(Serialize)]
struct ReviewRecipe {
    chunks: ChunkLimits,
    max_model_work: Int4Work,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
    aggregation: FaithfulnessAggregationPolicy,
}
impl From<Int4FaithfulnessMapLimits> for ReviewRecipe {
    fn from(limits: Int4FaithfulnessMapLimits) -> Self {
        let Int4FaithfulnessMapLimits { chunks, max_model_work, max_retained_result_bytes,
            max_result_bytes, aggregation } = limits;
        Self { chunks, max_model_work, max_retained_result_bytes, max_result_bytes, aggregation }
    }
}
#[derive(Serialize)]
struct WorkRecipe {
    max_model_work: Int4Work,
    max_mask_node_visits: u64,
    max_verification: GroundingBudget,
}
#[derive(Serialize)]
struct Settings<'a> {
    extraction: &'a ExtractionBatchArgs,
    compiler: CompilerRecipe,
    source: SourceRuntimeLimits,
    options: &'a Int4SemanticOptions,
    judge_planning: JudgeRecipe,
    encoding: &'a UntrustedEncoding,
    extraction_map: MappingRecipe,
    review_map: ReviewRecipe,
    work: WorkRecipe,
}
impl<'a> TryFrom<&'a Int4SemanticObjectDocumentJobConfig> for Settings<'a> {
    type Error = BatchFault;
    fn try_from(config: &'a Int4SemanticObjectDocumentJobConfig) -> Result<Self, Self::Error> {
        // Exhaustive projections make new declaration fields a source error
        // instead of silently omitting them from resume authentication.
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items,
            max_output_bytes, max_states, max_transitions, max_mask_bytes } = config.compiler_limits;
        let Int4SemanticObjectDocumentBatchLimits { max_model_work, max_mask_node_visits,
            max_verification } = config.work;
        Ok(Self { extraction: &config.extraction,
            compiler: CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items,
                max_output_bytes, max_states, max_transitions, max_mask_bytes },
            source: config.source_limits, options: &config.options,
            judge_planning: config.judge_planning.into(), encoding: &config.encoding,
            extraction_map: config.extraction_limits.try_into()?, review_map: config.review_limits.into(),
            work: WorkRecipe { max_model_work, max_mask_node_visits, max_verification } })
    }
}

pub struct Int4SemanticObjectDocumentJobRecipe {
    config: Int4SemanticObjectDocumentJobConfig,
    extraction_identity: ExecutionIdentity,
    judge_identity: ExecutionIdentity,
    group_width: usize,
}
#[derive(Serialize)]
struct RecipeView<'a> {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    reservation_policy: &'static str,
    execution: &'static str,
    semantic_scope: &'static str,
    extraction_prompt: &'static str,
    schema_derivation: &'static str,
    reconciliation: &'static str,
    chunk_profile: &'static str,
    json_runtime: &'static str,
    source_runtime: &'static str,
    semantic_verification: &'static str,
    claim_rendering: &'static str,
    judge_execution: &'static str,
    judge_prompt: &'static str,
    faithfulness: &'static str,
    faithfulness_map: &'static str,
    faithfulness_aggregation: &'static str,
    encoding_scope: &'static str,
    group_width: usize,
    extraction_identity: &'a ExecutionIdentity,
    judge_identity: &'a ExecutionIdentity,
    settings: Settings<'a>,
}
impl Serialize for Int4SemanticObjectDocumentJobRecipe {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let settings = Settings::try_from(&self.config).map_err(serde::ser::Error::custom)?;
        RecipeView { version: 1, dependency_scope: "item-local-complete-document",
            input_policy: INPUT_POLICY, reservation_policy: RESERVATION,
            execution: INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION,
            semantic_scope: SEMANTIC_OBJECT_DOCUMENT_SCOPE, extraction_prompt: ROOT_OBJECT_MAP_PROMPT,
            schema_derivation: ROOT_OBJECT_DERIVATION, reconciliation: ROOT_OBJECT_RECONCILIATION,
            chunk_profile: CHUNK_PROFILE, json_runtime: JSON_RUNTIME_VERSION,
            source_runtime: SOURCE_JSON_RUNTIME_VERSION, semantic_verification: SEMANTIC_VERIFICATION_VERSION,
            claim_rendering: CLAIM_RENDER_VERSION, judge_execution: INT4_JUDGE_EXECUTION,
            judge_prompt: JUDGE_PROMPT_VERSION, faithfulness: FAITHFULNESS_VERSION,
            faithfulness_map: INT4_FAITHFULNESS_MAP_EXECUTION,
            faithfulness_aggregation: FAITHFULNESS_AGGREGATION_PROFILE, encoding_scope: ENCODING_SCOPE,
            group_width: self.group_width, extraction_identity: &self.extraction_identity,
            judge_identity: &self.judge_identity, settings }.serialize(serializer)
    }
}

/// Genuine immutable model/compiler factory, not a deserializable task factory.
pub struct Int4SemanticObjectDocumentJobPlanner {
    compiler: Arc<Int4SemanticObjectDocumentBatchPlanner>,
    recipe: Int4SemanticObjectDocumentJobRecipe,
}
impl Int4SemanticObjectDocumentJobPlanner {
    pub fn for_model(model: &Int4ModelIdentity, config: Int4SemanticObjectDocumentJobConfig)
        -> Result<Self, BatchFault> {
        config.extraction.budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        config.options.validate().map_err(|_| BatchCode::InvalidLimits)?;
        encoding_workspace(&config.encoding)?;
        check_judge_limits(config.judge_planning)?;
        bounded_json(&Settings::try_from(&config)?, SEMANTIC_OBJECT_JOB_RECIPE_BYTES)
            .map_err(|_| BatchCode::InvalidLimits)?;
        let extraction = Int4ExtractionBatchPlanner::for_model(model, config.extraction.budget,
            config.compiler_limits, config.source_limits, None)?;
        let extraction = Int4ObjectExtractionPlanner::new(extraction)
            .map_err(|_| BatchCode::InvalidLimits)?.with_document_encoding(config.encoding.clone())?;
        let judge = Arc::new(Int4JudgePlanner::for_model(model, config.options.specification.judge_budget,
            config.judge_planning).and_then(|planner| planner.with_document_encoding(config.encoding.clone()))
            .map_err(|_| BatchCode::Admission)?);
        let compiler = Arc::new(Int4SemanticObjectDocumentBatchPlanner::new(extraction,
            config.extraction.clone(), judge, config.options.clone(), config.extraction_limits,
            config.review_limits, config.work)?);
        let recipe = Int4SemanticObjectDocumentJobRecipe { group_width: model.group_width(), config,
            extraction_identity: compiler.base_execution_identity().clone(),
            judge_identity: compiler.judge_identity().clone() };
        bounded_json(&recipe, SEMANTIC_OBJECT_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { compiler, recipe })
    }
    pub fn config(&self) -> &Int4SemanticObjectDocumentJobConfig { &self.recipe.config }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.recipe.extraction_identity }
    pub fn judge_identity(&self) -> &ExecutionIdentity { &self.recipe.judge_identity }
    pub fn max_result_bytes(&self) -> u64 { self.recipe.config.options.limits.max_output_bytes }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        check_job_limits(self.config(), limits)
    }
    pub(crate) fn retain_compiler(&self) -> Arc<Int4SemanticObjectDocumentBatchPlanner> {
        Arc::clone(&self.compiler)
    }
}

/// Retains the actual output guard through serialization and durable commit.
pub struct Int4SemanticObjectDocumentJobProcessor<'e, 'w, 'v, A: Int4SemanticObjectDocumentBatchAdmission> {
    native: NativeInt4SemanticObjectDocumentBatch<'e, 'w, 'v, A>,
    recipe: Int4SemanticObjectDocumentJobRecipe,
}
impl<'e, 'w, 'v, A: Int4SemanticObjectDocumentBatchAdmission> Int4SemanticObjectDocumentJobProcessor<'e, 'w, 'v, A> {
    pub fn new(planner: Int4SemanticObjectDocumentJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let Int4SemanticObjectDocumentJobPlanner { compiler, recipe } = planner;
        let native = NativeInt4SemanticObjectDocumentBatch::new_shared(compiler, engine, vocabulary, admission)?;
        Ok(Self { native, recipe })
    }
}
impl<A: Int4SemanticObjectDocumentBatchAdmission> BatchProcessor for Int4SemanticObjectDocumentJobProcessor<'_, '_, '_, A> {
    // Population authentication rejects non-null overrides before opening job
    // files. The config alone owns the exact schema and review declarations.
    type Args = ();
    type Prepared = PreparedInt4SemanticObjectBatchDocument;
    type Output = GuardedOutput<Int4SemanticObjectDocumentRun, A::Guard>;
    fn prepare(&mut self, input: BatchDocument<()>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(shared_document(input)?)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, input: BatchDocument<()>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_control(shared_document(input)?, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int4SemanticObjectDocumentBatchAdmission> DurableBatchProcessor for Int4SemanticObjectDocumentJobProcessor<'_, '_, '_, A> {
    type Recipe = Int4SemanticObjectDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.recipe.extraction_identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        check_job_limits(&self.recipe.config, limits)
    }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.native.verify_prepared(prepared)?;
        let config = &self.recipe.config;
        if prepared.model_work() != config.options.max_model_work
            || prepared.mask_node_visits() != config.extraction_limits.mapping.mapping.max_mask_visits
            || prepared.verification_budget() != config.extraction_limits.mapping.verification
            || prepared.max_result_bytes() != config.options.limits.max_output_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(JobWork::from_int4(prepared.model_work(), prepared.mask_node_visits()))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}
fn shared_document(document: BatchDocument<()>)
    -> Result<BatchDocument<crate::batch::extract::grouped_int4::long::stream::ExtractionDocumentArgs>, BatchItemFailure> {
    if document.task_args.is_some() { return Err(BatchItemFailure::fatal(BatchCode::InvalidEnvelope)); }
    Ok(BatchDocument { id: document.id, text: document.text, task_args: None })
}

#[cfg(test)]
mod tests;
