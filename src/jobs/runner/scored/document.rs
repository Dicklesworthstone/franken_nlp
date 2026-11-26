//! Authenticated whole-document INT8 scoring over the resident map adapters.
//! One original document is one durable item. Every chunk/head is prepared
//! before the five native work axes are journaled; only the complete guarded
//! map and aggregate can reach the existing spool-first commit protocol.
use super::{ClassificationJobRecipe, SentimentJobRecipe, SentimentPlanningRecipe,
    ScorerRecipe, RECIPE_BYTES};
use std::sync::Arc;
use serde::Serialize;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput,
        classify::{ClassificationBatchArgs, quantized::long::{
            Int8ClassificationDocumentBatchAdmission, Int8ClassificationDocumentBatchPlanner,
            NativeInt8ClassificationDocumentBatch, PreparedInt8ClassificationDocument}},
        sentiment::quantized::{Int8SentimentDocumentBatchAdmission, Int8SentimentDocumentBatchConfig,
            Int8SentimentDocumentBatchPlanner, NativeInt8SentimentDocumentBatch,
            PreparedInt8SentimentDocument},
        judge::quantized::{Int8FaithfulnessDocumentArgs, Int8FaithfulnessDocumentBatchAdmission,
            Int8FaithfulnessDocumentBatchPlanner, NativeInt8FaithfulnessDocumentBatch,
            PreparedInt8FaithfulnessDocument}},
    execution_identity::ExecutionIdentity,
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl,
        strict_int8::{Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, CHUNK_PROFILE},
        classify::{ClassificationAggregationPolicy, ClassificationLimits, ClassificationPlanner,
            CLASSIFICATION_AGGREGATION_PROFILE, quantized::long::{Int8ClassificationMapLimits,
                Int8ClassificationMapRun, INT8_CLASSIFICATION_MAP_EXECUTION}},
        sentiment::{SentimentAggregationPolicy, SentimentLimits, SentimentPlanner,
            SENTIMENT_AGGREGATION_PROFILE, SENTIMENT_PROMPT_VERSION, batch::SentimentBatchArgs,
            quantized::{INT8_SENTIMENT_EXECUTION, long::{Int8SentimentMapLimits,
                Int8SentimentMapRun, INT8_SENTIMENT_MAP_EXECUTION}}},
        judge::{FaithfulnessAggregationPolicy, FaithfulnessPolicy, JudgeLimits, JudgePlanner,
            FAITHFULNESS_AGGREGATION_PROFILE, JUDGE_PROMPT_VERSION,
            quantized::faithfulness_map::{Int8FaithfulnessMapLimits, Int8FaithfulnessMapRun,
                INT8_FAITHFULNESS_MAP_EXECUTION}}},
    tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding},
};

/// Serialized only inside an owned job's keyed recipe commitment. Private
/// fields and constructors prevent a wire recipe from replacing the actual
/// pinned planner. Physical worker/prompt widths grant no semantic authority.
#[derive(Serialize)]
pub struct CompleteDocumentScoringRecipe<R, A> {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    aggregation_profile: &'static str,
    reservation_policy: &'static str,
    scoring: R,
    chunks: ChunkLimits,
    per_document_model_work: Int8Work,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
    aggregation: A,
}
pub type Int8ClassificationDocumentJobRecipe =
    CompleteDocumentScoringRecipe<ClassificationJobRecipe, ClassificationAggregationPolicy>;
pub type Int8SentimentDocumentJobRecipe =
    CompleteDocumentScoringRecipe<SentimentJobRecipe, SentimentAggregationPolicy>;

impl<R: Serialize, A: Serialize> CompleteDocumentScoringRecipe<R, A> {
    fn checked(self) -> Result<Self, BatchFault> {
        bounded_json(&self, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
}

/// Owns the exact complete-document compiler and immutable resume recipe.
/// The original defaults allocation remains retained beside its recipe copy.
pub struct Int8ClassificationDocumentJobPlanner {
    compiler: Arc<Int8ClassificationDocumentBatchPlanner>,
    recipe: Int8ClassificationDocumentJobRecipe,
}
impl Int8ClassificationDocumentJobPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: Arc<ClassificationPlanner>, identity: ExecutionIdentity,
        ceiling: TaskBudget, planning: ClassificationLimits, mapping: Int8ClassificationMapLimits,
        defaults: Option<ClassificationBatchArgs>, max_model_work: Int8Work,
        max_tokenizer_work: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        Int8ClassificationDocumentBatchPlanner::validate_configuration(&planner, &identity,
            ceiling, planning, mapping, defaults.as_ref(), max_model_work, max_tokenizer_work)?;
        let mut scoring = ClassificationJobRecipe::new(ceiling, planning, defaults.clone(), max_model_work)?;
        scoring.input_encoding = Some(planner.document_encoding().clone());
        scoring.encoding_work_limits = max_tokenizer_work;
        // Exhaustive destructuring forces every new map setting to receive a
        // deliberate authenticated representation rather than a silent default.
        let Int8ClassificationMapLimits { chunks, max_model_work: per_document_model_work,
            max_retained_result_bytes, max_result_bytes, aggregation } = mapping;
        let recipe = CompleteDocumentScoringRecipe { version: 1,
            dependency_scope: "item-local-complete-document",
            input_policy: "typed-complete-document-arguments-v1",
            execution: INT8_CLASSIFICATION_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            aggregation_profile: CLASSIFICATION_AGGREGATION_PROFILE,
            reservation_policy: "all-chunk-five-axis-work-before-durable-attempt-v1",
            scoring, chunks, per_document_model_work, max_retained_result_bytes,
            max_result_bytes, aggregation }.checked()?;
        let compiler = Int8ClassificationDocumentBatchPlanner::new(planner, identity,
            ceiling, planning, mapping, defaults, max_model_work, max_tokenizer_work)?;
        Ok(Self { compiler: Arc::new(compiler), recipe })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.compiler.base_execution_identity() }
    pub fn job_recipe(&self) -> &Int8ClassificationDocumentJobRecipe { &self.recipe }
    pub fn task_ceiling(&self) -> TaskBudget { self.compiler.task_budget() }
    pub fn planning_limits(&self) -> ClassificationLimits { self.compiler.planning_limits() }
    pub fn mapping_limits(&self) -> Int8ClassificationMapLimits { self.compiler.mapping_limits() }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn max_run_work(&self) -> Int8Work { self.compiler.max_run_work() }
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        self.compiler.retained_defaults_bytes()?.checked_add(classification_defaults_bytes(
            self.recipe.scoring.defaults.as_ref())?).ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        check_limits(limits, self.recipe.chunks, self.recipe.max_result_bytes,
            self.recipe.per_document_model_work, self.max_run_work())
    }
}

fn classification_defaults_bytes(defaults: Option<&ClassificationBatchArgs>) -> Result<u64, BatchFault> {
    let Some(args) = defaults else { return Ok(0); };
    let base = (args.labels.capacity() as u64)
        .checked_mul(std::mem::size_of::<crate::tasks::classify::ClassificationLabel>() as u64)
        .ok_or(BatchCode::InvalidLimits)?;
    args.labels.iter().try_fold(base, |bytes, label| bytes.checked_add(label.id.capacity() as u64)
        .and_then(|n| n.checked_add(label.description.capacity() as u64))
        .ok_or_else(|| BatchCode::InvalidLimits.into()))
}

/// Owns the sentiment planner and source/axis defaults across physical join.
/// The transient borrowed batch compiler is constructed only inside execution.
pub struct Int8SentimentDocumentJobPlanner {
    planner: Arc<SentimentPlanner>,
    config: Int8SentimentDocumentBatchConfig,
    recipe: Int8SentimentDocumentJobRecipe,
}
impl Int8SentimentDocumentJobPlanner {
    pub fn new(planner: Arc<SentimentPlanner>, config: Int8SentimentDocumentBatchConfig)
        -> Result<Self, BatchFault> {
        config.validate(&planner)?;
        let scoring = SentimentJobRecipe { version: 1, dependency_scope: "item-local",
            execution: INT8_SENTIMENT_EXECUTION, prompt_version: SENTIMENT_PROMPT_VERSION,
            task_ceiling: config.task_ceiling, planning: SentimentPlanningRecipe::from(config.planning),
            max_item_work: config.mapping.max_model_work, max_model_work: config.max_model_work,
            defaults: config.defaults.clone(), input_encoding: Some(planner.document_encoding().clone()),
            encoding_work_limits: config.max_tokenizer_work };
        let Int8SentimentMapLimits { chunks, max_model_work: per_document_model_work,
            max_retained_result_bytes, max_result_bytes, aggregation } = config.mapping;
        let recipe = CompleteDocumentScoringRecipe { version: 1,
            dependency_scope: "item-local-complete-document",
            input_policy: "typed-complete-document-arguments-v1",
            execution: INT8_SENTIMENT_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            aggregation_profile: SENTIMENT_AGGREGATION_PROFILE,
            reservation_policy: "all-chunk-five-axis-work-before-durable-attempt-v1",
            scoring, chunks, per_document_model_work, max_retained_result_bytes,
            max_result_bytes, aggregation }.checked()?;
        Ok(Self { planner, config, recipe })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.config.identity }
    pub fn job_recipe(&self) -> &Int8SentimentDocumentJobRecipe { &self.recipe }
    pub fn task_ceiling(&self) -> TaskBudget { self.config.task_ceiling }
    pub fn planning_limits(&self) -> SentimentLimits { self.config.planning }
    pub fn mapping_limits(&self) -> Int8SentimentMapLimits { self.config.mapping }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.document_encoding() }
    pub fn max_run_work(&self) -> Int8Work { self.config.max_model_work }
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        // Configuration + transient compiler + immutable recipe coexist.
        let recipe = self.recipe.scoring.defaults.as_ref().map_or(Ok(0), |args|
            (args.axes.capacity() as u64).checked_mul(std::mem::size_of::<crate::tasks::sentiment::SentimentAxis>() as u64)
                .ok_or(BatchCode::InvalidLimits))?;
        self.config.retained_defaults_bytes()?.checked_add(recipe)
            .ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        check_limits(limits, self.recipe.chunks, self.recipe.max_result_bytes,
            self.recipe.per_document_model_work, self.max_run_work())
    }
    fn compiler(&self) -> Result<Int8SentimentDocumentBatchPlanner<'_>, BatchFault> {
        Int8SentimentDocumentBatchPlanner::new(&self.planner, Int8SentimentDocumentBatchConfig {
            identity: self.config.identity.clone(), task_ceiling: self.config.task_ceiling,
            planning: self.config.planning, mapping: self.config.mapping,
            defaults: self.config.defaults.clone(), max_model_work: self.config.max_model_work,
            max_tokenizer_work: self.config.max_tokenizer_work,
        })
    }
}

/// The complete finite-scoring configuration is authenticated separately from
/// physical prefill/team widths. Every new planner limit must be frozen here.
#[derive(Serialize)]
struct FaithfulnessPlanningRecipe {
    per_head: ScorerRecipe,
    max_total_prompt_tokens: usize,
    max_total_candidate_tokens: usize,
    max_total_projected_logits: u64,
    max_output_bytes: u64,
}
impl From<JudgeLimits> for FaithfulnessPlanningRecipe {
    fn from(limits: JudgeLimits) -> Self {
        let JudgeLimits { per_head, max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_projected_logits, max_output_bytes } = limits;
        Self { per_head: per_head.into(), max_total_prompt_tokens, max_total_candidate_tokens,
            max_total_projected_logits, max_output_bytes }
    }
}

/// A new complete-source recipe; existing one-context judgment job identities
/// are unchanged. Only this private projection of the actual pinned compiler
/// enters the journal's keyed commitment. Claims remain in authenticated items.
#[derive(Serialize)]
pub struct Int8FaithfulnessDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    prompt_version: &'static str,
    chunk_profile: &'static str,
    aggregation_profile: &'static str,
    reservation_policy: &'static str,
    task_ceiling: TaskBudget,
    planning: FaithfulnessPlanningRecipe,
    policy: FaithfulnessPolicy,
    chunks: ChunkLimits,
    per_document_model_work: Int8Work,
    max_model_work: Int8Work,
    max_retained_result_bytes: u64,
    max_result_bytes: u64,
    aggregation: FaithfulnessAggregationPolicy,
    input_encoding: UntrustedEncoding,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_work_limits: Option<SafeBpeWorkLimits>,
}

/// Owns the same sealed complete-source compiler as resident faithfulness.
/// No duplicate request builder, evidence reducer or native scorer is exposed.
pub struct Int8FaithfulnessDocumentJobPlanner {
    compiler: Arc<Int8FaithfulnessDocumentBatchPlanner>,
    recipe: Int8FaithfulnessDocumentJobRecipe,
}
impl Int8FaithfulnessDocumentJobPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: Arc<JudgePlanner>, identity: ExecutionIdentity,
        ceiling: TaskBudget, planning: JudgeLimits, mapping: Int8FaithfulnessMapLimits,
        policy: FaithfulnessPolicy, max_model_work: Int8Work,
        max_tokenizer_work: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        Int8FaithfulnessDocumentBatchPlanner::validate_configuration(&planner, &identity,
            ceiling, planning, mapping, policy, max_model_work, max_tokenizer_work)?;
        let Int8FaithfulnessMapLimits { chunks, max_model_work: per_document_model_work,
            max_retained_result_bytes, max_result_bytes, aggregation } = mapping;
        let recipe = Int8FaithfulnessDocumentJobRecipe { version: 1,
            dependency_scope: "item-local-complete-source-claim",
            input_policy: "fixed-faithfulness-policy-required-claim-item-args-v1",
            execution: INT8_FAITHFULNESS_MAP_EXECUTION, prompt_version: JUDGE_PROMPT_VERSION,
            chunk_profile: CHUNK_PROFILE, aggregation_profile: FAITHFULNESS_AGGREGATION_PROFILE,
            reservation_policy: "all-chunk-five-axis-work-before-durable-attempt-v1",
            task_ceiling: ceiling, planning: planning.into(), policy, chunks,
            per_document_model_work, max_model_work, max_retained_result_bytes,
            max_result_bytes, aggregation, input_encoding: planner.document_encoding().clone(),
            encoding_work_limits: max_tokenizer_work };
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        let compiler = Int8FaithfulnessDocumentBatchPlanner::new(planner, identity,
            ceiling, planning, mapping, policy, max_model_work, max_tokenizer_work)?;
        Ok(Self { compiler: Arc::new(compiler), recipe })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.compiler.base_execution_identity() }
    pub fn job_recipe(&self) -> &Int8FaithfulnessDocumentJobRecipe { &self.recipe }
    pub fn task_ceiling(&self) -> TaskBudget { self.compiler.task_budget() }
    pub fn planning_limits(&self) -> JudgeLimits { self.compiler.planning_limits() }
    pub fn mapping_limits(&self) -> Int8FaithfulnessMapLimits { self.compiler.mapping_limits() }
    pub fn policy(&self) -> FaithfulnessPolicy { self.compiler.policy() }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn max_run_work(&self) -> Int8Work { self.compiler.max_run_work() }
    pub fn max_tokenizer_work(&self) -> Option<SafeBpeWorkLimits> { self.compiler.max_tokenizer_work() }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        check_limits(limits, self.recipe.chunks, self.recipe.max_result_bytes,
            self.recipe.per_document_model_work, self.max_run_work())
    }
}

fn fits(work: Int8Work, ceiling: Int8Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn check_limits(limits: JobLimits, chunks: ChunkLimits, max_result_bytes: u64,
    per_document: Int8Work, invocation: Int8Work) -> Result<(), BatchFault> {
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if chunks.max_input_bytes > limits.max_input_bytes_per_item
        || max_result_bytes > limits.max_result_bytes as u64
        || !fits(per_document, invocation) || !fits(invocation, limits.max_work.model) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

/// The private invocation seal is checked before durable attempt accounting,
/// not only when the underlying native processor consumes the document.
pub struct PreparedInt8ClassificationDocumentJob {
    native: PreparedInt8ClassificationDocument,
    owner: Arc<()>,
}
pub struct PreparedInt8SentimentDocumentJob {
    native: PreparedInt8SentimentDocument,
    owner: Arc<()>,
}
pub struct PreparedInt8FaithfulnessDocumentJob {
    native: PreparedInt8FaithfulnessDocument,
    owner: Arc<()>,
}
impl PreparedInt8ClassificationDocumentJob {
    pub fn planned_work(&self) -> Int8Work { self.native.planned_work() }
    pub fn chunk_count(&self) -> usize { self.native.chunk_count() }
}
impl PreparedInt8SentimentDocumentJob {
    pub fn planned_work(&self) -> Int8Work { self.native.planned_work() }
    pub fn chunk_count(&self) -> usize { self.native.chunk_count() }
}
impl PreparedInt8FaithfulnessDocumentJob {
    pub fn planned_work(&self) -> Int8Work { self.native.planned_work() }
    pub fn chunk_count(&self) -> usize { self.native.chunk_count() }
}
fn verify_owner(owner: &Arc<()>, prepared: &Arc<()>) -> Result<(), BatchItemFailure> {
    if !Arc::ptr_eq(owner, prepared) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    Ok(())
}

pub struct Int8ClassificationDocumentJobProcessor<'e, 'w, A: Int8ClassificationDocumentBatchAdmission> {
    factory: Arc<Int8ClassificationDocumentJobPlanner>,
    native: NativeInt8ClassificationDocumentBatch<'e, 'w, A>,
    owner: Arc<()>,
}
impl<'e, 'w, A: Int8ClassificationDocumentBatchAdmission>
    Int8ClassificationDocumentJobProcessor<'e, 'w, A> {
    pub fn scheduled(factory: Arc<Int8ClassificationDocumentJobPlanner>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A, prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        let compiler = Arc::clone(&factory.compiler);
        let native = match prefill {
            None => NativeInt8ClassificationDocumentBatch::new(compiler, engine, admission),
            Some(prefill) => NativeInt8ClassificationDocumentBatch::new_layer_major(compiler, engine, admission, prefill),
        }?;
        Ok(Self { factory, native, owner: Arc::new(()) })
    }
}
impl<A: Int8ClassificationDocumentBatchAdmission> BatchProcessor
    for Int8ClassificationDocumentJobProcessor<'_, '_, A> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedInt8ClassificationDocumentJob;
    type Output = GuardedOutput<Int8ClassificationMapRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        let native = self.native.prepare(document)?;
        Ok(PreparedInt8ClassificationDocumentJob { native, owner: Arc::clone(&self.owner) })
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        let native = self.native.prepare_with_control(document, control)?;
        Ok(PreparedInt8ClassificationDocumentJob { native, owner: Arc::clone(&self.owner) })
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(&prepared.native) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        verify_owner(&self.owner, &prepared.owner)?;
        self.native.execute(prepared.native, control)
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        verify_owner(&self.owner, &prepared.owner)?;
        self.native.execute_with_context(prepared.native, context, control)
    }
}
impl<A: Int8ClassificationDocumentBatchAdmission> DurableBatchProcessor
    for Int8ClassificationDocumentJobProcessor<'_, '_, A> {
    type Recipe = Int8ClassificationDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.factory.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.factory.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.factory.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        verify_owner(&self.owner, &prepared.owner)?;
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork { model: prepared.planned_work(), mask_node_visits: 0 })
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.factory.mapping_limits().max_result_bytes }
}

pub struct Int8SentimentDocumentJobProcessor<'p, 'e, 'w, A: Int8SentimentDocumentBatchAdmission> {
    factory: &'p Int8SentimentDocumentJobPlanner,
    native: NativeInt8SentimentDocumentBatch<'p, 'e, 'w, A>,
    owner: Arc<()>,
}
impl<'p, 'e, 'w, A: Int8SentimentDocumentBatchAdmission> Int8SentimentDocumentJobProcessor<'p, 'e, 'w, A> {
    pub fn scheduled(factory: &'p Int8SentimentDocumentJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        admission: A, prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        let compiler = factory.compiler()?;
        let native = match prefill {
            None => NativeInt8SentimentDocumentBatch::new(compiler, engine, admission),
            Some(prefill) => NativeInt8SentimentDocumentBatch::new_layer_major(compiler, engine, admission, prefill),
        }?;
        Ok(Self { factory, native, owner: Arc::new(()) })
    }
}
impl<A: Int8SentimentDocumentBatchAdmission> BatchProcessor
    for Int8SentimentDocumentJobProcessor<'_, '_, '_, A> {
    type Args = SentimentBatchArgs;
    type Prepared = PreparedInt8SentimentDocumentJob;
    type Output = GuardedOutput<Int8SentimentMapRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        let native = self.native.prepare(document)?;
        Ok(PreparedInt8SentimentDocumentJob { native, owner: Arc::clone(&self.owner) })
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        let native = self.native.prepare_with_control(document, control)?;
        Ok(PreparedInt8SentimentDocumentJob { native, owner: Arc::clone(&self.owner) })
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(&prepared.native) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        verify_owner(&self.owner, &prepared.owner)?;
        self.native.execute(prepared.native, control)
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        verify_owner(&self.owner, &prepared.owner)?;
        self.native.execute_with_context(prepared.native, context, control)
    }
}
impl<A: Int8SentimentDocumentBatchAdmission> DurableBatchProcessor
    for Int8SentimentDocumentJobProcessor<'_, '_, '_, A> {
    type Recipe = Int8SentimentDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.factory.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.factory.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.factory.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        verify_owner(&self.owner, &prepared.owner)?;
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork { model: prepared.planned_work(), mask_node_visits: 0 })
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.factory.mapping_limits().max_result_bytes }
}

/// The complete source/claim, admission guard and exact evidence result stay
/// owned until the existing runner has synced and acknowledged the whole item.
pub struct Int8FaithfulnessDocumentJobProcessor<'e, 'w, A: Int8FaithfulnessDocumentBatchAdmission> {
    factory: Arc<Int8FaithfulnessDocumentJobPlanner>,
    native: NativeInt8FaithfulnessDocumentBatch<'e, 'w, A>,
    owner: Arc<()>,
}
impl<'e, 'w, A: Int8FaithfulnessDocumentBatchAdmission>
    Int8FaithfulnessDocumentJobProcessor<'e, 'w, A> {
    pub fn scheduled(factory: Arc<Int8FaithfulnessDocumentJobPlanner>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A, prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        let compiler = Arc::clone(&factory.compiler);
        let native = match prefill {
            None => NativeInt8FaithfulnessDocumentBatch::new(compiler, engine, admission),
            Some(prefill) => NativeInt8FaithfulnessDocumentBatch::new_layer_major(compiler, engine, admission, prefill),
        }?;
        Ok(Self { factory, native, owner: Arc::new(()) })
    }
}
impl<A: Int8FaithfulnessDocumentBatchAdmission> BatchProcessor
    for Int8FaithfulnessDocumentJobProcessor<'_, '_, A> {
    type Args = Int8FaithfulnessDocumentArgs;
    type Prepared = PreparedInt8FaithfulnessDocumentJob;
    type Output = GuardedOutput<Int8FaithfulnessMapRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        let native = self.native.prepare(document)?;
        Ok(PreparedInt8FaithfulnessDocumentJob { native, owner: Arc::clone(&self.owner) })
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        let native = self.native.prepare_with_control(document, control)?;
        Ok(PreparedInt8FaithfulnessDocumentJob { native, owner: Arc::clone(&self.owner) })
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(&prepared.native) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        verify_owner(&self.owner, &prepared.owner)?;
        self.native.execute(prepared.native, control)
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        verify_owner(&self.owner, &prepared.owner)?;
        self.native.execute_with_context(prepared.native, context, control)
    }
}
impl<A: Int8FaithfulnessDocumentBatchAdmission> DurableBatchProcessor
    for Int8FaithfulnessDocumentJobProcessor<'_, '_, A> {
    type Recipe = Int8FaithfulnessDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.factory.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.factory.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.factory.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        // The runner persists this debit BEFORE native execution, so reject a
        // foreign prepared item here rather than relying on its later check.
        verify_owner(&self.owner, &prepared.owner)?;
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork { model: prepared.planned_work(), mask_node_visits: 0 })
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.factory.mapping_limits().max_result_bytes }
}

#[cfg(test)]
mod tests;
