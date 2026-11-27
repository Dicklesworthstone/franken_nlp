//! Authenticated retained sampled populations on the resident portable INT8 task.
//! The actual compiler, vocabulary and tokenizer-work owner remain intact.
//! JobRunner authenticates the frozen population and durably debits complete
//! attempts before this adapter can admit native execution or publish a result.

use std::sync::Arc;
use serde::Serialize;
use super::{PlanningRecipe, PrefillRecipe};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, encoding_work::EncodingWorkScope,
        extract::ExtractionMaskBudget, generation::GuardedOutput,
        sentiment::sampled::{SampledSentimentBatchArgs, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES,
            sampled_sentiment_job_preparation_bytes}},
    execution_identity::ExecutionIdentity,
    grammar::{mask::MaskWorkLimits, runtime::JSON_RUNTIME_VERSION},
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{constrained::JsonWorkBudget,
        constrained_int8::{self, Int8JsonBudget, Int8JsonError, INT8_JSON_SAMPLED_EXECUTION,
            JSON_SAMPLER_VERSION, json_sampling_workspace_bytes},
        decode::DecodeStepControl, portable_int8::ProjectionWork,
        strict_int8::{Int8MemoryRequirement, Int8RunBudget, Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE,
            prefill::{Int8PrefillLimits, LAYER_MAJOR_PREFILL_VERSION}}},
    tasks::{ir::TaskBudget, sentiment::{SentimentAxis, SentimentRequest, sampled::{
        SampledSentimentError, SampledSentimentLimits, SampledSentimentOptions, SampledSentimentVocabulary,
        MAX_SENTIMENT_SAMPLES, SAMPLED_SENTIMENT_AGGREGATION_VERSION,
        SAMPLED_SENTIMENT_ITEM_ADDRESS_VERSION, SAMPLED_SENTIMENT_PROMPT_VERSION,
        quantized::{Int8SampledSentimentError, Int8SampledSentimentPlanner, Int8SampledSentimentRun,
            PreparedInt8SampledSentiment, INT8_SAMPLED_SENTIMENT_EXECUTION, check_model}}}},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
};

/// All native dimensions and mask work cover complete independent bundles.
/// Lifetime ceilings must equal the immutable job's durable work envelope.
#[derive(Clone, Copy, Debug)]
pub struct Int8SampledSentimentJobLimits {
    pub max_item_work: Int8Work,
    pub max_model_work: Int8Work,
    pub masks: ExtractionMaskBudget,
    pub max_sampler_bytes: u64,
}
impl Int8SampledSentimentJobLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        let sampler = json_sampling_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
        if !nonzero(self.max_item_work) || !nonzero(self.max_model_work)
            || !fits(self.max_item_work, self.max_model_work)
            || self.masks.per_mask.max_trie_node_visits == 0
            || self.masks.per_mask.checkpoint_interval_nodes == 0
            || self.masks.max_visits_per_item == 0
            || self.masks.max_visits_per_item > self.masks.max_visits_per_run
            || self.max_sampler_bytes < sampler {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct NativeRecipe {
    max_item_work: Int8Work,
    max_model_work: Int8Work,
    max_mask_trie_node_visits: usize,
    mask_checkpoint_interval_nodes: usize,
    max_mask_visits_per_item: u64,
    max_mask_visits_per_run: u64,
    max_sampler_bytes: u64,
}
impl From<Int8SampledSentimentJobLimits> for NativeRecipe {
    fn from(limits: Int8SampledSentimentJobLimits) -> Self {
        let Int8SampledSentimentJobLimits { max_item_work, max_model_work, masks, max_sampler_bytes } = limits;
        let ExtractionMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        Self { max_item_work, max_model_work, max_mask_trie_node_visits: max_trie_node_visits,
            mask_checkpoint_interval_nodes: checkpoint_interval_nodes,
            max_mask_visits_per_item: max_visits_per_item, max_mask_visits_per_run: max_visits_per_run,
            max_sampler_bytes }
    }
}

/// Private keyed-commitment material only: no Debug, Deserialize, public
/// fields or caller-supplied constructor. This is never an execution permit.
#[derive(Serialize)]
pub struct Int8SampledSentimentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    prompt_version: &'static str,
    aggregation_version: &'static str,
    sampler_version: &'static str,
    item_address_version: &'static str,
    sample_address_stride: usize,
    native_execution: &'static str,
    grammar_version: &'static str,
    task_spec: &'static str,
    task_budget: TaskBudget,
    default_axes: Vec<SentimentAxis>,
    sampling: SampledSentimentOptions,
    planning: PlanningRecipe,
    native: NativeRecipe,
    input_encoding: UntrustedEncoding,
    encoding_work_scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_work_limits: Option<SafeBpeWorkLimits>,
    job: JobLimits,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill: Option<PrefillRecipe>,
}
impl Int8SampledSentimentJobRecipe {
    fn new(task_budget: TaskBudget, sampling: SampledSentimentOptions,
        planning: SampledSentimentLimits, native: Int8SampledSentimentJobLimits,
        input_encoding: UntrustedEncoding, encoding_work_limits: Option<SafeBpeWorkLimits>,
        job: JobLimits) -> Result<Self, BatchFault> {
        task_budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        sampling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
        native.validate()?;
        input_encoding.validate().map_err(|_| BatchCode::InvalidLimits)?;
        EncodingWorkScope::workspace(&input_encoding, encoding_work_limits)?;
        let recipe = Self { version: 1, dependency_scope: "item-local",
            input_policy: "exact-record-independent-axis-sample-addresses-v1",
            execution: INT8_SAMPLED_SENTIMENT_EXECUTION,
            prompt_version: SAMPLED_SENTIMENT_PROMPT_VERSION,
            aggregation_version: SAMPLED_SENTIMENT_AGGREGATION_VERSION,
            sampler_version: JSON_SAMPLER_VERSION,
            item_address_version: SAMPLED_SENTIMENT_ITEM_ADDRESS_VERSION,
            sample_address_stride: MAX_SENTIMENT_SAMPLES,
            native_execution: INT8_JSON_SAMPLED_EXECUTION, grammar_version: JSON_RUNTIME_VERSION,
            task_spec: "sentiment-v1", task_budget, default_axes: SampledSentimentBatchArgs::default().axes,
            sampling, planning: planning.into(), native: native.into(), input_encoding,
            encoding_work_scope: "one-invocation-including-rejected-preparation-v1",
            encoding_work_limits, job, prefill: None };
        recipe.check_job_limits(job)?;
        bounded_json(&recipe, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
    fn with_prefill(mut self, limits: Int8PrefillLimits) -> Result<Self, BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let Int8PrefillLimits { max_batch_rows, max_extra_scratch_bytes } = limits;
        self.prefill = Some(PrefillRecipe { execution: LAYER_MAJOR_PREFILL_VERSION,
            max_batch_rows, max_extra_scratch_bytes });
        bounded_json(&self, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    fn prefill_limits(&self) -> Option<Int8PrefillLimits> {
        self.prefill.as_ref().map(|prefill| Int8PrefillLimits {
            max_batch_rows: prefill.max_batch_rows, max_extra_scratch_bytes: prefill.max_extra_scratch_bytes })
    }
    fn max_result_bytes(&self) -> u64 {
        self.task_budget.max_output_bytes.min(self.planning.max_result_bytes)
    }
    fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if limits != self.job || limits.max_id_bytes > 256
            || self.planning.max_input_bytes > limits.max_input_bytes_per_item
            || limits.max_work != (JobWork { model: self.native.max_model_work,
                mask_node_visits: self.native.max_mask_visits_per_run })
            || self.max_result_bytes() > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// A model-bound compiler and immutable vocabulary with one nonrenewable
/// tokenizer owner. The host retains this same Arc through native team join.
pub struct Int8SampledSentimentJobPlanner {
    planner: Int8SampledSentimentPlanner,
    vocabulary: Arc<SampledSentimentVocabulary>,
    native: Int8SampledSentimentJobLimits,
    recipe: Int8SampledSentimentJobRecipe,
    encoding: Arc<EncodingWorkScope>,
    preparation_bytes: u64,
    sampler_bytes: u64,
    seal: Arc<()>,
}
impl Int8SampledSentimentJobPlanner {
    pub fn new(planner: Int8SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        native: Int8SampledSentimentJobLimits, job: JobLimits) -> Result<Self, BatchFault> {
        Self::new_with_document_encoding(planner, vocabulary, native, job, None)
    }
    /// Compact input requires explicit invocation totals, as in durable scored
    /// jobs. Failed preparation spends that same owner; resume authenticates
    /// the original policy/totals and retains all journaled native debits.
    pub fn new_with_document_encoding(planner: Int8SampledSentimentPlanner,
        vocabulary: Arc<SampledSentimentVocabulary>, native: Int8SampledSentimentJobLimits,
        job: JobLimits, total: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        planner.execution_identity().validate().map_err(|_| BatchCode::Admission)?;
        constrained_int8::check_profile(planner.execution_identity()).map_err(|_| BatchCode::Admission)?;
        if planner.execution_identity().task_spec != "sentiment-v1"
            || planner.execution_identity().sampler_version != JSON_SAMPLER_VERSION {
            return Err(BatchCode::Admission.into());
        }
        let encoding = EncodingWorkScope::new(planner.input_encoding(), total)?;
        let recipe = Int8SampledSentimentJobRecipe::new(planner.task_budget(),
            planner.durable_options().clone(), planner.limits(), native,
            planner.input_encoding().clone(), total, job)?;
        let preparation_bytes = sampled_sentiment_job_preparation_bytes(
            planner.task_budget(), planner.limits(), planner.input_encoding())?
            .checked_add(if total.is_some() { 1024 } else { 0 }).ok_or(BatchCode::InvalidLimits)?;
        let sampler_bytes = json_sampling_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { planner, vocabulary, native, recipe, encoding, preparation_bytes,
            sampler_bytes, seal: Arc::new(()) })
    }
    /// Freeze the prompt schedule before storage is opened. Team width stays
    /// a physical host choice and cannot change private sample addresses.
    pub fn with_layer_major_prefill(mut self, limits: Int8PrefillLimits) -> Result<Self, BatchFault> {
        self.recipe = self.recipe.with_prefill(limits)?;
        Ok(self)
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    pub fn task_budget(&self) -> TaskBudget { self.planner.task_budget() }
    pub fn native_limits(&self) -> Int8SampledSentimentJobLimits { self.native }
    pub fn planning_limits(&self) -> SampledSentimentLimits { self.planner.limits() }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.input_encoding() }
    pub fn encoding_work_limits(&self) -> Option<SafeBpeWorkLimits> { self.encoding.limits }
    pub fn encoding_work(&self) -> Result<SafeBpeWork, BatchFault> { self.encoding.work() }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }
    pub fn sampler_workspace_bytes(&self) -> u64 { self.sampler_bytes }
    pub fn max_result_bytes(&self) -> u64 { self.recipe.max_result_bytes() }
    pub fn prefill_limits(&self) -> Option<Int8PrefillLimits> { self.recipe.prefill_limits() }
    pub fn job_limits(&self) -> JobLimits { self.recipe.job }
    /// Private configuration for the owned job's keyed commitment only.
    pub fn job_recipe(&self) -> &Int8SampledSentimentJobRecipe { &self.recipe }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.recipe.check_job_limits(limits)
    }

    /// Model-free preparation preserves stable item addressing and the actual
    /// spent tokenizer owner. An equivalent factory cannot execute its result.
    pub fn prepare_with_control<C: DecodeStepControl>(&self,
        document: BatchDocument<SampledSentimentBatchArgs>, control: &mut C)
        -> Result<PreparedInt8SampledSentimentJob, BatchItemFailure> {
        checkpoint(control)?;
        let args = document.task_args.unwrap_or_default();
        let request = SentimentRequest { document: document.text, axes: args.axes,
            budget: args.budget.unwrap_or(self.task_budget()) };
        let plan = self.encoding.prepare(control, |ceiling, work, control| {
            self.planner.plan_for_item_with_encoding_budget(&request, &document.id, control, work, ceiling)
                .map_err(task_failure)
        })?;
        if !fits(plan.planned_work(), self.native.max_item_work) {
            return Err(BatchItemFailure::reject(BatchCode::WorkLimit));
        }
        if plan.preparation_requirement_bytes() == 0
            || plan.preparation_requirement_bytes() > self.preparation_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let budget = native_budget(plan.planned_work(), plan.task_budget().max_kv_bytes,
            self.native.masks.max_visits_per_item, self.native);
        let masks = plan.complete_mask_charge(&self.vocabulary, budget, self.native.max_sampler_bytes)
            .map_err(task_failure)?;
        checkpoint(control)?;
        Ok(PreparedInt8SampledSentimentJob { plan, masks, seal: Arc::clone(&self.seal),
            encoding: Arc::clone(&self.encoding) })
    }
    fn verify(&self, prepared: &PreparedInt8SampledSentimentJob) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.seal, &prepared.seal) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        self.encoding.verify(&prepared.encoding)
    }
}

pub struct PreparedInt8SampledSentimentJob {
    plan: PreparedInt8SampledSentiment,
    masks: u64,
    seal: Arc<()>,
    encoding: Arc<EncodingWorkScope>,
}
impl PreparedInt8SampledSentimentJob {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn model_work(&self) -> Int8Work { self.plan.planned_work() }
    pub fn mask_node_visits(&self) -> u64 { self.masks }
    pub fn max_result_bytes(&self) -> u64 { self.plan.max_result_bytes() }
}

/// Embedding authority, not a model permit. The guard must cover the complete
/// output until its spool sync and journal acknowledgement have completed.
pub trait Int8SampledSentimentJobAdmission {
    type Guard;
    fn admit(&mut self, identity: &ExecutionIdentity, work: Int8Work, kv_bytes: u64,
        sampler_bytes: u64, max_result_bytes: u64)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

pub struct Int8SampledSentimentJobProcessor<'e, 'w, A: Int8SampledSentimentJobAdmission> {
    factory: Arc<Int8SampledSentimentJobPlanner>,
    engine: &'e mut StrictInt8Engine<'w>,
    admission: A,
    kv_bytes: u64,
    run: RunState,
}
impl<'e, 'w, A: Int8SampledSentimentJobAdmission> Int8SampledSentimentJobProcessor<'e, 'w, A> {
    pub fn new(factory: Arc<Int8SampledSentimentJobPlanner>, engine: &'e mut StrictInt8Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        factory.check_job_limits(factory.job_limits())?;
        check_model(factory.execution_identity(), engine.artifact_identity()).map_err(|_| BatchCode::Admission)?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(BatchCode::InvalidExecution.into());
        }
        let kv_bytes = Int8MemoryRequirement::for_context(engine.kv_cache().capacity_positions())
            .map_err(|_| BatchCode::Admission)?.kv_bytes;
        if kv_bytes > factory.task_budget().max_kv_bytes { return Err(BatchCode::Admission.into()); }
        let run = RunState::new(factory.native);
        Ok(Self { factory, engine, admission, kv_bytes, run })
    }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.run.state != State::Ready || self.engine.is_poisoned()
            || !self.engine.kv_cache().all_slots_have_len(0) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(())
    }
    fn preflight(&self, prepared: &PreparedInt8SampledSentimentJob) -> Result<(), BatchItemFailure> {
        self.factory.verify(prepared)?;
        if prepared.plan.required_context() > self.engine.kv_cache().capacity_positions()
            || self.kv_bytes > prepared.plan.task_budget().max_kv_bytes {
            return Err(BatchItemFailure::reject(BatchCode::Admission));
        }
        let budget = native_budget(prepared.model_work(), self.kv_bytes, prepared.masks, self.factory.native);
        let masks = prepared.plan.complete_mask_charge(&self.factory.vocabulary, budget,
            self.factory.native.max_sampler_bytes).map_err(task_failure)?;
        if masks != prepared.masks { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(())
    }
    fn settle<T>(&mut self, result: Result<T, BatchItemFailure>) -> Result<T, BatchItemFailure> {
        match result {
            Ok(value) if self.run.state == State::Running && !self.engine.is_poisoned()
                && self.engine.kv_cache().all_slots_have_len(0) => {
                self.run.state = State::Ready; Ok(value)
            }
            Ok(_) => {
                self.run.state = State::Failed; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
            }
            Err(mut failure) => {
                self.run.state = State::Failed; failure.stop = true; Err(failure)
            }
        }
    }
}
impl<A: Int8SampledSentimentJobAdmission> BatchProcessor for Int8SampledSentimentJobProcessor<'_, '_, A> {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedInt8SampledSentimentJob;
    type Output = GuardedOutput<Int8SampledSentimentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.ready()?;
        // A panic or refusal cannot leave a reusable processor around a spent
        // tokenizer owner. Model-free factory previews retain their own work.
        self.run.state = State::Running;
        let result = (|| {
            let prepared = self.factory.prepare_with_control(document, control)?;
            self.preflight(&prepared)?;
            self.run.available(prepared.model_work(), prepared.masks)?;
            checkpoint(control)?;
            Ok(prepared)
        })();
        self.settle(result)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.model_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ready()?;
        self.run.state = State::Running;
        let result = (|| {
            self.preflight(&prepared)?;
            // All six local ceilings debit atomically before host admission.
            // JobRunner has already journaled this same full attempt debit.
            self.run.charge(prepared.model_work(), prepared.masks, context)?;
            checkpoint(control)?;
            let plan = &prepared.plan;
            let (identity, guard) = self.admission.admit(plan.execution_identity(), plan.planned_work(),
                self.kv_bytes, self.factory.sampler_bytes, plan.max_result_bytes())?;
            plan.verify_identity(&identity).map_err(task_failure)?;
            checkpoint(control)?;
            let budget = native_budget(plan.planned_work(), self.kv_bytes, prepared.masks, self.factory.native);
            let run = match self.factory.prefill_limits() {
                Some(prefill) => plan.execute_layer_major_with_control(self.engine, &identity,
                    &self.factory.vocabulary, budget, self.factory.native.max_sampler_bytes, prefill, control),
                None => plan.execute_with_control(self.engine, &identity, &self.factory.vocabulary,
                    budget, self.factory.native.max_sampler_bytes, control),
            }.map_err(task_failure)?;
            if !valid_result(plan, &run, prepared.masks) {
                return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
            }
            checkpoint(control)?;
            Ok(GuardedOutput::new(run, guard))
        })();
        self.settle(result)
    }
}
impl<A: Int8SampledSentimentJobAdmission> DurableBatchProcessor for Int8SampledSentimentJobProcessor<'_, '_, A> {
    type Recipe = Int8SampledSentimentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.factory.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.factory.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.factory.check_job_limits(limits)
    }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.ready()?;
        self.preflight(prepared)?;
        self.run.available(prepared.model_work(), prepared.masks)?;
        Ok(JobWork { model: prepared.model_work(), mask_node_visits: prepared.masks })
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

fn native_budget(work: Int8Work, kv_bytes: u64, masks: u64,
    limits: Int8SampledSentimentJobLimits) -> Int8JsonBudget {
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: kv_bytes, mask_limits: limits.masks.per_mask, max_total_mask_node_visits: masks } }
}
fn valid_result(plan: &PreparedInt8SampledSentiment, run: &Int8SampledSentimentRun, masks: u64) -> bool {
    let samples = run.dimensions.iter().try_fold(0_usize, |n, axis| n.checked_add(axis.samples.len()));
    run.schema_version == 1 && run.execution == INT8_SAMPLED_SENTIMENT_EXECUTION
        && run.task_spec == "sentiment-v1" && run.sampler_version == JSON_SAMPLER_VERSION
        && run.prompt_template_sha256 == plan.execution_identity().template_digest
        && run.numerics_profile == STRICT_INT8_PROFILE
        && run.aggregation == SAMPLED_SENTIMENT_AGGREGATION_VERSION
        && run.samples_per_axis != 0 && samples == Some(plan.sample_count())
        && run.dimensions.iter().all(|axis| axis.samples.len() == run.samples_per_axis
            && axis.samples.iter().enumerate().all(|(index, sample)| sample.sample_index == index))
        && fits(run.model_work, plan.planned_work())
        && run.generated_tokens <= plan.max_output_tokens()
        && run.generated_tokens <= u64::from(plan.task_budget().max_output_tokens)
        && run.mask_node_visit_charge <= masks
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct RunState {
    remaining: Int8Work,
    remaining_masks: u64,
    last_sequence: u64,
    state: State,
}
impl RunState {
    fn new(limits: Int8SampledSentimentJobLimits) -> Self {
        Self { remaining: limits.max_model_work, remaining_masks: limits.masks.max_visits_per_run,
            last_sequence: 0, state: State::Ready }
    }
    fn available(&self, work: Int8Work, masks: u64) -> Result<(), BatchItemFailure> {
        subtract(self.remaining, work)?;
        if masks > self.remaining_masks { return Err(BatchItemFailure::reject(BatchCode::WorkLimit)); }
        Ok(())
    }
    fn charge(&mut self, work: Int8Work, masks: u64, context: BatchRequestContext)
        -> Result<(), BatchItemFailure> {
        if self.state != State::Running || context.request_seq <= self.last_sequence
            || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let remaining = subtract(self.remaining, work)?;
        let remaining_masks = self.remaining_masks.checked_sub(masks)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.remaining = remaining; self.remaining_masks = remaining_masks;
        self.last_sequence = context.request_seq;
        Ok(())
    }
}
fn nonzero(work: Int8Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn subtract(a: Int8Work, b: Int8Work) -> Result<Int8Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int8Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
fn task_failure(error: Int8SampledSentimentError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8SampledSentimentError::Shared(SampledSentimentError::WorkBudget | SampledSentimentError::TokenizerWork) =>
            BatchItemFailure::reject(BatchCode::WorkLimit),
        Int8SampledSentimentError::Shared(SampledSentimentError::OutputBudget) =>
            BatchItemFailure::reject(BatchCode::OutputLineLimit),
        Int8SampledSentimentError::Shared(SampledSentimentError::Limits | SampledSentimentError::Tokenizer
            | SampledSentimentError::Template | SampledSentimentError::Schema) =>
            BatchItemFailure::reject(BatchCode::Planning),
        Int8SampledSentimentError::Shared(SampledSentimentError::Identity)
            | Int8SampledSentimentError::Native(Int8JsonError::ModelIdentity) =>
            BatchItemFailure::fatal(BatchCode::Admission),
        Int8SampledSentimentError::Shared(SampledSentimentError::Accounting)
            | Int8SampledSentimentError::Native(Int8JsonError::WorkMismatch) =>
            BatchItemFailure::fatal(BatchCode::InvalidExecution),
        Int8SampledSentimentError::Shared(SampledSentimentError::Allocation) =>
            BatchItemFailure::fatal(BatchCode::Allocation),
        Int8SampledSentimentError::Shared(SampledSentimentError::Serialization) =>
            BatchItemFailure::fatal(BatchCode::Serialization),
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}

pub mod long;

#[cfg(test)] mod tests;
