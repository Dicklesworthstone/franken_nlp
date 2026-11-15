//! Ordered sampled-sentiment bundles on one resident grouped-INT4 engine.
//!
//! The existing task owns sampling, complete JSON validation and per-sample
//! session cleanup. This adapter reserves the whole axis-by-sample schedule
//! before execution, including failed attempts. Flush never renews any native
//! or mask allowance, and delivery retains the embedding host's output guard.

use std::sync::Arc;
use serde::Deserialize;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, extract::ExtractionMaskBudget, output::GuardedOutput},
    execution_identity::ExecutionIdentity,
    native_engine::{
        constrained::JsonWorkBudget,
        constrained_int4::{self, Int4JsonBudget, JSON_SAMPLER_VERSION, json_sampling_workspace_bytes},
        decode::DecodeStepControl,
        portable_int8::ProjectionWork,
        strict_int4::{Int4MemoryRequirement, Int4RunBudget, Int4Work, StrictInt4Engine, STRICT_INT4_PROFILE,
            prefill::Int4PrefillLimits},
    },
    tasks::{ir::TaskBudget, sentiment::{SentimentAxis, SentimentRequest,
        sampled::{PreparedSampledSentiment, SampledSentimentError, SampledSentimentLimits,
            SampledSentimentPlanner, SampledSentimentRun, SampledSentimentVocabulary,
            SAMPLED_SENTIMENT_EXECUTION}}},
    tokenizer::untrusted::UntrustedEncoding,
};

/// Per-record task data only. A budget can tighten every planner ceiling;
/// sampler policy, seed and model identity remain fixed for the invocation.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledSentimentBatchArgs {
    #[serde(default = "all_axes")]
    pub axes: Vec<SentimentAxis>,
    /// Omission uses the run ceiling. Every supplied field must fit it.
    #[serde(default)]
    pub budget: Option<TaskBudget>,
}
fn all_axes() -> Vec<SentimentAxis> { SentimentAxis::ALL.to_vec() }
impl Default for SampledSentimentBatchArgs {
    fn default() -> Self { Self { axes: all_axes(), budget: None } }
}

/// Native and mask work are nonrenewable for the whole invocation. The
/// sampler workspace is reused sequentially, not multiplied by sample count.
#[derive(Clone, Copy, Debug)]
pub struct SampledSentimentBatchLimits {
    pub max_item_work: Int4Work,
    pub max_model_work: Int4Work,
    pub masks: ExtractionMaskBudget,
    pub max_sampler_bytes: u64,
}
impl SampledSentimentBatchLimits {
    pub fn validate(self) -> Result<(), BatchFault> {
        let masks = self.masks;
        let sampler = json_sampling_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
        if !nonzero(self.max_item_work) || !nonzero(self.max_model_work)
            || !fits(self.max_item_work, self.max_model_work)
            || masks.per_mask.max_trie_node_visits == 0 || masks.per_mask.checkpoint_interval_nodes == 0
            || masks.max_visits_per_item == 0 || masks.max_visits_per_run < masks.max_visits_per_item
            || self.max_sampler_bytes < sampler {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

/// Pure upper bound shared by pre-IO callers and the hosted corpus adapter.
/// It includes one pinned compiler/vocabulary allowance, the largest retained
/// four-axis prompt bundle, result staging, and one compact-encoder workspace.
/// Transport, native buffers, sampler and allocator reserves remain separate.
pub fn sampled_sentiment_batch_preparation_bytes(task: TaskBudget, limits: SampledSentimentLimits,
    encoding: &UntrustedEncoding) -> Result<u64, BatchFault> {
    task.validate().map_err(|_| BatchCode::InvalidLimits)?;
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    encoding.validate().map_err(|_| BatchCode::InvalidLimits)?;
    let workspace = match encoding {
        UntrustedEncoding::ByteFallback => 0,
        UntrustedEncoding::SafeBpeV1(policy) => policy.requirements()
            .map_err(|_| BatchCode::InvalidLimits)?.workspace_bytes,
    };
    let result = task.max_output_bytes.min(limits.max_result_bytes);
    u64::from(task.max_input_tokens).checked_mul(SentimentAxis::ALL.len() as u64)
        .and_then(|n| n.checked_mul(64)).and_then(|n| n.checked_add(512 * 1024 * 1024))
        .and_then(|n| n.checked_add(result.checked_mul(4)?))
        .and_then(|n| n.checked_add((limits.max_sample_json_bytes as u64).checked_mul(2)?))
        .and_then(|n| n.checked_add(workspace)).ok_or_else(|| BatchCode::InvalidLimits.into())
}

/// A model-bound compiler plus one immutable vocabulary for the whole stream.
/// The caller's item ID is part of its private semantic sample address.
/// Request sequence, input order and flush epochs never enter the RNG key.
pub struct SampledSentimentBatchPlanner {
    planner: SampledSentimentPlanner,
    vocabulary: Arc<SampledSentimentVocabulary>,
    task_budget: TaskBudget,
    limits: SampledSentimentBatchLimits,
    preparation_bytes: u64,
    seal: Arc<()>,
}
impl SampledSentimentBatchPlanner {
    pub fn new(planner: SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        task_budget: TaskBudget, limits: SampledSentimentBatchLimits) -> Result<Self, BatchFault> {
        limits.validate()?;
        if task_budget != planner.task_budget() || planner.execution_identity().task_spec != "sentiment-v1"
            || planner.execution_identity().sampler_version != JSON_SAMPLER_VERSION {
            return Err(BatchCode::InvalidLimits.into());
        }
        constrained_int4::check_profile(planner.execution_identity(), planner.group_width())
            .map_err(|_| BatchCode::Admission)?;
        let preparation_bytes = sampled_sentiment_batch_preparation_bytes(
            task_budget, planner.limits(), planner.input_encoding())?;
        Ok(Self { planner, vocabulary, task_budget, limits, preparation_bytes, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    pub fn group_width(&self) -> usize { self.planner.group_width() }
    pub fn task_budget(&self) -> TaskBudget { self.task_budget }
    pub fn limits(&self) -> SampledSentimentBatchLimits { self.limits }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.input_encoding() }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }

    fn prepare<C: DecodeStepControl>(&self, document: BatchDocument<SampledSentimentBatchArgs>,
        control: &mut C) -> Result<PreparedSampledSentiment, BatchItemFailure> {
        checkpoint(control)?;
        let args = document.task_args.unwrap_or_default();
        let budget = item_budget(args.budget, self.task_budget)?;
        // Planning owns one cumulative compact allowance for this complete
        // record. The source is encoded once and reused across axes/samples.
        self.planner.plan_for_item_with_control(&SentimentRequest { document: document.text,
            axes: args.axes, budget }, &document.id, control).map_err(task_failure)
    }
}

fn item_budget(requested: Option<TaskBudget>, ceiling: TaskBudget) -> Result<TaskBudget, BatchItemFailure> {
    let budget = requested.unwrap_or(ceiling);
    if budget.validate().is_err() || budget.max_input_tokens > ceiling.max_input_tokens
        || budget.max_output_tokens > ceiling.max_output_tokens || budget.max_output_bytes > ceiling.max_output_bytes
        || budget.max_grammar_states > ceiling.max_grammar_states || budget.max_kv_bytes > ceiling.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Planning));
    }
    Ok(budget)
}

/// Read-only execution facts from the sealed task, never deserialized authority.
pub struct SampledSentimentPlanFacts<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int4Work,
    pub group_width: usize,
    pub sample_count: usize,
    pub required_context: usize,
    pub task_budget: TaskBudget,
    pub max_result_bytes: u64,
    pub preparation_bytes: u64,
}
pub struct SampledSentimentAdmission<'a> {
    pub facts: SampledSentimentPlanFacts<'a>,
    pub kv_reservation_bytes: u64,
    pub mask_node_visits: u64,
}
pub trait SampledSentimentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: SampledSentimentAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

struct PreparedRecord<P> { plan: P, mask_node_visits: u64, seal: Arc<()> }
pub struct PreparedSampledSentimentRecord { record: PreparedRecord<PreparedSampledSentiment> }
impl PreparedSampledSentimentRecord {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.record.plan.execution_identity() }
    pub fn model_work(&self) -> Int4Work { self.record.plan.planned_work() }
    pub fn mask_node_visits(&self) -> u64 { self.record.mask_node_visits }
}

/// One native engine serves every independently reset sample in every record.
/// Unwind, cancellation or an unclean engine permanently closes the adapter.
pub struct NativeSampledSentimentBatch<'e, 'w, A: SampledSentimentBatchAdmission> {
    compiler: SampledSentimentBatchPlanner,
    driver: NativeDriver<'e, 'w>,
    admission: A,
    run: RunState,
}
impl<'e, 'w, A: SampledSentimentBatchAdmission> NativeSampledSentimentBatch<'e, 'w, A> {
    pub fn new(compiler: SampledSentimentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        Self::new_scheduled(compiler, engine, admission, None)
    }
    /// Reuse one admitted layer-major prompt workspace across the whole stream.
    /// The caller must reserve its additional payload before constructing this
    /// adapter. Sample addresses, task budgets and run work remain unchanged.
    pub fn new_layer_major(compiler: SampledSentimentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A, prefill: Int4PrefillLimits) -> Result<Self, BatchFault> {
        Self::new_scheduled(compiler, engine, admission, Some(prefill))
    }
    fn new_scheduled(compiler: SampledSentimentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A, prefill: Option<Int4PrefillLimits>) -> Result<Self, BatchFault> {
        compiler.limits.validate()?;
        validate_prefill(prefill)?;
        constrained_int4::check_model(compiler.base_execution_identity(), engine.model_identity())
            .map_err(|_| BatchCode::Admission)?;
        if compiler.group_width() != engine.model_identity().group_width() || !clean(engine) {
            return Err(BatchCode::InvalidExecution.into());
        }
        let run = RunState::new(compiler.limits);
        let vocabulary = Arc::clone(&compiler.vocabulary);
        Ok(Self { compiler, driver: NativeDriver { engine, vocabulary, prefill }, admission, run })
    }
    pub fn remaining_model_work(&self) -> Int4Work { self.run.remaining }
    pub fn remaining_mask_visits(&self) -> u64 { self.run.remaining_masks }
    pub fn is_poisoned(&self) -> bool { self.run.state != State::Ready || !self.driver.clean() }
}
impl<A: SampledSentimentBatchAdmission> BatchProcessor for NativeSampledSentimentBatch<'_, '_, A> {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedSampledSentimentRecord;
    type Output = GuardedOutput<SampledSentimentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            let plan = self.compiler.prepare(document, control)?;
            let record = prepare_checked(plan, &self.compiler.seal, &self.driver,
                self.compiler.limits, self.compiler.preparation_bytes, control)?;
            self.run.check_available(record.plan.planned_work(), record.mask_node_visits)?;
            Ok(PreparedSampledSentimentRecord { record })
        })();
        settle(result, self.driver.clean(), &mut self.run)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        transport_work(prepared.model_work())
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_owned(prepared.record, &self.compiler.seal, &mut self.driver,
            &mut self.admission, &mut self.run, context, control)
    }
}

// Only these private seams allow synthetic lifecycle fixtures. Public callers
// can neither substitute a planner/native driver nor construct prepared plans.
trait Plan {
    fn facts(&self) -> SampledSentimentPlanFacts<'_>;
    fn verify(&self, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure>;
}
impl Plan for PreparedSampledSentiment {
    fn facts(&self) -> SampledSentimentPlanFacts<'_> {
        SampledSentimentPlanFacts { identity: self.execution_identity(), model_work: self.planned_work(),
            group_width: self.group_width(), sample_count: self.sample_count(),
            required_context: self.required_context(), task_budget: self.task_budget(),
            max_result_bytes: self.max_result_bytes(), preparation_bytes: self.preparation_requirement_bytes() }
    }
    fn verify(&self, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure> {
        self.verify_identity(admitted).map_err(task_failure)
    }
}
trait Driver<P: Plan> {
    fn prefill(&self) -> Option<Int4PrefillLimits> { None }
    fn clean(&self) -> bool;
    fn capacity(&self) -> usize;
    fn group_width(&self) -> usize;
    fn mask_charge(&self, plan: &P, budget: Int4JsonBudget, sampler_bytes: u64) -> Result<u64, BatchItemFailure>;
    fn execute<C: DecodeStepControl>(&mut self, plan: &P, identity: &ExecutionIdentity,
        budget: Int4JsonBudget, sampler_bytes: u64, control: &mut C) -> Result<SampledSentimentRun, BatchItemFailure>;
}
struct NativeDriver<'e, 'w> {
    engine: &'e mut StrictInt4Engine<'w>, vocabulary: Arc<SampledSentimentVocabulary>,
    prefill: Option<Int4PrefillLimits>,
}
fn validate_prefill(prefill: Option<Int4PrefillLimits>) -> Result<(), BatchFault> {
    if let Some(limits) = prefill {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    }
    Ok(())
}
fn clean(engine: &StrictInt4Engine<'_>) -> bool { !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0) }
impl Driver<PreparedSampledSentiment> for NativeDriver<'_, '_> {
    fn prefill(&self) -> Option<Int4PrefillLimits> { self.prefill }
    fn clean(&self) -> bool { clean(self.engine) }
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn group_width(&self) -> usize { self.engine.model_identity().group_width() }
    fn mask_charge(&self, plan: &PreparedSampledSentiment, budget: Int4JsonBudget, sampler_bytes: u64)
        -> Result<u64, BatchItemFailure> {
        plan.complete_mask_charge(&self.vocabulary, budget, sampler_bytes).map_err(task_failure)
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedSampledSentiment, identity: &ExecutionIdentity,
        budget: Int4JsonBudget, sampler_bytes: u64, control: &mut C) -> Result<SampledSentimentRun, BatchItemFailure> {
        match self.prefill {
            Some(prefill) => plan.execute_layer_major_with_control(self.engine, identity, &self.vocabulary,
                budget, sampler_bytes, prefill, control),
            None => plan.execute_with_control(self.engine, identity, &self.vocabulary,
                budget, sampler_bytes, control),
        }.map_err(task_failure)
    }
}

fn preflight<P: Plan, D: Driver<P>>(plan: &P, driver: &D, limits: SampledSentimentBatchLimits)
    -> Result<(u64, u64), BatchItemFailure> {
    validate_prefill(driver.prefill()).map_err(BatchItemFailure::fatal)?;
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let facts = plan.facts();
    if facts.group_width != driver.group_width() || facts.identity.task_spec != "sentiment-v1"
        || facts.identity.sampler_version != JSON_SAMPLER_VERSION || facts.sample_count == 0
        || facts.required_context == 0 || !nonzero(facts.model_work) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let required = Int4MemoryRequirement::for_context(driver.capacity(), driver.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if facts.required_context > driver.capacity() || required.kv_bytes > facts.task_budget.max_kv_bytes
        || !fits(facts.model_work, limits.max_item_work) {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    if facts.max_result_bytes == 0 || facts.max_result_bytes > facts.task_budget.max_output_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    let masks = driver.mask_charge(plan, native_budget(facts.model_work, required.kv_bytes,
        limits.masks.max_visits_per_item, limits), limits.max_sampler_bytes)?;
    if masks == 0 || masks > limits.masks.max_visits_per_item {
        return Err(BatchItemFailure::reject(BatchCode::WorkLimit));
    }
    Ok((required.kv_bytes, masks))
}
fn prepare_checked<P: Plan, D: Driver<P>, C: DecodeStepControl>(plan: P, seal: &Arc<()>,
    driver: &D, limits: SampledSentimentBatchLimits, preparation_bytes: u64, control: &mut C)
    -> Result<PreparedRecord<P>, BatchItemFailure> {
    checkpoint(control)?;
    if plan.facts().preparation_bytes == 0 || plan.facts().preparation_bytes > preparation_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let (_, mask_node_visits) = preflight(&plan, driver, limits)?;
    checkpoint(control)?;
    Ok(PreparedRecord { plan, mask_node_visits, seal: Arc::clone(seal) })
}
#[allow(clippy::too_many_arguments)]
fn execute_owned<P: Plan, D: Driver<P>, A: SampledSentimentBatchAdmission, C: DecodeStepControl>(
    prepared: PreparedRecord<P>, seal: &Arc<()>, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<SampledSentimentRun, A::Guard>, BatchItemFailure> {
    run.enter()?;
    let result = (|| {
        if !Arc::ptr_eq(&prepared.seal, seal) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let facts = prepared.plan.facts();
        // All six allowances debit atomically before an admission guard or
        // native session. Early EOS and every later failure retain this debit.
        run.charge(facts.model_work, prepared.mask_node_visits, context)?;
        checkpoint(control)?;
        let (kv_bytes, mask_node_visits) = preflight(&prepared.plan, driver, run.limits)?;
        if mask_node_visits != prepared.mask_node_visits {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let work = facts.model_work;
        let (identity, guard) = admission.admit(SampledSentimentAdmission {
            facts, kv_reservation_bytes: kv_bytes, mask_node_visits })?;
        prepared.plan.verify(&identity)?;
        checkpoint(control)?;
        let output = driver.execute(&prepared.plan, &identity,
            native_budget(work, kv_bytes, mask_node_visits, run.limits), run.limits.max_sampler_bytes, control)?;
        if !valid_result(&prepared.plan, &output, mask_node_visits) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        checkpoint(control)?;
        Ok(GuardedOutput::new(output, guard))
    })();
    settle(result, driver.clean(), run)
}
fn native_budget(work: Int4Work, kv_bytes: u64, masks: u64, limits: SampledSentimentBatchLimits) -> Int4JsonBudget {
    Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: kv_bytes, max_total_mask_node_visits: masks, mask_limits: limits.masks.per_mask } }
}
fn valid_result<P: Plan>(plan: &P, output: &SampledSentimentRun, masks: u64) -> bool {
    let facts = plan.facts();
    let samples = output.dimensions.iter().try_fold(0_usize, |count, axis| count.checked_add(axis.samples.len()));
    output.schema_version == 1 && output.execution == SAMPLED_SENTIMENT_EXECUTION
        && output.task_spec == "sentiment-v1" && output.sampler_version == JSON_SAMPLER_VERSION
        && output.prompt_template_sha256 == facts.identity.template_digest
        && output.numerics_profile == STRICT_INT4_PROFILE && output.group_width == facts.group_width
        && output.samples_per_axis != 0 && samples == Some(facts.sample_count)
        && output.dimensions.iter().all(|axis| axis.samples.len() == output.samples_per_axis
            && axis.samples.iter().enumerate().all(|(index, sample)| sample.sample_index == index))
        && fits(output.model_work, facts.model_work)
        && output.generated_tokens <= u64::from(facts.task_budget.max_output_tokens)
        && output.mask_node_visit_charge <= masks
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State { Ready, Running, Failed }
struct RunState {
    limits: SampledSentimentBatchLimits, remaining: Int4Work, remaining_masks: u64,
    last_sequence: u64, state: State,
}
impl RunState {
    fn new(limits: SampledSentimentBatchLimits) -> Self {
        Self { limits, remaining: limits.max_model_work, remaining_masks: limits.masks.max_visits_per_run,
            last_sequence: 0, state: State::Ready }
    }
    fn enter(&mut self) -> Result<(), BatchItemFailure> {
        if self.state != State::Ready { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.state = State::Running; Ok(())
    }
    fn check_available(&self, work: Int4Work, masks: u64) -> Result<(), BatchItemFailure> {
        subtract(self.remaining, work)?;
        if masks > self.remaining_masks { return Err(BatchItemFailure::reject(BatchCode::WorkLimit)); }
        Ok(())
    }
    fn charge(&mut self, work: Int4Work, masks: u64, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        if self.state != State::Running || context.request_seq <= self.last_sequence
            || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let remaining = subtract(self.remaining, work).map_err(|e| BatchItemFailure::fatal(e.fault))?;
        let remaining_masks = self.remaining_masks.checked_sub(masks)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.remaining = remaining; self.remaining_masks = remaining_masks;
        self.last_sequence = context.request_seq;
        Ok(())
    }
}
fn settle<T>(result: Result<T, BatchItemFailure>, clean: bool, run: &mut RunState) -> Result<T, BatchItemFailure> {
    match result {
        Ok(value) if clean && run.state == State::Running => { run.state = State::Ready; Ok(value) }
        Ok(_) => { run.state = State::Failed; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) }
        Err(mut error) => {
            if !clean || run.state != State::Running || error.must_stop() { error.stop = true; }
            run.state = if error.stop { State::Failed } else { State::Ready };
            Err(error)
        }
    }
}
fn nonzero(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn subtract(a: Int4Work, b: Int4Work) -> Result<Int4Work, BatchItemFailure> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit));
    Ok(Int4Work { forward_positions: sub(a.forward_positions, b.forward_positions)?,
        projected_logits: sub(a.projected_logits, b.projected_logits)?, attention_pairs: sub(a.attention_pairs, b.attention_pairs)?,
        projections: ProjectionWork { dot_products: sub(a.projections.dot_products, b.projections.dot_products)?,
            multiply_accumulates: sub(a.projections.multiply_accumulates, b.projections.multiply_accumulates)? } })
}
fn transport_work(work: Int4Work) -> BatchWork {
    BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
}
fn task_failure(error: SampledSentimentError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        SampledSentimentError::WorkBudget | SampledSentimentError::TokenizerWork =>
            BatchItemFailure::reject(BatchCode::WorkLimit),
        SampledSentimentError::OutputBudget => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        SampledSentimentError::Limits | SampledSentimentError::Tokenizer
            | SampledSentimentError::Template | SampledSentimentError::Schema => BatchItemFailure::reject(BatchCode::Planning),
        SampledSentimentError::NoResult => BatchItemFailure::reject(BatchCode::Execution),
        SampledSentimentError::Identity => BatchItemFailure::fatal(BatchCode::Admission),
        SampledSentimentError::Accounting => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        SampledSentimentError::Allocation => BatchItemFailure::fatal(BatchCode::Allocation),
        SampledSentimentError::Serialization => BatchItemFailure::fatal(BatchCode::Serialization),
        SampledSentimentError::Native(_) | SampledSentimentError::Cancelled(_) => BatchItemFailure::fatal(BatchCode::Execution),
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}

#[cfg(test)] mod tests;
