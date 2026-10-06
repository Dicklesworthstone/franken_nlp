//! One process-owned admission/drain path for both structured task families.
use super::super::*;
use crate::{native_engine::{decode::DecodeStepControl, portable_int8::batch::MAX_BATCH_ROWS,
    strict_int8::{Int8Work, cohort::Int8CohortEngine}, constrained::JsonWorkBudget,
    constrained_int8::sparse::cohort::Int8JsonCohortBudget},
    tasks::{extract::quantized::cohort::{self as extraction_group, Int8ExtractCohortRequest,
        Int8ExtractCohortBudget, Int8ExtractCohortRun},
        source_planning::quantized::{PreparedInt8SourceTask,
            cohort::{self as source_group, Int8SourceCohortRequest, Int8SourceCohortRun}}}};

impl NlpEngine {
    /// Run explicit selected-row extraction plans through shared decoder layers.
    /// native.context_tokens is PER ROW, preparation_reserve_bytes covers ALL
    /// transferred plans/vocabulary, and max_mask_node_visits is a per-row bound.
    /// One deadline/checkpoint quota covers the whole invocation. The complete
    /// retained output is separately admitted by max_result_bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_extract_cohort(&self, model: &ResidentInt8, prepared: Vec<Int8ExtractPlan>,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, max_result_bytes: u64,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ExtractCohortRun>, HostedError> {
        execute(self, model, prepared, vocabulary, limits, max_result_bytes, cancellation)
    }
    /// NER, keyphrases, summaries and passage QA with the same admission and
    /// shared-layer execution. Every semantic finalizer must succeed before any
    /// result returns. Different rows may use different source-task kinds.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_source_cohort(&self, model: &ResidentInt8, prepared: Vec<PreparedInt8SourceTask>,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, max_result_bytes: u64,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8SourceCohortRun>, HostedError> {
        execute(self, model, prepared, vocabulary, limits, max_result_bytes, cancellation)
    }
}

struct Quote { work: Int8Work, output_bytes: u64, output_tokens: u64, max_kv_bytes: u64 }
// Closed static dispatch: no external admission provider or result factory.
trait StructuredPlan: Sized + Send + 'static {
    type Output: Send + 'static;
    fn identity(&self) -> &ExecutionIdentity;
    fn quote(&self) -> Result<Quote, HostedError>;
    fn run<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>, plans: &[Self], contexts: &[usize],
        vocabulary: &ExtractionVocabulary, limits: SourceLimits, budget: Int8ExtractCohortBudget,
        control: &mut C) -> Result<Self::Output, HostedError>;
}
impl StructuredPlan for Int8ExtractPlan {
    type Output = Int8ExtractCohortRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn quote(&self) -> Result<Quote, HostedError> {
        self.verify_identity(self.execution_identity()).map_err(HostedError::Extraction)?;
        require_selected(self.selected_rows().is_some())?;
        Ok(Quote { work: self.planned_work(), output_bytes: self.max_result_bytes(),
            output_tokens: self.options().max_new_tokens as u64, max_kv_bytes: self.max_kv_bytes() })
    }
    fn run<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>, plans: &[Self], contexts: &[usize],
        vocabulary: &ExtractionVocabulary, limits: SourceLimits, budget: Int8ExtractCohortBudget,
        control: &mut C) -> Result<Self::Output, HostedError> {
        let mut requests = reserved(plans.len())?;
        for (slot, plan) in plans.iter().enumerate() {
            requests.push(Int8ExtractCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
                budget: row_budget(plan.planned_work(), contexts[slot], limits)? });
        }
        extraction_group::execute(engine, &requests, vocabulary, budget, control).map_err(HostedError::Extraction)
    }
}
impl StructuredPlan for PreparedInt8SourceTask {
    type Output = Int8SourceCohortRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn quote(&self) -> Result<Quote, HostedError> {
        self.verify_identity(self.execution_identity()).map_err(HostedError::Source)?;
        require_selected(self.selected_rows().is_some())?;
        Ok(Quote { work: self.planned_work(), output_bytes: self.max_result_bytes(),
            output_tokens: u64::from(self.task_budget().max_output_tokens), max_kv_bytes: self.task_budget().max_kv_bytes })
    }
    fn run<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>, plans: &[Self], contexts: &[usize],
        vocabulary: &ExtractionVocabulary, limits: SourceLimits, budget: Int8ExtractCohortBudget,
        control: &mut C) -> Result<Self::Output, HostedError> {
        let mut requests = reserved(plans.len())?;
        for (slot, plan) in plans.iter().enumerate() {
            requests.push(Int8SourceCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
                budget: row_budget(plan.planned_work(), contexts[slot], limits)? });
        }
        source_group::execute(engine, &requests, vocabulary, budget, control).map_err(HostedError::Source)
    }
}
struct Pricing { contexts: Vec<usize>, work: Int8Work, output_tokens: u64 }
struct Input<P> { prepared: Vec<P>, vocabulary: Arc<ExtractionVocabulary>, pricing: Pricing }

#[allow(clippy::too_many_arguments)]
fn execute<P: StructuredPlan>(host: &NlpEngine, model: &ResidentInt8, prepared: Vec<P>,
    vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, cap: u64, cancellation: CancellationToken)
    -> Result<HostedOutput<P::Output>, HostedError> {
    dispatch::preflight(host, limits.native.run)?;
    host.check_resident_domain(model)?;
    validate(limits, cap, prepared.len())?;
    for plan in &prepared { check_model_identity(model.artifact_identity(), plan.identity())?; }
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.preparation_reserve_bytes)?, || {
        let pricing = price(prepared.iter().map(P::quote), prepared.len(), limits, cap)?;
        Ok(Input { prepared, vocabulary, pricing })
    })?;
    let required = Int8MemoryRequirement::for_cohort(&input.value.pricing.contexts).map_err(HostedError::Native)?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
        sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
    let output = output_claim(&lease, cap, input.value.pricing.output_tokens)?;
    let model = model.clone();
    dispatch::run(host, limits.native.run, cancellation, move |control| {
        let input = input; // capture the charged aggregate, not disjoint fields
        let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
            .cohort_engine(&input.value.pricing.contexts, memory_budget(required)).map_err(HostedError::Model))?;
        let budget = Int8ExtractCohortBudget { decode: Int8JsonCohortBudget {
            native: Int8RunBudget::exact(input.value.pricing.work), max_kv_bytes: required.kv_bytes,
            max_mask_node_visits: aggregate_masks(limits.max_mask_node_visits, input.value.prepared.len())?,
            max_result_bytes: cap }, max_result_bytes: cap };
        // Source/task/envelope finalizers execute inside the native session.
        // A post-result commit failure drops result storage before its charge.
        let result = allocate(output, || P::run(&mut engine.value, &input.value.prepared,
            &input.value.pricing.contexts, &input.value.vocabulary, limits, budget, control))?;
        drop(engine); drop(input); drop(lease);
        Ok(GuardedOutput::new(result.value, result._memory))
    })
}
fn require_selected(selected: bool) -> Result<(), HostedError> {
    if selected { Ok(()) } else { Err(HostedError::Limits("explicit selected-row plan required")) }
}
fn aggregate_masks(per_row: u64, count: usize) -> Result<u64, HostedError> {
    per_row.checked_mul(count as u64).filter(|&n| n != 0).ok_or(HostedError::Limits("cohort mask arithmetic"))
}
fn validate(limits: SourceLimits, cap: u64, count: usize) -> Result<(), HostedError> {
    limits.native.run.validate()?;
    if count == 0 || count > MAX_BATCH_ROWS || cap == 0 || limits.preparation_reserve_bytes == 0
        || limits.native.allocator_reserve_bytes == 0 || limits.mask_limits.max_trie_node_visits == 0
        || limits.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(HostedError::Limits("structured cohort resources"));
    }
    aggregate_masks(limits.max_mask_node_visits, count)?;
    Int8MemoryRequirement::for_context(limits.native.context_tokens).map_err(HostedError::Native)?; Ok(())
}
fn price(quotes: impl Iterator<Item = Result<Quote, HostedError>>, count: usize,
    limits: SourceLimits, cap: u64) -> Result<Pricing, HostedError> {
    validate(limits, cap, count)?;
    let mut contexts = reserved(count)?; let mut work = Int8Work::default(); let mut output_tokens = 0;
    let mut output_bytes = sum(&[4096, count as u64])?;
    for quote in quotes {
        if contexts.len() >= count { return Err(HostedError::Limits("cohort row count")); }
        let quote = quote?;
        let context = usize::try_from(quote.work.forward_positions).map_err(|_| HostedError::Limits("cohort context arithmetic"))?;
        if context == 0 || context > limits.native.context_tokens { return Err(HostedError::Limits("cohort row context")); }
        let memory = Int8MemoryRequirement::for_context(context).map_err(HostedError::Native)?;
        if memory.kv_bytes > quote.max_kv_bytes { return Err(HostedError::Limits("cohort task KV authority")); }
        contexts.push(context); work = work.checked_add(quote.work).map_err(HostedError::Native)?;
        output_tokens = sum(&[output_tokens, quote.output_tokens])?;
        output_bytes = sum(&[output_bytes, quote.output_bytes])?;
    }
    if contexts.len() != count || output_bytes > cap { return Err(HostedError::Limits("all retained structured results")); }
    Ok(Pricing { contexts, work, output_tokens })
}
fn row_budget(work: Int8Work, context: usize, limits: SourceLimits) -> Result<Int8JsonBudget, HostedError> {
    Ok(Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: (context as u64).checked_mul(crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64)
            .ok_or(HostedError::Limits("row KV arithmetic"))?,
        max_total_mask_node_visits: limits.max_mask_node_visits, mask_limits: limits.mask_limits } })
}
fn reserved<T>(count: usize) -> Result<Vec<T>, HostedError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| HostedError::Native(StrictInt8Error::Allocation))?;
    Ok(values)
}
#[cfg(test)] mod tests;
