//! Finite prepared extraction/source collections under one physical owner.
use super::*;
use crate::{native_engine::{decode::DecodeStepControl, constrained::JsonWorkBudget,
    constrained_int4::{self, Int4JsonBudget, cohort::{Int4JsonCohortBudget, required_workspace_bytes}},
    strict_int4::{Int4Work, cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES}}},
    tasks::{extract::grouped_int4::{Int4ExtractPlan, Int4ExtractError,
        cohort::{self as extraction_group, Int4ExtractCohortRequest, Int4ExtractCohortBudget, Int4ExtractCohortRun}},
        source_planning::grouped_int4::{PreparedInt4SourceTask, Int4SourceError,
            cohort::{self as source_group, Int4SourceCohortRequest, Int4SourceCohortRun}}}};

impl NlpEngine {
    /// Execute complete schema/source plans through one dense native cohort.
    /// The preparation reserve covers ALL transferred plans/vocabulary and
    /// their variable grammar/source runtime state. Context and mask ceilings
    /// are per row; one deadline/checkpoint quota governs the whole collection.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_extract_cohort(&self, model: &ResidentInt4, prepared: Vec<Int4ExtractPlan>,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, max_result_bytes: u64,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ExtractCohortRun>, HostedError> {
        execute(self, model, prepared, vocabulary, limits, max_result_bytes, kernel, cancellation)
    }

    /// NER, keyphrases, summaries and passage QA retain their independent source
    /// languages and original semantic finalizers. Mixed source-task kinds are
    /// allowed; a failed later finalizer returns no partial collection.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_source_cohort(&self, model: &ResidentInt4, prepared: Vec<PreparedInt4SourceTask>,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, max_result_bytes: u64,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SourceCohortRun>, HostedError> {
        execute(self, model, prepared, vocabulary, limits, max_result_bytes, kernel, cancellation)
    }
}

struct Quote { work: Int4Work, output_bytes: u64, output_tokens: u64, max_kv_bytes: u64, group_width: usize }
trait StructuredPlan: Sized + Send + 'static {
    type Output: Send + 'static;
    fn identity(&self) -> &ExecutionIdentity;
    fn quote(&self) -> Result<Quote, HostedError>;
    fn preflight(&self, context: usize, vocabulary: &ExtractionVocabulary, limits: SourceLimits)
        -> Result<(), HostedError>;
    fn run<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>, plans: &[Self], contexts: &[usize],
        vocabulary: &ExtractionVocabulary, limits: SourceLimits, budget: Int4ExtractCohortBudget,
        control: &mut C) -> Result<Self::Output, HostedError>;
}
impl StructuredPlan for Int4ExtractPlan {
    type Output = Int4ExtractCohortRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn quote(&self) -> Result<Quote, HostedError> {
        self.verify_identity(self.execution_identity()).map_err(extract_error)?;
        Ok(Quote { work: self.planned_work(), output_bytes: self.max_result_bytes(),
            output_tokens: self.options().max_new_tokens as u64, max_kv_bytes: self.max_kv_bytes(),
            group_width: self.group_width() })
    }
    fn preflight(&self, context: usize, vocabulary: &ExtractionVocabulary, limits: SourceLimits)
        -> Result<(), HostedError> {
        extraction_group::preflight(&Int4ExtractCohortRequest { prepared: self,
            admitted_identity: self.execution_identity(), budget: row_budget(self.planned_work(), context, limits)? },
            vocabulary).map_err(extract_error)
    }
    fn run<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>, plans: &[Self], contexts: &[usize],
        vocabulary: &ExtractionVocabulary, limits: SourceLimits, budget: Int4ExtractCohortBudget,
        control: &mut C) -> Result<Self::Output, HostedError> {
        let mut requests = reserved(plans.len())?;
        for (slot, plan) in plans.iter().enumerate() {
            requests.push(Int4ExtractCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
                budget: row_budget(plan.planned_work(), contexts[slot], limits)? });
        }
        extraction_group::execute(engine, &requests, vocabulary, budget, control).map_err(extract_error)
    }
}
impl StructuredPlan for PreparedInt4SourceTask {
    type Output = Int4SourceCohortRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn quote(&self) -> Result<Quote, HostedError> {
        self.verify_identity(self.execution_identity()).map_err(source_error)?;
        Ok(Quote { work: self.planned_work(), output_bytes: self.max_result_bytes(),
            output_tokens: u64::from(self.task_budget().max_output_tokens), max_kv_bytes: self.max_kv_bytes(),
            group_width: self.group_width() })
    }
    fn preflight(&self, context: usize, vocabulary: &ExtractionVocabulary, limits: SourceLimits)
        -> Result<(), HostedError> {
        source_group::preflight(&Int4SourceCohortRequest { prepared: self,
            admitted_identity: self.execution_identity(), budget: row_budget(self.planned_work(), context, limits)? },
            vocabulary).map_err(source_error)
    }
    fn run<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>, plans: &[Self], contexts: &[usize],
        vocabulary: &ExtractionVocabulary, limits: SourceLimits, budget: Int4ExtractCohortBudget,
        control: &mut C) -> Result<Self::Output, HostedError> {
        let mut requests = reserved(plans.len())?;
        for (slot, plan) in plans.iter().enumerate() {
            requests.push(Int4SourceCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
                budget: row_budget(plan.planned_work(), contexts[slot], limits)? });
        }
        source_group::execute(engine, &requests, vocabulary, budget, control).map_err(source_error)
    }
}
struct Pricing { contexts: Vec<usize>, work: Int4Work, output_tokens: u64 }
struct Input<P> { plans: Vec<P>, vocabulary: Arc<ExtractionVocabulary>, pricing: Pricing }

#[allow(clippy::too_many_arguments)]
fn execute<P: StructuredPlan>(host: &NlpEngine, model: &ResidentInt4, plans: Vec<P>,
    vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, cap: u64,
    kernel: PackedInt4Kernel, cancellation: CancellationToken) -> Result<HostedOutput<P::Output>, HostedError> {
    dispatch::preflight(host, limits.native.run)?;
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    validate(limits, cap, plans.len())?;
    for plan in &plans {
        constrained_int4::check_model(plan.identity(), model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
    }
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.preparation_reserve_bytes)?, || {
        let pricing = price(plans.iter().map(P::quote), plans.len(), model.model_identity().group_width(), limits, cap)?;
        for (plan, &context) in plans.iter().zip(&pricing.contexts) {
            plan.preflight(context, &vocabulary, limits)?;
        }
        Ok(Input { plans, vocabulary, pricing })
    })?;
    let required = Int4MemoryRequirement::for_cohort(&input.value.pricing.contexts, model.model_identity().group_width())
        .map_err(native_error)?;
    let decode_workspace = required_workspace_bytes(input.value.plans.len(), input.value.pricing.output_tokens, cap)
        .map_err(|error| extract_error(error.into()))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
        sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes, decode_workspace])?)?;
    let output = output_claim(&lease, cap, input.value.pricing.output_tokens)?;
    let model = model.clone();
    dispatch::run(host, limits.native.run, cancellation, move |control| {
        let input = input;
        let mut engine = allocate_native(kv, scratch, || {
            Int4CohortEngine::new(&model.inner.loaded.value, &input.value.pricing.contexts,
                Int4MemoryBudget { max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound })
                .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
        })?;
        let budget = Int4ExtractCohortBudget { decode: Int4JsonCohortBudget {
            native: Int4RunBudget::exact(input.value.pricing.work), max_kv_bytes: required.kv_bytes,
            max_mask_node_visits: aggregate_masks(limits.max_mask_node_visits, input.value.plans.len())?,
            max_result_bytes: cap }, max_result_bytes: cap };
        let result = allocate(output, || {
            let result = P::run(&mut engine.value, &input.value.plans, &input.value.pricing.contexts,
                &input.value.vocabulary, limits, budget, control)?;
            engine.value.check_idle().map_err(native_error)?;
            Ok(result)
        })?;
        drop(engine); drop(input); drop(lease);
        Ok(GuardedOutput::new(result.value, result._memory))
    })
}
fn validate(limits: SourceLimits, cap: u64, count: usize) -> Result<(), HostedError> {
    limits.native.run.validate()?;
    if count == 0 || count > MAX_INT4_SEQUENCES || cap == 0
        || cap > crate::batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES
        || limits.preparation_reserve_bytes == 0 || limits.native.allocator_reserve_bytes == 0
        || limits.mask_limits.max_trie_node_visits == 0 || limits.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(HostedError::Limits("INT4 structured cohort resources"));
    }
    aggregate_masks(limits.max_mask_node_visits, count)?; Ok(())
}
fn aggregate_masks(per_row: u64, count: usize) -> Result<u64, HostedError> {
    per_row.checked_mul(count as u64).filter(|&n| n != 0).ok_or(HostedError::Limits("INT4 cohort mask arithmetic"))
}
fn price(quotes: impl Iterator<Item = Result<Quote, HostedError>>, count: usize, width: usize,
    limits: SourceLimits, cap: u64) -> Result<Pricing, HostedError> {
    validate(limits, cap, count)?;
    Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    let mut contexts = reserved(count)?; let mut work = Int4Work::default(); let mut output_tokens = 0;
    let mut output_bytes = sum(&[4096, count as u64])?;
    for quote in quotes {
        if contexts.len() >= count { return Err(HostedError::Limits("INT4 cohort row count")); }
        let quote = quote?;
        let context = usize::try_from(quote.work.forward_positions)
            .map_err(|_| HostedError::Limits("INT4 cohort context arithmetic"))?;
        if context == 0 || context > limits.native.context_tokens || quote.group_width != width
            || quote.output_tokens == 0 || quote.output_tokens > context as u64 || quote.output_bytes == 0 {
            return Err(HostedError::Limits("INT4 cohort row bounds"));
        }
        let memory = Int4MemoryRequirement::for_context(context, width).map_err(native_error)?;
        if memory.kv_bytes > quote.max_kv_bytes { return Err(HostedError::Limits("INT4 cohort task KV authority")); }
        contexts.push(context); work = work.checked_add(quote.work).map_err(native_error)?;
        output_tokens = sum(&[output_tokens, quote.output_tokens])?;
        output_bytes = sum(&[output_bytes, quote.output_bytes])?;
    }
    if contexts.len() != count || output_bytes > cap { return Err(HostedError::Limits("all retained INT4 structured results")); }
    Ok(Pricing { contexts, work, output_tokens })
}
fn row_budget(work: Int4Work, context: usize, limits: SourceLimits) -> Result<Int4JsonBudget, HostedError> {
    Ok(Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: (context as u64).checked_mul(crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64)
            .ok_or(HostedError::Limits("INT4 row KV arithmetic"))?,
        max_total_mask_node_visits: limits.max_mask_node_visits, mask_limits: limits.mask_limits } })
}
fn extract_error(error: Int4ExtractError) -> HostedError { HostedError::Int4(Int4HostError::Extraction(error)) }
fn source_error(error: Int4SourceError) -> HostedError { HostedError::Int4(Int4HostError::SourceTask(error)) }
fn reserved<T>(count: usize) -> Result<Vec<T>, HostedError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| native_error(StrictInt4Error::Allocation))?; Ok(values)
}

#[cfg(test)] mod tests;
