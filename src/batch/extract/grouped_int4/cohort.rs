//! Ordered raw-document extraction through one actual native cohort per window.
use super::*;
use std::sync::Arc;
use crate::{batch::grouped::{GroupedBatchProcessor, GroupedOutput, GroupedRequest, GroupedRow},
    native_engine::{int4::PackedInt4Kernel,
        constrained_int4::cohort::{Int4JsonCohortBudget, required_workspace_bytes},
        strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4ModelWeights, Int4RunBudget, StrictInt4Error,
            cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES}}},
    tasks::extract::grouped_int4::{Int4ExtractRun, cohort::{self as extraction,
        Int4ExtractCohortRequest, Int4ExtractCohortBudget, INT4_EXTRACT_COHORT_EXECUTION}}};

#[derive(Clone, Copy, Debug)]
pub struct Int4ExtractionCohortBatchLimits {
    pub max_documents: usize,
    pub context_limit: usize,
    pub native_memory: Int4MemoryBudget,
    pub max_workspace_bytes: u64,
    pub max_item_result_bytes: u64,
    pub max_result_bytes: u64,
}
impl Int4ExtractionCohortBatchLimits {
    pub fn validate(self, width: usize) -> Result<(), BatchFault> {
        if !(1..=MAX_INT4_SEQUENCES).contains(&self.max_documents) || self.max_workspace_bytes == 0
            || self.max_item_result_bytes == 0 || self.max_item_result_bytes > self.max_result_bytes
            || self.max_result_bytes > memory::MAX_COHORT_RESULT_BYTES {
            return Err(BatchCode::InvalidLimits.into());
        }
        let contexts = [self.context_limit; MAX_INT4_SEQUENCES];
        let required = Int4MemoryRequirement::for_cohort(&contexts[..self.max_documents], width)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if !memory_fits(required, self.native_memory) { return Err(BatchCode::InvalidLimits.into()); }
        Ok(())
    }
}

/// Complete native work, separate from conservative transport reservations.
/// Group counts measure execution structure, never elapsed time or throughput.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4ExtractionCohortBatchStats {
    pub cohorts: u64,
    pub documents: u64,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}

/// A model-free prepared item with its unforgeable in-process factory seal.
/// A wire request cannot supply a replacement plan, identity or result.
pub struct PreparedInt4CohortExtraction { prepared: PreparedInt4BatchExtraction, seal: Arc<()> }
impl PreparedInt4CohortExtraction {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.prepared.execution_identity() }
    pub fn planned_work(&self) -> Int4Work { self.prepared.planned_work() }
}

pub struct NativeInt4ExtractionCohortBatch<'w, 'v, A: Int4ExtractionBatchAdmission> {
    compiler: Int4ExtractionBatchPlanner,
    weights: &'w Int4ModelWeights,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    work: Int4ExtractionBatchLimits,
    limits: Int4ExtractionCohortBatchLimits,
    kernel: PackedInt4Kernel,
    seal: Arc<()>,
    account: Account,
    last_context: Option<BatchRequestContext>,
    failed: bool,
    statistics: Int4ExtractionCohortBatchStats,
}
impl<'w, 'v, A: Int4ExtractionBatchAdmission> NativeInt4ExtractionCohortBatch<'w, 'v, A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(compiler: Int4ExtractionBatchPlanner, weights: &'w Int4ModelWeights,
        vocabulary: &'v ExtractionVocabulary, admission: A, work: Int4ExtractionBatchLimits,
        limits: Int4ExtractionCohortBatchLimits, kernel: PackedInt4Kernel) -> Result<Self, BatchFault> {
        work.validate()?; limits.validate(compiler.group_width())?;
        crate::native_engine::constrained_int4::check_model(compiler.base_execution_identity(), weights.identity())
            .map_err(|_| BatchCode::Admission)?;
        if compiler.group_width() != weights.group_width() { return Err(BatchCode::Admission.into()); }
        Ok(Self { compiler, weights, vocabulary, admission, work, limits, kernel, seal: Arc::new(()),
            account: Account::default(), last_context: None, failed: false,
            statistics: Int4ExtractionCohortBatchStats::default() })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.account.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.account.masks }
    pub fn is_poisoned(&self) -> bool { self.failed }
    pub fn cohort_stats(&self) -> Int4ExtractionCohortBatchStats { self.statistics }

    fn execute<C: DecodeStepControl>(&mut self, requests: Vec<GroupedRequest<PreparedInt4CohortExtraction>>,
        control: &mut C) -> Result<GroupedOutput<Int4ExtractRun, Vec<A::Guard>>, BatchFault> {
        validate_contexts(&requests, self.last_context, self.limits.max_documents)?;
        checkpoint(control).map_err(|error| error.fault)?;
        // All result storage declared later drops before these output claims.
        // Transport receives every guard together, through the final flush.
        let mut guards = reserve(requests.len())?;
        let mut identities = reserve(requests.len())?;
        let mut contexts = reserve(requests.len())?;
        let mut work = Int4Work::default(); let mut tokens = 0_u64;
        for request in &requests {
            if !Arc::ptr_eq(&request.prepared.seal, &self.seal) { return Err(BatchCode::Admission.into()); }
            let prepared = &request.prepared.prepared;
            let (context, kv) = price_plan(prepared, self.limits).map_err(|error| error.fault)?;
            let row = prepared.planned_work();
            work = work.checked_add(row).map_err(native_error)?;
            tokens = tokens.checked_add(prepared.plan.options().max_new_tokens as u64).ok_or(BatchCode::Admission)?;
            let (identity, guard) = self.admission.admit(Int4ExtractionAdmission {
                identity: prepared.execution_identity(), model_work: row,
                mask_node_visits: self.work.masks.max_visits_per_item, mask_limits: self.work.masks.per_mask,
                kv_reservation_bytes: kv, max_result_bytes: prepared.plan.max_result_bytes(),
            }).map_err(|error| error.fault)?;
            guards.push(guard);
            prepared.verify_identity(&identity).map_err(|_| BatchCode::Admission)?;
            identities.push(identity); contexts.push(context);
        }
        let required = Int4MemoryRequirement::for_cohort(&contexts, self.compiler.group_width()).map_err(native_error)?;
        if !memory_fits(required, self.limits.native_memory) { return Err(BatchCode::Admission.into()); }
        let workspace = required_workspace_bytes(requests.len(), tokens, self.limits.max_result_bytes)
            .map_err(|_| BatchCode::Admission)?;
        if workspace > self.limits.max_workspace_bytes { return Err(BatchCode::Admission.into()); }
        let mask_visits = self.work.masks.max_visits_per_item.checked_mul(requests.len() as u64)
            .filter(|&count| count <= self.work.masks.max_visits_per_run).ok_or(BatchCode::WorkLimit)?;
        let mut rows = reserve(requests.len())?;
        for ((request, identity), &context) in requests.iter().zip(&identities).zip(&contexts) {
            let prepared = &request.prepared.prepared;
            let mut budget = native::native_budget(prepared, self.work.masks);
            budget.json.max_kv_bytes = (context as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(BatchCode::Admission)?;
            let row = Int4ExtractCohortRequest { prepared: &prepared.plan, admitted_identity: identity, budget };
            extraction::preflight(&row, self.vocabulary).map_err(|error| native::execution_failure(error).fault)?;
            rows.push(row);
        }
        checkpoint(control).map_err(|error| error.fault)?;
        let mut engine = Int4CohortEngine::new(self.weights, &contexts, self.limits.native_memory)
            .map_err(native_error)?.with_kernel(self.kernel);
        let raw = extraction::execute(&mut engine, &rows, self.vocabulary, Int4ExtractCohortBudget {
            decode: Int4JsonCohortBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes,
                max_mask_node_visits: mask_visits, max_result_bytes: self.limits.max_result_bytes },
            max_result_bytes: self.limits.max_result_bytes,
        }, control).map_err(|error| native::execution_failure(error).fault)?;
        engine.check_idle().map_err(native_error)?; drop(engine);
        if raw.schema_version != 1 || raw.execution != INT4_EXTRACT_COHORT_EXECUTION
            || raw.sequences.len() != requests.len() || raw.planned_work != work {
            return Err(BatchCode::InvalidExecution.into());
        }
        for (request, result) in requests.iter().zip(&raw.sequences) {
            if !native::valid_result(&request.prepared.prepared, result, self.work.masks.max_visits_per_item) {
                return Err(BatchCode::InvalidExecution.into());
            }
        }
        checkpoint(control).map_err(|error| error.fault)?;
        let statistics = add_statistics(self.statistics, requests.len(), raw.group_steps,
            raw.projection_groups, raw.planned_work, raw.model_work)?;
        let mut output = reserve(requests.len())?;
        for (request, value) in requests.iter().zip(raw.sequences) {
            output.push(GroupedRow { request_seq: request.context.request_seq, result: Ok(value) });
        }
        self.last_context = requests.last().map(|request| request.context);
        self.statistics = statistics;
        Ok(GroupedOutput::new(output, guards))
    }
}
impl<A: Int4ExtractionBatchAdmission> GroupedBatchProcessor for NativeInt4ExtractionCohortBatch<'_, '_, A> {
    type Args = ExtractionBatchArgs;
    type Prepared = PreparedInt4CohortExtraction;
    type Output = Int4ExtractRun;
    type Guard = Vec<A::Guard>;
    fn max_group_size(&self) -> usize { self.limits.max_documents }
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        if self.failed { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.failed = true;
        let result = prepare_document(&self.compiler, document, self.vocabulary, self.limits,
            self.work, &mut self.account, control)
            .map(|prepared| PreparedInt4CohortExtraction { prepared, seal: Arc::clone(&self.seal) });
        if !result.as_ref().is_err_and(|error| error.stop) { self.failed = false; }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> Result<BatchWork, BatchItemFailure> {
        if self.failed || !Arc::ptr_eq(&prepared.seal, &self.seal) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let work = prepared.planned_work();
        Ok(BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits })
    }
    fn execute_group<C: DecodeStepControl>(&mut self, requests: Vec<GroupedRequest<Self::Prepared>>, control: &mut C)
        -> Result<GroupedOutput<Self::Output, Self::Guard>, BatchFault> {
        if self.failed { return Err(BatchCode::InvalidExecution.into()); }
        self.failed = true;
        let result = self.execute(requests, control);
        if result.is_ok() { self.failed = false; }
        result
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Account { work: Int4Work, masks: u64 }
impl Account {
    fn reserve(&mut self, work: Int4Work, limits: Int4ExtractionBatchLimits) -> Result<(), BatchItemFailure> {
        let total = self.work.checked_add(work).ok().filter(|&value| fits(value, limits.max_model_work))
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit))?;
        let masks = self.masks.checked_add(limits.masks.max_visits_per_item)
            .filter(|&value| value <= limits.masks.max_visits_per_run)
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::WorkLimit))?;
        self.work = total; self.masks = masks; Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
fn prepare_document<C: DecodeStepControl>(compiler: &Int4ExtractionBatchPlanner,
    document: BatchDocument<ExtractionBatchArgs>, vocabulary: &ExtractionVocabulary,
    limits: Int4ExtractionCohortBatchLimits, work: Int4ExtractionBatchLimits,
    account: &mut Account, control: &mut C) -> Result<PreparedInt4BatchExtraction, BatchItemFailure> {
    let prepared = compiler.prepare_with_control(document, control)?;
    let (_, kv) = price_plan(&prepared, limits)?;
    let mut budget = native::native_budget(&prepared, work.masks); budget.json.max_kv_bytes = kv;
    extraction::preflight(&Int4ExtractCohortRequest { prepared: &prepared.plan,
        admitted_identity: prepared.execution_identity(), budget }, vocabulary).map_err(native::execution_failure)?;
    account.reserve(prepared.planned_work(), work)?;
    checkpoint(control)?;
    Ok(prepared)
}
fn price_plan(prepared: &PreparedInt4BatchExtraction, limits: Int4ExtractionCohortBatchLimits)
    -> Result<(usize, u64), BatchItemFailure> {
    let context = usize::try_from(prepared.planned_work().forward_positions)
        .map_err(|_| BatchItemFailure::reject(BatchCode::Admission))?;
    if context == 0 || context > limits.context_limit { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
    let required = Int4MemoryRequirement::for_context(context, prepared.plan.group_width())
        .map_err(|_| BatchItemFailure::reject(BatchCode::Admission))?;
    if required.kv_bytes > prepared.plan.max_kv_bytes() || required.kv_bytes > limits.native_memory.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    if prepared.plan.max_result_bytes() == 0 || prepared.plan.max_result_bytes() > limits.max_item_result_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok((context, required.kv_bytes))
}
fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn memory_fits(required: Int4MemoryRequirement, budget: Int4MemoryBudget) -> bool {
    required.kv_bytes <= budget.max_kv_bytes && required.rope_bytes <= budget.max_rope_bytes
        && required.scratch_payload_bound <= budget.max_scratch_payload_bytes
}
fn validate_contexts<T>(requests: &[GroupedRequest<T>], previous: Option<BatchRequestContext>, maximum: usize)
    -> Result<(), BatchFault> {
    if requests.is_empty() || requests.len() > maximum { return Err(BatchCode::InvalidExecution.into()); }
    let epoch = requests[0].context.epoch; let mut previous = previous;
    for request in requests {
        let context = request.context;
        if context.epoch == 0 || context.epoch != epoch || context.request_seq == 0 || context.input_line == 0
            || previous.is_some_and(|last| context.epoch < last.epoch || context.request_seq <= last.request_seq
                || context.input_line <= last.input_line || context.byte_offset <= last.byte_offset) {
            return Err(BatchCode::InvalidExecution.into());
        }
        previous = Some(context);
    }
    Ok(())
}
fn add_statistics(previous: Int4ExtractionCohortBatchStats, documents: usize, steps: u64, projections: u64,
    planned: Int4Work, actual: Int4Work) -> Result<Int4ExtractionCohortBatchStats, BatchFault> {
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(BatchCode::SequenceOverflow);
    Ok(Int4ExtractionCohortBatchStats { cohorts: add(previous.cohorts, 1)?,
        documents: add(previous.documents, documents as u64)?, group_steps: add(previous.group_steps, steps)?,
        projection_groups: add(previous.projection_groups, projections)?,
        planned_work: previous.planned_work.checked_add(planned).map_err(native_error)?,
        model_work: previous.model_work.checked_add(actual).map_err(native_error)? })
}
fn native_error(error: StrictInt4Error) -> BatchFault {
    match error {
        StrictInt4Error::Cancelled(cause) => BatchFault::cancelled(cause),
        StrictInt4Error::Allocation => BatchCode::Allocation.into(),
        StrictInt4Error::Context | StrictInt4Error::Memory => BatchCode::Admission.into(),
        StrictInt4Error::Work => BatchCode::WorkLimit.into(),
        _ => BatchCode::InvalidExecution.into(),
    }
}
fn reserve<T>(count: usize) -> Result<Vec<T>, BatchFault> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| BatchCode::Allocation)?; Ok(values)
}

#[cfg(test)] mod tests;
