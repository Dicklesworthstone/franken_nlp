//! Independently selected discovery and pair cohorts over one complete entity
//! snapshot. Native phases drain in order; source, graph and result ownership
//! remain with the caller until the whole snapshot completes.
use super::*;
use crate::{
    native_engine::{
        constrained_int4::cohort::Int4JsonCohortBudget,
        int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4ModelWeights,
            cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES}},
    },
    tasks::{extract::grouped_int4::cohort::Int4ExtractCohortBudget,
        source_planning::grouped_int4::{
            cohort::{self as source_group, Int4SourceCohortRequest, Int4SourceCohortRun,
                INT4_SOURCE_COHORT_EXECUTION},
            long::cohort::Int4SourceMapCohortMemoryLimits,
        }},
    corpus::native_resolve::grouped_int4::cohort::{Int4ResolutionCohortBudget,
        Int4ResolutionCohortMemoryLimits, INT4_RESOLUTION_COHORT_EXECUTION},
};

pub const INT4_ENTITY_COHORT_EXECUTION: &str = "grouped-int4-entity-discovery-pair-cohorts-v1";
const ENVELOPE_BYTES: u64 = 4096;

/// Native scheduling only. Neither width changes a source task, pair language,
/// graph policy, single-head KV authority or whole-snapshot work allowance.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Int4EntitySchedule {
    pub ner_cohort_chunks: Option<usize>,
    pub pair_cohort_pairs: Option<usize>,
}

/// Complete declared phase peak. Retained source/witness input, the original
/// one-plan preparation, graph/collector storage, output staging and allocator
/// slack remain separate owners; this is a payload quote, not measured RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4EntityCohortRequirement {
    pub native: Int4MemoryRequirement,
    pub additional_preparation_bytes: u64,
    pub workspace_bytes: u64,
    pub max_ner_result_bytes: u64,
    pub max_pair_result_bytes: u64,
    pub max_result_bytes: u64,
}

/// The one-plan preparation declaration is the same input used by the host's
/// pure quote. Additional simultaneous source plans must fit their separately
/// admitted cap. Starting another native phase or window grants no new bytes.
#[derive(Clone, Copy, Debug)]
pub struct Int4EntityCohortBudget {
    pub context_tokens: usize,
    pub native_memory: Int4MemoryBudget,
    pub preparation_reserve_bytes: u64,
    pub max_preparation_bytes: u64,
    pub max_workspace_bytes: u64,
    pub max_result_bytes: u64,
}

/// Physical schedule information is outside the unchanged semantic result.
/// Zero counters identify a phase that retained serial execution or had no
/// rows; the selected schedule remains explicit even for an empty snapshot.
#[derive(Serialize)]
pub struct Int4EntityCohortRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub schedule: Int4EntitySchedule,
    pub ner_cohorts: usize,
    pub ner_group_steps: u64,
    pub ner_projection_groups: u64,
    pub pair_cohorts: usize,
    pub pair_group_steps: u64,
    pub pair_projection_groups: u64,
    pub output: Int4EntityRun,
}

impl Int4EntitySchedule {
    pub fn validate(self) -> Result<(), Int4EntityError> {
        if self.ner_cohort_chunks.is_some_and(|n| !(1..=MAX_INT4_SEQUENCES).contains(&n))
            || self.pair_cohort_pairs.is_some_and(|n| !(1..=MAX_INT4_SEQUENCES / 2).contains(&n)) {
            return Err(Int4EntityError::InvalidLimits);
        }
        Ok(())
    }

    /// No source/model IO or native allocation. Each source row and pair head
    /// keeps its original single-context budget. The two native phases never
    /// overlap, so the native and cursor quotes use componentwise maxima.
    pub fn requirements(self, config: &Int4EntityConfig, context_tokens: usize,
        preparation_reserve_bytes: u64, group_width: usize)
        -> Result<Int4EntityCohortRequirement, Int4EntityError> {
        self.validate()?;
        config.validate()?;
        config.source_planning.validate().map_err(Int4SourceError::from)?;
        config.graph.validate(config.resolution)?;
        if preparation_reserve_bytes == 0
            || config.source_planning.max_context_tokens > context_tokens
            || config.scoring.planning.max_context_tokens > context_tokens {
            return Err(Int4EntityError::InvalidLimits);
        }
        let row = Int4MemoryRequirement::for_context(context_tokens, group_width)?;
        if row.kv_bytes > config.ner_budget.max_kv_bytes
            || row.kv_bytes > config.scoring.planning.per_head.max_kv_bytes {
            return Err(Int4EntityError::WorkBudget);
        }
        let mut native = row;
        let mut workspace_bytes = 0;
        let mut additional_preparation_bytes = 0;
        let mut max_ner_result_bytes = config.ner_budget.max_output_bytes;
        if let Some(count) = self.ner_cohort_chunks {
            let quote = Int4SourceMapCohortMemoryLimits { max_chunks: count, context_tokens,
                max_output_tokens_per_chunk: u64::from(config.ner_budget.max_output_tokens),
                max_result_bytes_per_chunk: config.ner_budget.max_output_bytes }
                .requirements(group_width).map_err(|_| Int4EntityError::InvalidLimits)?;
            native = native_max(native, quote.native);
            workspace_bytes = quote.workspace_bytes;
            max_ner_result_bytes = quote.max_result_bytes;
            additional_preparation_bytes = mul(preparation_reserve_bytes, (count - 1) as u64)?;
        }
        let mut max_pair_result_bytes = (config.scoring.planning.max_result_bytes as u64)
            .min(config.scoring.planning.per_head.max_output_bytes);
        if let Some(max_pairs) = self.pair_cohort_pairs {
            let quote = Int4ResolutionCohortMemoryLimits { max_pairs, context_tokens,
                max_model_work: meet(config.scoring.max_model_work, config.max_model_work),
                max_result_bytes: max_pair_result_bytes }.requirements(group_width)?;
            native = native_max(native, quote.native);
            workspace_bytes = workspace_bytes.max(quote.workspace_bytes);
            max_pair_result_bytes = quote.max_result_bytes;
        }
        let max_result_bytes = add(&[config.max_result_bytes as u64, ENVELOPE_BYTES])?;
        let result = Int4EntityCohortRequirement { native, additional_preparation_bytes,
            workspace_bytes, max_ner_result_bytes, max_pair_result_bytes, max_result_bytes };
        // Components can each fit while the simultaneous declared peak does
        // not. Refuse before the embedding host sums wrapped reservations.
        add(&[native.kv_bytes, native.rope_bytes, native.scratch_payload_bound,
            preparation_reserve_bytes, additional_preparation_bytes, workspace_bytes,
            max_result_bytes])?;
        Ok(result)
    }
}

impl PreparedInt4EntityCorpus {
    /// Rebuild and authenticate every original chunk before the first native
    /// allocation. Pair inputs become available only after complete source
    /// discovery; their own executor admits every pair/order before scoring.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, weights: &Int4ModelWeights,
        vocabulary: &ExtractionVocabulary, kernel: PackedInt4Kernel,
        schedule: Int4EntitySchedule, budget: Int4EntityCohortBudget, control: &mut C)
        -> Result<Int4EntityCohortRun, Int4EntityError> {
        resolve::checkpoint(control)?;
        if schedule.ner_cohort_chunks.is_none() && schedule.pair_cohort_pairs.is_none() {
            return Err(Int4EntityError::InvalidLimits);
        }
        validate(&self.config, &self.source, &self.source_identity, &self.resolution_identity, weights.identity())?;
        if weights.identity() != &self.model { return Err(Int4EntityError::Identity); }
        let required = schedule.requirements(&self.config, budget.context_tokens,
            budget.preparation_reserve_bytes, self.group_width())?;
        check_budget(required, budget)?;
        preflight_discovery(&self.compiled, &self.source, &self.source_identity, &self.config,
            weights.group_width(), vocabulary, budget.context_tokens, control)?;
        let Self { compiled, source, resolver, source_identity, resolution_identity, config, model } = self;
        let CompiledInputs { inputs, ner_work, mask_visits, chunks, .. } = compiled;
        let width = model.group_width();
        let mut discovery = DiscoveryStats::default();
        let maps = if let Some(count) = schedule.ner_cohort_chunks {
            collect_windows(inputs, &config, width, count, control, |requests, control| {
                let run = execute_discovery_window(requests, &source, &source_identity,
                    &config, weights, vocabulary, kernel, budget, required.max_ner_result_bytes, control)?;
                discovery.add(&run, requests.len())?;
                Ok(run.sequences)
            })?
        } else if inputs.is_empty() {
            collect(inputs, &config, width, control, |_, _, _| Err(Int4EntityError::Accounting))?
        } else {
            let row = Int4MemoryRequirement::for_context(budget.context_tokens, width)?;
            let mut engine = StrictInt4Engine::new(weights, budget.context_tokens, memory_budget(row))?.with_kernel(kernel);
            let maps = collect(inputs, &config, width, control, |text, chunk, control| {
                let plan = source_plan(text, &source, &source_identity, &config, width, control)?;
                check_rebuilt(chunk, &plan)?;
                let run = plan.execute_with_control(plan.execution_identity(), &mut engine, vocabulary,
                    row_budget(&plan, &config), control)?;
                if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                    return Err(Int4EntityError::Accounting);
                }
                Ok(run)
            })?;
            drop(engine);
            maps
        };
        discovery.check(schedule.ner_cohort_chunks, chunks, ner_work, maps.work)?;
        resolve::checkpoint(control)?;
        let graph = ResolutionPlan::prepare(&maps.documents, config.resolution, config.graph, control)?;
        // Native EOS, empty discoveries and narrower windows never refund the
        // complete NER reservation into the remaining pair-scoring authority.
        let prepared = resolver.prepare_int4(&graph, &resolution_identity,
            remaining_scoring(&config, ner_work)?, &model, control)?;
        let pair_work = prepared.planned_work();
        let pair_count = prepared.pair_count();
        let mut admitted = reserved(pair_count)?;
        for identity in prepared.execution_identities() {
            resolve::checkpoint(control)?;
            admitted.push(identity.clone());
        }
        let (resolution, pair_cohorts, pair_group_steps, pair_projection_groups) =
            if let Some(count) = schedule.pair_cohort_pairs {
                let run = prepared.execute_cohorts_with_control(&admitted, weights, kernel, count,
                    Int4ResolutionCohortBudget { native: Int4RunBudget::exact(pair_work),
                        memory: budget.native_memory, max_task_workspace_bytes: budget.max_workspace_bytes,
                        max_output_bytes: required.max_pair_result_bytes }, control)?;
                if run.schema_version != 1 || run.execution != INT4_RESOLUTION_COHORT_EXECUTION
                    || run.pair_cohorts != pair_count.div_ceil(count) {
                    return Err(Int4EntityError::Accounting);
                }
                (run.output, run.pair_cohorts, run.group_steps, run.projection_groups)
            } else if pair_count == 0 {
                (prepared.finalize_without_model(control)?, 0, 0, 0)
            } else {
                let row = Int4MemoryRequirement::for_context(budget.context_tokens, width)?;
                let mut engine = StrictInt4Engine::new(weights, budget.context_tokens, memory_budget(row))?.with_kernel(kernel);
                let output = prepared.execute_with_control(&admitted, &mut engine,
                    Int4ScoringBudget { native: Int4RunBudget::exact(pair_work),
                        max_kv_bytes: config.scoring.planning.per_head.max_kv_bytes }, control)?;
                if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                    return Err(Int4EntityError::Accounting);
                }
                drop(engine);
                (output, 0, 0, 0)
            };
        drop(admitted); drop(graph);
        let output = finish(maps, resolution, ner_work, mask_visits, chunks, &config, width, control)?;
        let result = Int4EntityCohortRun { schema_version: 1, execution: INT4_ENTITY_COHORT_EXECUTION,
            schedule, ner_cohorts: discovery.cohorts, ner_group_steps: discovery.group_steps,
            ner_projection_groups: discovery.projection_groups,
            pair_cohorts, pair_group_steps, pair_projection_groups, output };
        resolve::check_output(&result, usize::try_from(required.max_result_bytes)
            .map_err(|_| Int4EntityError::WorkBudget)?)?;
        resolve::checkpoint(control)?;
        Ok(result)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn preflight_discovery<C: DecodeStepControl>(compiled: &CompiledInputs, source: &SourceTaskPlanner,
    identity: &ExecutionIdentity, config: &Int4EntityConfig, width: usize,
    vocabulary: &ExtractionVocabulary, context_tokens: usize, control: &mut C)
    -> Result<(), Int4EntityError> {
    let mut count = 0_usize; let mut work = Int4Work::default();
    let mut masks = 0_u64; let mut maximum_positions = 0;
    for input in &compiled.inputs {
        let (mut cursor, mut scalar) = (0, 0);
        if input.chunks.is_empty() { return Err(Int4EntityError::Accounting); }
        for chunk in &input.chunks {
            resolve::checkpoint(control)?;
            let text = chunk_text(&input.document.text, chunk.span, cursor, scalar)?;
            let plan = source_plan(text, source, identity, config, width, control)?;
            check_rebuilt(chunk, &plan)?;
            let positions = usize::try_from(chunk.work.forward_positions).map_err(|_| Int4EntityError::WorkBudget)?;
            if positions == 0 || positions > context_tokens
                || Int4MemoryRequirement::for_context(positions, width)?.kv_bytes > config.ner_budget.max_kv_bytes {
                return Err(Int4EntityError::WorkBudget);
            }
            source_group::preflight(&Int4SourceCohortRequest { prepared: &plan,
                admitted_identity: plan.execution_identity(), budget: row_budget(&plan, config) }, vocabulary)?;
            work = plus(work, chunk.work)?;
            count = count.checked_add(1).ok_or(Int4EntityError::WorkBudget)?;
            masks = masks.checked_add(config.masks.max_visits_per_item).ok_or(Int4EntityError::WorkBudget)?;
            maximum_positions = maximum_positions.max(positions);
            cursor = chunk.span.byte_end; scalar = chunk.span.scalar_end;
        }
        if cursor != input.document.text.len() { return Err(Int4EntityError::Accounting); }
    }
    if count != compiled.chunks || work != compiled.ner_work || masks != compiled.mask_visits
        || maximum_positions != compiled.maximum_positions || !within(work, config.max_model_work)
        || masks > config.masks.max_visits_per_run || count > config.max_snapshot_chunks {
        return Err(Int4EntityError::Accounting);
    }
    resolve::checkpoint(control)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_discovery_window<C: DecodeStepControl>(rows: &[ChunkRequest<'_>], source: &SourceTaskPlanner,
    identity: &ExecutionIdentity, config: &Int4EntityConfig, weights: &Int4ModelWeights,
    vocabulary: &ExtractionVocabulary, kernel: PackedInt4Kernel, budget: Int4EntityCohortBudget,
    max_result_bytes: u64, control: &mut C) -> Result<Int4SourceCohortRun, Int4EntityError> {
    if rows.is_empty() || rows.len() > MAX_INT4_SEQUENCES { return Err(Int4EntityError::InvalidLimits); }
    let mut plans = reserved(rows.len())?;
    let mut contexts = [0; MAX_INT4_SEQUENCES];
    let mut work = Int4Work::default();
    for (slot, request) in rows.iter().enumerate() {
        resolve::checkpoint(control)?;
        let plan = source_plan(request.text, source, identity, config, weights.group_width(), control)?;
        check_rebuilt(request.witness, &plan)?;
        contexts[slot] = usize::try_from(plan.planned_work().forward_positions).map_err(|_| Int4EntityError::WorkBudget)?;
        if contexts[slot] == 0 || contexts[slot] > budget.context_tokens {
            return Err(Int4EntityError::WorkBudget);
        }
        work = plus(work, plan.planned_work())?;
        plans.push(plan);
    }
    let required = Int4MemoryRequirement::for_cohort(&contexts[..rows.len()], weights.group_width())?;
    check_native(required, budget.native_memory)?;
    let mut requests = reserved(rows.len())?;
    for plan in &plans {
        requests.push(Int4SourceCohortRequest { prepared: plan,
            admitted_identity: plan.execution_identity(), budget: row_budget(plan, config) });
    }
    let masks = mul(config.masks.max_visits_per_item, rows.len() as u64)?;
    let mut engine = Int4CohortEngine::new(weights, &contexts[..rows.len()], memory_budget(required))?.with_kernel(kernel);
    let result = source_group::execute(&mut engine, &requests, vocabulary,
        Int4ExtractCohortBudget { decode: Int4JsonCohortBudget { native: Int4RunBudget::exact(work),
            max_kv_bytes: required.kv_bytes, max_mask_node_visits: masks, max_result_bytes },
            max_result_bytes }, control)?;
    engine.check_idle()?;
    drop(engine);
    if result.planned_work != work { return Err(Int4EntityError::Accounting); }
    resolve::checkpoint(control)?;
    Ok(result)
}

#[derive(Default)]
struct DiscoveryStats {
    cohorts: usize, chunks: usize, group_steps: u64, projection_groups: u64,
    planned_work: Int4Work, model_work: Int4Work,
}
impl DiscoveryStats {
    fn add(&mut self, run: &Int4SourceCohortRun, count: usize) -> Result<(), Int4EntityError> {
        if run.schema_version != 1 || run.execution != INT4_SOURCE_COHORT_EXECUTION
            || run.sequences.len() != count || count == 0 || count > MAX_INT4_SEQUENCES {
            return Err(Int4EntityError::Accounting);
        }
        let mut actual = Int4Work::default(); let mut longest = 0;
        let mut selections = 0_u64; let mut longest_selections = 0;
        for row in &run.sequences {
            let SourceTaskResult::Ner(ner) = &row.result else { return Err(Int4EntityError::Accounting); };
            let tokens = ner.generated_token_ids.len() as u64;
            if tokens == 0 { return Err(Int4EntityError::Accounting); }
            actual = plus(actual, row.model_work)?;
            longest = longest.max(row.model_work.forward_positions);
            selections = selections.checked_add(tokens).ok_or(Int4EntityError::WorkBudget)?;
            longest_selections = longest_selections.max(tokens);
        }
        if actual != run.model_work || run.group_steps != longest
            || run.projection_groups < longest_selections || run.projection_groups > selections
            || !within(run.model_work, run.planned_work) {
            return Err(Int4EntityError::Accounting);
        }
        self.cohorts = self.cohorts.checked_add(1).ok_or(Int4EntityError::WorkBudget)?;
        self.chunks = self.chunks.checked_add(count).ok_or(Int4EntityError::WorkBudget)?;
        self.group_steps = self.group_steps.checked_add(run.group_steps).ok_or(Int4EntityError::WorkBudget)?;
        self.projection_groups = self.projection_groups.checked_add(run.projection_groups).ok_or(Int4EntityError::WorkBudget)?;
        self.planned_work = plus(self.planned_work, run.planned_work)?;
        self.model_work = plus(self.model_work, run.model_work)?;
        Ok(())
    }
    fn check(&self, window: Option<usize>, chunks: usize, planned: Int4Work, actual: Int4Work)
        -> Result<(), Int4EntityError> {
        if let Some(window) = window {
            if self.cohorts != chunks.div_ceil(window) || self.chunks != chunks
                || self.planned_work != planned || self.model_work != actual {
                return Err(Int4EntityError::Accounting);
            }
        } else if self.cohorts != 0 || self.chunks != 0 || self.group_steps != 0 || self.projection_groups != 0
            || self.planned_work != Int4Work::default() || self.model_work != Int4Work::default() {
            return Err(Int4EntityError::Accounting);
        }
        Ok(())
    }
}

fn row_budget(plan: &PreparedInt4SourceTask, config: &Int4EntityConfig) -> Int4JsonBudget {
    let work = plan.planned_work();
    Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: config.ner_budget.max_kv_bytes,
        max_total_mask_node_visits: config.masks.max_visits_per_item, mask_limits: config.masks.per_mask,
    } }
}
pub(super) fn check_budget(required: Int4EntityCohortRequirement, budget: Int4EntityCohortBudget)
    -> Result<(), Int4EntityError> {
    check_native(required.native, budget.native_memory)?;
    if required.additional_preparation_bytes > budget.max_preparation_bytes
        || required.workspace_bytes > budget.max_workspace_bytes || required.max_result_bytes > budget.max_result_bytes {
        return Err(Int4EntityError::WorkBudget);
    }
    Ok(())
}
fn check_native(required: Int4MemoryRequirement, cap: Int4MemoryBudget) -> Result<(), Int4EntityError> {
    if required.kv_bytes > cap.max_kv_bytes || required.rope_bytes > cap.max_rope_bytes
        || required.scratch_payload_bound > cap.max_scratch_payload_bytes {
        return Err(Int4EntityError::WorkBudget);
    }
    Ok(())
}
fn memory_budget(required: Int4MemoryRequirement) -> Int4MemoryBudget {
    Int4MemoryBudget { max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
        max_scratch_payload_bytes: required.scratch_payload_bound }
}
fn native_max(a: Int4MemoryRequirement, b: Int4MemoryRequirement) -> Int4MemoryRequirement {
    Int4MemoryRequirement { kv_bytes: a.kv_bytes.max(b.kv_bytes), rope_bytes: a.rope_bytes.max(b.rope_bytes),
        scratch_payload_bound: a.scratch_payload_bound.max(b.scratch_payload_bound) }
}
fn mul(a: u64, b: u64) -> Result<u64, Int4EntityError> {
    a.checked_mul(b).ok_or(Int4EntityError::WorkBudget)
}
fn add(values: &[u64]) -> Result<u64, Int4EntityError> {
    values.iter().try_fold(0_u64, |sum, value| sum.checked_add(*value).ok_or(Int4EntityError::WorkBudget))
}
