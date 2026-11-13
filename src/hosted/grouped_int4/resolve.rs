//! One owned INT4 corpus snapshot: source verification, both presentation
//! orders of every comparison, and deterministic complete-link clustering.
use super::*;
mod entities;
mod corpus;
pub use corpus::{HostedInt4EntityCorpus, Int4EntityCorpusError, Int4EntityCorpusHostConfig,
    Int4EntityCorpusHostRequirements, Int4EntityCorpusSummary, Int4EntityCorpusCohortSummary};
pub use entities::Int4EntityCohortHostRequirements;
use crate::{
    corpus::{
        native_resolve::{ResolutionPlanner,
            grouped_int4::{Int4ResolveError, Int4ResolveLimits, Int4ResolutionRun, INT4_RESOLUTION_EXECUTION,
                cohort::{Int4ResolutionCohortBudget, Int4ResolutionCohortMemoryLimits,
                    Int4ResolutionCohortRequirements, Int4ResolutionCohortRun, INT4_RESOLUTION_COHORT_EXECUTION}}},
        resolve::{ResolutionDocument, ResolutionPlan, ResolveOptions, ResolveLimits, RESOLVE_VERSION},
    },
    native_engine::{constrained_int4, strict_int4::{Int4Work, STRICT_INT4_PROFILE,
        scoring::Int4ScoringBudget}},
};

/// Whole-snapshot bounds, never renewed between pairs or presentation orders.
/// Preparation prices the pinned planner, retained prompts/identities/schedules
/// and temporary compilation storage. Graph reserve prices lexical blocks,
/// candidate tickets, frozen scores, clustering and allocator overhead. These
/// caller-modeled reservations are not measurements of process RSS.
pub struct Int4ResolveConfig {
    pub identity: ExecutionIdentity,
    pub options: ResolveOptions,
    pub graph: ResolveLimits,
    pub scoring: Int4ResolveLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub graph_reserve_bytes: u64,
}

/// Pure admission for a bounded window containing both orders of every pair.
/// Source payload and resident weights are separate; preparation, graph,
/// native windows, full output staging and cleanup are included here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ResolveCohortHostRequirements {
    pub preparation_bytes: u64,
    pub temporary_bytes: u64,
    pub output_bytes: u64,
    pub kv_bytes: u64,
    pub scratch_bytes: u64,
    pub cleanup_bytes: u64,
    pub max_result_bytes: u64,
    cohort: Int4ResolutionCohortRequirements,
}
impl Int4ResolveCohortHostRequirements {
    #[allow(clippy::too_many_arguments)]
    pub fn for_config(scoring: Int4ResolveLimits, native: NativeLimits, preparation: u64,
        graph: u64, cohort_pairs: usize, width: usize) -> Result<Self, HostedError> {
        native.run.validate()?;
        scoring.validate().map_err(resolution_error)?;
        let per_head = Int4MemoryRequirement::for_context(native.context_tokens, width).map_err(native_error)?;
        if preparation == 0 || graph == 0 || native.allocator_reserve_bytes == 0
            || scoring.planning.max_context_tokens > native.context_tokens
            || per_head.kv_bytes > scoring.planning.per_head.max_kv_bytes {
            return Err(HostedError::Limits("INT4 resolution cohort preparation, graph or per-head context"));
        }
        let cohort = Int4ResolutionCohortMemoryLimits { max_pairs: cohort_pairs,
            context_tokens: native.context_tokens, max_model_work: scoring.max_model_work,
            max_result_bytes: (scoring.planning.max_result_bytes as u64)
                .min(scoring.planning.per_head.max_output_bytes) }.requirements(width).map_err(resolution_error)?;
        let result = Self { preparation_bytes: preparation,
            temporary_bytes: sum(&[graph, cohort.workspace_bytes])?,
            output_bytes: cohort_output_bytes(cohort.max_result_bytes)?,
            kv_bytes: cohort.native.kv_bytes,
            scratch_bytes: sum(&[cohort.native.rope_bytes, cohort.native.scratch_payload_bound,
                native.allocator_reserve_bytes])?,
            cleanup_bytes: native.run.cleanup_reserve_bytes, max_result_bytes: cohort.max_result_bytes, cohort };
        result.total_bytes()?;
        Ok(result)
    }
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.preparation_bytes, self.temporary_bytes, self.output_bytes,
            self.kv_bytes, self.scratch_bytes, self.cleanup_bytes])
    }
}
fn cohort_output_bytes(bytes: u64) -> Result<u64, HostedError> {
    sum(&[bytes.checked_mul(4).ok_or(HostedError::Limits("INT4 cohort output arithmetic"))?, 4096])
}

struct ResolveInput {
    documents: Vec<ResolutionDocument>,
    planner: Arc<ResolutionPlanner>,
    config: Int4ResolveConfig,
}

impl NlpEngine {
    /// Resolve exact source-anchored mentions using the resident INT4 model.
    /// Planning, scoring and finalization share one cancellation budget and
    /// physical invocation. A failed pair/order yields no partial graph.
    /// Empty candidate graphs need no native KV, scratch or engine allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_int4(&self, model: &ResidentInt4, documents: Vec<ResolutionDocument>,
        planner: Arc<ResolutionPlanner>, config: Int4ResolveConfig, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4ResolutionRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        // Authenticate even an empty graph against the actual resident model.
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let (required, scratch_bytes) = requirements(&config, width,
            planner.tokenizer_digest(), planner.template_digest())?;
        let input_bytes = sum(&[super::super::resolve::input_payload(&documents, config.graph)?,
            config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(ResolveInput { documents, planner, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Capture the complete charged aggregate so disjoint captures can
            // never free a reservation before its source/plan storage.
            let input = input;
            let config = &input.value.config;
            let graph_charge = Pending::reserve(&lease, MemoryClass::JobBuffers, config.graph_reserve_bytes)?;
            let plan = ResolutionPlan::prepare(&input.value.documents, config.options, config.graph, control)
                .map_err(|error| resolution_error(error.into()))?;
            let prepared = input.value.planner.prepare_int4(&plan, &config.identity,
                config.scoring, model.model_identity(), control).map_err(resolution_error)?;
            let pairs = prepared.pair_count();
            let heads = prepared.head_count();
            let work = prepared.planned_work();
            check_work(work, config.scoring.max_model_work)?;
            let output = output_claim(&lease, prepared.max_result_bytes(), 0)?;
            let result = if pairs == 0 {
                allocate(output, || {
                    let result = prepared.finalize_without_model(control).map_err(resolution_error)?;
                    check_result(&result, width, heads, work)?;
                    checkpoint(control)?;
                    Ok(result)
                })?
            } else {
                if prepared.required_context_tokens() > config.native.context_tokens {
                    return Err(HostedError::Limits("INT4 resolution complete context"));
                }
                let mut admitted = Vec::new();
                admitted.try_reserve_exact(pairs)
                    .map_err(|_| resolution_error(Int4ResolveError::WorkBudget))?;
                // Every pair is authenticated before the first native forward.
                for identity in prepared.execution_identities() {
                    checkpoint(control)?;
                    constrained_int4::check_model(identity, model.model_identity())
                        .map_err(|_| HostedError::ModelIdentity)?;
                    admitted.push(identity.clone());
                }
                let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
                let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound,
                    }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
                })?;
                let result = allocate(output, || {
                    let result = prepared.execute_with_control(&admitted, &mut engine.value,
                        Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes }, control)
                        .map_err(resolution_error)?;
                    check_result(&result, width, heads, work)?;
                    if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                        return Err(resolution_error(Int4ResolveError::Accounting));
                    }
                    checkpoint(control)?;
                    Ok(result)
                })?;
                drop(engine);
                drop(admitted);
                result
            };
            drop(plan);
            drop(graph_charge);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
    /// Keep one fixed numerical row team across both orders of every pair
    /// and the complete snapshot finalizer. The ordinary sealed plans, graph,
    /// five-axis work allowance and output owner remain unchanged. An empty
    /// graph authenticates the model and reports zero completed projections.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_int4_with_threads(&self, model: &ResidentInt4, documents: Vec<ResolutionDocument>,
        planner: Arc<ResolutionPlanner>, config: Int4ResolveConfig, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4ResolutionRun>, HostedError> {
        dispatch::preflight_int4_team(self, config.native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let (required, scratch_bytes) = requirements(&config, width,
            planner.tokenizer_digest(), planner.template_digest())?;
        sum(&[scratch_bytes, int4_task_team_workspace_bytes(compute_threads, width)?])?;
        let input_bytes = sum(&[super::super::resolve::input_payload(&documents, config.graph)?,
            config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(ResolveInput { documents, planner, config }))?;
        dispatch::run_int4_team(self, run, cancellation, model.clone(), compute_threads,
            kernel, move |weights, control, team| {
                // Keep source/planner storage charged through native cleanup
                // and clustering. The dispatcher owns workers until join.
                let input = input;
                let config = &input.value.config;
                let graph_charge = Pending::reserve(&lease, MemoryClass::JobBuffers, config.graph_reserve_bytes)?;
                let plan = ResolutionPlan::prepare(&input.value.documents, config.options, config.graph, control)
                    .map_err(|error| resolution_error(error.into()))?;
                let prepared = input.value.planner.prepare_int4(&plan, &config.identity,
                    config.scoring, weights.identity(), control).map_err(resolution_error)?;
                let pairs = prepared.pair_count();
                let heads = prepared.head_count();
                let work = prepared.planned_work();
                check_work(work, config.scoring.max_model_work)?;
                let output = output_claim(&lease, prepared.max_result_bytes(), 0)?;
                let result = if pairs == 0 {
                    allocate(output, || {
                        let result = prepared.finalize_without_model(control).map_err(resolution_error)?;
                        check_result(&result, width, heads, work)?;
                        checkpoint(control)?;
                        Ok(result)
                    })?
                } else {
                    if prepared.required_context_tokens() > config.native.context_tokens {
                        return Err(HostedError::Limits("INT4 resolution complete context"));
                    }
                    let mut admitted = Vec::new();
                    admitted.try_reserve_exact(pairs)
                        .map_err(|_| resolution_error(Int4ResolveError::WorkBudget))?;
                    for identity in prepared.execution_identities() {
                        checkpoint(control)?;
                        constrained_int4::check_model(identity, weights.identity())
                            .map_err(|_| HostedError::ModelIdentity)?;
                        admitted.push(identity.clone());
                    }
                    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
                    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
                    let mut engine = allocate_native(kv, scratch, || {
                        StrictInt4Engine::new(weights, config.native.context_tokens, Int4MemoryBudget {
                            max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                            max_scratch_payload_bytes: required.scratch_payload_bound,
                        }).map_err(native_error)?.with_kernel(kernel)
                            .with_projection_executor(team).map_err(native_error)
                    })?;
                    let result = allocate(output, || {
                        let result = prepared.execute_with_control(&admitted, &mut engine.value,
                            Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes }, control)
                            .map_err(resolution_error)?;
                        check_result(&result, width, heads, work)?;
                        if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                            return Err(resolution_error(Int4ResolveError::Accounting));
                        }
                        checkpoint(control)?;
                        Ok(result)
                    })?;
                    drop(engine);
                    drop(admitted);
                    result
                };
                drop(plan);
                drop(graph_charge);
                drop(input);
                let compute = team.stats().into();
                drop(lease);
                Ok(HostedInt4TeamResult::new(GuardedOutput::new(result.value, result._memory), compute))
            })
    }

    /// Score independently bounded windows of pairs, keeping both presentation
    /// orders together. Every window shares the original complete-snapshot work
    /// allowance and failure leaves no partial clustering result.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_int4_cohorts(&self, model: &ResidentInt4, documents: Vec<ResolutionDocument>,
        planner: Arc<ResolutionPlanner>, config: Int4ResolveConfig, cohort_pairs: usize,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ResolutionCohortRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        requirements(&config, width, planner.tokenizer_digest(), planner.template_digest())?;
        let required = Int4ResolveCohortHostRequirements::for_config(config.scoring, config.native,
            config.preparation_reserve_bytes, config.graph_reserve_bytes, cohort_pairs, width)?;
        let input_bytes = sum(&[super::super::resolve::input_payload(&documents, config.graph)?,
            required.preparation_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(ResolveInput { documents, planner, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, required.temporary_bytes)?;
            let plan = ResolutionPlan::prepare(&input.value.documents, config.options, config.graph, control)
                .map_err(|error| resolution_error(error.into()))?;
            let prepared = input.value.planner.prepare_int4(&plan, &config.identity,
                config.scoring, model.model_identity(), control).map_err(resolution_error)?;
            let heads = prepared.head_count();
            let work = prepared.planned_work();
            check_work(work, config.scoring.max_model_work)?;
            if prepared.required_context_tokens() > config.native.context_tokens {
                return Err(HostedError::Limits("INT4 resolution cohort complete context"));
            }
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(prepared.pair_count())
                .map_err(|_| resolution_error(Int4ResolveError::WorkBudget))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, model.model_identity())
                    .map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let budget = Int4ResolutionCohortBudget { native: Int4RunBudget::exact(work),
                memory: Int4MemoryBudget { max_kv_bytes: required.cohort.native.kv_bytes,
                    max_rope_bytes: required.cohort.native.rope_bytes,
                    max_scratch_payload_bytes: required.cohort.native.scratch_payload_bound },
                max_task_workspace_bytes: required.cohort.workspace_bytes,
                max_output_bytes: required.max_result_bytes };
            prepared.preflight_cohorts_with_control(&admitted, model.model_identity(), cohort_pairs, budget, control)
                .map_err(resolution_error)?;
            let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_bytes)?;
            // The core allocates exactly its phase engines. There is no spare
            // scalar engine beside the window; empty graphs allocate neither.
            let native = if heads == 0 { None } else {
                let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
                let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
                Some((kv.commit()?, scratch.commit()?))
            };
            let result = allocate(output, || {
                let result = prepared.execute_cohorts_with_control(&admitted, &model.inner.loaded.value,
                    kernel, cohort_pairs, budget, control).map_err(resolution_error)?;
                check_cohort_result(&result, width, heads, work, cohort_pairs)?;
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(native);
            drop(admitted);
            drop(plan);
            drop(temporary);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

}

fn requirements(config: &Int4ResolveConfig, width: usize, tokenizer: Sha256Digest,
    template: Sha256Digest) -> Result<(Int4MemoryRequirement, u64), HostedError> {
    config.native.run.validate()?;
    config.scoring.planning.per_head.validate()
        .map_err(|_| HostedError::Limits("INT4 resolution task budget"))?;
    constrained_int4::check_profile(&config.identity, width).map_err(|_| HostedError::ModelIdentity)?;
    if config.identity.task_spec != RESOLVE_VERSION || config.identity.tokenizer_digest != tokenizer
        || config.identity.template_digest != template { return Err(HostedError::ModelIdentity); }
    let required = Int4MemoryRequirement::for_context(config.native.context_tokens, width).map_err(native_error)?;
    if config.preparation_reserve_bytes == 0 || config.graph_reserve_bytes == 0
        || config.native.allocator_reserve_bytes == 0
        || config.scoring.planning.max_context_tokens > config.native.context_tokens
        || required.kv_bytes > config.scoring.planning.per_head.max_kv_bytes {
        return Err(HostedError::Limits("INT4 resolution preparation, graph or full resident context"));
    }
    let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?;
    Ok((required, scratch))
}

fn check_work(work: Int4Work, cap: Int4Work) -> Result<(), HostedError> {
    if work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(HostedError::Limits("INT4 resolution complete model work"));
    }
    Ok(())
}

fn check_result(result: &Int4ResolutionRun, width: usize, heads: usize, work: Int4Work) -> Result<(), HostedError> {
    if result.schema_version != 1 || result.execution != INT4_RESOLUTION_EXECUTION
        || result.numerics_profile != STRICT_INT4_PROFILE
        || result.scoring_execution != crate::native_engine::strict_int4::scoring::INT4_SCORING_EXECUTION
        || result.group_width != width || result.head_count != heads || result.model_evaluated != (heads != 0)
        || result.planned_model_work != work || result.model_work != work
        || (heads as u64).checked_mul(2) != Some(result.rewound_positions) {
        return Err(resolution_error(Int4ResolveError::Accounting));
    }
    Ok(())
}


fn check_cohort_result(result: &Int4ResolutionCohortRun, width: usize, heads: usize,
    work: Int4Work, cohort_pairs: usize) -> Result<(), HostedError> {
    check_result(&result.output, width, heads, work)?;
    if !(1..=32).contains(&cohort_pairs) || heads % 2 != 0 {
        return Err(resolution_error(Int4ResolveError::Accounting));
    }
    let windows = (heads / 2).div_ceil(cohort_pairs);
    if result.schema_version != 1 || result.execution != INT4_RESOLUTION_COHORT_EXECUTION
        || result.pair_cohorts != windows
        || (windows == 0) != (result.group_steps == 0)
        || (windows == 0) != (result.projection_groups == 0)
        || result.projection_groups > result.group_steps || result.group_steps > work.forward_positions {
        return Err(resolution_error(Int4ResolveError::Accounting));
    }
    Ok(())
}

fn resolution_error(error: Int4ResolveError) -> HostedError { HostedError::Int4(Int4HostError::Resolution(error)) }

#[cfg(test)]
mod tests;
