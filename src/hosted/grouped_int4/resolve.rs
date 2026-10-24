//! One owned INT4 corpus snapshot: source verification, both presentation
//! orders of every comparison, and deterministic complete-link clustering.
use super::*;
mod entities;
use crate::{
    corpus::{
        native_resolve::{ResolutionPlanner,
            grouped_int4::{Int4ResolveError, Int4ResolveLimits, Int4ResolutionRun, INT4_RESOLUTION_EXECUTION}},
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

fn resolution_error(error: Int4ResolveError) -> HostedError { HostedError::Int4(Int4HostError::Resolution(error)) }

#[cfg(test)]
mod tests;
