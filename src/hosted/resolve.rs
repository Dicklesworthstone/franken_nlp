//! One process-owned snapshot: source verification, pair planning, INT8 scoring
//! and conservative complete-link clustering, with output ownership retained.
use super::*;
use std::mem::size_of;
use crate::{
    corpus::{
        native_resolve::{ResolutionPlanner, quantized::{Int8ResolveError, Int8ResolveLimits, Int8ResolutionRun}},
        resolve::{ResolutionDocument, MentionInput, ResolutionPlan, ResolveOptions, ResolveLimits, ResolveError, RESOLVE_VERSION},
    },
    native_engine::{constrained_int8, decode::DecodeStepControl, strict_int8::scoring::Int8ScoringBudget},
};

/// Fixed snapshot policy and complete work ceilings, not per-pair defaults.
/// Preparation prices the pinned planner, all retained prompt/identity/schedule
/// allocations and temporary compilation storage. Graph reserve prices lexical
/// blocks, candidate tickets, frozen scores, clustering and allocator overhead.
/// These are explicit caller-modeled commitments, NOT measured RSS bounds.
pub struct ResolveConfig {
    pub identity: ExecutionIdentity,
    pub options: ResolveOptions,
    pub graph: ResolveLimits,
    pub scoring: Int8ResolveLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub graph_reserve_bytes: u64,
}
struct ResolveInput {
    documents: Vec<ResolutionDocument>,
    planner: Arc<ResolutionPlanner>,
    config: ResolveConfig,
}

impl NlpEngine {
    /// Resolve source-anchored mentions across documents using the resident
    /// INT8 model and existing process runtime. All source validation, blocking,
    /// two-order planning/scoring and clustering use ONE cancellation budget.
    /// No per-pair model loads, runtimes, output publication or retry exist.
    /// Caller-provided mention offsets are checked against original source.
    pub fn resolve_int8(&self, model: &ResidentInt8, documents: Vec<ResolutionDocument>,
        planner: Arc<ResolutionPlanner>, config: ResolveConfig, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ResolutionRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        validate(&config, planner.tokenizer_digest(), planner.template_digest(), required.kv_bytes)?;
        let bytes = sum(&[input_payload(&documents, config.graph)?, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ResolveInput { documents, planner, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Capture the complete Charged aggregate, never disjoint fields
            // that could free their reservation before the retained storage.
            let input = input;
            let config = &input.value.config;
            let graph_charge = Pending::reserve(&lease, MemoryClass::JobBuffers, config.graph_reserve_bytes)?;
            let plan = ResolutionPlan::prepare(&input.value.documents, config.options, config.graph, control)
                .map_err(|e| HostedError::Resolution(e.into()))?;
            let prepared = input.value.planner.prepare_int8(&plan, &config.identity, config.scoring, control)
                .map_err(HostedError::Resolution)?;
            // The final result contains scores/mentions/clusters, not retained
            // vocabulary-sized head logits or generated-token transcripts.
            let output = output_claim(&lease, prepared.max_result_bytes(), 0)?;
            let result = if prepared.pair_count() == 0 {
                // No lexical candidates means no neural scoring. Do not spend
                // a resident context's KV/RoPE/scratch just to emit singletons.
                allocate(output, || prepared.finalize_without_model(control).map_err(HostedError::Resolution))?
            } else {
                let work = prepared.planned_work();
                if prepared.required_context_tokens() > config.native.context_tokens {
                    return Err(HostedError::Limits("resolution complete context"));
                }
                let mut admitted = Vec::new();
                admitted.try_reserve_exact(prepared.pair_count())
                    .map_err(|_| HostedError::Resolution(Int8ResolveError::WorkBudget))?;
                for identity in prepared.execution_identities() {
                    if let Some(cause) = control.prefill_checkpoint(0) {
                        return Err(HostedError::Resolution(ResolveError::Cancelled(cause).into()));
                    }
                    // This is the existing process/model authority, not a new
                    // fake admission provider whose receipt always succeeds.
                    check_model_identity(model.artifact_identity(), identity)?;
                    admitted.push(identity.clone());
                }
                let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
                let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                    sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
                let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                    .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
                let result = allocate(output, || prepared.execute_with_control(&admitted, &mut engine.value,
                    Int8ScoringBudget { native: Int8RunBudget::exact(work), max_kv_bytes: required.kv_bytes }, control)
                    .map_err(HostedError::Resolution))?;
                drop(engine);
                drop(admitted);
                result
            };
            // No borrowed source or preparation storage escapes with the
            // result. On error/unwind the same declaration order applies.
            drop(plan);
            drop(graph_charge);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn validate(config: &ResolveConfig, tokenizer: Sha256Digest, template: Sha256Digest, kv_bytes: u64)
    -> Result<(), HostedError> {
    config.native.run.validate()?;
    config.identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int8::check_profile(&config.identity).map_err(|_| HostedError::ModelIdentity)?;
    if config.identity.task_spec != RESOLVE_VERSION || config.identity.tokenizer_digest != tokenizer
        || config.identity.template_digest != template { return Err(HostedError::ModelIdentity); }
    if config.preparation_reserve_bytes == 0 || config.graph_reserve_bytes == 0
        || config.scoring.planning.max_context_tokens > config.native.context_tokens
        || kv_bytes > config.scoring.planning.per_head.max_kv_bytes {
        return Err(HostedError::Limits("resolution preparation, graph or full resident context"));
    }
    // The authoritative graph and task compilers validate their remaining
    // semantic limits inside the owned, checkpointed invocation before native
    // allocation. This preliminary pass neither invents nor repairs a plan.
    Ok(())
}

fn allocation_bytes(capacity: usize, element: usize) -> Result<u64, HostedError> {
    (capacity as u64).checked_mul(element as u64).ok_or(HostedError::Limits("resolution allocation arithmetic"))
}
/// Price actual Vec/String CAPACITY, including empty spare capacity and nested
/// mention vectors. Logical input limits remain independent of allocation size.
fn input_payload(documents: &Vec<ResolutionDocument>, limits: ResolveLimits) -> Result<u64, HostedError> {
    if documents.len() > limits.max_documents || documents.len() > 65_536 {
        return Err(HostedError::Limits("resolution document count"));
    }
    let mut payload = allocation_bytes(documents.capacity(), size_of::<ResolutionDocument>())?;
    let mut logical = 0_u64; let mut mentions = 0_usize;
    for document in documents {
        mentions = mentions.checked_add(document.mentions.len()).ok_or(HostedError::Limits("resolution mention arithmetic"))?;
        if mentions > limits.max_mentions || mentions > 65_536 { return Err(HostedError::Limits("resolution mention count")); }
        logical = sum(&[logical, document.id.len() as u64, document.text.len() as u64])?;
        if logical > limits.max_input_bytes as u64 || logical > 64 * 1024 * 1024 {
            return Err(HostedError::Limits("resolution logical source bytes"));
        }
        payload = sum(&[payload, document.id.capacity() as u64, document.text.capacity() as u64,
            allocation_bytes(document.mentions.capacity(), size_of::<MentionInput>())?])?;
        for mention in &document.mentions {
            logical = sum(&[logical, mention.entity_type.len() as u64, mention.surface.len() as u64])?;
            if logical > limits.max_input_bytes as u64 || logical > 64 * 1024 * 1024 {
                return Err(HostedError::Limits("resolution logical mention bytes"));
            }
            payload = sum(&[payload, mention.entity_type.capacity() as u64, mention.surface.capacity() as u64])?;
        }
    }
    Ok(payload)
}

#[cfg(test)] mod tests;
