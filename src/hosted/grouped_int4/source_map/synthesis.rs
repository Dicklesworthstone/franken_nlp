//! One native engine and optional numerical team through every synthesis pass.
//! Task-specific modes retain their original planning, workspace and transcript
//! quotes. Chunk cohorts and hierarchy-group cohorts use a separate dispatcher.
use super::*;
use super::cohort::synthesis::{SynthesisMode, SynthesisPlan as CohortSynthesisPlan};

/// The four closed synthesis plans execute all discovery, intermediate and
/// final passes on the same engine. A plan must check its original complete
/// receipt before returning; changing the physical schedule grants no authority.
pub(super) trait SynthesisPlan: Sized {
    type Output: Send + 'static;
    fn validate_metadata(&self) -> Result<(), HostedError>;
    fn run_on_engine(self, admitted: &[ExecutionIdentity], planner: &SourceTaskPlanner,
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        control: &mut RunControl) -> Result<Self::Output, HostedError>;
}

// The complete charged input drops before its invocation lease on queued
// cancellation, and remains owned through all dynamically prepared passes.
struct SynthesisAdmission<T> {
    input: Charged<SourceMapInput<T>>,
    required: Int4MemoryRequirement,
    lease: EngineLease,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute<M, O>(host: &NlpEngine, model: &ResidentInt4, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    config: Int4SourceMapConfig<M::Task>, mode: M, kernel: PackedInt4Kernel,
    cancellation: CancellationToken) -> Result<HostedOutput<O>, HostedError>
where
    M: SynthesisMode,
    O: Send + 'static,
    for<'s> M::Prepared<'s>: SynthesisPlan<Output = O>,
{
    let run = config.native.run;
    let admission = admit(host, model, source, planner, vocabulary, config, mode)?;
    let model = model.clone();
    dispatch::run(host, run, cancellation, move |control| {
        execute_on_engine::<M, O>(admission, model.weights(), kernel, control, None, mode)
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_with_threads<M, O>(host: &NlpEngine, model: &ResidentInt4, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    config: Int4SourceMapConfig<M::Task>, mode: M, kernel: PackedInt4Kernel,
    compute_threads: usize, cancellation: CancellationToken)
    -> Result<HostedInt4TeamResult<O>, HostedError>
where
    M: SynthesisMode,
    O: Send + 'static,
    for<'s> M::Prepared<'s>: SynthesisPlan<Output = O>,
{
    // Installed process width and native-plus-worker scratch arithmetic are
    // checked before any private source/question capture. One worker workspace
    // serves every allowed discovery/final/hierarchy pass without multiplication.
    dispatch::preflight_int4_team(host, config.native.run, compute_threads)?;
    team::team_requirements(&config, model.model_identity().group_width(), compute_threads)?;
    let run = config.native.run;
    let admission = admit(host, model, source, planner, vocabulary, config, mode)?;
    dispatch::run_int4_team(host, run, cancellation, model.clone(), compute_threads,
        kernel, move |weights, control, team| {
            let output = execute_on_engine::<M, O>(admission, weights, kernel, control, Some(&mut *team), mode)?;
            // The engine released its executor borrow; the dispatcher retains
            // the model and worker charge until every child physically joins.
            let compute = team.stats().into();
            Ok(HostedInt4TeamResult::new(output, compute))
        })
}

#[allow(clippy::too_many_arguments)]
fn admit<M: SynthesisMode>(host: &NlpEngine, model: &ResidentInt4, source: String,
    planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
    config: Int4SourceMapConfig<M::Task>, mode: M)
    -> Result<SynthesisAdmission<M::Task>, HostedError> {
    dispatch::preflight(host, config.native.run)?;
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    constrained_int4::check_model(&config.identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
    let required = mode.requirements(&config, source.len(), model.model_identity().group_width(), &planner)?;
    let bytes = mode.input_bytes(source.capacity(), &config)?;
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
        || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
    Ok(SynthesisAdmission { input, required, lease })
}

fn execute_on_engine<'weights, M, O>(admission: SynthesisAdmission<M::Task>,
    weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel, control: &mut RunControl,
    team: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>, mode: M)
    -> Result<HostedOutput<O>, HostedError>
where
    M: SynthesisMode,
    O: Send + 'static,
    for<'s> M::Prepared<'s>: SynthesisPlan<Output = O>,
{
    let SynthesisAdmission { input, required, lease } = admission;
    let config = &input.value.config;
    let width = weights.group_width();
    checkpoint(control)?;
    let context = PlanContext::new(&config.identity, config.budget)
        .map_err(|_| HostedError::Limits("INT4 source synthesis task context"))?;
    let prepared = mode.prepare(&input.value.source, &input.value.planner, config, &context, width, control)?;
    prepared.validate_metadata()?;
    let count = prepared.source_chunks();
    let native = prepared.native_chunks();
    if prepared.group_width() != width || count == 0 || count > config.mapping.chunks.max_chunks
        || native == 0 || native > count || prepared.execution_identities().len() != native
        || required.kv_bytes > prepared.max_kv_bytes() {
        return Err(HostedError::ModelIdentity);
    }
    // The original complete frontier and every allowed native transcript stay
    // admitted even if discovery uses fewer tokens or later evidence is empty.
    let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, mode.workspace_bytes(config, count)?)?;
    let output = output_claim(&lease, prepared.max_result_bytes(), mode.output_tokens(config, count)?)?;
    let mut admitted = Vec::new();
    admitted.try_reserve_exact(native).map_err(|_| map_error(Int4SourceMapError::Allocation))?;
    for identity in prepared.execution_identities() {
        checkpoint(control)?;
        constrained_int4::check_model(identity, weights.identity()).map_err(|_| HostedError::ModelIdentity)?;
        if identity.task_spec != config.identity.task_spec
            || identity.tokenizer_digest != config.identity.tokenizer_digest
            || identity.template_digest != config.identity.template_digest {
            return Err(HostedError::ModelIdentity);
        }
        admitted.push(identity.clone());
    }
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
        sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
    let mut engine = allocate_native(kv, scratch, || {
        let engine = StrictInt4Engine::new(weights, config.native.context_tokens, Int4MemoryBudget {
            max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
            max_scratch_payload_bytes: required.scratch_payload_bound,
        }).map_err(native_error)?.with_kernel(kernel);
        match team {
            Some(team) => engine.with_projection_executor(team).map_err(native_error),
            None => Ok(engine),
        }
    })?;
    let result = allocate(output, || {
        let result = prepared.run_on_engine(&admitted, &input.value.planner,
            &mut engine.value, &input.value.vocabulary, control)?;
        check_clean(&engine.value)?;
        checkpoint(control)?;
        Ok(result)
    })?;
    drop(engine);
    drop(admitted);
    drop(input);
    drop(reduction);
    drop(lease);
    Ok(GuardedOutput::new(result.value, result._memory))
}

#[cfg(test)]
mod tests;
