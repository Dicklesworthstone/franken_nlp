//! One charged native partial-object map and original-schema reconciliation.
use super::*;
use crate::{
    batch::extract::quantized::long::object::{Int8ObjectExtractionPlanner,
        Int8ObjectExtractionMapLimits, Int8ObjectExtractionMapRun,
        Int8ObjectExtractionMapError, PreparedInt8ObjectExtractionMap},
    grammar::CompileLimits,
    tasks::extract::object_reconcile::{RootObjectLimits, RootObjectSchema},
};

/// Explicit opt-in to partial root objects. The original caller schema stays
/// in mapping.request; only the sealed object planner derives chunk schemas.
pub struct Int8ObjectExtractionMapConfig {
    pub mapping: ExtractionMapConfig,
    pub reconciliation: RootObjectLimits,
}
impl Int8ObjectExtractionMapConfig {
    pub fn limits(&self) -> Int8ObjectExtractionMapLimits {
        Int8ObjectExtractionMapLimits { mapping: self.mapping.limits, reconciliation: self.reconciliation }
    }
    /// Complete retained map frontier plus both schema declarations and exact
    /// reconciliation. Native engine, input and final output have separate
    /// reservations; none of these bounds renew for another source chunk.
    pub fn workspace_bytes(&self, chunks: usize, compiler: CompileLimits) -> Result<u64, HostedError> {
        sum(&[reduction_bytes(&self.mapping, chunks)?, self.object_workspace_bytes(compiler)?])
    }
    fn object_workspace_bytes(&self, compiler: CompileLimits) -> Result<u64, HostedError> {
        sum(&[self.limits().reconciliation_workspace_bytes().map_err(object_error)?,
            RootObjectSchema::workspace_bytes(compiler).map_err(|error| object_error(error.into()))?])
    }
}
struct ObjectInput {
    source: String,
    planner: Arc<Int8ObjectExtractionPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: Int8ObjectExtractionMapConfig,
}

/// Kept outside the scoped callback by dispatch. On error, cancellation or
/// unwind, source/schema/reducer/native charges outlive physical worker join.
struct ObjectAdmission {
    input: Charged<ObjectInput>,
    _object_workspace: Pending,
    kv: Option<Pending>,
    scratch: Option<Pending>,
    output: Option<Pending>,
    reduction: Option<Pending>,
    _kv: Option<CommittedMemory>,
    _scratch: Option<CommittedMemory>,
    lease: EngineLease,
}

impl NlpEngine {
    /// Reconcile complete partial root objects from every original chunk.
    /// Equal exact values retain all origins; conflicts or missing final
    /// required fields refuse the invocation. Independent final source checks
    /// spend the remaining original verification allowance, never a fresh one.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_object_document(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionMapConfig, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ObjectExtractionMapRun>, HostedError> {
        self.extract_int8_object_document_scheduled(model, source, planner, vocabulary, config, None, cancellation)
    }

    /// Reuse one bounded prompt-morsel workspace through the complete map;
    /// object reconciliation does not repeat model inference.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_object_document_layer_major(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionMapConfig, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ObjectExtractionMapRun>, HostedError> {
        self.extract_int8_object_document_scheduled(model, source, planner, vocabulary, config, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_int8_object_document_scheduled(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionMapConfig, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ObjectExtractionMapRun>, HostedError> {
        let base = &config.mapping;
        dispatch::preflight(self, base.native.run)?;
        self.check_resident_domain(model)?;
        planner.base_planner().map_document_encoding_workspace_bytes().map_err(HostedError::BatchSetup)?;
        check_model_identity(model.artifact_identity(), planner.base_execution_identity())?;
        let required = requirements(base.native)?;
        validate(base, planner.base_execution_identity(), source.len(), required.kv_bytes)?;
        let scratch_bytes = scratch_bytes(base.native, required, prefill)?;
        let object_bytes = config.object_workspace_bytes(planner.compiler_limits())?;
        let bytes = input_bytes(source.capacity(), base, planner.base_planner())?;
        let run = base.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ObjectInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config.mapping;
            // Derivation allocates both original and partial declarations;
            // reserve this independent schema/reducer peak before preparation.
            let object_workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, object_bytes)?;
            let (prepared, admitted) = prepare_object(&input.value, model.artifact_identity(), control)?;
            let count = prepared.preflight_metadata().chunk_count();
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, reduction_bytes(config, count)?)?;
            let tokens = (count as u64).checked_mul(u64::from(config.request.budget.max_output_tokens))
                .ok_or(HostedError::Limits("INT8 object map token arithmetic"))?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || execute_object(prepared, &admitted, &mut engine.value,
                &input.value.vocabulary, prefill, control))?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(object_workspace);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Keep one persistent numerical team across every partial-object chunk.
    /// The complete sealed map and final object return only after worker join.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_object_document_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionMapConfig, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt8TeamResult<Int8ObjectExtractionMapRun>, HostedError> {
        self.extract_int8_object_document_team(model, source, planner, vocabulary, config,
            kernel, compute_threads, None, cancellation)
    }

    /// Compose prompt morsels with the same fixed team used for all constrained
    /// decoding. Both coordinator and worker prompt storage remain charged.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_int8_object_document_layer_major_with_threads(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionMapConfig, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ObjectExtractionMapRun>, HostedError> {
        self.extract_int8_object_document_team(model, source, planner, vocabulary, config,
            kernel, compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_int8_object_document_team(&self, model: &ResidentInt8, source: String,
        planner: Arc<Int8ObjectExtractionPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int8ObjectExtractionMapConfig, kernel: Int8Kernel, compute_threads: usize,
        prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ObjectExtractionMapRun>, HostedError> {
        let base = &config.mapping;
        dispatch::preflight_int8_team(self, base.native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        planner.base_planner().map_document_encoding_workspace_bytes().map_err(HostedError::BatchSetup)?;
        check_model_identity(model.artifact_identity(), planner.base_execution_identity())?;
        let required = requirements(base.native)?;
        validate(base, planner.base_execution_identity(), source.len(), required.kv_bytes)?;
        let scratch_bytes = scratch_bytes(base.native, required, prefill)?;
        let object_bytes = config.object_workspace_bytes(planner.compiler_limits())?;
        let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
        sum(&[scratch_bytes, object_bytes,
            dispatch::int8_prefill_team_workspace_bytes(compute_threads, max_batch_rows)?])?;
        let bytes = input_bytes(source.capacity(), base, planner.base_planner())?;
        let run = base.native.run;
        let lease = self.resources().acquire_lease();
        let object_workspace = Pending::reserve(&lease, MemoryClass::JobBuffers, object_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(ObjectInput { source, planner, vocabulary, config }))?;
        let admission = ObjectAdmission { input, _object_workspace: object_workspace, kv: Some(kv), scratch: Some(scratch),
            output: None, reduction: None, _kv: None, _scratch: None, lease };
        let artifact = model.artifact_identity().clone();
        dispatch::run_int8_prefill_team(self, run, cancellation, model.clone(), compute_threads,
            kernel, max_batch_rows, admission, move |model, admission, control, team| {
                let input = &admission.input.value;
                let config = &input.config.mapping;
                let (prepared, admitted) = prepare_object(input, &artifact, control)?;
                let count = prepared.preflight_metadata().chunk_count();
                admission.reduction = Some(Pending::reserve(&admission.lease, MemoryClass::JobBuffers,
                    reduction_bytes(config, count)?)?);
                let tokens = (count as u64).checked_mul(u64::from(config.request.budget.max_output_tokens))
                    .ok_or(HostedError::Limits("INT8 object map token arithmetic"))?;
                admission.output = Some(output_claim(&admission.lease, prepared.max_result_bytes(), tokens)?);
                let mut engine = model.engine(config.native.context_tokens, memory_budget(required))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission._kv = Some(admission.kv.take()
                    .ok_or(HostedError::Limits("INT8 object map KV admission consumed"))?.commit()?);
                admission._scratch = Some(admission.scratch.take()
                    .ok_or(HostedError::Limits("INT8 object map scratch admission consumed"))?.commit()?);
                let result = execute_object(prepared, &admitted, &mut engine, &input.vocabulary, prefill, control)?;
                drop(engine);
                let compute = team.stats().into();
                let output = admission.output.take()
                    .ok_or(HostedError::Limits("INT8 object map output admission consumed"))?;
                let memory = match output.commit() {
                    Ok(memory) => memory,
                    Err(error) => { drop(result); return Err(error); }
                };
                Ok(HostedInt8TeamResult::from_guarded(GuardedOutput::new(result, memory), compute))
            })
    }
}

fn prepare_object<'s>(input: &'s ObjectInput, artifact: &ArtifactIdentity, control: &mut RunControl)
    -> Result<(PreparedInt8ObjectExtractionMap<'s>, Vec<ExecutionIdentity>), HostedError> {
    let prepared = input.planner.plan_document_with_control(&input.source,
        &input.config.mapping.request, input.config.limits(), control).map_err(object_error)?;
    let mut admitted = Vec::new();
    admitted.try_reserve_exact(prepared.preflight_metadata().chunk_count())
        .map_err(|_| HostedError::Limits("INT8 object map identity storage"))?;
    for identity in prepared.execution_identities() {
        poll(control)?;
        check_model_identity(artifact, identity)?;
        check_identity(identity)?;
        let base = input.planner.base_execution_identity();
        if identity.tokenizer_digest != base.tokenizer_digest || identity.template_digest != base.template_digest {
            return Err(HostedError::ModelIdentity);
        }
        admitted.push(identity.clone());
    }
    poll(control)?;
    Ok((prepared, admitted))
}
fn execute_object(prepared: PreparedInt8ObjectExtractionMap<'_>, admitted: &[ExecutionIdentity],
    engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary,
    prefill: Option<Int8PrefillLimits>, control: &mut RunControl) -> Result<Int8ObjectExtractionMapRun, HostedError> {
    let expected = prepared.preflight_metadata();
    let result = match prefill {
        None => prepared.execute_with_control(admitted, engine, vocabulary, control),
        Some(prefill) => prepared.execute_layer_major_with_control(admitted, engine, vocabulary, prefill, control),
    }.map_err(object_error)?;
    expected.verify_completed_with_control(&result, control).map_err(object_error)?;
    poll(control)?;
    Ok(result)
}
fn object_error(error: Int8ObjectExtractionMapError) -> HostedError {
    HostedError::Int8ObjectExtractionMap(error)
}

#[cfg(test)] mod tests;
