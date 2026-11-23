//! Process-owned prompt scheduling and persistent row teams for raw schemas.
//! The existing prepared extractor owns JSON, source, EOS and selected-row policy.
use super::*;
use crate::{
    native_engine::{constrained::JsonWorkBudget, simd::Int8Kernel,
        strict_int8::{Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::extract::ExtractError,
};

struct ExtractionInput {
    prepared: Int8ExtractPlan,
    vocabulary: Arc<ExtractionVocabulary>,
}

/// This aggregate lives outside the scoped team callback. Native storage
/// drains before its charges; transferred schema/source/vocabulary ownership
/// and all reservations remain live until every persistent child has joined.
struct ExtractionAdmission {
    input: Charged<ExtractionInput>,
    kv: Option<Pending>, scratch: Option<Pending>, output: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>, _lease: EngineLease,
}
impl ExtractionAdmission {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::Limits("INT8 extraction KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::Limits("INT8 extraction scratch admission consumed"))?.commit()?);
        Ok(())
    }
    fn finish(&mut self, result: Int8ExtractRun) -> Result<HostedOutput<Int8ExtractRun>, HostedError> {
        let output = self.output.take().ok_or(HostedError::Limits("INT8 extraction output admission consumed"))?;
        match output.commit() {
            Ok(memory) => Ok(GuardedOutput::new(result, memory)),
            Err(error) => { drop(result); Err(error) }
        }
    }
}

struct ExtractionRequirements { native: Int8MemoryRequirement, scratch_bytes: u64, max_batch_rows: usize }
fn extraction_requirements(limits: SourceLimits, work: Int8Work,
    prefill: Option<Int8PrefillLimits>, team_width: Option<usize>) -> Result<ExtractionRequirements, HostedError> {
    let native = requirements(limits.native)?;
    if limits.preparation_reserve_bytes == 0 || limits.mask_limits.max_trie_node_visits == 0
        || limits.mask_limits.checkpoint_interval_nodes == 0 || limits.max_mask_node_visits == 0
        || work.forward_positions > limits.native.context_tokens as u64 {
        return Err(HostedError::Limits("native extraction preparation/context/mask limits"));
    }
    let extra = prefill.map(Int8PrefillLimits::validate).transpose().map_err(HostedError::Native)?.unwrap_or(0);
    let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    let scratch_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes, extra])?;
    if let Some(width) = team_width {
        // The dispatcher separately reserves child storage. It never replaces
        // coordinator scratch, whole-context KV or either preparation owner.
        sum(&[scratch_bytes, dispatch::int8_prefill_team_workspace_bytes(width, max_batch_rows)?])?;
    }
    Ok(ExtractionRequirements { native, scratch_bytes, max_batch_rows })
}
fn native_budget(limits: SourceLimits, work: Int8Work, kv_bytes: u64) -> Int8JsonBudget {
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: kv_bytes, max_total_mask_node_visits: limits.max_mask_node_visits,
        mask_limits: limits.mask_limits,
    } }
}
fn admit(host: &NlpEngine, prepared: Int8ExtractPlan, vocabulary: Arc<ExtractionVocabulary>,
    limits: SourceLimits, required: &ExtractionRequirements) -> Result<ExtractionAdmission, HostedError> {
    let lease = host.resources().acquire_lease();
    let output = output_claim(&lease, prepared.max_result_bytes(), prepared.options().max_new_tokens as u64)?;
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers,
        limits.preparation_reserve_bytes)?, || Ok(ExtractionInput { prepared, vocabulary }))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
    Ok(ExtractionAdmission { input, kv: Some(kv), scratch: Some(scratch), output: Some(output),
        _kv: None, _scratch: None, _lease: lease })
}
fn execute_prepared(input: &ExtractionInput, engine: &mut StrictInt8Engine<'_>,
    budget: Int8JsonBudget, prefill: Option<Int8PrefillLimits>, control: &mut RunControl)
    -> Result<Int8ExtractRun, HostedError> {
    // The sealed plan additionally clips KV authority to its original task
    // ceiling, and the native driver checks the COMPLETE resident capacity
    // before opening a session. Both head strategies share its finalizer.
    let prepared = &input.prepared;
    let result = match prefill {
        None => prepared.execute(engine, prepared.execution_identity(), &input.vocabulary, budget, control),
        Some(prefill) => prepared.execute_layer_major(engine, prepared.execution_identity(),
            &input.vocabulary, budget, prefill, control),
    }.map_err(HostedError::Extraction)?;
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
        return Err(HostedError::Extraction(Int8ExtractError::Extraction(ExtractError::InvalidResult)));
    }
    Ok(result)
}

impl NlpEngine {
    /// Process-admitted layer-major prompt morsels for an exact compiled schema.
    /// SourceLimits prices transferred schema/source/vocabulary ownership as
    /// well as the existing native and mask bounds. The caller still reserves
    /// preparation before constructing the plan or vocabulary.
    ///
    /// Finalization, exact JSON text, source evidence and complete legal head
    /// rows stay under the original sealed plan. Output memory remains guarded
    /// through serialization/delivery. The ordinary sequential API is unchanged.
    pub fn execute_int8_extract_layer_major(&self, model: &ResidentInt8, prepared: Int8ExtractPlan,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, prefill: Int8PrefillLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ExtractRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let work = prepared.planned_work();
        let required = extraction_requirements(limits, work, Some(prefill), None)?;
        let admission = admit(self, prepared, vocabulary, limits, &required)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the complete ownership aggregate, never its fields.
            let mut admission = admission;
            let mut engine = model.inner.loaded.value.engine(limits.native.context_tokens,
                memory_budget(required.native)).map_err(HostedError::Model)?;
            admission.commit_native()?;
            let result = execute_prepared(&admission.input.value, &mut engine,
                native_budget(limits, work, required.native.kv_bytes), Some(prefill), control)?;
            drop(engine);
            admission.finish(result)
        })
    }

    /// One persistent numerical row team across sequential prompt processing
    /// and schema-constrained decode. Width includes the coordinator and must
    /// be admitted by the installed resource host before the team is formed.
    /// Scheduling never changes source binding, exact schema numbers, selected
    /// head policy or EOS; every legal selected row remains eligible.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_extract_with_threads(&self, model: &ResidentInt8, prepared: Int8ExtractPlan,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, kernel: Int8Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ExtractRun>, HostedError> {
        self.execute_int8_extract_team(model, prepared, vocabulary, limits, kernel, compute_threads, None, cancellation)
    }

    /// Share the same admitted team across bounded layer-major prompt morsels
    /// and constrained decode. Coordinator and child prompt storage are charged
    /// independently before work. One cancellation authority covers the entire
    /// request; no result escapes before finalization and physical worker join.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_extract_layer_major_with_threads(&self, model: &ResidentInt8, prepared: Int8ExtractPlan,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, kernel: Int8Kernel,
        compute_threads: usize, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ExtractRun>, HostedError> {
        self.execute_int8_extract_team(model, prepared, vocabulary, limits, kernel,
            compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int8_extract_team(&self, model: &ResidentInt8, prepared: Int8ExtractPlan,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, kernel: Int8Kernel,
        compute_threads: usize, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8ExtractRun>, HostedError> {
        dispatch::preflight_int8_team(self, limits.native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let work = prepared.planned_work();
        let required = extraction_requirements(limits, work, prefill, Some(compute_threads))?;
        let admission = admit(self, prepared, vocabulary, limits, &required)?;
        dispatch::run_int8_prefill_team(self, limits.native.run, cancellation, model.clone(), compute_threads,
            kernel, required.max_batch_rows, admission, move |model, admission, control, team| {
                let mut engine = model.engine(limits.native.context_tokens, memory_budget(required.native))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission.commit_native()?;
                let result = execute_prepared(&admission.input.value, &mut engine,
                    native_budget(limits, work, required.native.kv_bytes), prefill, control)?;
                drop(engine);
                let compute = team.stats().into();
                Ok(HostedInt8TeamResult::from_guarded(admission.finish(result)?, compute))
            })
    }
}

#[cfg(test)]
mod tests;
