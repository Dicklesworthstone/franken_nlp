//! One admitted INT8 extraction and explicitly requested correlated review.
//! Both stages share one native engine, physical invocation and work ceiling.
use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{decode::DecodeStepControl, simd::Int8Kernel,
        strict_int8::{Int8Work, StrictInt8Engine, STRICT_INT8_PROFILE, prefill::Int8PrefillLimits}},
    tasks::{ir::TaskBudget, judge::JudgeLimits,
        extract::{quantized::INT8_EXTRACT_VERSION, semantic::{SemanticLimits, int8::{
            Int8SemanticError, Int8SemanticExtractionRun, PreparedInt8SemanticExtraction,
            INT8_SEMANTIC_EXECUTION}}}},
    tokenizer::untrusted::UntrustedEncoding,
};

/// Explicit native, mask and preparation admission for both stages. Caller
/// reserve covers pinned compilers/vocabulary, original schema/source graphs,
/// allocator slack and caller-owned capacities. Retained second-reader data
/// adds the independently checked floors returned by `requirements`.
#[derive(Clone, Copy, Debug)]
pub struct Int8SemanticHostLimits {
    pub native: NativeLimits,
    pub masks: MaskWorkLimits,
    pub max_mask_node_visits: u64,
    pub preparation_reserve_bytes: u64,
}

/// Modeled simultaneous ledger occupancy, not measured allocator RSS. Resident
/// weights, caller preparation and runtime cleanup remain separate charges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SemanticHostRequirements {
    pub native: Int8MemoryRequirement,
    pub native_workspace_bytes: u64,
    pub preparation_bytes: u64,
    pub output_bytes: u64,
}
impl Int8SemanticHostRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.native_workspace_bytes, self.preparation_bytes, self.output_bytes])
    }
}
impl Int8SemanticHostLimits {
    /// Pure quote usable before model/source IO. Every field judge retains its
    /// full source and complete head plans; all plans coexist before any review
    /// forward and while results accumulate. Final output capacity does not
    /// replace or discount these intermediates. Full-vocabulary projection
    /// remains the scoring contract for both stages.
    pub fn requirements(self, extraction: TaskBudget, judge: TaskBudget,
        semantic: SemanticLimits, judge_planning: JudgeLimits, source_bytes: usize)
        -> Result<Int8SemanticHostRequirements, HostedError> {
        self.requirements_with_encoding(extraction, judge, semantic, judge_planning,
            source_bytes, &UntrustedEncoding::ByteFallback)
    }

    /// Original extraction and all field reviews reuse one transient encoder
    /// workspace. Its charge is additional to every retained prompt and result.
    #[allow(clippy::too_many_arguments)]
    pub fn requirements_with_encoding(self, extraction: TaskBudget, judge: TaskBudget,
        semantic: SemanticLimits, judge_planning: JudgeLimits, source_bytes: usize, encoding: &UntrustedEncoding)
        -> Result<Int8SemanticHostRequirements, HostedError> {
        self.requirements_with_schedule(extraction, judge, semantic, judge_planning,
            source_bytes, encoding, None)
    }

    /// Quote the same retained preparation and output plus one reusable prompt
    /// morsel workspace. Persistent worker scratch is admitted independently by
    /// the process dispatcher and never discounts this coordinator reservation.
    #[allow(clippy::too_many_arguments)]
    pub fn requirements_with_schedule(self, extraction: TaskBudget, judge: TaskBudget,
        semantic: SemanticLimits, judge_planning: JudgeLimits, source_bytes: usize,
        encoding: &UntrustedEncoding, prefill: Option<Int8PrefillLimits>)
        -> Result<Int8SemanticHostRequirements, HostedError> {
        self.native.run.validate()?;
        let extra = prefill.map(Int8PrefillLimits::validate).transpose()
            .map_err(HostedError::Native)?.unwrap_or(0);
        let encoding_workspace_bytes = match encoding {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map_err(|_| HostedError::Limits("INT8 semantic input encoding policy"))?.workspace_bytes,
        };
        extraction.validate().map_err(|_| HostedError::Limits("INT8 semantic extraction budget"))?;
        judge.validate().map_err(|_| HostedError::Limits("INT8 semantic judge budget"))?;
        if self.preparation_reserve_bytes == 0 || self.native.allocator_reserve_bytes == 0
            || self.masks.max_trie_node_visits == 0 || self.masks.checkpoint_interval_nodes == 0
            || self.max_mask_node_visits == 0 || source_bytes == 0
            || !(1..=4096).contains(&semantic.max_fields) || semantic.max_claim_bytes == 0
            || semantic.max_total_claim_bytes == 0 || semantic.max_template_bytes == 0
            || semantic.max_walk_nodes == 0 || semantic.max_input_json_bytes == 0
            || semantic.max_total_prompt_tokens == 0 || semantic.max_total_projected_logits == 0
            || semantic.max_output_bytes == 0 || judge_planning.max_total_prompt_tokens == 0
            || judge_planning.max_total_candidate_tokens == 0 || judge_planning.max_total_projected_logits == 0
            || judge_planning.max_output_bytes == 0 || judge_planning.per_head.max_candidates == 0
            || judge_planning.per_head.max_candidate_id_bytes == 0 {
            return Err(HostedError::Limits("INT8 semantic complete preparation and mask limits"));
        }
        let native = requirements(self.native)?;
        if native.kv_bytes > extraction.max_kv_bytes || native.kv_bytes > judge.max_kv_bytes {
            return Err(HostedError::Limits("INT8 semantic both stages need full resident KV"));
        }
        let fields = semantic.max_fields as u64;
        let sources = product(fields.checked_add(2).ok_or(HostedError::Limits("INT8 semantic field arithmetic"))?,
            source_bytes as u64)?;
        // TaskIR copies/serialization, scorer/native schedule nodes and complete
        // plan/bundle/evidence storage. The judge compiler has at most 32 heads
        // and 33 candidates per head, with no caller-provided extra heads.
        let prompts = product(semantic.max_total_prompt_tokens as u64, 64)?;
        let candidates = product(product(fields, judge_planning.max_total_candidate_tokens as u64)?, 256)?;
        let ids = product(product(product(fields, 32 * 33)?,
            judge_planning.per_head.max_candidate_id_bytes as u64)?, 2)?;
        let plans = product(fields, 65_536)?;
        let judge_results = product(product(fields,
            judge.max_output_bytes.min(judge_planning.max_output_bytes))?, 4)?;
        // Escaped JSON pointers can repeat almost twice the entire input for
        // each leaf. The parsed tree/walk and exact claims remain live together.
        let pointers = product(product(fields, semantic.max_input_json_bytes as u64)?, 2)?;
        let preparation_bytes = sum(&[
            self.preparation_reserve_bytes, encoding_workspace_bytes, sources, prompts, candidates, ids, plans, judge_results, pointers,
            product(extraction.max_output_bytes, 4)?, product(semantic.max_input_json_bytes as u64, 128)?,
            product(semantic.max_walk_nodes as u64, 128)?, product(semantic.max_total_claim_bytes as u64, 4)?,
            product(semantic.max_template_bytes as u64, 4)?,
        ])?;
        let native_workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
            self.native.allocator_reserve_bytes, extra])?;
        let output_bytes = sum(&[product(semantic.max_output_bytes, 4)?,
            product(u64::from(extraction.max_output_tokens), 8)?, 4096])?;
        let required = Int8SemanticHostRequirements { native, native_workspace_bytes, preparation_bytes, output_bytes };
        required.total_bytes()?;
        Ok(required)
    }
}
fn product(left: u64, right: u64) -> Result<u64, HostedError> {
    left.checked_mul(right).ok_or(HostedError::Limits("INT8 semantic memory arithmetic"))
}

struct SemanticInput {
    prepared: Option<PreparedInt8SemanticExtraction>,
    vocabulary: Arc<ExtractionVocabulary>,
}

/// The dispatcher owns this aggregate outside its scoped-team callback.
/// Native storage drops before its charges; preparation, output and native
/// reservations remain owned until every persistent child has physically joined.
struct SemanticAdmission {
    input: Charged<SemanticInput>,
    kv: Option<Pending>, scratch: Option<Pending>, output: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>, _lease: EngineLease,
}
impl SemanticAdmission {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::CompletionMissing)?.commit()?);
        Ok(())
    }
    fn finish(&mut self, result: Int8SemanticExtractionRun)
        -> Result<HostedOutput<Int8SemanticExtractionRun>, HostedError> {
        let output = self.output.take().ok_or(HostedError::CompletionMissing)?;
        match output.commit() {
            Ok(memory) => Ok(GuardedOutput::new(result, memory)),
            Err(error) => { drop(result); Err(error) }
        }
    }
}
fn admit(host: &NlpEngine, prepared: PreparedInt8SemanticExtraction,
    vocabulary: Arc<ExtractionVocabulary>, required: Int8SemanticHostRequirements)
    -> Result<SemanticAdmission, HostedError> {
    let lease = host.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.preparation_bytes)?,
        || Ok(SemanticInput { prepared: Some(prepared), vocabulary }))?;
    let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_bytes)?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.native_workspace_bytes)?;
    Ok(SemanticAdmission { input, kv: Some(kv), scratch: Some(scratch), output: Some(output),
        _kv: None, _scratch: None, _lease: lease })
}
fn execute_prepared(input: &mut SemanticInput, engine: &mut StrictInt8Engine<'_>,
    limits: Int8SemanticHostLimits, prefill: Option<Int8PrefillLimits>, control: &mut RunControl)
    -> Result<Int8SemanticExtractionRun, HostedError> {
    checkpoint(control)?;
    let prepared = input.prepared.take().ok_or(HostedError::CompletionMissing)?;
    let extraction_work = prepared.extraction_reserved_work();
    let max_work = prepared.max_model_work();
    let encoding = prepared.input_encoding().clone();
    let initial_encoding_work = prepared.input_encoding_work().unwrap_or_default();
    let result = match prefill {
        None => prepared.execute_with_control(engine, &input.vocabulary,
            limits.masks, limits.max_mask_node_visits, control),
        Some(prefill) => prepared.execute_layer_major_with_control(engine, &input.vocabulary,
            limits.masks, limits.max_mask_node_visits, prefill, control),
    }.map_err(HostedError::SemanticExtraction)?;
    check_completion(&result, extraction_work, max_work)?;
    result.verify_input_encoding(&encoding, initial_encoding_work).map_err(HostedError::SemanticExtraction)?;
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
        return Err(HostedError::SemanticExtraction(Int8SemanticError::Accounting));
    }
    checkpoint(control)?;
    Ok(result)
}

impl NlpEngine {
    /// Preserve the real extraction and attach each configured scalar's
    /// complete-source faithfulness judgment. The same-model diagnostic is
    /// experimental and uncalibrated; it neither repairs nor accepts output.
    /// All dynamic judges compile and preflight before any review forward.
    /// Original maximum extraction work is never refunded after early EOS.
    /// A failed stage returns no combined result. Native/preparation guards
    /// remain owned until physical completion; output stays charged to its
    /// returned guard through publication.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_semantic_extract(&self, model: &ResidentInt8,
        prepared: PreparedInt8SemanticExtraction, vocabulary: Arc<ExtractionVocabulary>,
        limits: Int8SemanticHostLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SemanticExtractionRun>, HostedError> {
        self.execute_int8_semantic_extract_with_prefill(model, prepared, vocabulary, limits, None, cancellation)
    }

    /// Reuse bounded prompt morsels through extraction and all correlated
    /// review heads while retaining complete-source and full-head contracts.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_semantic_extract_layer_major(&self, model: &ResidentInt8,
        prepared: PreparedInt8SemanticExtraction, vocabulary: Arc<ExtractionVocabulary>,
        limits: Int8SemanticHostLimits, prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SemanticExtractionRun>, HostedError> {
        self.execute_int8_semantic_extract_with_prefill(model, prepared, vocabulary, limits, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int8_semantic_extract_with_prefill(&self, model: &ResidentInt8,
        prepared: PreparedInt8SemanticExtraction, vocabulary: Arc<ExtractionVocabulary>,
        limits: Int8SemanticHostLimits, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SemanticExtractionRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        let required = checked_requirements(self, model, &prepared, limits, prefill)?;
        let admission = admit(self, prepared, vocabulary, required)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut admission = admission;
            checkpoint(control)?;
            let mut engine = model.inner.loaded.value.engine(limits.native.context_tokens,
                memory_budget(required.native)).map_err(HostedError::Model)?;
            admission.commit_native()?;
            let result = execute_prepared(&mut admission.input.value, &mut engine, limits, prefill, control)?;
            drop(engine);
            admission.finish(result)
        })
    }

    /// One persistent numerical team covers extraction and every review.
    /// Width includes the coordinator; no child is admitted anew per field,
    /// and no combined output escapes before complete physical worker join.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_semantic_extract_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8SemanticExtraction, vocabulary: Arc<ExtractionVocabulary>,
        limits: Int8SemanticHostLimits, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt8TeamResult<Int8SemanticExtractionRun>, HostedError> {
        self.execute_int8_semantic_extract_team(model, prepared, vocabulary, limits,
            kernel, compute_threads, None, cancellation)
    }

    /// Keep that same team and one admitted prompt workspace across the whole
    /// request. Shared cancellation, exact JSON, full-vocabulary scores and
    /// the all-judges preflight barrier remain unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_semantic_extract_layer_major_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8SemanticExtraction, vocabulary: Arc<ExtractionVocabulary>,
        limits: Int8SemanticHostLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SemanticExtractionRun>, HostedError> {
        self.execute_int8_semantic_extract_team(model, prepared, vocabulary, limits,
            kernel, compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int8_semantic_extract_team(&self, model: &ResidentInt8,
        prepared: PreparedInt8SemanticExtraction, vocabulary: Arc<ExtractionVocabulary>,
        limits: Int8SemanticHostLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SemanticExtractionRun>, HostedError> {
        dispatch::preflight_int8_team(self, limits.native.run, compute_threads)?;
        let required = checked_requirements(self, model, &prepared, limits, prefill)?;
        let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
        sum(&[required.total_bytes()?,
            dispatch::int8_prefill_team_workspace_bytes(compute_threads, max_batch_rows)?])?;
        let admission = admit(self, prepared, vocabulary, required)?;
        dispatch::run_int8_prefill_team(self, limits.native.run, cancellation, model.clone(),
            compute_threads, kernel, max_batch_rows, admission, move |model, admission, control, team| {
                checkpoint(control)?;
                let mut engine = model.engine(limits.native.context_tokens, memory_budget(required.native))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission.commit_native()?;
                let result = execute_prepared(&mut admission.input.value, &mut engine, limits, prefill, control)?;
                drop(engine);
                let compute = team.stats().into();
                Ok(HostedInt8TeamResult::from_guarded(admission.finish(result)?, compute))
            })
    }
}

fn checked_requirements(host: &NlpEngine, model: &ResidentInt8,
    prepared: &PreparedInt8SemanticExtraction, limits: Int8SemanticHostLimits, prefill: Option<Int8PrefillLimits>)
    -> Result<Int8SemanticHostRequirements, HostedError> {
    host.check_resident_domain(model)?;
    for identity in [prepared.execution_identity(), prepared.judge_identity()] {
        check_model_identity(model.artifact_identity(), identity)?;
    }
    let work = prepared.extraction_reserved_work();
    if prepared.execution_identity().task_spec != "extract-v1" || prepared.judge_identity().task_spec != "judge-v1"
        || prepared.extraction_selected_rows().is_some()
        || work.forward_positions == 0 || work.forward_positions > limits.native.context_tokens as u64
        || !fits(work, prepared.max_model_work()) {
        return Err(HostedError::Limits("INT8 semantic original extraction reservation and full-head policy"));
    }
    limits.requirements_with_schedule(prepared.extraction_budget(), prepared.judge_budget(),
        prepared.semantic_limits(), prepared.judge_planning_limits(), prepared.source_bytes(),
        prepared.input_encoding(), prefill)
}
fn check_completion(run: &Int8SemanticExtractionRun, extraction: Int8Work, ceiling: Int8Work)
    -> Result<(), HostedError> {
    let refused = || HostedError::SemanticExtraction(Int8SemanticError::Accounting);
    let reserved = extraction.checked_add(run.semantic_model_work).map_err(|_| refused())?;
    let actual = run.extraction.model_work.checked_add(run.semantic_model_work).map_err(|_| refused())?;
    if run.schema_version != 1 || run.execution != INT8_SEMANTIC_EXECUTION || run.numerics_profile != STRICT_INT8_PROFILE
        || run.extraction.schema_version != 1 || run.extraction.execution != INT8_EXTRACT_VERSION
        || run.extraction.result.output.numerics_profile != STRICT_INT8_PROFILE
        || run.extraction_reserved_work != extraction || run.reserved_model_work != reserved
        || run.model_work != actual || !fits(run.extraction.model_work, extraction)
        || !fits(actual, reserved) || !fits(reserved, ceiling) || !run.semantic.same_model_correlated
        || run.semantic.calibration != "experimental_uncalibrated_diagnostic_only" {
        return Err(refused());
    }
    Ok(())
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn checkpoint(control: &mut RunControl) -> Result<(), HostedError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(HostedError::SemanticExtraction(Int8SemanticError::Cancelled(cause))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
