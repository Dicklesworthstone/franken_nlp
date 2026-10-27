//! One charged native extraction and explicitly requested correlated review.
use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int4, strict_int4::{Int4Work, STRICT_INT4_PROFILE}},
    tasks::{ir::TaskBudget, judge::JudgeLimits,
        extract::{ExtractionVocabulary, semantic::{SemanticLimits, grouped_int4::{
            Int4SemanticError, Int4SemanticExtractionRun, PreparedInt4SemanticExtraction,
            INT4_SEMANTIC_EXECUTION}}}},
};

/// Explicit native, mask and preparation admission for BOTH stages. Preparation
/// reserve prices pinned compilers/vocabulary, original schema/source graphs,
/// allocator slack and caller-owned capacities; variable retained second-reader
/// data adds the independent checked floors returned by `requirements`.
#[derive(Clone, Copy, Debug)]
pub struct Int4SemanticHostLimits {
    pub native: NativeLimits,
    pub masks: MaskWorkLimits,
    pub max_mask_node_visits: u64,
    pub preparation_reserve_bytes: u64,
}

/// Modeled simultaneous ledger occupancy, not measured/enforced allocator RSS.
/// Resident model, caller preparation and runtime cleanup are additional.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SemanticHostRequirements {
    pub native: Int4MemoryRequirement,
    pub native_workspace_bytes: u64,
    pub preparation_bytes: u64,
    pub output_bytes: u64,
}
impl Int4SemanticHostRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.native_workspace_bytes, self.preparation_bytes, self.output_bytes])
    }
}
impl Int4SemanticHostLimits {
    /// Pure preflight quote shared with the CLI before model/snapshot IO. Each
    /// field judge retains its own full source, complete head plans and scores;
    /// all plans coexist before any second-reader forward and while results
    /// accumulate. The final output never discounts those intermediates.
    #[allow(clippy::too_many_arguments)]
    pub fn requirements(self, width: usize, extraction: TaskBudget, judge: TaskBudget,
        semantic: SemanticLimits, judge_planning: JudgeLimits, source_bytes: usize)
        -> Result<Int4SemanticHostRequirements, HostedError> {
        self.native.run.validate()?;
        extraction.validate().map_err(|_| HostedError::Limits("INT4 semantic extraction budget"))?;
        judge.validate().map_err(|_| HostedError::Limits("INT4 semantic judge budget"))?;
        if self.preparation_reserve_bytes == 0 || self.native.allocator_reserve_bytes == 0
            || self.masks.max_trie_node_visits == 0 || self.masks.checkpoint_interval_nodes == 0
            || self.max_mask_node_visits == 0 || source_bytes == 0
            || !(1..=4096).contains(&semantic.max_fields) || semantic.max_total_claim_bytes == 0
            || semantic.max_template_bytes == 0 || semantic.max_walk_nodes == 0
            || semantic.max_input_json_bytes == 0 || semantic.max_total_prompt_tokens == 0
            || semantic.max_output_bytes == 0 || judge_planning.max_total_prompt_tokens == 0
            || judge_planning.max_total_candidate_tokens == 0 || judge_planning.max_output_bytes == 0
            || judge_planning.per_head.max_candidates == 0 || judge_planning.per_head.max_candidate_id_bytes == 0 {
            return Err(HostedError::Limits("INT4 semantic complete preparation and mask limits"));
        }
        let native = Int4MemoryRequirement::for_context(self.native.context_tokens, width).map_err(native_error)?;
        if native.kv_bytes > extraction.max_kv_bytes || native.kv_bytes > judge.max_kv_bytes {
            return Err(HostedError::Limits("INT4 semantic both stages need full resident KV"));
        }
        let fields = semantic.max_fields as u64;
        let sources = product(fields.checked_add(2).ok_or(HostedError::Limits("INT4 semantic field arithmetic"))?,
            source_bytes as u64)?;
        // TaskIR prompt copies/serialization, retained scorer and native
        // schedule nodes, candidate IDs, and plan/bundle/evidence bookkeeping.
        // The shared judge compiler admits at most 32 heads and 33 candidates
        // per head. All multipliers are conservative modeled storage floors.
        let prompts = product(semantic.max_total_prompt_tokens as u64, 64)?;
        let candidates = product(product(fields, judge_planning.max_total_candidate_tokens as u64)?, 256)?;
        let ids = product(product(product(fields, 32 * 33)?,
            judge_planning.per_head.max_candidate_id_bytes as u64)?, 2)?;
        let plans = product(fields, 65_536)?;
        let judge_results = product(product(fields,
            judge.max_output_bytes.min(judge_planning.max_output_bytes))?, 4)?;
        // JSON pointers may repeat ancestor keys across leaves; escaped paths
        // can approach twice the complete input size per field. Parser/tree
        // and bounded recursive walk storage remain live alongside claims.
        let pointers = product(product(fields, semantic.max_input_json_bytes as u64)?, 2)?;
        let preparation_bytes = sum(&[
            self.preparation_reserve_bytes, sources, prompts, candidates, ids, plans, judge_results, pointers,
            product(extraction.max_output_bytes, 4)?, product(semantic.max_input_json_bytes as u64, 128)?,
            product(semantic.max_walk_nodes as u64, 128)?, product(semantic.max_total_claim_bytes as u64, 4)?,
            product(semantic.max_template_bytes as u64, 4)?,
        ])?;
        let native_workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
            self.native.allocator_reserve_bytes])?;
        let output_bytes = sum(&[product(semantic.max_output_bytes, 4)?,
            product(u64::from(extraction.max_output_tokens), 8)?, 4096])?;
        let required = Int4SemanticHostRequirements { native, native_workspace_bytes, preparation_bytes, output_bytes };
        required.total_bytes()?;
        Ok(required)
    }
}
fn product(left: u64, right: u64) -> Result<u64, HostedError> {
    left.checked_mul(right).ok_or(HostedError::Limits("INT4 semantic memory arithmetic"))
}

struct SemanticInput {
    prepared: Option<PreparedInt4SemanticExtraction>,
    vocabulary: Arc<ExtractionVocabulary>,
}

impl NlpEngine {
    /// Execute a real extraction followed by every configured present scalar's
    /// full-source faithfulness judgment. Null/verbatim/unmapped fields remain
    /// explicitly not checked. The original extraction is preserved; the
    /// correlated, uncalibrated diagnostic neither repairs nor accepts it.
    ///
    /// All dynamic judges must compile and preflight before the first review
    /// forward. The original maximum extraction work is never refunded after
    /// early EOS. One failed stage returns no combined result. Preparation,
    /// native and output guards stay owned through physical cancellation/drain.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_semantic_extract(&self, model: &ResidentInt4,
        prepared: PreparedInt4SemanticExtraction, vocabulary: Arc<ExtractionVocabulary>,
        limits: Int4SemanticHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SemanticExtractionRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        for identity in [prepared.execution_identity(), prepared.judge_identity()] {
            constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
        }
        let width = prepared.group_width();
        let extraction_work = prepared.extraction_reserved_work();
        let max_work = prepared.max_model_work();
        if width != model.model_identity().group_width() || prepared.execution_identity().task_spec != "extract-v1"
            || prepared.judge_identity().task_spec != "judge-v1" || extraction_work.forward_positions == 0
            || extraction_work.forward_positions > limits.native.context_tokens as u64 || !fits(extraction_work, max_work) {
            return Err(HostedError::Limits("INT4 semantic actual model and original extraction reservation"));
        }
        let required = limits.requirements(width, prepared.extraction_budget(), prepared.judge_budget(),
            prepared.semantic_limits(), prepared.judge_planning_limits(), prepared.source_bytes())?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.preparation_bytes)?,
            || Ok(SemanticInput { prepared: Some(prepared), vocabulary }))?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.native_workspace_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                        max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&mut engine.value, &input.value.vocabulary,
                    limits.masks, limits.max_mask_node_visits, control).map_err(semantic_error)?;
                check_completion(&result, width, extraction_work, max_work)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(semantic_error(Int4SemanticError::Accounting));
                }
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn check_completion(run: &Int4SemanticExtractionRun, width: usize, extraction: Int4Work, ceiling: Int4Work)
    -> Result<(), HostedError> {
    let reserved = extraction.checked_add(run.semantic_model_work).map_err(|_| semantic_error(Int4SemanticError::Accounting))?;
    let actual = run.extraction.model_work.checked_add(run.semantic_model_work)
        .map_err(|_| semantic_error(Int4SemanticError::Accounting))?;
    if run.schema_version != 1 || run.execution != INT4_SEMANTIC_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
        || run.group_width != width || run.extraction.group_width != width || run.extraction_reserved_work != extraction
        || run.reserved_model_work != reserved || run.model_work != actual || !fits(run.extraction.model_work, extraction)
        || !fits(actual, reserved) || !fits(reserved, ceiling) || !run.semantic.same_model_correlated
        || run.semantic.calibration != "experimental_uncalibrated_diagnostic_only" {
        return Err(semantic_error(Int4SemanticError::Accounting));
    }
    Ok(())
}
fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn semantic_error(error: Int4SemanticError) -> HostedError {
    HostedError::Int4(Int4HostError::SemanticExtraction(error))
}

#[cfg(test)]
mod tests;
