//! One charged native source task and explicitly selected whole-source review.
use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    execution_identity::Sha256Digest,
    native_engine::{constrained_int4, strict_int4::{Int4Work, STRICT_INT4_PROFILE}},
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding},
    tasks::{extract::ExtractionVocabulary, ir::TaskBudget, judge::JudgeLimits,
        extract::semantic::grouped_int4::Int4SemanticEncoding,
        source_planning::{SourceTaskResult, grouped_int4::{INT4_SOURCE_EXECUTION, semantic::{
            Int4SourceSemanticError, Int4SourceSemanticRun, PreparedInt4SourceSemantic,
            SourceSemanticInputFacts, SourceSemanticLimits, SourceSemanticStatus,
            INT4_SOURCE_SEMANTIC_EXECUTION, SOURCE_SEMANTIC_VERSION, SOURCE_SEMANTIC_CLAIM_VERSION,
            SOURCE_SEMANTIC_SCOPE, SOURCE_SEMANTIC_CALIBRATION}}}},
};

/// Admission for generation, every full-source judge and complete publication.
/// These are modeled storage floors, not an allocator or operating-system cap.
#[derive(Clone, Copy, Debug)]
pub struct Int4SourceSemanticHostLimits {
    pub native: NativeLimits,
    pub masks: MaskWorkLimits,
    pub max_mask_node_visits: u64,
    pub preparation_reserve_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceSemanticHostRequirements {
    pub native: Int4MemoryRequirement,
    pub native_workspace_bytes: u64,
    pub preparation_bytes: u64,
    pub output_bytes: u64,
}
impl Int4SourceSemanticHostRequirements {
    pub fn total_bytes(self) -> Result<u64, HostedError> {
        sum(&[self.native.kv_bytes, self.native_workspace_bytes, self.preparation_bytes, self.output_bytes])
    }
}
impl Int4SourceSemanticHostLimits {
    /// Shared pre-model quote. All dynamic judges coexist before the first
    /// review forward, so source copies, complete prompts and accumulated
    /// results are counted for the ENTIRE statement ceiling simultaneously.
    #[allow(clippy::too_many_arguments)]
    pub fn requirements(self, width: usize, source: TaskBudget, judge: TaskBudget,
        semantic: SourceSemanticLimits, planning: JudgeLimits, input: SourceSemanticInputFacts)
        -> Result<Int4SourceSemanticHostRequirements, HostedError> {
        self.requirements_with_encoding(width, source, judge, semantic, planning, input,
            &UntrustedEncoding::ByteFallback)
    }
    /// Quote one shared encoder workspace before capturing private input or
    /// starting native work. Original source and every review head use the same
    /// immutable compact policy and reuse this allocation sequentially.
    #[allow(clippy::too_many_arguments)]
    pub fn requirements_with_encoding(self, width: usize, source: TaskBudget, judge: TaskBudget,
        semantic: SourceSemanticLimits, planning: JudgeLimits, input: SourceSemanticInputFacts,
        encoding: &UntrustedEncoding) -> Result<Int4SourceSemanticHostRequirements, HostedError> {
        self.native.run.validate()?;
        let encoding_bytes = match encoding {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map_err(|_| HostedError::Limits("INT4 semantic source encoding workspace"))?.workspace_bytes,
        };
        source.validate().map_err(|_| HostedError::Limits("INT4 semantic source budget"))?;
        judge.validate().map_err(|_| HostedError::Limits("INT4 semantic source judge budget"))?;
        semantic.validate().map_err(task_error)?;
        if self.preparation_reserve_bytes == 0 || self.native.allocator_reserve_bytes == 0
            || self.masks.max_trie_node_visits == 0 || self.masks.checkpoint_interval_nodes == 0
            || self.max_mask_node_visits == 0 || input.source_bytes == 0
            || input.source_bytes > semantic.max_source_bytes || input.question_bytes > semantic.max_claim_bytes
            || input.passage_count > 1024 || (input.passage_count == 0) != (input.question_bytes == 0)
            || source.max_output_bytes > semantic.max_output_bytes || judge.max_output_bytes > semantic.max_output_bytes
            || planning.max_total_prompt_tokens == 0 || planning.max_total_candidate_tokens == 0
            || planning.max_total_projected_logits == 0 || planning.max_output_bytes == 0
            || planning.per_head.max_candidates == 0 || planning.per_head.max_total_tokens == 0
            || planning.per_head.max_nodes == 0 || planning.per_head.max_depth == 0
            || planning.per_head.max_candidate_id_bytes == 0 || planning.per_head.max_projected_logits == 0 {
            return Err(HostedError::Limits("INT4 semantic source complete preparation and mask limits"));
        }
        let native = Int4MemoryRequirement::for_context(self.native.context_tokens, width).map_err(native_error)?;
        if native.kv_bytes > source.max_kv_bytes || native.kv_bytes > judge.max_kv_bytes {
            return Err(HostedError::Limits("INT4 semantic source both stages need full resident KV"));
        }
        let statements = semantic.max_statements as u64;
        let sources = product(statements.checked_add(4).ok_or(HostedError::Limits("INT4 review statement arithmetic"))?,
            input.source_bytes as u64)?;
        // TaskIR copies/serialization, native schedule nodes and all retained
        // scorer state. Faithfulness has <=32 heads and exactly 3 candidates.
        let prompts = product(semantic.max_total_prompt_tokens as u64, 64)?;
        let candidates = product(product(statements, planning.max_total_candidate_tokens as u64)?, 256)?;
        let ids = product(product(product(statements, 32 * 3)?,
            planning.per_head.max_candidate_id_bytes as u64)?, 2)?;
        let plans = product(statements, 65_536)?;
        let judge_results = product(product(statements, judge.max_output_bytes.min(planning.max_output_bytes))?, 4)?;
        let preparation_bytes = sum(&[self.preparation_reserve_bytes, sources, prompts, candidates, ids, plans,
            judge_results, product(source.max_output_bytes, 6)?,
            product(semantic.max_total_claim_bytes as u64, 4)?, product(input.question_bytes as u64, 8)?,
            product(input.passage_count as u64, 2048)?, encoding_bytes])?;
        let native_workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
            self.native.allocator_reserve_bytes])?;
        let output_bytes = sum(&[product(semantic.max_output_bytes, 4)?,
            product(u64::from(source.max_output_tokens), 8)?, 4096])?;
        let requirements = Int4SourceSemanticHostRequirements { native, native_workspace_bytes, preparation_bytes, output_bytes };
        requirements.total_bytes()?;
        Ok(requirements)
    }
}
fn product(a: u64, b: u64) -> Result<u64, HostedError> {
    a.checked_mul(b).ok_or(HostedError::Limits("INT4 semantic source memory arithmetic"))
}
struct Input { prepared: Option<PreparedInt4SourceSemantic>, vocabulary: Arc<ExtractionVocabulary> }

impl NlpEngine {
    /// Generate a cited summary or passage answer, then run every requested
    /// complete-statement faithfulness judgment on the SAME native engine.
    /// All dynamic judges preflight before any review forward. No source work
    /// is refunded after early EOS; a failed stage publishes no combined result.
    /// Preparation/native/output guards remain owned through physical drain.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_semantic_source(&self, model: &ResidentInt4, prepared: PreparedInt4SourceSemantic,
        vocabulary: Arc<ExtractionVocabulary>, limits: Int4SourceSemanticHostLimits, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4SourceSemanticRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        let required = checked_requirements(self, model, &prepared, limits)?;
        let width = prepared.group_width(); let source_work = prepared.source_reserved_work();
        let max_work = prepared.max_model_work(); let policy = prepared.policy_digest();
        let encoding = prepared.input_encoding().clone(); let initial_encoding = prepared.input_encoding_work();
        let task_spec = prepared.execution_identity().task_spec.clone();
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.preparation_bytes)?,
            || Ok(Input { prepared: Some(prepared), vocabulary }))?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.native_workspace_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            checkpoint(control)?;
            let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let run = prepared.execute_with_control(&mut engine.value, &input.value.vocabulary,
                    limits.masks, limits.max_mask_node_visits, control).map_err(task_error)?;
                check_completion(&run, width, &task_spec, source_work, max_work, policy)?;
                check_encoding_completion(run.input_encoding.as_ref(), &encoding, initial_encoding)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(task_error(Int4SourceSemanticError::Accounting));
                }
                checkpoint(control)?;
                Ok(run)
            })?;
            drop(engine); drop(input); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Keep one fixed numerical team through source generation and every
    /// complete-statement judgment. The unchanged sealed composition owns
    /// one source reservation, review allowance and cancellation lifetime.
    /// Worker workspace is admitted once and workers physically join before
    /// the complete guarded source result and diagnostics return.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_semantic_source_with_threads(&self, model: &ResidentInt4,
        prepared: PreparedInt4SourceSemantic, vocabulary: Arc<ExtractionVocabulary>,
        limits: Int4SourceSemanticHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4SourceSemanticRun>, HostedError> {
        dispatch::preflight_int4_team(self, limits.native.run, compute_threads)?;
        let required = checked_requirements(self, model, &prepared, limits)?;
        let width = prepared.group_width(); let source_work = prepared.source_reserved_work();
        let max_work = prepared.max_model_work(); let policy = prepared.policy_digest();
        let encoding = prepared.input_encoding().clone(); let initial_encoding = prepared.input_encoding_work();
        let task_spec = prepared.execution_identity().task_spec.clone();
        sum(&[required.total_bytes()?, int4_task_team_workspace_bytes(compute_threads, width)?])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.preparation_bytes)?,
            || Ok(Input { prepared: Some(prepared), vocabulary }))?;
        let output = Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_bytes)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.native_workspace_bytes)?;
        dispatch::run_int4_team(self, limits.native.run, cancellation, model.clone(), compute_threads,
            kernel, move |weights, control, team| {
                let mut input = input;
                checkpoint(control)?;
                let prepared = input.value.prepared.take().ok_or(HostedError::CompletionMissing)?;
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(weights, limits.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                        max_scratch_payload_bytes: required.native.scratch_payload_bound,
                    }).map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let result = allocate(output, || {
                    let run = prepared.execute_with_control(&mut engine.value, &input.value.vocabulary,
                        limits.masks, limits.max_mask_node_visits, control).map_err(task_error)?;
                    check_completion(&run, width, &task_spec, source_work, max_work, policy)?;
                    check_encoding_completion(run.input_encoding.as_ref(), &encoding, initial_encoding)?;
                    if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                        return Err(task_error(Int4SourceSemanticError::Accounting));
                    }
                    checkpoint(control)?;
                    Ok(run)
                })?;
                drop(engine);
                let compute = team.stats().into();
                drop(input); drop(lease);
                Ok(HostedInt4TeamResult::new(GuardedOutput::new(result.value, result._memory), compute))
            })
    }
}

fn checked_requirements(host: &NlpEngine, model: &ResidentInt4,
    prepared: &PreparedInt4SourceSemantic, limits: Int4SourceSemanticHostLimits)
    -> Result<Int4SourceSemanticHostRequirements, HostedError> {
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    for identity in [prepared.execution_identity(), prepared.judge_identity()] {
        constrained_int4::check_model(identity, model.model_identity()).map_err(|_| HostedError::ModelIdentity)?;
    }
    let width = prepared.group_width(); let work = prepared.source_reserved_work();
    if width != model.model_identity().group_width()
        || !matches!(prepared.execution_identity().task_spec.as_str(), "summarize-v1" | "answer-v1")
        || prepared.judge_identity().task_spec != "judge-v1" || work.forward_positions == 0
        || work.forward_positions > limits.native.context_tokens as u64 || !fits(work, prepared.max_model_work()) {
        return Err(HostedError::Limits("INT4 semantic source model and complete generation reservation"));
    }
    limits.requirements_with_encoding(width, prepared.source_budget(), prepared.judge_budget(),
        prepared.semantic_limits(), prepared.judge_planning_limits(), prepared.input_facts(), prepared.input_encoding())
}
fn check_encoding_completion(report: Option<&Int4SemanticEncoding>, policy: &UntrustedEncoding,
    original: Option<SafeBpeWork>) -> Result<(), HostedError> {
    let failure = || task_error(Int4SourceSemanticError::Accounting);
    let work = report.map_or_else(SafeBpeWork::default, |report| report.work);
    let expected = Int4SemanticEncoding::from_work(policy, work).map_err(|_| failure())?;
    if report != expected.as_ref() || original.is_some() != expected.is_some()
        || work.checked_sub(original.unwrap_or_default()).is_none() {
        return Err(failure());
    }
    Ok(())
}
fn check_completion(run: &Int4SourceSemanticRun, width: usize, task_spec: &str, source: Int4Work,
    ceiling: Int4Work, policy: Sha256Digest) -> Result<(), HostedError> {
    let reserved = source.checked_add(run.semantic_model_work).map_err(|_| task_error(Int4SourceSemanticError::Accounting))?;
    let actual = run.source_task.model_work.checked_add(run.semantic_model_work)
        .map_err(|_| task_error(Int4SourceSemanticError::Accounting))?;
    let s = &run.semantic;
    let task_matches = matches!((&run.source_task.result, task_spec),
        (SourceTaskResult::Summarize(_), "summarize-v1") | (SourceTaskResult::Answer(_), "answer-v1"));
    if run.schema_version != 1 || run.execution != INT4_SOURCE_SEMANTIC_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
        || run.group_width != width || run.source_task.schema_version != 1 || run.source_task.execution != INT4_SOURCE_EXECUTION
        || run.source_task.group_width != width || !task_matches || run.source_reserved_work != source
        || run.reserved_model_work != reserved || run.model_work != actual || !fits(run.source_task.model_work, source)
        || !fits(actual, reserved) || !fits(reserved, ceiling) || !s.same_model_correlated || s.policy_digest != policy
        || s.schema_version != 1 || s.algorithm != SOURCE_SEMANTIC_VERSION || s.claim_rendering != SOURCE_SEMANTIC_CLAIM_VERSION
        || s.evidence_scope != SOURCE_SEMANTIC_SCOPE || s.calibration != SOURCE_SEMANTIC_CALIBRATION
        || s.work.projected_logits != run.semantic_model_work.projected_logits
        || s.attempted_statements != s.decided_statements + s.abstained_statements
        || s.statements.len() != s.attempted_statements + s.not_checked_statements
        || s.statements.iter().filter(|r| r.status == SourceSemanticStatus::NotChecked).count() != s.not_checked_statements {
        return Err(task_error(Int4SourceSemanticError::Accounting));
    }
    Ok(())
}
fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn task_error(error: Int4SourceSemanticError) -> HostedError { HostedError::Int4(Int4HostError::SourceSemantic(error)) }

#[cfg(test)] mod tests;
