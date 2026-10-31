//! Frozen source spans -> actual constrained native NER -> guarded Assay.
//!
//! One physical dispatch owns both comparison arms. Immutable tokenizer and
//! vocabulary assets are shared, native engines reuse one maximum envelope,
//! and model, mask and independent verification allowances never renew.

use super::*;
use crate::{
    assay::{self as metrics, native_ner::{self, Collector, Dataset, Evaluation,
        EvalOptions, Gates, NativeReplay, NerLimits, QualificationReport}, AssayError},
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained::JsonWorkBudget, constrained_int4::{self, Int4JsonBudget},
        strict_int4::Int4Work},
    receipt::{CommitmentDomain, CommitmentKey},
    tasks::{BuiltInTask, ir::TaskBudget, ner::NerResult,
        source_planning::{SourcePlanningLimits, SourceTaskResult,
            grouped_int4::{Int4SourceError, Int4SourcePlanner}}},
    validation::grounded_fields::GroundingBudget,
};

/// Complete native NER operation bounds. Source compiler and per-row source
/// verification limits are fixed policy inputs; global work caps cover every
/// locked-test row and both arms of a paired qualification.
#[derive(Clone, Copy, Debug)]
pub struct Int4NerAssayHostLimits {
    pub native: NativeLimits,
    pub task: TaskBudget,
    pub planning: SourcePlanningLimits,
    /// Explicit fixed allowance for the pinned tokenizer, vocabulary trie and
    /// their construction. Variable compiler/source/result storage is priced
    /// separately by `requirements`.
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int4Work,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
}

/// Modeled transient reservations, excluding both resident weight charges,
/// outer caller input staging and the separately retained cleanup allowance.
/// Native payload is the maximum sequential arm, not their sum. This is not
/// a process-RSS guarantee.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4NerAssayHostRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
}
impl Int4NerAssayHostLimits {
    /// Pure complete pre-IO quote: one width for evaluation, two for an
    /// explicit candidate/baseline comparison. No tokenizer or model loads.
    pub fn requirements(self, group_widths: &[usize], limits: metrics::Limits, ner: NerLimits)
        -> Result<Int4NerAssayHostRequirements, HostedError> {
        limits.validate().map_err(HostedError::Assay)?;
        ner.validate().map_err(HostedError::Assay)?;
        self.native.run.validate()?;
        self.task.validate().map_err(|_| HostedError::Limits("INT4 NER Assay task budget"))?;
        self.planning.validate().map_err(|_| HostedError::Limits("INT4 NER Assay planning limits"))?;
        let compiler = self.planning.compiler;
        let source = self.planning.source;
        if !(1..=2).contains(&group_widths.len()) || self.preparation_reserve_bytes == 0
            || self.native.allocator_reserve_bytes == 0 || !nonzero(self.max_model_work)
            || self.planning.max_context_tokens > self.native.context_tokens
            || self.task.max_output_tokens as usize >= self.planning.max_context_tokens
            || compiler.max_schema_bytes == 0 || compiler.max_string_bytes == 0
            || compiler.max_array_items == 0 || compiler.max_states == 0 || compiler.max_transitions == 0
            || compiler.max_mask_bytes == 0 || compiler.max_output_bytes == 0
            || compiler.max_output_bytes as u64 > self.task.max_output_bytes
            || compiler.max_states as u64 > u64::from(self.task.max_grammar_states)
            || source.index.max_source_bytes == 0 || source.index.max_index_bytes == 0
            || source.index.max_build_steps == 0 || source.index.max_matches == 0
            || source.verification.max_fields == 0 || source.verification.max_matches == 0
            || source.verification.max_scan_steps == 0
            || source.verification.max_matches.min(source.index.max_matches) > ner.max_occurrences_per_row
            || self.mask_limits.max_trie_node_visits == 0 || self.mask_limits.checkpoint_interval_nodes == 0
            || self.max_mask_node_visits == 0 {
            return Err(HostedError::Limits("INT4 NER Assay complete resource envelope"));
        }
        let mut native = Int4MemoryRequirement::for_context(self.native.context_tokens, group_widths[0])
            .map_err(native_error)?;
        for &width in &group_widths[1..] {
            let arm = Int4MemoryRequirement::for_context(self.native.context_tokens, width).map_err(native_error)?;
            native.kv_bytes = native.kv_bytes.max(arm.kv_bytes);
            native.rope_bytes = native.rope_bytes.max(arm.rope_bytes);
            native.scratch_payload_bound = native.scratch_payload_bound.max(arm.scratch_payload_bound);
        }
        if native.kv_bytes > self.task.max_kv_bytes {
            return Err(HostedError::Limits("INT4 NER Assay complete reusable KV allocation"));
        }
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
            self.native.allocator_reserve_bytes])?;
        let replay = native_ner::required_replay_bytes(limits, ner, group_widths.len()).map_err(HostedError::Assay)?;
        let gold = multiply(multiply(limits.max_rows as u64, ner.max_spans_per_row as u64)?, 64)?;
        let input_bytes = sum(&[multiply(limits.max_input_bytes as u64, 8)?,
            multiply(limits.max_manifest_bytes as u64, 8)?, multiply(limits.max_rows as u64, 1024)?,
            gold, replay, self.preparation_reserve_bytes, preparation_bytes(self, ner)?])?;
        let output_reservation_bytes = sum(&[multiply(limits.max_output_bytes as u64, 4)?, 4096])?;
        Ok(Int4NerAssayHostRequirements { native, workspace_bytes, input_bytes, output_reservation_bytes })
    }
}

struct Input { dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: Option<String> }
struct Ledger { model: Int4Work, masks: u64, verification: GroundingBudget }

impl NlpEngine {
    /// Evaluate only frozen locked-test documents. Gold labels never enter the
    /// source task request, and private mention/token observations do not leave
    /// the physical dispatch or appear in the aggregate public scorecard.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_int4_ner(&self, model: &ResidentInt4, dataset: Dataset,
        key: CommitmentKey, namespace: String, options: EvalOptions, limits: metrics::Limits,
        ner: NerLimits, host: Int4NerAssayHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Evaluation>, HostedError> {
        dispatch::preflight(self, host.native.run)?;
        check_domain(self, model)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, ner, false)?;
        let required = host.requirements(&[model.model_identity().group_width()], limits, ner)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(Input { dataset, key, namespace, host_scope: None }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = output_claim(&lease, limits.max_output_bytes as u64, 0)?;
        let model = model.clone();
        dispatch::run(self, host.native.run, cancellation, move |control| {
            let input = input;
            let native_claims = (kv, workspace);
            let result = allocate(output, || {
                let mut ledger = new_ledger(input.value.dataset.replay_len(), 1, ner)?;
                checkpoint(control)?;
                let planner = Int4SourcePlanner::for_model(model.model_identity(), BuiltInTask::Ner,
                    host.task, host.planning).map_err(task_error)?;
                checkpoint(control)?;
                let replay = replay_arm(&model, &planner, &input.value.dataset, host, limits,
                    ner, required.native, kernel, &mut ledger, control)?;
                drop(planner);
                checkpoint(control)?;
                native_ner::evaluate_with_checkpoint::<HostedError>(&replay, &input.value.key,
                    &input.value.namespace, options, limits, &mut || checkpoint(control))
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Compare two explicitly selected loaded candidates over the identical
    /// frozen locked-test population under one deadline and all work ledgers.
    /// This produces bounded audit evidence, never deployment authority.
    #[allow(clippy::too_many_arguments)]
    pub fn qualify_int4_ner(&self, candidate: &ResidentInt4, baseline: &ResidentInt4,
        dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: String,
        gates: Gates, options: EvalOptions, limits: metrics::Limits, ner: NerLimits,
        host: Int4NerAssayHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<QualificationReport>, HostedError> {
        dispatch::preflight(self, host.native.run)?;
        check_domain(self, candidate)?; check_domain(self, baseline)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, ner, true)?;
        gates.validate(limits).map_err(HostedError::Assay)?;
        native_ner::validate_scope(&namespace, Some(&host_scope)).map_err(HostedError::Assay)?;
        let required = host.requirements(&[candidate.model_identity().group_width(),
            baseline.model_identity().group_width()], limits, ner)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(Input { dataset, key, namespace, host_scope: Some(host_scope) }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = output_claim(&lease, limits.max_output_bytes as u64, 0)?;
        let candidate = candidate.clone(); let baseline = baseline.clone();
        dispatch::run(self, host.native.run, cancellation, move |control| {
            let input = input;
            let native_claims = (kv, workspace);
            let result = allocate(output, || {
                let mut ledger = new_ledger(input.value.dataset.replay_len(), 2, ner)?;
                checkpoint(control)?;
                let candidate_planner = Int4SourcePlanner::for_model(candidate.model_identity(), BuiltInTask::Ner,
                    host.task, host.planning).map_err(task_error)?;
                let baseline_planner = candidate_planner.with_model(baseline.model_identity()).map_err(task_error)?;
                checkpoint(control)?;
                let candidate = replay_arm(&candidate, &candidate_planner, &input.value.dataset, host,
                    limits, ner, required.native, kernel, &mut ledger, control)?;
                let baseline = replay_arm(&baseline, &baseline_planner, &input.value.dataset, host,
                    limits, ner, required.native, kernel, &mut ledger, control)?;
                drop(candidate_planner); drop(baseline_planner);
                checkpoint(control)?;
                native_ner::compare_with_checkpoint::<HostedError>(&candidate, &baseline, &input.value.key,
                    &input.value.namespace, input.value.host_scope.as_deref().ok_or(HostedError::CompletionMissing)?,
                    gates, options, limits, &mut || checkpoint(control))
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_arm(model: &ResidentInt4, planner: &Int4SourcePlanner, dataset: &Dataset,
    host: Int4NerAssayHostLimits, limits: metrics::Limits, ner: NerLimits,
    memory: Int4MemoryRequirement, kernel: PackedInt4Kernel, ledger: &mut Ledger,
    control: &mut RunControl) -> Result<NativeReplay, HostedError> {
    checkpoint(control)?;
    let mut collector = Collector::new(dataset, model.model_identity(), limits, ner).map_err(HostedError::Assay)?;
    let mut engine = StrictInt4Engine::new(&model.inner.loaded.value, host.native.context_tokens,
        Int4MemoryBudget { max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
            max_scratch_payload_bytes: memory.scratch_payload_bound }).map_err(native_error)?.with_kernel(kernel);
    for index in 0..dataset.replay_len() {
        checkpoint(control)?;
        let row = &dataset.replay_rows()[index];
        preflight_verification(row.text.len(), dataset.manifest().options.max_entities,
            dataset.manifest().options.max_mention_scalars, host, &ledger.verification)?;
        let request = dataset.source_request(index, host.task).map_err(HostedError::Assay)?;
        let prepared = planner.plan_with_control(&request, control).map_err(task_error)?;
        drop(request);
        let planned = prepared.planned_work();
        if prepared.group_width() != planner.group_width() || prepared.task_budget() != host.task
            || prepared.prompt_tokens() == 0 || planned.forward_positions > host.native.context_tokens as u64
            || prepared.max_kv_bytes() < memory.kv_bytes {
            return Err(task_error(Int4SourceError::InvalidResult));
        }
        // Admission uses the complete worst-case row; successful work is
        // debited at its actual generated length. Neither axis is renewed.
        checked_work(ledger.model, planned, host.max_model_work)?;
        let remaining_masks = host.max_mask_node_visits.checked_sub(ledger.masks)
            .filter(|&remaining| remaining != 0).ok_or_else(|| budget_error("aggregate NER mask work"))?;
        let run = prepared.execute_with_control(prepared.execution_identity(), &mut engine, planner.vocabulary(),
            Int4JsonBudget { native: Int4RunBudget::exact(planned), json: JsonWorkBudget {
                max_forward_positions: planned.forward_positions, max_projected_logits: planned.projected_logits,
                max_kv_bytes: memory.kv_bytes, max_total_mask_node_visits: remaining_masks,
                mask_limits: host.mask_limits,
            } }, control).map_err(task_error)?;
        let SourceTaskResult::Ner(result) = &run.result else { return Err(task_error(Int4SourceError::InvalidResult)); };
        let actual = constrained_int4::planned_work(prepared.group_width(), prepared.prompt_tokens(),
            result.generated_token_ids.len()).map_err(|_| task_error(Int4SourceError::InvalidResult))?;
        if run.group_width != prepared.group_width() || run.model_work != actual
            || !work_fits(actual, planned) || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(task_error(Int4SourceError::InvalidResult));
        }
        ledger.model = checked_work(ledger.model, actual, host.max_model_work)?;
        ledger.masks = ledger.masks.checked_add(result.mask_node_visit_charge)
            .filter(|&total| total <= host.max_mask_node_visits)
            .ok_or_else(|| budget_error("aggregate NER mask work"))?;
        let native_verification_steps = charge_native_verification(row.text.len(), result, &mut ledger.verification)?;
        collector.accept_with_checkpoint::<HostedError>(index, &prepared, run, native_verification_steps,
            &mut ledger.verification, &mut || checkpoint(control))?;
    }
    checkpoint(control)?;
    drop(engine);
    collector.finish().map_err(HostedError::Assay)
}

fn validate_inputs(dataset: &Dataset, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: metrics::Limits, ner: NerLimits, paired: bool) -> Result<(), HostedError> {
    native_ner::validate_scope(namespace, None).map_err(HostedError::Assay)?;
    dataset.validate_limits(limits, ner).map_err(HostedError::Assay)?;
    if paired { options.validate_paired_rows(dataset.replay_len(), limits) }
    else { options.validate_rows(dataset.replay_len(), limits) }.map_err(HostedError::Assay)?;
    key.commit(CommitmentDomain::Configuration, namespace, "native_ner_assay_preflight_v1", &[])
        .map_err(|_| HostedError::Assay(AssayError::Commitment))?;
    Ok(())
}

fn preparation_bytes(host: Int4NerAssayHostLimits, ner: NerLimits) -> Result<u64, HostedError> {
    let compiler = host.planning.compiler;
    // One row is prepared at a time. Explicit caps cover source-index build
    // scratch, prompt/TaskIR copies, compiler graph/literals, legal masks,
    // current result/JSON trees, KMP scratch and all candidate intervals.
    // Immutable tokenizer/vocabulary construction remains the separate fixed
    // caller allowance; model allocations remain the native envelope.
    sum(&[multiply(host.planning.source.index.max_index_bytes as u64, 2)?,
        multiply(host.planning.max_input_bytes as u64, 16)?,
        multiply(host.native.context_tokens as u64, 256)?,
        multiply(compiler.max_schema_bytes as u64, 64)?,
        multiply(compiler.max_states as u64, 512)?, multiply(compiler.max_transitions as u64, 128)?,
        compiler.max_mask_bytes as u64, multiply(compiler.max_string_bytes as u64, 16)?,
        multiply(host.task.max_output_bytes, 16)?, multiply(host.task.max_output_tokens as u64, 16)?,
        multiply(ner.max_spans_per_row as u64, 1024)?, multiply(ner.max_occurrences_per_row as u64, 256)?])
}

fn new_ledger(rows: usize, arms: usize, ner: NerLimits) -> Result<Ledger, HostedError> {
    if rows == 0 || !(1..=2).contains(&arms) { return Err(HostedError::Assay(AssayError::Limits)); }
    let passes = rows.checked_mul(arms).and_then(|count| count.checked_mul(3))
        .ok_or(HostedError::Limits("INT4 NER Assay verification arithmetic"))?;
    let max_fields = passes.checked_mul(ner.max_spans_per_row)
        .ok_or(HostedError::Limits("INT4 NER Assay verification arithmetic"))?;
    let max_matches = passes.checked_mul(ner.max_occurrences_per_row)
        .ok_or(HostedError::Limits("INT4 NER Assay verification arithmetic"))?;
    Ok(Ledger { model: Int4Work::default(), masks: 0,
        verification: GroundingBudget { max_fields, max_matches, max_scan_steps: ner.max_verification_steps } })
}

fn preflight_verification(source_bytes: usize, entities: usize, mention_scalars: usize,
    host: Int4NerAssayHostLimits, remaining: &GroundingBudget) -> Result<(), HostedError> {
    let cap = host.planning.source.verification;
    let fields = entities.min(cap.max_fields);
    let mention_bytes = mention_scalars.checked_mul(4)
        .ok_or(HostedError::Limits("INT4 NER Assay verification arithmetic"))?
        .min(source_bytes).min(host.planning.compiler.max_string_bytes);
    let scan = sum(&[source_bytes as u64, mention_bytes as u64, 1])?;
    let steps = multiply(multiply(scan, 8)?, fields as u64)?.min(cap.max_scan_steps);
    // JsonProgram::compile_with_source stores this minimum as the verifier's
    // aggregate row allowance, before either independent native source pass.
    // It is not a per-field renewal of the source-index occurrence bound.
    let matches = cap.max_matches.min(host.planning.source.index.max_matches);
    if multiply(steps, 3)? > remaining.max_scan_steps
        || fields.checked_mul(3).is_none_or(|count| count > remaining.max_fields)
        || matches.checked_mul(3).is_none_or(|count| count > remaining.max_matches) {
        return Err(budget_error("aggregate NER source verification"));
    }
    Ok(())
}

fn charge_native_verification(source_bytes: usize, result: &NerResult, remaining: &mut GroundingBudget)
    -> Result<u64, HostedError> {
    // Existing native source validation performs two KMP passes: decoder
    // whole-value validation, then extraction evidence finalization. Collector
    // performs the third independently and checks this exact two-pass charge.
    let mut fields = 0_usize; let mut matches = 0_usize; let mut steps = 0_u64;
    for entity in &result.entities {
        fields = fields.checked_add(2).ok_or(HostedError::Limits("INT4 NER Assay verification arithmetic"))?;
        matches = entity.spans.len().checked_mul(2).and_then(|count| matches.checked_add(count))
            .ok_or(HostedError::Limits("INT4 NER Assay verification arithmetic"))?;
        steps = sum(&[steps, multiply(sum(&[source_bytes as u64, entity.text.len() as u64, 1])?, 16)?])?;
    }
    let next_fields = remaining.max_fields.checked_sub(fields)
        .ok_or_else(|| budget_error("aggregate NER source verification"))?;
    let next_matches = remaining.max_matches.checked_sub(matches)
        .ok_or_else(|| budget_error("aggregate NER source verification"))?;
    let next_steps = remaining.max_scan_steps.checked_sub(steps)
        .ok_or_else(|| budget_error("aggregate NER source verification"))?;
    *remaining = GroundingBudget {
        max_fields: next_fields, max_matches: next_matches, max_scan_steps: next_steps,
    };
    Ok(steps)
}

fn check_domain(host: &NlpEngine, model: &ResidentInt4) -> Result<(), HostedError> {
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    Ok(())
}
fn nonzero(work: Int4Work) -> bool {
    work.forward_positions != 0 && work.projected_logits != 0 && work.attention_pairs != 0
        && work.projections.dot_products != 0 && work.projections.multiply_accumulates != 0
}
fn work_fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn checked_work(total: Int4Work, row: Int4Work, cap: Int4Work) -> Result<Int4Work, HostedError> {
    if !nonzero(row) { return Err(task_error(Int4SourceError::InvalidResult)); }
    let next = total.checked_add(row).map_err(|_| budget_error("aggregate NER model work"))?;
    if !work_fits(next, cap) { return Err(budget_error("aggregate NER model work")); }
    Ok(next)
}
fn multiply(value: u64, factor: u64) -> Result<u64, HostedError> {
    value.checked_mul(factor).ok_or(HostedError::Limits("INT4 NER Assay memory arithmetic"))
}
fn task_error(error: Int4SourceError) -> HostedError { HostedError::Int4(Int4HostError::SourceTask(error)) }
fn budget_error(reason: &'static str) -> HostedError {
    task_error(constrained_int4::Int4JsonError::Decode(
        crate::native_engine::constrained::JsonDecodeError::BudgetExceeded(reason)).into())
}

#[cfg(test)] mod tests;
