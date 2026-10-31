//! Exact-schema native extraction over frozen user-owned Assay documents.
//!
//! One physical invocation owns both comparison arms, shared immutable
//! preparation assets, private gold/results and statistical finalization.
//! Every native, mask and source-verification allowance is nonrenewable.

use super::*;
use crate::{
    assay::{self as metrics, native_extract::{self, Collector, Dataset, Evaluation,
        EvalOptions, ExtractLimits, Gates, NativeReplay, QualificationReport}, AssayError},
    batch::extract::grouped_int4::Int4ExtractionBatchPlanner,
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
    native_engine::{constrained::{JsonDecodeError, JsonWorkBudget},
        constrained_int4::{self, Int4JsonBudget, Int4JsonError}, strict_int4::Int4Work},
    receipt::{CommitmentDomain, CommitmentKey},
    tasks::{extract::{ExtractError, ExtractResult, grouped_int4::Int4ExtractError}, ir::TaskBudget},
    tokenizer::pinned_controls,
    validation::grounded_fields::GroundingBudget,
};

/// Complete operation limits. Compiler and source policies must exactly
/// match those used to validate the Dataset before model IO. Global source
/// work includes gold preparation once and all three native/collector passes.
#[derive(Clone, Copy, Debug)]
pub struct Int4ExtractAssayHostLimits {
    pub native: NativeLimits,
    pub task: TaskBudget,
    pub compiler: CompileLimits,
    pub source: SourceRuntimeLimits,
    /// Fixed caller allowance for pinned tokenizer/vocabulary construction.
    /// Source, parsed gold, grammar, current results and replay are additional.
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int4Work,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
}

/// Modeled transient reservations, excluding resident weights, outer caller
/// parsing staging and RunLimits cleanup. Sequential arm engines use the
/// maximum native envelope. This is not a process-RSS measurement or cap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ExtractAssayHostRequirements {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
}
impl Int4ExtractAssayHostLimits {
    /// Pure complete pre-IO quotation: one group width for evaluation, two
    /// for an explicitly selected candidate and baseline. No model loads.
    pub fn requirements(self, group_widths: &[usize], limits: metrics::Limits, extract: ExtractLimits)
        -> Result<Int4ExtractAssayHostRequirements, HostedError> {
        limits.validate().map_err(HostedError::Assay)?;
        extract.validate().map_err(HostedError::Assay)?;
        self.native.run.validate()?;
        self.task.validate().map_err(|_| HostedError::Limits("INT4 extraction Assay task budget"))?;
        if !(1..=2).contains(&group_widths.len()) || self.preparation_reserve_bytes == 0
            || self.native.allocator_reserve_bytes == 0 || !nonzero(self.max_model_work)
            || self.task.max_output_tokens as usize >= self.native.context_tokens
            || self.compiler.max_schema_bytes == 0 || self.compiler.max_string_bytes == 0
            || self.compiler.max_array_items == 0 || self.compiler.max_states == 0
            || self.compiler.max_transitions == 0 || self.compiler.max_mask_bytes == 0
            || self.compiler.max_output_bytes == 0 || self.compiler.max_output_bytes as u64 > self.task.max_output_bytes
            || self.compiler.max_states as u64 > u64::from(self.task.max_grammar_states)
            || self.source.index.max_source_bytes == 0 || self.source.index.max_index_bytes == 0
            || self.source.index.max_build_steps == 0 || self.source.index.max_matches == 0
            || self.source.verification.max_fields == 0 || self.source.verification.max_fields > extract.max_fields_per_row
            || self.source.verification.max_matches == 0 || self.source.verification.max_scan_steps == 0
            || self.source.verification.max_scan_steps > extract.max_verification_steps
            || self.source.verification.max_matches.min(self.source.index.max_matches) > extract.max_occurrences_per_row
            || self.mask_limits.max_trie_node_visits == 0 || self.mask_limits.checkpoint_interval_nodes == 0
            || self.max_mask_node_visits == 0 {
            return Err(HostedError::Limits("INT4 extraction Assay complete resource envelope"));
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
            return Err(HostedError::Limits("INT4 extraction Assay complete reusable KV allocation"));
        }
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
            self.native.allocator_reserve_bytes])?;
        let replay = native_extract::required_replay_bytes(limits, extract, group_widths.len()).map_err(HostedError::Assay)?;
        // Each parsed node consumes at least one input byte. The wire cap and
        // per-row node caps jointly bound retained gold without multiplying
        // independent maxima that cannot occur in the admitted input.
        let gold_nodes = multiply(limits.max_rows as u64, extract.max_json_nodes_per_row as u64)?
            .min(limits.max_input_bytes as u64);
        let input_bytes = sum(&[multiply(limits.max_input_bytes as u64, 16)?,
            multiply(limits.max_manifest_bytes as u64, 8)?, multiply(limits.max_rows as u64, 1024)?,
            multiply(gold_nodes, 128)?, replay, self.preparation_reserve_bytes,
            preparation_bytes(self, limits, extract)?])?;
        let output_reservation_bytes = sum(&[multiply(limits.max_output_bytes as u64, 4)?, 4096])?;
        Ok(Int4ExtractAssayHostRequirements { native, workspace_bytes, input_bytes, output_reservation_bytes })
    }
}

struct Input { dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: Option<String> }
struct Ledger { model: Int4Work, masks: u64, verification: GroundingBudget }

impl NlpEngine {
    /// Execute only locked-test documents through actual native extraction.
    /// Exact gold remains private and never enters model prompts. All schemas,
    /// splits and gold were already validated and bound by the sealed Dataset.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_int4_extraction(&self, model: &ResidentInt4, dataset: Dataset,
        key: CommitmentKey, namespace: String, options: EvalOptions, limits: metrics::Limits,
        extract: ExtractLimits, host: Int4ExtractAssayHostLimits, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Evaluation>, HostedError> {
        dispatch::preflight(self, host.native.run)?;
        check_domain(self, model)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, extract, host, false)?;
        let required = host.requirements(&[model.model_identity().group_width()], limits, extract)?;
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
                let mut ledger = new_ledger(&input.value.dataset, 1, extract)?;
                checkpoint(control)?;
                let planner = Int4ExtractionBatchPlanner::for_model(model.model_identity(), host.task,
                    host.compiler, host.source, None).map_err(HostedError::BatchSetup)?;
                let vocabulary = vocabulary()?;
                checkpoint(control)?;
                let replay = replay_arm(&model, &planner, &vocabulary, &input.value.dataset, host,
                    limits, extract, required.native, kernel, &mut ledger, control)?;
                drop(planner); drop(vocabulary);
                checkpoint(control)?;
                native_extract::evaluate_with_checkpoint::<HostedError>(&replay, &input.value.key,
                    &input.value.namespace, options, limits, &mut || checkpoint(control))
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Paired actual native extraction under one deadline and all work caps.
    /// Both explicit residents remain charged; immutable assets are shared
    /// and sequential arm engines reuse one native allocation envelope.
    #[allow(clippy::too_many_arguments)]
    pub fn qualify_int4_extraction(&self, candidate: &ResidentInt4, baseline: &ResidentInt4,
        dataset: Dataset, key: CommitmentKey, namespace: String, host_scope: String,
        gates: Gates, options: EvalOptions, limits: metrics::Limits, extract: ExtractLimits,
        host: Int4ExtractAssayHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<QualificationReport>, HostedError> {
        dispatch::preflight(self, host.native.run)?;
        check_domain(self, candidate)?; check_domain(self, baseline)?;
        validate_inputs(&dataset, &key, &namespace, options, limits, extract, host, true)?;
        gates.validate(limits).map_err(HostedError::Assay)?;
        native_extract::validate_scope(&namespace, Some(&host_scope)).map_err(HostedError::Assay)?;
        let required = host.requirements(&[candidate.model_identity().group_width(),
            baseline.model_identity().group_width()], limits, extract)?;
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
                let mut ledger = new_ledger(&input.value.dataset, 2, extract)?;
                checkpoint(control)?;
                let candidate_planner = Int4ExtractionBatchPlanner::for_model(candidate.model_identity(), host.task,
                    host.compiler, host.source, None).map_err(HostedError::BatchSetup)?;
                let baseline_planner = candidate_planner.with_model(baseline.model_identity()).map_err(HostedError::BatchSetup)?;
                let vocabulary = vocabulary()?;
                checkpoint(control)?;
                let candidate = replay_arm(&candidate, &candidate_planner, &vocabulary, &input.value.dataset,
                    host, limits, extract, required.native, kernel, &mut ledger, control)?;
                let baseline = replay_arm(&baseline, &baseline_planner, &vocabulary, &input.value.dataset,
                    host, limits, extract, required.native, kernel, &mut ledger, control)?;
                drop(candidate_planner); drop(baseline_planner); drop(vocabulary);
                checkpoint(control)?;
                native_extract::compare_with_checkpoint::<HostedError>(&candidate, &baseline, &input.value.key,
                    &input.value.namespace, input.value.host_scope.as_deref().ok_or(HostedError::CompletionMissing)?,
                    gates, options, limits, &mut || checkpoint(control))
            })?;
            drop(input); drop(native_claims); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_arm(model: &ResidentInt4, planner: &Int4ExtractionBatchPlanner, vocabulary: &ExtractionVocabulary,
    dataset: &Dataset, host: Int4ExtractAssayHostLimits, limits: metrics::Limits, extract: ExtractLimits,
    memory: Int4MemoryRequirement, kernel: PackedInt4Kernel, ledger: &mut Ledger,
    control: &mut RunControl) -> Result<NativeReplay, HostedError> {
    checkpoint(control)?;
    let mut collector = Collector::new(dataset, model.model_identity(), limits, extract).map_err(HostedError::Assay)?;
    let mut engine = StrictInt4Engine::new(&model.inner.loaded.value, host.native.context_tokens,
        Int4MemoryBudget { max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
            max_scratch_payload_bytes: memory.scratch_payload_bound }).map_err(native_error)?.with_kernel(kernel);
    for index in 0..dataset.replay_len() {
        checkpoint(control)?;
        let row = &dataset.replay_rows()[index];
        let source_fields = dataset.max_source_fields(index).map_err(HostedError::Assay)?;
        preflight_verification(row.text.len(), source_fields, host, &ledger.verification)?;
        let request = dataset.extraction_request(index, host.task).map_err(HostedError::Assay)?;
        let prepared = planner.prepare_with_control(request, control)
            .map_err(|error| HostedError::BatchSetup(error.fault))?;
        let plan = prepared.extraction_plan();
        let planned = plan.planned_work();
        if plan.group_width() != planner.group_width() || *prepared.task_plan().ir().budget() != host.task
            || plan.prompt_tokens() == 0 || planned.forward_positions > host.native.context_tokens as u64
            || plan.max_kv_bytes() < memory.kv_bytes {
            return Err(extraction_error(Int4ExtractError::Identity));
        }
        // Full worst-case admission precedes every row; successful native
        // work is charged at its actual generated length, without renewal.
        checked_work(ledger.model, planned, host.max_model_work)?;
        let remaining_masks = host.max_mask_node_visits.checked_sub(ledger.masks)
            .filter(|&remaining| remaining != 0).ok_or_else(|| budget_error("aggregate extraction mask work"))?;
        let run = plan.execute(&mut engine, plan.execution_identity(), vocabulary,
            Int4JsonBudget { native: Int4RunBudget::exact(planned), json: JsonWorkBudget {
                max_forward_positions: planned.forward_positions, max_projected_logits: planned.projected_logits,
                max_kv_bytes: memory.kv_bytes, max_total_mask_node_visits: remaining_masks,
                mask_limits: host.mask_limits,
            } }, control).map_err(extraction_error)?;
        let actual = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(),
            run.result.output.token_ids.len()).map_err(|error| extraction_error(error.into()))?;
        if run.group_width != plan.group_width() || run.model_work != actual || !work_fits(actual, planned)
            || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(extraction_error(ExtractError::InvalidResult.into()));
        }
        ledger.model = checked_work(ledger.model, actual, host.max_model_work)?;
        ledger.masks = ledger.masks.checked_add(run.result.output.mask_node_visit_charge)
            .filter(|&total| total <= host.max_mask_node_visits)
            .ok_or_else(|| budget_error("aggregate extraction mask work"))?;
        let native_verification_steps = charge_native_verification(row.text.len(), &run.result, &mut ledger.verification)?;
        collector.accept_with_checkpoint::<HostedError>(index, &prepared, run, native_verification_steps,
            &mut ledger.verification, &mut || checkpoint(control))?;
    }
    checkpoint(control)?;
    drop(engine);
    collector.finish().map_err(HostedError::Assay)
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(dataset: &Dataset, key: &CommitmentKey, namespace: &str, options: EvalOptions,
    limits: metrics::Limits, extract: ExtractLimits, host: Int4ExtractAssayHostLimits, paired: bool)
    -> Result<(), HostedError> {
    native_extract::validate_scope(namespace, None).map_err(HostedError::Assay)?;
    dataset.validate_limits(limits, extract, host.compiler, host.source).map_err(HostedError::Assay)?;
    if paired { options.validate_paired_rows(dataset.replay_len(), limits) }
    else { options.validate_rows(dataset.replay_len(), limits) }.map_err(HostedError::Assay)?;
    key.commit(CommitmentDomain::Configuration, namespace, "native_extraction_assay_preflight_v1", &[])
        .map_err(|_| HostedError::Assay(AssayError::Commitment))?;
    Ok(())
}

fn vocabulary() -> Result<ExtractionVocabulary, HostedError> {
    let registry = pinned_controls::pinned().map_err(|_| HostedError::BatchSetup(crate::batch::BatchCode::Planning.into()))?;
    ExtractionVocabulary::pinned(registry.template_controls()).map_err(|error| extraction_error(error.into()))
}

fn preparation_bytes(host: Int4ExtractAssayHostLimits, limits: metrics::Limits, extract: ExtractLimits)
    -> Result<u64, HostedError> {
    // One source/grammar/result is live at a time. Native validation may parse
    // any JSON within its original output cap before the Assay-specific node
    // bound is checked, so its full byte-derived tree bound is charged too.
    let observed_nodes = host.task.max_output_bytes.min(extract.max_json_nodes_per_row as u64);
    // An object key occurs once in the schema/wire but may prefix every
    // repeated array field. Native evidence is constructed before its outer
    // result-size check. Price both evidence sets and one depth-128 traversal
    // stack explicitly; never multiply these current-row paths by row count.
    let pointer_bytes = sum(&[multiply((host.compiler.max_schema_bytes as u64)
        .min(host.task.max_output_bytes), 2)?, 128 * 32, 1])?;
    let pointer_count = sum(&[multiply(host.source.verification.max_fields as u64, 2)?, 128])?;
    let source_paths = multiply(multiply(pointer_bytes, pointer_count)?, 2)?;
    sum(&[multiply(host.source.index.max_index_bytes as u64, 2)?,
        multiply(limits.max_line_bytes as u64, 16)?, multiply(host.native.context_tokens as u64, 256)?,
        multiply(host.compiler.max_schema_bytes as u64, 64)?, multiply(host.compiler.max_states as u64, 512)?,
        multiply(host.compiler.max_transitions as u64, 128)?, host.compiler.max_mask_bytes as u64,
        multiply(host.compiler.max_string_bytes as u64, 16)?, multiply(host.task.max_output_bytes, 144)?,
        multiply(host.task.max_output_tokens as u64, 16)?, multiply(observed_nodes, 128)?,
        multiply(extract.max_fields_per_row as u64, 512)?, multiply(extract.max_occurrences_per_row as u64, 256)?,
        source_paths])
}

fn new_ledger(dataset: &Dataset, arms: usize, extract: ExtractLimits) -> Result<Ledger, HostedError> {
    let gold = GroundingBudget { max_fields: dataset.gold_verification_fields(),
        max_matches: dataset.gold_verification_matches(), max_scan_steps: dataset.gold_verification_steps() };
    let verification = verification_budget(dataset.len(), dataset.replay_len(), arms, extract, gold)?;
    Ok(Ledger { model: Int4Work::default(), masks: 0, verification })
}
fn verification_budget(validated_rows: usize, replay_rows: usize, arms: usize,
    extract: ExtractLimits, preparation: GroundingBudget) -> Result<GroundingBudget, HostedError> {
    if replay_rows == 0 || replay_rows > validated_rows || !(1..=2).contains(&arms) {
        return Err(HostedError::Assay(AssayError::Limits));
    }
    let passes = replay_rows.checked_mul(arms).and_then(|count| count.checked_mul(3))
        .and_then(|count| count.checked_add(validated_rows))
        .ok_or(HostedError::Limits("INT4 extraction Assay verification arithmetic"))?;
    let max_fields = passes.checked_mul(extract.max_fields_per_row)
        .and_then(|count| count.checked_sub(preparation.max_fields))
        .ok_or_else(|| budget_error("aggregate extraction source fields"))?;
    let max_matches = passes.checked_mul(extract.max_occurrences_per_row)
        .and_then(|count| count.checked_sub(preparation.max_matches))
        .ok_or_else(|| budget_error("aggregate extraction source occurrences"))?;
    let max_scan_steps = extract.max_verification_steps.checked_sub(preparation.max_scan_steps)
        .ok_or_else(|| budget_error("aggregate extraction source verification"))?;
    Ok(GroundingBudget { max_fields, max_matches, max_scan_steps })
}

fn preflight_verification(source_bytes: usize, schema_fields: usize, host: Int4ExtractAssayHostLimits, remaining: &GroundingBudget)
    -> Result<(), HostedError> {
    let cap = host.source.verification;
    let fields = schema_fields.min(cap.max_fields);
    if fields == 0 { return Ok(()); }
    let mention_bytes = source_bytes.min(host.compiler.max_string_bytes);
    let steps = multiply(multiply(sum(&[source_bytes as u64, mention_bytes as u64, 1])?, 8)?,
        fields as u64)?.min(cap.max_scan_steps);
    // Source compilation stores this minimum as the aggregate row cap; the
    // occurrence index bound is not renewed independently for every field.
    let matches = cap.max_matches.min(host.source.index.max_matches);
    if multiply(steps, 3)? > remaining.max_scan_steps
        || fields.checked_mul(3).is_none_or(|count| count > remaining.max_fields)
        || matches.checked_mul(3).is_none_or(|count| count > remaining.max_matches) {
        return Err(budget_error("aggregate extraction source verification"));
    }
    Ok(())
}

fn charge_native_verification(source_bytes: usize, result: &ExtractResult, remaining: &mut GroundingBudget)
    -> Result<u64, HostedError> {
    let mut fields = 0_usize; let mut matches = 0_usize; let mut steps = 0_u64;
    for field in &result.source_fields {
        // Native finalization has already proved these complete occurrences.
        // The collector rechecks the sealed whole result before accepting it.
        let span = field.spans.first().ok_or_else(|| extraction_error(ExtractError::InvalidResult.into()))?;
        let bytes = span.byte_end.checked_sub(span.byte_start)
            .filter(|&bytes| bytes <= source_bytes).ok_or_else(|| extraction_error(ExtractError::InvalidResult.into()))?;
        fields = fields.checked_add(2).ok_or_else(|| budget_error("aggregate extraction source fields"))?;
        matches = field.spans.len().checked_mul(2).and_then(|count| matches.checked_add(count))
            .ok_or_else(|| budget_error("aggregate extraction source occurrences"))?;
        steps = sum(&[steps, multiply(sum(&[source_bytes as u64, bytes as u64, 1])?, 16)?])?;
    }
    let next_fields = remaining.max_fields.checked_sub(fields)
        .ok_or_else(|| budget_error("aggregate extraction source fields"))?;
    let next_matches = remaining.max_matches.checked_sub(matches)
        .ok_or_else(|| budget_error("aggregate extraction source occurrences"))?;
    let next_steps = remaining.max_scan_steps.checked_sub(steps)
        .ok_or_else(|| budget_error("aggregate extraction source verification"))?;
    *remaining = GroundingBudget { max_fields: next_fields, max_matches: next_matches, max_scan_steps: next_steps };
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
    if !nonzero(row) { return Err(extraction_error(ExtractError::InvalidResult.into())); }
    let next = total.checked_add(row).map_err(|_| budget_error("aggregate extraction model work"))?;
    if !work_fits(next, cap) { return Err(budget_error("aggregate extraction model work")); }
    Ok(next)
}
fn multiply(value: u64, factor: u64) -> Result<u64, HostedError> {
    value.checked_mul(factor).ok_or(HostedError::Limits("INT4 extraction Assay memory arithmetic"))
}
fn extraction_error(error: Int4ExtractError) -> HostedError { HostedError::Int4(Int4HostError::Extraction(error)) }
fn budget_error(reason: &'static str) -> HostedError {
    extraction_error(Int4JsonError::Decode(JsonDecodeError::BudgetExceeded(reason)).into())
}

#[cfg(test)] mod tests;
