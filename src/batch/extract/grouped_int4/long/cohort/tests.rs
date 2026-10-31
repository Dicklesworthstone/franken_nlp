//! Pinned plans and private grouped fault injection, never native parity evidence.
use super::*;
use super::super::tests::{args, source_args, planner, prepare, Script};
use std::rc::Rc;
use crate::tokenizer::pinned_controls;

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn limits(width: usize) -> Int4ExtractionMapLimits {
    let mut limits = super::super::tests::limits();
    limits.mapping.reduction.map_batch_chunks = width;
    // Real vocabulary preflight is intentionally exercised by this fixture.
    limits.mapping.mask_visits_per_chunk = 1 << 32;
    limits.mapping.max_mask_visits = 1 << 40;
    limits
}
fn vocabulary() -> ExtractionVocabulary {
    let controls = pinned_controls::pinned().unwrap();
    ExtractionVocabulary::pinned(controls.template_controls()).unwrap()
}
fn budget(need: Int4ExtractionMapCohortRequirement) -> Int4ExtractionMapCohortBudget {
    Int4ExtractionMapCohortBudget { context_limit: 8192,
        native_memory: Int4MemoryBudget { max_kv_bytes: need.native.kv_bytes, max_rope_bytes: need.native.rope_bytes,
            max_scratch_payload_bytes: need.native.scratch_payload_bound },
        max_workspace_bytes: need.workspace_bytes, max_result_bytes: need.max_result_bytes }
}
struct Scripted {
    scalar: Script,
    windows: Rc<RefCell<Vec<usize>>>,
    corrupt_group: Option<usize>,
}
impl Scripted {
    fn new(args: &ExtractionBatchArgs) -> Self {
        Self { scalar: Script::new(args), windows: Rc::new(RefCell::new(Vec::new())), corrupt_group: None }
    }
}
impl CohortDriver for Scripted {
    fn execute<C: DecodeStepControl>(&mut self, plans: &[PreparedInt4BatchExtraction], admitted: &[ExecutionIdentity],
        masks: ExtractionMaskBudget, control: &mut C) -> Result<Int4ExtractCohortRun, Int4ExtractError> {
        self.windows.borrow_mut().push(plans.len());
        let mut sequences = Vec::new(); let mut planned_work = Int4Work::default();
        let mut model_work = Int4Work::default(); let mut group_steps = 0; let mut projection_groups = 0;
        for (plan, identity) in plans.iter().zip(admitted) {
            let mut row = self.scalar.execute(plan, identity, native_budget(plan, masks), control)?;
            // This fixture funds real all-row vocabulary preflight, unlike the
            // smaller scalar fixture; preserve its over-budget fault meaning.
            if self.scalar.corrupt == Some(7) && self.scalar.calls.get() == 2 {
                row.result.output.mask_node_visit_charge = masks.max_visits_per_item + 1;
            }
            planned_work = add_work(planned_work, plan.planned_work()).unwrap();
            model_work = add_work(model_work, row.model_work).unwrap();
            group_steps = group_steps.max(row.model_work.forward_positions);
            projection_groups = projection_groups.max(row.result.output.token_ids.len() as u64);
            sequences.push(row);
        }
        let mut run = Int4ExtractCohortRun { schema_version: 1, execution: INT4_EXTRACT_COHORT_EXECUTION.to_owned(),
            sequences, group_steps, projection_groups, planned_work, model_work };
        if let Some(axis) = self.corrupt_group { match axis {
            0 => run.schema_version += 1,
            1 => run.execution.push('x'),
            2 => { run.sequences.pop(); },
            3 => run.planned_work.attention_pairs += 1,
            4 => run.model_work.projections.dot_products += 1,
            5 => run.group_steps += 1,
            6 => run.projection_groups = 0,
            _ => run.projection_groups = u64::MAX,
        } }
        Ok(run)
    }
}
fn run(prepared: PreparedInt4ExtractionMap<'_>, script: Scripted, vocabulary: &ExtractionVocabulary)
    -> Result<Int4ExtractionMapCohortRun, Int4ExtractionMapError> {
    let ids: Vec<_> = prepared.execution_identities().cloned().collect();
    let required = prepared.cohort_requirements(8192)?;
    prepared.execute_cohorts_with_driver(&ids, vocabulary, budget(required), script, &mut Continue)
}
fn without_schedule(run: &Int4ExtractionMapRun) -> serde_json::Value {
    let mut value = serde_json::to_value(run).unwrap();
    value["mapped"].as_object_mut().unwrap().remove("map_batches");
    value
}

#[test]
fn actual_windows_include_a_ragged_tail_and_preserve_every_serial_semantic_field() {
    let planner = planner(); let args = source_args(); let vocabulary = vocabulary();
    let source = "éAéA      éAéA Alice Bob Carol";
    let serial_plan = prepare(&planner, source, &args, limits(1));
    let ids: Vec<_> = serial_plan.execution_identities().cloned().collect();
    let serial = serial_plan.execute_with_driver(&ids, Script::new(&args), &mut Continue).unwrap();
    let count = serial.mapped.root().chunk_range().len();
    assert!(count > 2);
    for width in [1, 2, 4, 64] {
        let prepared = prepare(&planner, source, &args, limits(width)); let expected = prepared.preflight_metadata();
        let script = Scripted::new(&args); let windows = Rc::clone(&script.windows);
        let completed = run(prepared, script, &vocabulary).unwrap();
        expected.verify_completed(&completed.document).unwrap();
        assert_eq!(without_schedule(&completed.document), without_schedule(&serial));
        assert_eq!(completed.document.mapped.map_batches(), count.div_ceil(width));
        assert_eq!(completed.cohorts.cohorts, count.div_ceil(width) as u64);
        assert_eq!(completed.cohorts.chunks, count as u64);
        assert_eq!(completed.cohorts.planned_work, serial.planned_model_work);
        assert_eq!(completed.cohorts.model_work, serial.model_work);
        let expected_windows: Vec<_> = (0..count).step_by(width).map(|start| width.min(count - start)).collect();
        assert_eq!(*windows.borrow(), expected_windows);
    }
}

#[test]
fn actual_component_peaks_fit_the_shared_declared_window_quote_without_padded_tail_work() {
    let planner = planner(); let args = source_args(); let source = "Alice Bob Carol";
    let prepared = prepare(&planner, source, &args, limits(2));
    let required = prepared.cohort_requirements(8192).unwrap();
    let declared = Int4ExtractionMapCohortMemoryLimits { max_chunks: 2, context_tokens: 8192,
        max_output_tokens_per_chunk: args.budget.max_output_tokens as u64,
        max_result_bytes_per_chunk: args.budget.max_output_bytes }.requirements(32).unwrap();
    assert!(required.native.kv_bytes < declared.native.kv_bytes);
    assert!(required.native.rope_bytes <= declared.native.rope_bytes);
    assert!(required.native.scratch_payload_bound <= declared.native.scratch_payload_bound);
    assert_eq!(required.workspace_bytes, declared.workspace_bytes);
    assert_eq!(required.max_result_bytes, declared.max_result_bytes);
    let expected_kv = prepared.plans.chunks(2).map(|rows| rows.iter()
        .map(|plan| plan.planned_work().forward_positions * KV_BYTES_PER_TOKEN as u64).sum::<u64>()).max().unwrap();
    assert_eq!(required.native.kv_bytes, expected_kv);
    assert_eq!(prepared.preflight_metadata().planned_work(), prepared.plans.iter()
        .fold(Int4Work::default(), |work, plan| work.checked_add(plan.planned_work()).unwrap()));
}

#[test]
fn declared_quotations_refuse_zero_overflow_and_unfunded_complete_result_envelopes() {
    let valid = Int4ExtractionMapCohortMemoryLimits { max_chunks: 4, context_tokens: 4096,
        max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: 65536 };
    let need = valid.requirements(32).unwrap();
    assert_eq!(need.native.kv_bytes, 4 * 4096 * KV_BYTES_PER_TOKEN as u64);
    assert_eq!(need.max_result_bytes, 4 * 65536 + COHORT_ENVELOPE_BYTES + 4);
    for axis in 0..8 {
        let mut limits = valid;
        match axis { 0 => limits.max_chunks = 0, 1 => limits.max_chunks = 65,
            2 => limits.context_tokens = 0, 3 => limits.max_output_tokens_per_chunk = 0,
            4 => limits.max_output_tokens_per_chunk = 4097, 5 => limits.max_result_bytes_per_chunk = 0,
            6 => limits.max_result_bytes_per_chunk = u64::MAX, _ => limits.max_result_bytes_per_chunk = 64 << 20 }
        assert!(limits.requirements(32).is_err(), "axis {axis}");
    }
}

#[test]
fn final_identity_kv_and_mask_failures_are_rejected_before_any_native_window() {
    let planner = planner(); let args = source_args(); let vocabulary = vocabulary(); let source = "Alice Bob Carol";
    let prepared = prepare(&planner, source, &args, limits(2));
    let mut ids: Vec<_> = prepared.execution_identities().cloned().collect();
    ids.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final chunk");
    let required = prepared.cohort_requirements(8192).unwrap();
    let script = Scripted::new(&args); let calls = Rc::clone(&script.scalar.calls);
    assert!(prepared.execute_cohorts_with_driver(&ids, &vocabulary, budget(required), script, &mut Continue).is_err());
    assert_eq!(calls.get(), 0);

    let mut prepared = prepare(&planner, source, &args, limits(2));
    let last = prepared.plans.last().unwrap();
    let mut small = args.clone(); small.budget.max_kv_bytes = last.planned_work().forward_positions * KV_BYTES_PER_TOKEN as u64 - 1;
    let text = prepared.chunks.chunks().last().unwrap().text().to_owned();
    let replacement = planner.prepare(BatchDocument { id: "tail".to_owned(), text, task_args: Some(small) }).unwrap();
    *prepared.plans.last_mut().unwrap() = replacement;
    assert!(prepared.cohort_requirements(8192).is_err());

    let mut insufficient_masks = limits(2); insufficient_masks.mapping.mask_visits_per_chunk = 1;
    let prepared = prepare(&planner, source, &args, insufficient_masks);
    let script = Scripted::new(&args); let calls = Rc::clone(&script.scalar.calls);
    assert!(run(prepared, script, &vocabulary).is_err()); assert_eq!(calls.get(), 0);
}

#[test]
fn every_peak_memory_axis_is_enforced_before_the_first_forward() {
    let planner = planner(); let args = source_args(); let vocabulary = vocabulary();
    for axis in 0..6 {
        let prepared = prepare(&planner, "Alice Bob Carol", &args, limits(2));
        let ids: Vec<_> = prepared.execution_identities().cloned().collect();
        let need = prepared.cohort_requirements(8192).unwrap(); let mut cap = budget(need);
        match axis { 0 => cap.native_memory.max_kv_bytes -= 1, 1 => cap.native_memory.max_rope_bytes -= 1,
            2 => cap.native_memory.max_scratch_payload_bytes -= 1, 3 => cap.max_workspace_bytes -= 1,
            4 => cap.max_result_bytes -= 1, _ => cap.context_limit = 1 }
        let script = Scripted::new(&args); let calls = Rc::clone(&script.scalar.calls);
        assert!(prepared.execute_cohorts_with_driver(&ids, &vocabulary, cap, script, &mut Continue).is_err(), "axis {axis}");
        assert_eq!(calls.get(), 0);
    }
}

#[test]
fn corrupt_later_chunk_or_group_metadata_returns_no_partial_document() {
    let planner = planner(); let args = source_args(); let vocabulary = vocabulary();
    for axis in 0..11 {
        let prepared = prepare(&planner, "Alice Bob Carol", &args, limits(2));
        let mut script = Scripted::new(&args); script.scalar.corrupt = Some(axis);
        assert!(run(prepared, script, &vocabulary).is_err(), "row axis {axis}");
    }
    for axis in 0..8 {
        let prepared = prepare(&planner, "Alice Bob Carol", &args, limits(2));
        let mut script = Scripted::new(&args); script.corrupt_group = Some(axis);
        assert!(run(prepared, script, &vocabulary).is_err(), "group axis {axis}");
    }
    let prepared = prepare(&planner, "Alice Bob Carol", &args, limits(2));
    let mut script = Scripted::new(&args); script.scalar.fail_at = Some(3); let calls = Rc::clone(&script.scalar.calls);
    let error = run(prepared, script, &vocabulary).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls.get(), 3);
}

#[test]
fn repeated_unicode_and_empty_quotes_keep_producing_chunk_coordinates_and_one_verification_allowance() {
    let planner = planner(); let args = source_args(); let vocabulary = vocabulary(); let source = "éAéAéAéA";
    for quote in ["é", ""] {
        let prepared = prepare(&planner, source, &args, limits(2));
        let mut script = Scripted::new(&args); script.scalar.quote = Some(quote.to_owned());
        let run = run(prepared, script, &vocabulary).unwrap();
        for chunk in run.document.mapped.root().value().chunks() {
            for field in &chunk.original_fields { for span in &field.spans {
                assert_eq!(&source[span.byte_start..span.byte_end], quote);
                assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
                assert!(span.byte_start >= chunk.source_span.byte_start && span.byte_end <= chunk.source_span.byte_end);
            } }
        }
    }
    let mut limited = limits(1); limited.verification.max_fields = 1;
    let prepared = prepare(&planner, source, &args, limited);
    assert!(run(prepared, Scripted::new(&args), &vocabulary).is_err());
}

#[test]
fn exact_decimal_chunk_json_is_preserved_without_cross_chunk_object_merging() {
    let planner = planner(); let vocabulary = vocabulary();
    let number = "9007199254740993.0000000000000000001";
    let schema = format!(r#"{{"type":"number","enum":[{number}]}}"#);
    let args = args(&schema, ExtractionBatchGrounding::Structural);
    let prepared = prepare(&planner, "Alice Bob Carol", &args, limits(2));
    let mut script = Scripted::new(&args); script.scalar.json = Some(number.to_owned());
    let run = run(prepared, script, &vocabulary).unwrap();
    for chunk in run.document.mapped.root().value().chunks() {
        assert_eq!(chunk.native.result.output.json, number); assert!(chunk.original_fields.is_empty());
    }
}

#[test]
fn final_cancellation_checkpoint_discards_the_complete_result_with_original_cause() {
    struct Count { calls: usize, stop: Option<usize> }
    impl DecodeStepControl for Count {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.stop == Some(self.calls)).then_some(DecodeCancellationKind::CostBudget)
        }
    }
    let planner = planner(); let args = source_args(); let vocabulary = vocabulary();
    let prepared = prepare(&planner, "Alice Bob Carol", &args, limits(2));
    let ids: Vec<_> = prepared.execution_identities().cloned().collect();
    let need = prepared.cohort_requirements(8192).unwrap();
    let mut counter = Count { calls: 0, stop: None };
    prepared.execute_cohorts_with_driver(&ids, &vocabulary, budget(need), Scripted::new(&args), &mut counter).unwrap();
    let prepared = prepare(&planner, "Alice Bob Carol", &args, limits(2));
    let error = prepared.execute_cohorts_with_driver(&ids, &vocabulary, budget(need), Scripted::new(&args),
        &mut Count { calls: 0, stop: Some(counter.calls) }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
}
