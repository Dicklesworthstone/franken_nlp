//! Pinned admission and independent source-map checks over scripted native
//! records. These fixtures do not establish native parity or throughput.
use super::*;
use super::super::{execution::SourceDriver, tests::{self as fixtures, Continue, Script}};
use std::{cell::Cell, rc::Rc};

fn limits(width: usize) -> Int4SourceMapLimits {
    let mut limits = fixtures::limits();
    limits.reduction.map_batch_chunks = width;
    limits.mask_visits_per_chunk = 1_000_000_000;
    limits.max_mask_visits = limits.mask_visits_per_chunk * limits.chunks.max_chunks as u64;
    limits
}
fn vocabulary(planner: &SourceTaskPlanner) -> ExtractionVocabulary {
    ExtractionVocabulary::pinned(&planner.controls).unwrap()
}
fn execution_budget(prepared: &PreparedInt4SourceMap<'_>) -> Int4SourceMapCohortBudget {
    let quote = prepared.cohort_requirements(8192).unwrap();
    Int4SourceMapCohortBudget { context_limit: 8192, native_memory: Int4MemoryBudget {
        max_kv_bytes: quote.native.kv_bytes, max_rope_bytes: quote.native.rope_bytes,
        max_scratch_payload_bytes: quote.native.scratch_payload_bound },
        max_workspace_bytes: quote.workspace_bytes, max_result_bytes: quote.max_result_bytes }
}
struct Scripted {
    inner: Script,
    groups: Rc<Cell<usize>>,
    widths: Rc<RefCell<Vec<usize>>>,
    corrupt: Option<usize>,
}
fn scripted(prepared: &PreparedInt4SourceMap<'_>) -> Scripted {
    Scripted { inner: fixtures::script(prepared), groups: Rc::new(Cell::new(0)),
        widths: Rc::new(RefCell::new(Vec::new())), corrupt: None }
}
impl CohortDriver for Scripted {
    fn execute<C: DecodeStepControl>(&mut self, plans: &[PreparedInt4SourceTask], admitted: &[ExecutionIdentity],
        limits: Int4SourceMapLimits, control: &mut C) -> Result<Int4SourceCohortRun, Int4SourceError> {
        checkpoint(control)?;
        self.groups.set(self.groups.get() + 1); self.widths.borrow_mut().push(plans.len());
        let mut sequences = Vec::new(); let mut planned_work = Int4Work::default();
        let mut model_work = Int4Work::default(); let mut group_steps = 0;
        for (plan, identity) in plans.iter().zip(admitted) {
            checkpoint(control)?;
            let row = self.inner.run(plan, identity)?;
            planned_work = add_work(planned_work, plan.planned_work()).unwrap();
            model_work = add_work(model_work, row.model_work).unwrap();
            group_steps = group_steps.max(row.model_work.forward_positions);
            sequences.push(row);
        }
        let mut run = Int4SourceCohortRun { schema_version: 1, execution: INT4_SOURCE_COHORT_EXECUTION.to_owned(),
            sequences, group_steps, projection_groups: 2, planned_work, model_work };
        if let Some(axis) = self.corrupt {
            match axis {
                0 => { run.sequences.pop(); },
                1 => run.planned_work.attention_pairs += 1,
                2 => run.model_work.projections.multiply_accumulates += 1,
                3 => run.group_steps += 1,
                4 => run.projection_groups = 0,
                5 => run.execution.push('x'),
                6 => run.sequences[0].group_width = 16,
                7 => { let SourceTaskResult::Ner(r) = &mut run.sequences[0].result else { unreachable!() };
                    r.mask_node_visit_charge = limits.mask_visits_per_chunk + 1; },
                _ => { let SourceTaskResult::Ner(r) = &mut run.sequences[0].result else { unreachable!() };
                    r.entities[0].spans[0].scalar_end += 1; },
            }
        }
        Ok(run)
    }
}
fn work_axis(work: &mut Int4Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn static_quote_is_shared_but_source_errors_and_row_permissions_remain_separate() {
    for width in [1, 2, 4, 64] {
        let limits = Int4SourceMapCohortMemoryLimits { max_chunks: width, context_tokens: 8192,
            max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: 4096 };
        let source = limits.requirements(32).unwrap();
        let extraction = Int4ExtractionMapCohortMemoryLimits { max_chunks: width, context_tokens: 8192,
            max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: 4096 }.requirements(32).unwrap();
        assert_eq!(source.native, extraction.native);
        assert_eq!(source.workspace_bytes, extraction.workspace_bytes);
        assert_eq!(source.max_result_bytes, extraction.max_result_bytes);
        assert_eq!(source.native.kv_bytes, Int4MemoryRequirement::for_context(8192, 32).unwrap().kv_bytes * width as u64);
    }
    for width in [0, 65, usize::MAX] {
        let limits = Int4SourceMapCohortMemoryLimits { max_chunks: width, context_tokens: 8192,
            max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: 4096 };
        assert!(matches!(limits.requirements(32), Err(Int4SourceMapError::InvalidLimits)));
    }
}

#[test]
fn ner_keyphrase_and_summary_cohorts_keep_the_serial_semantics_and_chunk_identity() {
    let planner = fixtures::planner(); let vocabulary = vocabulary(&planner);
    let source = "éééééééééééééééééé";
    for task in [SourceMapTask::Ner(NerOptions::default()), SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        SourceMapTask::Summarize(SummaryOptions::default())] {
        let mut identities = None;
        for width in [1, 2, 4] {
            let serial = fixtures::prepare_task(&planner, source, &task, limits(width), 32);
            let admitted = fixtures::admitted(&serial);
            if let Some(previous) = &identities { assert_eq!(&admitted, previous); }
            else { identities = Some(admitted.clone()); }
            let driver = fixtures::script(&serial);
            let serial = serial.execute_with_driver(&admitted, driver).unwrap();
            let prepared = fixtures::prepare_task(&planner, source, &task, limits(width), 32);
            let expected = prepared.preflight_metadata(); let count = prepared.chunk_count();
            let budget = execution_budget(&prepared); let driver = scripted(&prepared);
            let widths = Rc::clone(&driver.widths);
            let grouped = prepared.execute_cohorts_with_driver(&admitted, &vocabulary, budget, driver, &mut Continue).unwrap();
            expected.verify_completed(&grouped.document).unwrap();
            assert_eq!(canonjson::canonical_bytes(&grouped.document).unwrap(), canonjson::canonical_bytes(&serial).unwrap());
            assert_eq!(grouped.cohorts.cohorts, count.div_ceil(width) as u64);
            assert_eq!(grouped.cohorts.chunks, count as u64);
            assert_eq!(grouped.cohorts.planned_work, grouped.document.planned_model_work);
            assert_eq!(grouped.cohorts.model_work, grouped.document.model_work);
            assert_eq!(*widths.borrow(), (0..count).collect::<Vec<_>>().chunks(width).map(<[usize]>::len).collect::<Vec<_>>());
            for chunk in grouped.document.mapped.root().value().chunks() {
                for field in &chunk.original_spans { for span in &field.spans {
                    assert_eq!(&source[span.byte_start..span.byte_end], "é");
                    assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
                    assert_eq!(source[..span.byte_end].chars().count(), span.scalar_end);
                } }
            }
        }
    }
}

#[test]
fn actual_peak_prices_ragged_contexts_and_the_short_final_window_without_padding() {
    let planner = fixtures::planner(); let task = SourceMapTask::Ner(NerOptions::default());
    for group_width in [16, 32, 127] {
        let prepared = fixtures::prepare_task(&planner, "ééééAlice東京 Bob Carol Dave End", &task, limits(4), group_width);
        let quote = prepared.cohort_requirements(8192).unwrap();
        let mut peak = None;
        for plans in prepared.plans.chunks(4) {
            let actual = window(plans, group_width, 8192).unwrap();
            assert_eq!(actual.count, plans.len());
            for (plan, context) in plans.iter().zip(actual.contexts) {
                assert_eq!(context as u64, plan.planned_work().forward_positions);
                assert!(context < 8192);
                assert_eq!(plan.task_budget(), fixtures::budget());
            }
            let exact_kv: u64 = plans.iter().map(|p| Int4MemoryRequirement::for_context(
                p.planned_work().forward_positions as usize, group_width).unwrap().kv_bytes).sum();
            assert_eq!(actual.requirement.native.kv_bytes, exact_kv);
            peak = Some(match peak { None => actual.requirement, Some(previous) => maximum(previous, actual.requirement) });
        }
        assert_eq!(Some(quote), peak);
        let declared = Int4SourceMapCohortMemoryLimits { max_chunks: 4, context_tokens: 8192,
            max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: fixtures::budget().max_output_bytes }
            .requirements(group_width).unwrap();
        assert!(quote.native.kv_bytes < declared.native.kv_bytes);
        assert!(quote.native.rope_bytes <= declared.native.rope_bytes);
        assert!(quote.native.scratch_payload_bound <= declared.native.scratch_payload_bound);
        assert!(quote.workspace_bytes <= declared.workspace_bytes);
    }
}

#[test]
fn final_chunk_identity_or_task_kv_refusal_prevents_every_native_group() {
    let planner = fixtures::planner(); let vocabulary = vocabulary(&planner);
    let task = SourceMapTask::Ner(NerOptions::default());
    for axis in 0..4 {
        let mut prepared = fixtures::prepare_task(&planner, "Alice Bob Carol Dave End", &task, limits(2), 32);
        let mut admitted = fixtures::admitted(&prepared); let budget = execution_budget(&prepared);
        let driver = scripted(&prepared); let groups = Rc::clone(&driver.groups);
        match axis {
            0 => admitted.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final chunk"),
            1 => { admitted.pop(); },
            2 => { let last = prepared.plans.last_mut().unwrap();
                last.budget.max_kv_bytes = Int4MemoryRequirement::for_context(
                    last.planned_work().forward_positions as usize, 32).unwrap().kv_bytes - 1; },
            _ => prepared.plans.last_mut().unwrap().budget.max_kv_bytes = 0,
        }
        assert!(prepared.execute_cohorts_with_driver(&admitted, &vocabulary, budget, driver, &mut Continue).is_err());
        assert_eq!(groups.get(), 0);
    }
}

#[test]
fn every_peak_memory_and_whole_document_work_axis_is_admitted_before_execution() {
    let planner = fixtures::planner(); let vocabulary = vocabulary(&planner);
    let task = SourceMapTask::Ner(NerOptions::default());
    for axis in 0..11 {
        let mut prepared = fixtures::prepare_task(&planner, "Alice Bob Carol Dave End", &task, limits(2), 32);
        let admitted = fixtures::admitted(&prepared); let mut budget = execution_budget(&prepared);
        let driver = scripted(&prepared); let groups = Rc::clone(&driver.groups);
        match axis {
            0 => budget.native_memory.max_kv_bytes -= 1,
            1 => budget.native_memory.max_rope_bytes -= 1,
            2 => budget.native_memory.max_scratch_payload_bytes -= 1,
            3 => budget.max_workspace_bytes -= 1,
            4 => budget.max_result_bytes -= 1,
            5..=9 => { prepared.limits.max_model_work = prepared.planned_work();
                *work_axis(&mut prepared.limits.max_model_work, axis - 5) -= 1; },
            _ => prepared.limits.max_mask_visits = prepared.reserved_mask_visits() - 1,
        }
        assert!(prepared.execute_cohorts_with_driver(&admitted, &vocabulary, budget, driver, &mut Continue).is_err(), "axis {axis}");
        assert_eq!(groups.get(), 0);
    }
}

#[test]
fn native_group_counters_rows_and_source_evidence_are_independently_rechecked() {
    let planner = fixtures::planner(); let vocabulary = vocabulary(&planner);
    let task = SourceMapTask::Ner(NerOptions::default());
    for axis in 0..9 {
        let prepared = fixtures::prepare_task(&planner, "éééééééééééé", &task, limits(2), 32);
        let admitted = fixtures::admitted(&prepared); let budget = execution_budget(&prepared);
        let mut driver = scripted(&prepared); driver.corrupt = Some(axis);
        assert!(prepared.execute_cohorts_with_driver(&admitted, &vocabulary, budget, driver, &mut Continue).is_err(), "axis {axis}");
    }
}

#[test]
fn later_window_cancellation_and_global_grounding_exhaustion_publish_no_document() {
    let planner = fixtures::planner(); let vocabulary = vocabulary(&planner);
    let task = SourceMapTask::Ner(NerOptions::default()); let source = "éééééééééééééééééé";
    let prepared = fixtures::prepare_task(&planner, source, &task, limits(2), 32);
    let admitted = fixtures::admitted(&prepared); let budget = execution_budget(&prepared);
    let driver = scripted(&prepared);
    let used = prepared.execute_cohorts_with_driver(&admitted, &vocabulary, budget, driver, &mut Continue)
        .unwrap().document.verification_used;
    for axis in 0..4 {
        let mut prepared = fixtures::prepare_task(&planner, source, &task, limits(2), 32);
        let admitted = fixtures::admitted(&prepared); let budget = execution_budget(&prepared);
        let mut driver = scripted(&prepared); let groups = Rc::clone(&driver.groups);
        match axis { 0 => prepared.limits.verification.max_fields = used.fields - 1,
            1 => prepared.limits.verification.max_matches = used.matches - 1,
            2 => prepared.limits.verification.max_scan_steps = used.scan_steps - 1,
            _ => driver.inner.fail_run_at = Some(prepared.chunk_count()),
        }
        let error = prepared.execute_cohorts_with_driver(&admitted, &vocabulary, budget, driver, &mut Continue).err().unwrap();
        assert!(groups.get() > 1);
        if axis == 3 { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
    }
}

#[test]
fn cancellation_before_admission_and_after_the_final_result_remains_typed() {
    struct StopAt { calls: usize, stop: usize }
    impl DecodeStepControl for StopAt {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.calls == self.stop).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let planner = fixtures::planner(); let vocabulary = vocabulary(&planner);
    let task = SourceMapTask::Ner(NerOptions::default()); let source = "Alice Bob Carol Dave End";
    let prepared = fixtures::prepare_task(&planner, source, &task, limits(2), 32);
    let admitted = fixtures::admitted(&prepared); let budget = execution_budget(&prepared); let driver = scripted(&prepared);
    let mut control = StopAt { calls: 0, stop: usize::MAX };
    prepared.execute_cohorts_with_driver(&admitted, &vocabulary, budget, driver, &mut control).unwrap();
    for stop in [1, control.calls] {
        let prepared = fixtures::prepare_task(&planner, source, &task, limits(2), 32);
        let admitted = fixtures::admitted(&prepared); let budget = execution_budget(&prepared); let driver = scripted(&prepared);
        let groups = Rc::clone(&driver.groups);
        let error = prepared.execute_cohorts_with_driver(&admitted, &vocabulary, budget,
            driver, &mut StopAt { calls: 0, stop }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        if stop == 1 { assert_eq!(groups.get(), 0); }
    }
}
