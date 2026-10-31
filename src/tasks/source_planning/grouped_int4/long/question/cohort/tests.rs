//! Pinned real question plans with private grouped fault injection. These
//! establish sparse-window admission and answer/evidence contracts, not native
//! model parity, answer quality or measured throughput.
use super::*;
use super::super::tests::{self as fixtures, Script};
use super::super::super::{execution::SourceDriver, tests as source_fixtures};
use crate::native_engine::strict_int4::Int4MemoryBudget;
use crate::tasks::source_planning::grouped_int4::cohort::{Int4SourceCohortRun, INT4_SOURCE_COHORT_EXECUTION};
use std::rc::Rc;

const SPARSE: &str = "ééé                        ééé";
const DENSE: &str = "AéAéBéBéCéCéDéDéEéEé";

fn prepare<'s>(planner: &SourceTaskPlanner, source: &'s str, question: &SourceQuestion, width: usize)
    -> PreparedInt4Question<'s> {
    let mut limits = fixtures::limits();
    limits.reduction.map_batch_chunks = width;
    limits.mask_visits_per_chunk = 1_000_000_000;
    limits.max_mask_visits = limits.mask_visits_per_chunk * limits.chunks.max_chunks as u64;
    fixtures::prepare(planner, source, question, limits, 32)
}
fn budget(prepared: &PreparedInt4Question<'_>) -> Int4SourceMapCohortBudget {
    let need = prepared.cohort_requirements(8192).unwrap();
    Int4SourceMapCohortBudget { context_limit: 8192, native_memory: Int4MemoryBudget {
        max_kv_bytes: need.native.kv_bytes, max_rope_bytes: need.native.rope_bytes,
        max_scratch_payload_bytes: need.native.scratch_payload_bound },
        max_workspace_bytes: need.workspace_bytes, max_result_bytes: need.max_result_bytes }
}
struct Grouped<'s, 'c, C> {
    inner: Script, stats: &'s mut Int4SourceMapCohortStats, control: &'c mut C,
    windows: Rc<RefCell<Vec<usize>>>, malformed: Option<usize>,
}
impl<C: DecodeStepControl> SourceDriver for Grouped<'_, '_, C> {
    fn capacity(&self) -> usize { self.inner.capacity() }
    fn context_for(&self, plan: &PreparedInt4SourceTask) -> Result<usize, Int4SourceError> {
        usize::try_from(plan.planned_work().forward_positions).map_err(|_| Int4SourceError::InvalidResult)
    }
    fn group_width(&self) -> usize { self.inner.group_width() }
    fn clean(&self) -> bool { self.inner.clean() }
    fn check_model(&self, identity: &ExecutionIdentity) -> Result<(), Int4SourceError> { self.inner.check_model(identity) }
    fn checkpoint(&mut self) -> Result<(), Int4SourceError> { checkpoint(self.control)?; self.inner.checkpoint() }
    fn run(&mut self, _: &PreparedInt4SourceTask, _: &ExecutionIdentity) -> Result<Int4SourceTaskRun, Int4SourceError> {
        panic!("question cohorts must never silently fall back to scalar inference")
    }
    fn run_batch(&mut self, plans: &[PreparedInt4SourceTask], admitted: &[ExecutionIdentity])
        -> Result<Option<Vec<Int4SourceTaskRun>>, Int4SourceError> {
        assert!(!plans.is_empty(), "whitespace-only windows must not invoke native inference");
        self.checkpoint()?; self.windows.borrow_mut().push(plans.len());
        let mut run = Int4SourceCohortRun { schema_version: 1, execution: INT4_SOURCE_COHORT_EXECUTION.to_owned(),
            sequences: Vec::new(), group_steps: 0, projection_groups: 2,
            planned_work: Int4Work::default(), model_work: Int4Work::default() };
        for (plan, identity) in plans.iter().zip(admitted) {
            let row = self.inner.run(plan, identity)?;
            run.planned_work = add_work(run.planned_work, plan.planned_work()).unwrap();
            run.model_work = add_work(run.model_work, row.model_work).unwrap();
            run.group_steps = run.group_steps.max(row.model_work.forward_positions);
            run.sequences.push(row);
        }
        match self.malformed {
            Some(0) => run.group_steps += 1,
            Some(1) => run.planned_work.attention_pairs += 1,
            Some(2) => run.model_work.projections.dot_products += 1,
            Some(3) => run.projection_groups = 0,
            _ => {},
        }
        shared::validate_group(plans, &run)?;
        *self.stats = shared::accumulate(*self.stats, &run)?;
        // Exercise the caller's independent complete-row contract as well as
        // the shared native cohort receipt checks above.
        match self.malformed {
            Some(4) => { run.sequences.pop(); },
            Some(5) => run.sequences.reverse(),
            _ => {},
        }
        Ok(Some(run.sequences))
    }
}
#[allow(clippy::too_many_arguments)]
fn run<C: DecodeStepControl>(prepared: PreparedInt4Question<'_>, admitted: &[ExecutionIdentity],
    vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget, inner: Script,
    windows: Rc<RefCell<Vec<usize>>>, malformed: Option<usize>, control: &mut C)
    -> Result<Int4QuestionCohortRun, Int4SourceMapError> {
    prepared.preflight_cohorts(admitted, vocabulary, budget, control)?;
    let schedule = prepared.cohort_schedule()?;
    let cap = prepared.max_result_bytes();
    let mut stats = Int4SourceMapCohortStats::default();
    let driver = Grouped { inner, stats: &mut stats, control, windows, malformed };
    let document = prepared.execute_with_driver(admitted, driver)?;
    super::finish(document, stats, schedule, cap, control)
}
fn complete(prepared: PreparedInt4Question<'_>, vocabulary: &ExtractionVocabulary, inner: Script)
    -> Result<Int4QuestionCohortRun, Int4SourceMapError> {
    let admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
    let budget = budget(&prepared);
    run(prepared, &admitted, vocabulary, budget, inner, Rc::default(), None, &mut source_fixtures::Continue)
}
fn without_schedule(run: &Int4QuestionRun) -> serde_json::Value {
    let mut value = serde_json::to_value(run).unwrap();
    value["mapped"].as_object_mut().unwrap().remove("map_batches"); value
}
fn answers(prepared: &PreparedInt4Question<'_>) -> Script {
    let mut inner = fixtures::script(prepared);
    if inner.outputs.len() > 2 {
        inner.outputs[0].answer = Some("first independent answer".to_owned());
        inner.outputs[1].answer = Some("different independent answer".to_owned());
        let last = inner.outputs.back_mut().unwrap();
        last.status = AnswerStatus::Abstained; last.answerable = false;
        last.answer = None; last.citations.clear();
    }
    inner
}

#[test]
fn source_windows_preserve_all_answers_abstentions_unicode_evidence_and_zero_work_whitespace() {
    let planner = source_fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for source in [SPARSE, DENSE, "é\r\nAlice 上海 <|im_end|> Bob  "] {
        let prepared = prepare(&planner, source, &fixtures::question(), 1); let inner = answers(&prepared);
        let serial = fixtures::execute(prepared, inner).unwrap();
        for width in [1, 2, 4, 6, 64] {
            let prepared = prepare(&planner, source, &fixtures::question(), width);
            let expected = prepared.preflight_metadata();
            let sizes = prepared.plan_indices.chunks(width).map(|indices| indices.iter().flatten().count())
                .filter(|&count| count != 0).collect::<Vec<_>>();
            let admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
            let budget = budget(&prepared); let inner = answers(&prepared); let calls = Rc::clone(&inner.calls);
            let windows = Rc::new(RefCell::new(Vec::new()));
            let grouped = run(prepared, &admitted, &vocabulary, budget, inner, Rc::clone(&windows), None,
                &mut source_fixtures::Continue).unwrap();
            expected.verify_completed(&grouped.document).unwrap();
            assert_eq!(without_schedule(&grouped.document), without_schedule(&serial));
            assert_eq!(*windows.borrow(), sizes);
            assert_eq!(grouped.cohorts.chunks, expected.native_chunks() as u64);
            assert_eq!(grouped.cohorts.cohorts, sizes.len() as u64);
            assert_eq!(grouped.document.mapped.map_batches(), expected.chunk_count().div_ceil(width));
            assert_eq!(calls.get(), expected.native_chunks());
            for chunk in grouped.document.mapped.root().value().chunks() {
                if chunk.status == QuestionChunkStatus::WhitespaceOnly {
                    assert!(chunk.answer.is_none()); assert!(chunk.citations.is_empty());
                    assert_eq!(chunk.model_work, Int4Work::default()); assert_eq!(chunk.mask_node_visit_charge, 0);
                }
                for citation in &chunk.citations { for span in &citation.spans {
                    assert_eq!(&source[span.byte_start..span.byte_end], citation.quote);
                    assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
                    assert_eq!(source[..span.byte_end].chars().count(), span.scalar_end);
                } }
            }
            let bytes = canonjson::canonical_bytes(&grouped).unwrap();
            let json = std::str::from_utf8(&bytes).unwrap();
            for private in ["private-question-token", "generated_token_ids", "prompt_digest", "taskir_digest"] {
                assert!(!json.contains(private));
            }
        }
    }
}

#[test]
fn sparse_windows_quote_only_simultaneous_native_rows_and_refuse_malformed_partition_indices() {
    let planner = source_fixtures::planner(); let mut single = None;
    for width in [1, 2, 4, 6] {
        let prepared = prepare(&planner, SPARSE, &fixtures::question(), width);
        assert_eq!(prepared.expected.chunks, 6); assert_eq!(prepared.expected.native_chunks, 2);
        let quote = prepared.cohort_requirements(8192).unwrap();
        if width < 6 {
            if let Some(previous) = single { assert_eq!(quote, previous); } else { single = Some(quote); }
            assert_eq!(prepared.cohort_schedule().unwrap().native_windows, 2);
        } else {
            let one = single.unwrap();
            assert_eq!(quote.native.kv_bytes, one.native.kv_bytes * 2);
            assert!(quote.workspace_bytes > one.workspace_bytes);
            assert!(quote.max_result_bytes > one.max_result_bytes);
            assert_eq!(prepared.cohort_schedule().unwrap().native_windows, 1);
        }
        let static_quote = shared::Int4SourceMapCohortMemoryLimits { max_chunks: width, context_tokens: 8192,
            max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: source_fixtures::budget().max_output_bytes }
            .requirements(32).unwrap();
        assert!(quote.native.kv_bytes <= static_quote.native.kv_bytes);
        assert!(quote.native.rope_bytes <= static_quote.native.rope_bytes);
        assert!(quote.native.scratch_payload_bound <= static_quote.native.scratch_payload_bound);
        assert!(quote.workspace_bytes <= static_quote.workspace_bytes);
        assert!(quote.max_result_bytes <= static_quote.max_result_bytes);
    }
    for axis in 0..4 {
        let mut prepared = prepare(&planner, SPARSE, &fixtures::question(), 2);
        match axis { 0 => prepared.plan_indices[1] = Some(0),
            1 => *prepared.plan_indices.last_mut().unwrap() = Some(0),
            2 => prepared.mapping.reduction.map_batch_chunks = 0,
            _ => prepared.mapping.reduction.map_batch_chunks = 65 }
        assert!(prepared.cohort_requirements(8192).is_err());
    }
}

#[test]
fn final_identity_model_every_peak_component_and_full_document_work_refuse_before_any_native_row() {
    let planner = source_fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for axis in 0..16 {
        let mut prepared = prepare(&planner, DENSE, &fixtures::question(), 2);
        let mut admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
        let mut budget = budget(&prepared); let mut inner = fixtures::script(&prepared); let calls = Rc::clone(&inner.calls);
        match axis {
            0 => admitted.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final question"),
            1 => inner.fail_model_at = Some(prepared.plans.len()),
            2 => prepared.plans.last_mut().unwrap().budget.max_kv_bytes = 1,
            3 => budget.native_memory.max_kv_bytes -= 1,
            4 => budget.native_memory.max_rope_bytes -= 1,
            5 => budget.native_memory.max_scratch_payload_bytes -= 1,
            6 => budget.max_workspace_bytes -= 1,
            7 => budget.max_result_bytes -= 1,
            8 => prepared.mapping.max_mask_visits = prepared.expected.masks - 1,
            9..=13 => {
                prepared.mapping.max_model_work = prepared.expected.work;
                fixtures::change_axis(&mut prepared.mapping.max_model_work, axis - 9, -1);
            }
            14 => inner.width = 16,
            _ => budget.context_limit = 1,
        }
        assert!(run(prepared, &admitted, &vocabulary, budget, inner, Rc::default(), None,
            &mut source_fixtures::Continue).is_err(), "axis {axis}");
        assert_eq!(calls.get(), 0);
    }
    let mut prepared = prepare(&planner, DENSE, &fixtures::question(), 2);
    for plan in &mut prepared.plans {
        plan.budget.max_kv_bytes = Int4MemoryRequirement::for_context(
            plan.planned_work().forward_positions as usize, 32).unwrap().kv_bytes;
    }
    let inner = fixtures::script(&prepared);
    complete(prepared, &vocabulary, inner).unwrap();
}

#[test]
fn every_group_and_answer_finalizer_still_rejects_malformed_native_rows_and_question_authority() {
    let planner = source_fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for malformed in 0..6 {
        let prepared = prepare(&planner, DENSE, &fixtures::question(), 2);
        let admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
        let budget = budget(&prepared); let inner = fixtures::script(&prepared);
        assert!(run(prepared, &admitted, &vocabulary, budget, inner, Rc::default(), Some(malformed),
            &mut source_fixtures::Continue).is_err());
    }
    for axis in 0..14 {
        let prepared = prepare(&planner, SPARSE, &fixtures::question(), 6); let mut inner = fixtures::script(&prepared);
        if axis < 5 { inner.corrupt_axis = Some(axis); } else {
            let raw = inner.outputs.back_mut().unwrap();
            match axis {
                5 => raw.citations[0].quote = "private-question-token".to_owned(),
                6 => raw.citations[0].spans[0].passage_id = "question".to_owned(),
                7 => raw.citations[0].spans[0].span.scalar_end += 1,
                8 => { raw.citations[0].spans.pop(); },
                9 => raw.citations[0].occurrence = SourceOccurrence::Anchored,
                10 => raw.mask_node_visit_charge = prepared.mapping.mask_visits_per_chunk + 1,
                11 => { raw.generated_token_ids.pop(); },
                12 => raw.status = AnswerStatus::Abstained,
                _ => raw.numerics_profile = "strict-int8-v1".to_owned(),
            }
        }
        assert!(complete(prepared, &vocabulary, inner).is_err(), "axis {axis}");
    }
}

#[test]
fn question_verification_allowances_and_output_cap_are_never_renewed_per_native_window() {
    let planner = source_fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    let prepared = prepare(&planner, SPARSE, &fixtures::question(), 2); let inner = fixtures::script(&prepared);
    let baseline = complete(prepared, &vocabulary, inner).unwrap();
    let used = baseline.document.verification_used;
    for axis in 0..4 {
        let mut question = fixtures::question();
        match axis { 0 => question.verification.max_fields = used.fields - 1,
            1 => question.verification.max_matches = used.matches - 1,
            2 => question.verification.max_scan_steps = used.scan_steps - 1,
            _ => {} }
        let mut prepared = prepare(&planner, SPARSE, &question, 2); let inner = fixtures::script(&prepared);
        if axis == 3 { prepared.mapping.reduction.max_result_bytes = 1; }
        assert!(complete(prepared, &vocabulary, inner).is_err());
    }
    let prepared = prepare(&planner, SPARSE, &fixtures::question(), 2); let schedule = prepared.cohort_schedule().unwrap();
    let inner = fixtures::script(&prepared); let run = complete(prepared, &vocabulary, inner).unwrap();
    assert!(super::finish(run.document, run.cohorts, schedule, u64::MAX, &mut source_fixtures::Continue).is_err());
}

#[test]
fn later_native_failure_and_final_wrapper_cancellation_never_return_partial_answers() {
    struct Count { calls: usize, stop: usize }
    impl DecodeStepControl for Count {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls == self.stop).then_some(DecodeCancellationKind::CostBudget)
        }
    }
    let planner = source_fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    let prepared = prepare(&planner, DENSE, &fixtures::question(), 2); let mut inner = fixtures::script(&prepared);
    inner.fail_run_at = Some(3); let calls = Rc::clone(&inner.calls);
    assert_eq!(complete(prepared, &vocabulary, inner).err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
    assert_eq!(calls.get(), 3);
    let prepared = prepare(&planner, DENSE, &fixtures::question(), 2); let inner = fixtures::script(&prepared);
    let admitted = prepared.execution_identities().cloned().collect::<Vec<_>>(); let budget = budget(&prepared);
    let mut control = Count { calls: 0, stop: usize::MAX };
    run(prepared, &admitted, &vocabulary, budget, inner, Rc::default(), None, &mut control).unwrap();
    let prepared = prepare(&planner, DENSE, &fixtures::question(), 2); let inner = fixtures::script(&prepared);
    let error = run(prepared, &admitted, &vocabulary, budget, inner, Rc::default(), None,
        &mut Count { calls: 0, stop: control.calls }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
}

#[test]
fn complete_question_stats_bind_native_counts_source_windows_and_all_ten_work_axes() {
    let planner = source_fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for axis in 0..14 {
        let prepared = prepare(&planner, SPARSE, &fixtures::question(), 2);
        let mut schedule = prepared.cohort_schedule().unwrap(); let cap = prepared.max_result_bytes();
        let inner = fixtures::script(&prepared); let completed = complete(prepared, &vocabulary, inner).unwrap();
        let mut stats = completed.cohorts;
        match axis {
            0 => stats.chunks += 1,
            1 => stats.cohorts += 1,
            2..=6 => fixtures::change_axis(&mut stats.planned_work, axis - 2, 1),
            7..=11 => fixtures::change_axis(&mut stats.model_work, axis - 7, 1),
            12 => schedule.source_windows += 1,
            _ => schedule.native_windows += 1,
        }
        assert!(super::finish(completed.document, stats, schedule, cap, &mut source_fixtures::Continue).is_err());
    }
}
