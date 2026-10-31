//! Private grouped fault injection over real pinned plans. These regressions
//! exercise composition and exact reducers, not native parity or throughput.
use super::*;
use super::super::{execution::SourceDriver, tests::{self as fixtures, Script},
    summary::{self, Int4SummaryLimits, Int4CorpusSummaryCohortRun, Int4CorpusSummaryError},
    keyphrases::{self, Int4KeyphraseLimits, Int4CorpusKeyphraseCohortRun, Int4CorpusKeyphraseError}};
use std::rc::Rc;
use crate::tasks::keyphrases::RankedKeyphrase;

const SOURCE: &str = "AéAéBéBéCéCéDéDéEéEé";
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
#[derive(Clone, Copy)]
enum Aggregation { Summary(Int4SummaryLimits), Keyphrases(Int4KeyphraseLimits) }
impl Aggregation {
    fn all() -> [Self; 2] {
        [Self::Summary(Int4SummaryLimits { max_bullets: 1, ..Default::default() }),
            Self::Keyphrases(Int4KeyphraseLimits { max_phrases: 1, ..Default::default() })]
    }
    fn task(self) -> SourceMapTask {
        match self { Self::Summary(_) => SourceMapTask::Summarize(SummaryOptions::default()),
            Self::Keyphrases(_) => SourceMapTask::Keyphrases(KeyphraseOptions::default()) }
    }
}
enum Completed { Summary(Int4CorpusSummaryCohortRun), Keyphrases(Int4CorpusKeyphraseCohortRun) }
impl Completed {
    fn document(&self) -> serde_json::Value {
        match self { Self::Summary(run) => serde_json::to_value(&run.document).unwrap(),
            Self::Keyphrases(run) => serde_json::to_value(&run.document).unwrap() }
    }
    fn stats(&self) -> Int4SourceMapCohortStats {
        match self { Self::Summary(run) => run.cohorts, Self::Keyphrases(run) => run.cohorts }
    }
}
#[derive(Debug)]
enum Failure { Map(Int4SourceMapError), Summary(Int4CorpusSummaryError), Keyphrases(Int4CorpusKeyphraseError) }
impl Failure {
    fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Map(e) => e.cancellation(), Self::Summary(e) => e.cancellation(), Self::Keyphrases(e) => e.cancellation() }
    }
}
fn prepare(planner: &SourceTaskPlanner, aggregation: Aggregation, width: usize) -> PreparedInt4SourceMap<'static> {
    let mut limits = fixtures::limits();
    limits.chunks.max_chunk_bytes = 6; limits.chunks.max_chunk_tokens = 6;
    limits.reduction.map_batch_chunks = width;
    limits.mask_visits_per_chunk = 1_000_000_000;
    limits.max_mask_visits = limits.mask_visits_per_chunk * limits.chunks.max_chunks as u64;
    fixtures::prepare_task(planner, SOURCE, &aggregation.task(), limits, 32)
}
fn script(prepared: &PreparedInt4SourceMap<'_>) -> Script {
    let mut script = fixtures::script(prepared);
    for (raw, chunk) in script.outputs.iter_mut().zip(prepared.chunks.chunks()) {
        if let SourceTaskResult::Keyphrases(result) = raw {
            let local = chunk.text().chars().next().unwrap().to_string();
            result.phrases = [&*local, "é"].iter().enumerate().map(|(index, &text)| {
                let spans = scan_occurrences(chunk.text(), text, &mut GroundingBudget::default()).unwrap();
                RankedKeyphrase { rank: index + 1, text: text.to_owned(),
                    occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
            }).collect();
        }
    }
    script
}
fn budget(prepared: &PreparedInt4SourceMap<'_>) -> Int4SourceMapCohortBudget {
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
        panic!("an opted-in aggregate cohort must never silently fall back to scalar inference")
    }
    fn run_batch(&mut self, plans: &[PreparedInt4SourceTask], admitted: &[ExecutionIdentity])
        -> Result<Option<Vec<Int4SourceTaskRun>>, Int4SourceError> {
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
        if let Some(axis) = self.malformed {
            match axis { 0 => run.group_steps += 1, 1 => run.planned_work.attention_pairs += 1,
                2 => run.model_work.projections.dot_products += 1, _ => run.projection_groups = 0 }
        }
        validate_group(plans, &run)?;
        *self.stats = accumulate(*self.stats, &run)?;
        Ok(Some(run.sequences))
    }
}
#[allow(clippy::too_many_arguments)]
fn run<C: DecodeStepControl>(prepared: PreparedInt4SourceMap<'_>, admitted: &[ExecutionIdentity],
    vocabulary: &ExtractionVocabulary, aggregation: Aggregation, budget: Int4SourceMapCohortBudget,
    inner: Script, windows: Rc<RefCell<Vec<usize>>>, malformed: Option<usize>, control: &mut C)
    -> Result<Completed, Failure> {
    prepared.preflight_cohorts(admitted, vocabulary, budget, control).map_err(Failure::Map)?;
    let mut stats = Int4SourceMapCohortStats::default();
    let driver = Grouped { inner, stats: &mut stats, control, windows, malformed };
    match aggregation {
        Aggregation::Summary(limits) => {
            let result = prepared.execute_summary_with_driver(admitted, limits, driver).map_err(Failure::Summary)?;
            summary::finish_cohorts(result, stats, limits.max_result_bytes, control).map(Completed::Summary).map_err(Failure::Summary)
        }
        Aggregation::Keyphrases(limits) => {
            let result = prepared.execute_keyphrases_with_driver(admitted, limits, driver).map_err(Failure::Keyphrases)?;
            keyphrases::finish_cohorts(result, stats, limits.max_result_bytes, control).map(Completed::Keyphrases).map_err(Failure::Keyphrases)
        }
    }
}
fn finish(prepared: PreparedInt4SourceMap<'_>, vocabulary: &ExtractionVocabulary, aggregation: Aggregation,
    inner: Script) -> Result<Completed, Failure> {
    let ids = fixtures::admitted(&prepared); let budget = budget(&prepared);
    run(prepared, &ids, vocabulary, aggregation, budget, inner, Rc::default(), None, &mut Continue)
}
fn without_schedule(mut value: serde_json::Value) -> serde_json::Value {
    value.as_object_mut().unwrap().remove("map_batches"); value
}

#[test]
fn grouped_exact_reducers_keep_serial_candidates_ranks_work_and_all_original_occurrences() {
    let planner = fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for aggregation in Aggregation::all() {
        let prepared = prepare(&planner, aggregation, 1); let ids = fixtures::admitted(&prepared); let driver = script(&prepared);
        let serial = match aggregation {
            Aggregation::Summary(limits) => serde_json::to_value(prepared.execute_summary_with_driver(&ids, limits, driver).unwrap()).unwrap(),
            Aggregation::Keyphrases(limits) => serde_json::to_value(prepared.execute_keyphrases_with_driver(&ids, limits, driver).unwrap()).unwrap(),
        };
        for width in [1, 2, 4, 64] {
            let prepared = prepare(&planner, aggregation, width); let count = prepared.chunk_count();
            assert_eq!(count, 5);
            let ids = fixtures::admitted(&prepared); let driver = script(&prepared); let budget = budget(&prepared);
            let windows = Rc::new(RefCell::new(Vec::new()));
            let grouped = run(prepared, &ids, &vocabulary, aggregation, budget, driver, Rc::clone(&windows), None, &mut Continue).unwrap();
            assert_eq!(without_schedule(grouped.document()), without_schedule(serial.clone()));
            assert_eq!(grouped.stats().chunks, count as u64);
            assert_eq!(grouped.stats().cohorts, count.div_ceil(width) as u64);
            assert_eq!(*windows.borrow(), (0..count).step_by(width).map(|at| width.min(count - at)).collect::<Vec<_>>());
            match grouped {
                Completed::Summary(run) => {
                    assert_eq!(run.document.summary.bullets.len(), 1);
                    assert_eq!(run.document.summary.bullets[0].text, "common");
                    assert_eq!(run.document.summary.bullets[0].evidence.len(), 5);
                }
                Completed::Keyphrases(run) => {
                    let phrase = &run.document.keyphrases.phrases[0];
                    assert_eq!(phrase.text, "é"); assert_eq!(phrase.rank_sum, 10); assert_eq!(phrase.evidence.len(), 5);
                    assert_eq!(run.document.keyphrases.omitted_candidates, 5);
                    for evidence in &phrase.evidence { for span in &evidence.spans {
                        assert_eq!(&SOURCE[span.byte_start..span.byte_end], "é");
                        assert_eq!(SOURCE[..span.byte_start].chars().count(), span.scalar_start);
                    } }
                }
            }
        }
    }
}

#[test]
fn every_last_identity_model_capacity_and_peak_is_checked_before_grouped_aggregate_inference() {
    let planner = fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for aggregation in Aggregation::all() { for axis in 0..5 {
        let mut prepared = prepare(&planner, aggregation, 2); let mut ids = fixtures::admitted(&prepared);
        let mut budget = budget(&prepared); let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        match axis { 0 => ids.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final aggregate row"),
            1 => driver.fail_model_at = Some(prepared.chunk_count()),
            2 => prepared.plans.last_mut().unwrap().budget.max_kv_bytes = 1,
            3 => budget.max_workspace_bytes -= 1,
            _ => prepared.limits.max_mask_visits = prepared.reserved_mask_visits() - 1 }
        assert!(run(prepared, &ids, &vocabulary, aggregation, budget, driver, Rc::default(), None, &mut Continue).is_err());
        assert_eq!(calls.get(), 0);
    } }
    for aggregation in Aggregation::all() {
        let mut prepared = prepare(&planner, aggregation, 2);
        for plan in &mut prepared.plans { plan.budget.max_kv_bytes =
            Int4MemoryRequirement::for_context(plan.planned_work().forward_positions as usize, 32).unwrap().kv_bytes; }
        let driver = script(&prepared);
        finish(prepared, &vocabulary, aggregation, driver).unwrap();
    }
}

#[test]
fn grouped_native_metadata_and_unselected_final_candidates_cannot_escape_verification() {
    let planner = fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for aggregation in Aggregation::all() {
        for axis in 0..4 {
            let prepared = prepare(&planner, aggregation, 2); let ids = fixtures::admitted(&prepared);
            let budget = budget(&prepared); let driver = script(&prepared);
            assert!(run(prepared, &ids, &vocabulary, aggregation, budget, driver, Rc::default(), Some(axis), &mut Continue).is_err());
        }
        let prepared = prepare(&planner, aggregation, 2); let mut driver = script(&prepared);
        match driver.outputs.back_mut().unwrap() {
            SourceTaskResult::Summarize(result) => result.bullets[0].citations[0].spans[0].scalar_end += 1,
            SourceTaskResult::Keyphrases(result) => result.phrases[0].spans[0].scalar_end += 1,
            _ => unreachable!(),
        }
        let calls = Rc::clone(&driver.calls);
        assert!(finish(prepared, &vocabulary, aggregation, driver).is_err()); assert_eq!(calls.get(), 5);
    }
}

#[test]
fn final_top_k_never_hides_whole_candidate_or_nonrenewable_verification_overflow() {
    let planner = fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for aggregation in Aggregation::all() {
        let prepared = prepare(&planner, aggregation, 2); let driver = script(&prepared);
        let baseline = finish(prepared, &vocabulary, aggregation, driver).unwrap().document();
        for axis in 0..6 {
            let mut prepared = prepare(&planner, aggregation, 2); let driver = script(&prepared); let mut limits = aggregation;
            match axis {
                0 => prepared.limits.verification.max_fields = baseline["verification_used"]["fields"].as_u64().unwrap() as usize - 1,
                1 => prepared.limits.verification.max_matches = baseline["verification_used"]["matches"].as_u64().unwrap() as usize - 1,
                2 => prepared.limits.verification.max_scan_steps = baseline["verification_used"]["scan_steps"].as_u64().unwrap() - 1,
                3 => match &mut limits {
                    Aggregation::Summary(l) => l.aggregation.max_scan_steps = baseline["verification_scan_steps"].as_u64().unwrap() - 1,
                    Aggregation::Keyphrases(l) => l.aggregation.max_scan_work = baseline["verification_scan_work"].as_u64().unwrap() - 1 },
                4 => match &mut limits { Aggregation::Summary(l) => l.aggregation.max_unique_bullets = 5,
                    Aggregation::Keyphrases(l) => l.aggregation.max_unique_phrases = 5 },
                _ => match &mut limits { Aggregation::Summary(l) => l.max_result_bytes = 1,
                    Aggregation::Keyphrases(l) => l.max_result_bytes = 1 },
            }
            assert!(finish(prepared, &vocabulary, limits, driver).is_err(), "axis {axis}");
        }
    }
}

#[test]
fn later_native_window_and_final_wrapper_cancellation_keep_the_original_cause() {
    struct Count { calls: usize, stop: usize }
    impl DecodeStepControl for Count {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls == self.stop).then_some(DecodeCancellationKind::CostBudget)
        }
    }
    let planner = fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for aggregation in Aggregation::all() {
        let prepared = prepare(&planner, aggregation, 2); let mut driver = script(&prepared); driver.fail_run_at = Some(5);
        assert_eq!(finish(prepared, &vocabulary, aggregation, driver).err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
        let prepared = prepare(&planner, aggregation, 2); let driver = script(&prepared);
        let ids = fixtures::admitted(&prepared); let need = budget(&prepared); let mut control = Count { calls: 0, stop: usize::MAX };
        run(prepared, &ids, &vocabulary, aggregation, need, driver, Rc::default(), None, &mut control).unwrap();
        let prepared = prepare(&planner, aggregation, 2); let driver = script(&prepared);
        let error = run(prepared, &ids, &vocabulary, aggregation, need, driver, Rc::default(), None,
            &mut Count { calls: 0, stop: control.calls }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
    }
}

#[test]
fn final_physical_ledger_must_match_complete_aggregate_chunks_batches_and_all_work_axes() {
    let planner = fixtures::planner(); let vocabulary = ExtractionVocabulary::pinned(&planner.controls).unwrap();
    for aggregation in Aggregation::all() { for axis in 0..12 {
        let prepared = prepare(&planner, aggregation, 2); let driver = script(&prepared);
        let completed = finish(prepared, &vocabulary, aggregation, driver).unwrap(); let mut stats = completed.stats();
        if axis == 0 { stats.chunks += 1; } else if axis == 1 { stats.cohorts += 1; }
        else { let work = if axis < 7 { &mut stats.planned_work } else { &mut stats.model_work };
            match (axis - 2) % 5 { 0 => work.forward_positions += 1, 1 => work.projected_logits += 1,
                2 => work.attention_pairs += 1, 3 => work.projections.dot_products += 1,
                _ => work.projections.multiply_accumulates += 1 } }
        match completed {
            Completed::Summary(run) => assert!(summary::finish_cohorts(run.document, stats, 4 << 20, &mut Continue).is_err()),
            Completed::Keyphrases(run) => assert!(keyphrases::finish_cohorts(run.document, stats, 4 << 20, &mut Continue).is_err()),
        }
    } }
}
