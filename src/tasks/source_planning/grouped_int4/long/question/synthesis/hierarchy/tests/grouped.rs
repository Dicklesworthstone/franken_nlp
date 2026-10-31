//! UNRUN private scheduling fixtures. Synthetic native rows exercise ordered
//! lineage and refusal contracts, never model parity, accuracy or throughput.
use super::*;
use super::super::grouped as entry;
use crate::tasks::source_planning::grouped_int4::long::cohort::{
    Int4SourceMapCohortMemoryLimits, Int4SourceMapCohortStats,
    hierarchy::{Int4HierarchyCohortSchedule, Int4HierarchyCohortStats},
};
use execution::Driver;

struct BatchScript {
    inner: Script, width: usize, windows: Vec<usize>, stats: Int4HierarchyCohortStats,
    corrupt_last_witness: bool, omit_row: bool, swap_rows: bool,
}
fn grouped(expected: Int4QuestionHierarchyPreflight, width: usize) -> BatchScript {
    BatchScript { inner: script(expected), width, windows: vec![], stats: Int4HierarchyCohortStats::default(),
        corrupt_last_witness: false, omit_row: false, swap_rows: false }
}
impl Driver for BatchScript {
    type Witness = <Script as Driver>::Witness;
    fn batch_width(&self) -> usize { self.width }
    fn count<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], bytes: usize, control: &mut C)
        -> Result<Option<usize>, Int4SourceMapError> { self.inner.count(passages, bytes, control) }
    fn prepare<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], prompt: usize, control: &mut C)
        -> Result<Self::Witness, Int4SourceMapError> {
        let mut witness = self.inner.prepare(passages, prompt, control)?;
        if self.corrupt_last_witness && self.inner.preparations.len() == 2 {
            witness.0 = Sha256Digest::of_bytes(b"different private question or last passage");
        }
        Ok(witness)
    }
    fn run<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], witness: &Self::Witness, control: &mut C)
        -> Result<(Int4SourceTaskRun, Int4Work), Int4SourceMapError> { self.inner.run(passages, witness, control) }
    fn run_batch<C: DecodeStepControl>(&mut self, passages: &[&[AnswerPassage]], witnesses: &[Self::Witness], control: &mut C)
        -> Result<Vec<(Int4SourceTaskRun, Int4Work)>, Int4SourceMapError> {
        assert_eq!(passages.len(), witnesses.len()); assert!(passages.len() <= self.width);
        // Model the production ALL-witness check before entering any row.
        for (passages, witness) in passages.iter().zip(witnesses) {
            if witness.0 != self.inner.binding(passages, witness.1) { return Err(Int4SourceMapError::Admission); }
        }
        self.windows.push(passages.len());
        let mut rows = Vec::new();
        for (passages, witness) in passages.iter().zip(witnesses) { rows.push(self.inner.run(passages, witness, control)?); }
        self.stats.cohorts += 1; self.stats.passes += rows.len() as u64;
        self.stats.group_steps += rows.iter().map(|(run, _)| run.model_work.forward_positions).max().unwrap();
        self.stats.projection_groups += rows.iter().map(|(run, _)|
            run.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64).max().unwrap();
        for (run, planned) in &rows {
            self.stats.planned_work = add_work(self.stats.planned_work, *planned).unwrap();
            self.stats.model_work = add_work(self.stats.model_work, run.model_work).unwrap();
        }
        if self.omit_row { rows.pop(); }
        if self.swap_rows && rows.len() > 1 { rows.swap(0, 1); }
        Ok(rows)
    }
}
fn discovery_stats(discovery: &Int4QuestionRun) -> Int4SourceMapCohortStats {
    Int4SourceMapCohortStats { cohorts: discovery.native_chunks as u64, chunks: discovery.native_chunks as u64,
        group_steps: discovery.model_work.forward_positions,
        projection_groups: discovery.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64,
        planned_work: discovery.planned_model_work, model_work: discovery.model_work }
}

#[test]
fn static_peak_covers_every_future_group_without_changing_semantic_reservations() {
    let planner = maps::planner();
    let mut prepared = prepare(&planner, SOURCE, maps::limits(), limits(), 32, &mut go()).unwrap();
    let original = prepared.preflight_metadata(); let context = prepared.base.planning.max_context_tokens;
    for width in [1, 2, 4] { for discovery in [false, true] {
        let schedule = Int4HierarchyCohortSchedule { max_groups: width, discovery_cohorts: discovery };
        let need = prepared.hierarchy_cohort_requirements(context, schedule).unwrap();
        let groups = Int4SourceMapCohortMemoryLimits { max_chunks: width, context_tokens: context,
            max_output_tokens_per_chunk: u64::from(prepared.base.budget.max_output_tokens),
            max_result_bytes_per_chunk: prepared.base.budget.max_output_bytes }.requirements(32).unwrap();
        assert!(need.native.kv_bytes >= groups.native.kv_bytes);
        assert!(need.native.rope_bytes >= groups.native.rope_bytes);
        assert!(need.native.scratch_payload_bound >= groups.native.scratch_payload_bound);
        assert!(need.workspace_bytes >= groups.workspace_bytes);
        assert!(need.max_result_bytes >= groups.max_result_bytes);
        assert_eq!(prepared.preflight_metadata(), original);
        assert!(prepared.hierarchy_cohort_requirements(context - 1, schedule).is_err());
    } }
    for width in [0, 5, 65] {
        assert!(prepared.hierarchy_cohort_requirements(context,
            Int4HierarchyCohortSchedule { max_groups: width, discovery_cohorts: false }).is_err());
    }
    prepared.base.budget.max_kv_bytes = Int4MemoryRequirement::for_context(context, 32).unwrap().kv_bytes - 1;
    assert!(prepared.hierarchy_cohort_requirements(context,
        Int4HierarchyCohortSchedule { max_groups: 2, discovery_cohorts: false }).is_err());
}

#[test]
fn serial_discovery_does_not_acquire_a_native_map_width_limit() {
    let planner = maps::planner(); let mut mapping = maps::limits(); mapping.reduction.map_batch_chunks = 65;
    let prepared = prepare(&planner, SOURCE, mapping, limits(), 32, &mut go()).unwrap();
    let context = prepared.base.planning.max_context_tokens;
    prepared.hierarchy_cohort_requirements(context,
        Int4HierarchyCohortSchedule { max_groups: 2, discovery_cohorts: false }).unwrap();
    assert!(prepared.hierarchy_cohort_requirements(context,
        Int4HierarchyCohortSchedule { max_groups: 2, discovery_cohorts: true }).is_err());
}

#[test]
fn complete_results_keep_question_lineage_statuses_transcripts_and_verifier_usage() {
    for mode in 0..4 { for width in [1, 2, 4] { for discovery in [false, true] {
        let (expected, baseline) = complete(mode);
        let original = serde_json::to_value(&baseline).unwrap();
        let mut remaining = remaining_verification(expected.base.discovery.verification, baseline.discovery.verification_used).unwrap();
        let collection = evidence::collect(SOURCE, baseline.discovery.mapped.root().value(),
            expected.base.limits, &mut remaining, &mut go()).unwrap();
        let initial = EvidenceSize::of(&collection);
        let mut driver = grouped(expected, width); driver.inner.empty = mode == 2; driver.inner.final_abstain = mode == 3;
        let reduced = execution::reduce(SOURCE, collection, expected, 16 << 20,
            &mut remaining, &mut driver, &mut go()).unwrap();
        let document = finish(baseline.discovery, initial, reduced, remaining, expected, 16 << 20, &mut go()).unwrap();
        assert_eq!(serde_json::to_value(&document).unwrap(), original);
        let maps = discovery.then(|| discovery_stats(&document.discovery));
        let output = entry::finish(document, maps, driver.stats,
            Int4HierarchyCohortSchedule { max_groups: width, discovery_cohorts: discovery }, 16 << 20, &mut go()).unwrap();
        assert_eq!(serde_json::to_value(&output.document).unwrap(), original);
        assert_eq!(output.discovery_cohorts.is_some(), discovery);
        assert_eq!(output.hierarchy_cohorts.passes, output.document.passes.len() as u64);
        if mode == 1 {
            assert!(driver.windows.is_empty()); assert!(driver.inner.preparations.is_empty());
            assert_eq!(driver.stats, Int4HierarchyCohortStats::default());
        } else {
            assert_eq!(driver.inner.events[..2], ["prepare", "prepare"]);
            assert_eq!(driver.inner.question, "Who is mentioned?");
            assert_eq!(driver.windows[0], width.min(2));
            assert!(driver.inner.inputs.iter().flatten().all(|passage| !passage.text.contains("generated-")));
            if mode != 2 { assert_eq!(driver.windows.last(), Some(&1)); }
        }
    } } }
}

#[test]
fn last_group_refusal_or_changed_witness_prevents_the_entire_first_window() {
    for changed_witness in [false, true] {
        let (expected, _, collection, mut remaining) = setup(false);
        let mut driver = grouped(expected, 2);
        if changed_witness { driver.corrupt_last_witness = true; } else { driver.inner.fail_prepare = Some(2); }
        assert!(execution::reduce(SOURCE, collection, expected, 16 << 20,
            &mut remaining, &mut driver, &mut go()).is_err());
        assert_eq!(driver.inner.preparations.len(), 2);
        assert!(driver.inner.inputs.is_empty()); assert!(driver.windows.is_empty());
    }
}

#[test]
fn missing_or_reordered_rows_cancellation_and_nonprogress_never_escape_a_partial_level() {
    for fault in 0..4 {
        let (expected, _, collection, mut remaining) = setup(false); let mut driver = grouped(expected, 2);
        match fault { 0 => driver.omit_row = true, 1 => driver.swap_rows = true,
            2 => driver.inner.fail_run = Some(2), _ => driver.inner.full = true }
        assert!(execution::reduce(SOURCE, collection, expected, 16 << 20,
            &mut remaining, &mut driver, &mut go()).is_err());
        assert_eq!(driver.windows, [2]); assert_eq!(driver.inner.preparations.len(), 2);
    }
}

#[test]
fn wrapper_rejects_wrong_physical_passes_and_discovery_work_in_hierarchy_counters() {
    let schedule = Int4HierarchyCohortSchedule { max_groups: 2, discovery_cohorts: false };
    for fault in 0..4 {
        let (expected, baseline) = complete(0);
        let (_, _, collection, mut remaining) = setup(false); let initial = EvidenceSize::of(&collection);
        let mut driver = grouped(expected, 2);
        let reduced = execution::reduce(SOURCE, collection, expected, 16 << 20,
            &mut remaining, &mut driver, &mut go()).unwrap();
        let document = finish(baseline.discovery, initial, reduced, remaining, expected, 16 << 20, &mut go()).unwrap();
        match fault { 0 => driver.stats.cohorts += 1, 1 => driver.stats.passes -= 1,
            2 => driver.stats.model_work = document.model_work, _ => driver.stats.projection_groups = 0 }
        assert!(entry::finish(document, None, driver.stats, schedule, 16 << 20, &mut go()).is_err());
    }
}
