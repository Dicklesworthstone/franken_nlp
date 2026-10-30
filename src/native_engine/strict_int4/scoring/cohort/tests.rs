//! Synthetic-logit differential fixtures; these do not qualify a native model.
use super::*;
use crate::{tasks::ir::TokenSequence,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{lmhead::scoring::SequenceScoreRule, strict_int4::STRICT_INT4_EXECUTION}};

fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"int4-scoring-cohort-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
        artifact_format: "synthetic-only".into(), quant_recipe: "fixture-int4".into(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "classify-v1".into(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.into(),
        host_class: None, compiler_identity: None }
}
fn plan(width: usize, prompt: Vec<u32>, mode: ScoringMode) -> Int4CandidatePlan {
    let candidate = |id, tokens: &[u32]| Candidate::new(id, TokenSequence::new(tokens.to_vec()));
    Int4CandidatePlan::compile(prompt, &[candidate("short", &[1]), candidate("long", &[1, 2, 3]),
        candidate("sibling", &[1, 4]), candidate("other", &[5, 6])], 0, mode, width,
        ScoringLimits::default(), 1_000_000).unwrap()
}
fn requests<'a>(plans: &'a [Int4CandidatePlan], identity: &'a ExecutionIdentity) -> Vec<CompiledRequest<'a>> {
    plans.iter().map(|plan| CompiledRequest { identity, prompt: &plan.prompt, scorer: &plan.scorer,
        mode: plan.mode, schedule: plan.schedule, max_output_bytes: plan.max_output_bytes,
        budget: Int4ScoringBudget { native: Int4RunBudget::exact(plan.planned_work()),
            max_kv_bytes: plan.required_context() as u64 * KV_BYTES_PER_TOKEN as u64 } }).collect()
}
fn budget(requests: &[CompiledRequest<'_>]) -> Int4ScoringCohortBudget {
    Int4ScoringCohortBudget { native: Int4RunBudget::exact(requests.iter().fold(Int4Work::default(),
        |sum, request| sum.checked_add(request.schedule.model).unwrap())),
        max_kv_bytes: requests.iter().map(|request| request.budget.max_kv_bytes).sum(), max_output_bytes: 8_000_000 }
}
fn values(tokens: &[u32], rows: LinearRows<'_>) -> Vec<f32> {
    let h = tokens.iter().fold(19_u64, |h, &t| h.wrapping_mul(31).wrapping_add(u64::from(t)));
    let value = |r: u32| (h.wrapping_add(u64::from(r) * 13) % 37) as f32 * 0.25;
    match rows { LinearRows::All => (0..V as u32).map(value).collect(),
        LinearRows::Selected(ids) => ids.iter().map(|&id| value(id)).collect() }
}
struct Replay<'a>(&'a [u32]);
impl CandidateLogits for Replay<'_> {
    type Error = Int4ScoringError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let mut tokens = self.0.to_vec(); tokens.extend_from_slice(prefix); Ok(values(&tokens, checked_rows(rows)?))
    }
}
#[derive(Default)]
struct Control { polls: usize, cancel_at: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.cancel_at == Some(self.polls)).then_some(DecodeCancellationKind::Deadline)
    }
}
struct Projection { slots: Vec<usize>, selected: Option<Vec<u32>>, positions: Vec<usize> }
struct Model {
    width: usize, tokens: Vec<Vec<u32>>, work: Vec<Int4Work>, rewound: Vec<u64>, control: Control,
    steps: Vec<Vec<(usize, usize, u32)>>, projections: Vec<Projection>, aborted: bool,
    fail_group: Option<usize>, corrupt: bool, short_projection: bool, skip_last_forward: bool,
    forged_work: bool, forged_rewind: bool,
}
impl Model {
    fn new(width: usize, count: usize) -> Self {
        Self { width, tokens: vec![Vec::new(); count], work: vec![Int4Work::default(); count], rewound: vec![0; count],
            control: Control::default(), steps: Vec::new(), projections: Vec::new(), aborted: false,
            fail_group: None, corrupt: false, short_projection: false, skip_last_forward: false,
            forged_work: false, forged_rewind: false }
    }
}
impl GroupDriver for Model {
    type Control = Control;
    fn control(&mut self) -> &mut Control { &mut self.control }
    fn position(&self, slot: usize) -> Result<usize, Int4ScoringError> { Ok(self.tokens[slot].len()) }
    fn rewind(&mut self, slot: usize, retain: usize) -> Result<(), Int4ScoringError> {
        let current = self.tokens[slot].len();
        if retain > current { return Err(Int4ScoringError::Traversal); }
        self.rewound[slot] += (current - retain) as u64; self.tokens[slot].truncate(retain); Ok(())
    }
    fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), Int4ScoringError> {
        if self.fail_group == Some(self.steps.len()) {
            return Err(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline).into());
        }
        assert!(!steps.is_empty()); assert!(steps.windows(2).all(|w| w[0].sequence < w[1].sequence));
        self.steps.push(steps.iter().map(|s| (s.sequence, self.tokens[s.sequence].len(), s.token)).collect());
        for (i, step) in steps.iter().enumerate() {
            if self.skip_last_forward && i + 1 == steps.len() { continue; }
            let slot = step.sequence;
            self.work[slot] = self.work[slot].checked_add(Int4Work::for_sequence(self.width, self.tokens[slot].len(), 1, 0)?)?;
            self.tokens[slot].push(step.token);
        }
        Ok(())
    }
    fn logits_group(&mut self, slots: &[usize], rows: LinearRows<'_>) -> Result<Vec<f32>, Int4ScoringError> {
        assert!(!slots.is_empty()); assert!(slots.windows(2).all(|w| w[0] < w[1]));
        let width = rows.checked_count(V).map_err(StrictInt4Error::from)?;
        self.projections.push(Projection { slots: slots.to_vec(), positions: slots.iter().map(|&slot| self.tokens[slot].len()).collect(),
            selected: match rows { LinearRows::All => None, LinearRows::Selected(ids) => Some(ids.to_vec()) } });
        let mut output = Vec::new();
        for &slot in slots {
            self.work[slot] = self.work[slot].checked_add(Int4Work::for_sequence(self.width, self.tokens[slot].len(), 0, width)?)?;
            output.extend(values(&self.tokens[slot], rows));
        }
        if self.corrupt { *output.last_mut().unwrap() = f32::NAN; }
        if self.short_projection { output.pop(); }
        Ok(output)
    }
    fn work(&self, slot: usize) -> Result<Int4Work, Int4ScoringError> {
        let mut work = self.work[slot];
        if self.forged_work && slot + 1 == self.work.len() { work.attention_pairs += 1; } Ok(work)
    }
    fn rewound_positions(&self, slot: usize) -> Result<u64, Int4ScoringError> {
        Ok(self.rewound[slot] + u64::from(self.forged_rewind && slot + 1 == self.work.len()))
    }
    fn abort(&mut self) { self.aborted = true; }
}
fn run(requests: &[CompiledRequest<'_>], model: &mut Model) -> Result<Int4CandidateCohortRun, Int4ScoringError> {
    execution::drive_with(requests, budget(requests), model, |run, _| Ok(run))
}
fn modes() -> [ScoringMode; 5] {
    [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit }]
}
#[test]
fn all_score_spaces_and_group_partitions_match_independent_prefix_replay() {
    let id = identity();
    for width in [3, 32, 128] {
        let plans: Vec<_> = modes().into_iter().enumerate()
            .map(|(i, mode)| plan(width, vec![7 + i as u32; i + 1], mode)).collect();
        let requests = requests(&plans, &id); let mut model = Model::new(width, plans.len());
        let output = run(&requests, &mut model).unwrap();
        for (slot, (head, plan)) in output.heads.iter().zip(&plans).enumerate() {
            assert_eq!(head.scores, plan.scorer.score(&mut Replay(&plan.prompt), plan.mode).unwrap());
            assert_eq!(head.model_work, plan.planned_work()); assert_eq!(head.rewound_positions, 4);
            assert_eq!(model.work[slot], head.model_work); assert_eq!(head.group_width, width);
        }
        assert_eq!(output.group_width, width);
        assert!(output.group_steps < output.model_work.forward_positions); assert!(!model.aborted);
    }
}
#[test]
fn identical_head_geometry_shares_decoder_and_head_calls_without_mixing_normalizers() {
    let id = identity();
    let plans: Vec<_> = (0..4).map(|i| plan(32, vec![7 + i, 8, 9], ScoringMode::FullVocabulary)).collect();
    let mut model = Model::new(32, 4); let output = run(&requests(&plans, &id), &mut model).unwrap();
    assert_eq!(output.group_steps, 9); assert_eq!(output.projection_groups, 7);
    assert!(model.steps.iter().all(|step| step.len() == 4));
    assert!(model.projections.iter().all(|projection| projection.slots == [0, 1, 2, 3] && projection.selected.is_none()));
    for (head, plan) in output.heads.iter().zip(&plans) {
        assert_eq!(head.scores, plan.scorer.score(&mut Replay(&plan.prompt), plan.mode).unwrap());
    }
}
#[test]
fn ragged_prompts_finish_without_replaying_padding_or_forwarding_eos() {
    let id = identity(); let plans = [plan(3, vec![7], ScoringMode::TrieConditional),
        plan(3, vec![8; 20], ScoringMode::TrieConditional)];
    let mut model = Model::new(3, 2); let output = run(&requests(&plans, &id), &mut model).unwrap();
    assert_eq!(output.group_steps, 26);
    assert!(model.steps[..7].iter().all(|step| step.len() == 2));
    assert!(model.steps[7..].iter().all(|step| step.len() == 1 && step[0].0 == 1));
    assert!(model.projections.iter().any(|projection| projection.slots == [0]));
    assert!(model.projections.iter().filter(|projection| projection.slots.contains(&1)).all(|projection|
        projection.slots.iter().zip(&projection.positions).all(|(&slot, &position)| slot != 1 || position >= 20)));
    assert!(model.steps.iter().flatten().all(|&(_, _, token)| token != 0));
}
#[test]
fn selected_projections_share_only_exact_row_sets_and_keep_distinct_score_spaces() {
    let id = identity(); let plans = [plan(128, vec![7; 3], ScoringMode::TrieConditional),
        plan(128, vec![8; 3], ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits })];
    let mut model = Model::new(128, 2); let output = run(&requests(&plans, &id), &mut model).unwrap();
    assert_eq!(output.projection_groups, 7);
    assert!(model.projections.iter().all(|projection| projection.slots == [0, 1] && projection.selected.is_some()));
    assert_ne!(output.heads[0].scores.score_space, output.heads[1].scores.score_space);
    assert!(!execution::same_rows(ProjectionRows::Selected(&[0, 1]), ProjectionRows::Selected(&[0, 2])));
    assert!(!execution::same_rows(ProjectionRows::Selected(&[0]), ProjectionRows::FullVocabulary { vocabulary_size: V }));
}
#[test]
fn every_cooperative_cancellation_returns_its_cause_and_no_partial_cohort() {
    let id = identity(); let plans = [plan(32, vec![7; 2], ScoringMode::TrieConditional),
        plan(32, vec![8; 4], ScoringMode::TrieConditional)];
    let requests = requests(&plans, &id); let mut baseline = Model::new(32, 2); run(&requests, &mut baseline).unwrap();
    for stop in 1..=baseline.control.polls {
        let mut model = Model::new(32, 2); model.control.cancel_at = Some(stop);
        assert_eq!(run(&requests, &mut model).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(model.aborted);
    }
}
#[test]
fn each_native_group_failure_and_bad_later_row_discards_all_heads() {
    let id = identity(); let plans = [plan(32, vec![7; 2], ScoringMode::TrieConditional),
        plan(32, vec![8; 4], ScoringMode::TrieConditional)];
    let requests = requests(&plans, &id); let baseline = run(&requests, &mut Model::new(32, 2)).unwrap();
    for group in 0..baseline.group_steps as usize {
        let mut model = Model::new(32, 2); model.fail_group = Some(group);
        assert_eq!(run(&requests, &mut model).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(model.aborted);
    }
    for fault in 0..5 {
        let mut model = Model::new(32, 2);
        match fault { 0 => model.corrupt = true, 1 => model.short_projection = true, 2 => model.skip_last_forward = true,
            3 => model.forged_work = true, _ => model.forged_rewind = true }
        assert!(run(&requests, &mut model).is_err()); assert!(model.aborted);
    }
}
#[test]
fn exact_branch_attention_and_group_work_do_not_refund_rewound_positions() {
    let id = identity(); let plans = [plan(3, vec![7, 8, 9], ScoringMode::TrieConditional)];
    let mut model = Model::new(3, 1); let output = run(&requests(&plans, &id), &mut model).unwrap();
    let steps: Vec<_> = model.steps.iter().flatten().map(|&(_, position, token)| (position, token)).collect();
    assert_eq!(steps, [(0, 7), (1, 8), (2, 9), (3, 1), (4, 2), (5, 3), (4, 4), (3, 5), (4, 6)]);
    assert_eq!(output.model_work.attention_pairs, (1 + 2 + 3 + 4 + 5 + 6 + 5 + 4 + 5) * (KV_SLOT_COUNT * QUERY_HEAD_COUNT) as u64);
    assert_eq!(output.heads[0].rewound_positions, 4); assert_eq!(output.model_work.projected_logits, 10);
    let mut wrong = Model::new(32, 1);
    assert_eq!(run(&requests(&plans, &id), &mut wrong).unwrap_err(), Int4ScoringError::Accounting);
    assert!(wrong.aborted);
}
#[test]
fn aggregate_work_full_resident_kv_and_group_identity_refuse_before_execution() {
    let id = identity(); let plans = [plan(32, vec![7; 3], ScoringMode::FullVocabulary),
        plan(32, vec![8; 5], ScoringMode::FullVocabulary)];
    let mut requests = requests(&plans, &id); let cap = budget(&requests);
    let contexts: Vec<_> = requests.iter().map(|request| request.schedule.context).collect();
    check_geometry(&requests, &contexts, cap).unwrap();
    for axis in 0..5 {
        let mut lower = cap;
        match axis { 0 => lower.max_kv_bytes -= 1, 1 => lower.native.max_forward_positions -= 1,
            2 => lower.native.max_attention_pairs -= 1, 3 => lower.native.max_projection_work.dot_products -= 1,
            _ => lower.native.max_projection_work.multiply_accumulates -= 1 }
        assert!(check_geometry(&requests, &contexts, lower).is_err());
    }
    let mut bigger = contexts.clone(); bigger[1] += 1;
    assert!(check_geometry(&requests, &bigger, cap).is_err());
    let mut smaller = contexts.clone(); smaller[1] -= 1;
    assert!(check_geometry(&requests, &smaller, cap).is_err());
    requests[1].budget.native.max_forward_positions -= 1;
    assert!(check_geometry(&requests, &contexts, cap).is_err());
    requests[1].budget.native.max_forward_positions += 1; requests[1].schedule.group_width = 128;
    assert_eq!(check_geometry(&requests, &contexts, cap), Err(Int4ScoringError::Identity));
    let mut model = Model::new(32, 2); assert!(run(&requests, &mut model).is_err());
    assert!(model.steps.is_empty()); assert!(model.projections.is_empty());
    for count in [0, MAX_INT4_SEQUENCES + 1, usize::MAX] { assert!(check_count(count).is_err()); }
    check_count(MAX_INT4_SEQUENCES).unwrap();
}
#[test]
fn head_and_outer_output_limits_are_exact_and_independent() {
    let id = identity(); let plans = [plan(32, vec![7; 3], ScoringMode::TrieConditional),
        plan(32, vec![8; 4], ScoringMode::TrieConditional)];
    let mut requests = requests(&plans, &id); let baseline = run(&requests, &mut Model::new(32, 2)).unwrap();
    for (request, head) in requests.iter_mut().zip(&baseline.heads) {
        request.max_output_bytes = canonjson::canonical_bytes(head).unwrap().len() as u64;
    }
    let mut cap = budget(&requests); cap.max_output_bytes = canonjson::canonical_bytes(&baseline).unwrap().len() as u64;
    execution::drive_with(&requests, cap, &mut Model::new(32, 2), |run, _| Ok::<_, Int4ScoringError>(run)).unwrap();
    cap.max_output_bytes -= 1; let mut model = Model::new(32, 2);
    assert_eq!(execution::drive_with(&requests, cap, &mut model, |run, _| Ok::<_, Int4ScoringError>(run)).unwrap_err(),
        Int4ScoringError::OutputBudget);
    assert!(model.aborted); requests[1].max_output_bytes -= 1;
    assert_eq!(run(&requests, &mut Model::new(32, 2)).unwrap_err(), Int4ScoringError::OutputBudget);
}
#[test]
fn semantic_failure_and_late_cancellation_poison_the_whole_owner() {
    let id = identity(); let plans = [plan(32, vec![7], ScoringMode::TrieConditional)]; let requests = requests(&plans, &id);
    let mut model = Model::new(32, 1);
    assert_eq!(execution::drive_with(&requests, budget(&requests), &mut model,
        |_, _| Err::<(), _>(Int4ScoringError::Accounting)).unwrap_err(), Int4ScoringError::Accounting);
    assert!(model.aborted);
    let mut model = Model::new(32, 1);
    let error = execution::drive_with(&requests, budget(&requests), &mut model,
        |run, control| { control.cancel_at = Some(control.polls + 1); Ok::<_, Int4ScoringError>(run) }).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert!(model.aborted);
}
#[test]
fn malformed_compiler_schedule_refuses_before_projection_or_forward() {
    let id = identity(); let plans = [plan(32, vec![7], ScoringMode::TrieConditional)]; let mut requests = requests(&plans, &id);
    requests[0].schedule.scoring.prefix_evaluations += 1; let mut model = Model::new(32, 1);
    assert_eq!(run(&requests, &mut model).unwrap_err(), Int4ScoringError::Accounting);
    assert!(model.steps.is_empty()); assert!(model.projections.is_empty()); assert!(model.aborted);
}
#[test]
fn task_differences_do_not_erase_model_host_or_group_partition_fields() {
    let a = identity(); let mut b = a.clone(); b.task_spec = "sentiment-v1".into();
    b.template_digest = Sha256Digest::of_bytes(b"other-task"); same_model(&a, &b).unwrap();
    for axis in 0..6 {
        let mut changed = b.clone();
        match axis { 0 => changed.logical_model_digest = b.template_digest, 1 => changed.tokenizer_digest = b.template_digest,
            2 => changed.packing_set_digest = b.template_digest, 3 => changed.quant_recipe = "different-group".into(),
            4 => changed.source_revision = "other".into(), _ => changed.backend_semantic_version = "other".into() }
        assert_eq!(same_model(&a, &changed), Err(Int4ScoringError::Identity));
    }
}
#[test]
fn task_workspace_quote_prices_all_heads_checks_arithmetic_and_bounds() {
    let mut work = Int4Work::default(); work.forward_positions = 1024;
    for heads in [1, 4, 64] {
        assert_eq!(required_task_workspace_bytes(work, heads, 8192).unwrap(),
            64 * 1024 + 8 * 8192 + 65536 * heads as u64);
    }
    for heads in [0, 65, usize::MAX] { assert!(required_task_workspace_bytes(work, heads, 8192).is_err()); }
    for cap in [0, 64 * 1024 * 1024 + 1, u64::MAX] { assert!(required_task_workspace_bytes(work, 1, cap).is_err()); }
    work.forward_positions = u64::MAX;
    assert!(required_task_workspace_bytes(work, 1, 8192).is_err());
    work.forward_positions = u64::MAX / 64;
    assert!(required_task_workspace_bytes(work, 1, 8192).is_err());
}
