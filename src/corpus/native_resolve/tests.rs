use super::*;
use crate::{corpus::resolve::{ResolutionDocument, MentionInput, ResolveOptions, ResolveLimits},
    native_engine::{decode::DecodeCancellationKind, lmhead::scoring::{CandidateScore, ScoringWork}},
    tokenizer::specials::ArchivedControlRegistries, validation::grounded_fields::VerifiedSourceSpan};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn fixture() -> (ResolutionPlanner, ArchivedControlRegistries) {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    // Synthetic census for model-free tests; not production control authority.
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    (ResolutionPlanner::pinned(controls.template_controls(), eos).unwrap(), controls)
}
fn identity(p: &ResolutionPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"native-resolution-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: p.template_digest(), task_spec: RESOLVE_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn docs() -> Vec<ResolutionDocument> {
    ["a", "b", "c"].iter().map(|&id| ResolutionDocument { id: id.to_owned(), text: "Smith <think> instructions".to_owned(),
        mentions: vec![MentionInput { entity_type: "PERSON".to_owned(), surface: "Smith".to_owned(),
            span: VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 } }] }).collect()
}
fn core(d: &[ResolutionDocument]) -> ResolutionPlan<'_> {
    ResolutionPlan::prepare(d, ResolveOptions::default(), ResolveLimits::default(), &mut Continue).unwrap()
}
#[test]
fn all_pairs_and_both_orders_have_exact_reserved_costs() {
    let (p, _) = fixture(); let d = docs(); let plan = core(&d); let id = identity(&p);
    let prepared = p.prepare(&plan, &id, NativeResolveLimits::default(), &mut Continue).unwrap();
    assert_eq!(prepared.pair_count(), 3);
    let mut total = BatchWork::default();
    for pair in &prepared.pairs { for prompt in &pair.prompts { total = plus(total, head_cost(prompt.len()).unwrap()).unwrap(); } }
    assert_eq!(prepared.planned_work(), total); assert_eq!(total.projected_logits, 6 * HEAD_ROWS);
}
#[test]
fn complete_admitted_identity_cannot_be_substituted() {
    let (p, _) = fixture(); let d = docs(); let plan = core(&d);
    let prepared = p.prepare(&plan, &identity(&p), NativeResolveLimits::default(), &mut Continue).unwrap();
    let expected = &prepared.pairs[0].identity; verify_identity(expected, expected).unwrap();
    for axis in 0..5 {
        let mut changed = expected.clone(); let hash = Sha256Digest::of_bytes(b"changed");
        match axis { 0 => changed.logical_model_digest = hash, 1 => changed.prompt_digest = hash,
            2 => changed.taskir_digest = hash, 3 => changed.decision_policy_digest = hash,
            _ => changed.backend_semantic_version = "another".to_owned() }
        assert!(verify_identity(expected, &changed).is_err());
    }
}
#[test]
fn context_window_targets_repeated_unicode_mention_without_first_match_guess() {
    let text = "é Smith Smith tail";
    let d = vec![ResolutionDocument { id: "d".to_owned(), text: text.to_owned(), mentions: vec![MentionInput {
        entity_type: "PERSON".to_owned(), surface: "Smith".to_owned(), span: VerifiedSourceSpan {
            byte_start: 9, byte_end: 14, scalar_start: 8, scalar_end: 13 } }] }];
    let plan = core(&d); let record: serde_json::Value = serde_json::from_str(&mention_record(&plan.mentions()[0], 128).unwrap()).unwrap();
    assert_eq!(record["before"], "é Smith "); assert_eq!(record["surface"], "Smith"); assert_eq!(record["after"], " tail");
    let plan = ResolutionPlan::prepare(&d, ResolveOptions { context_scalars: 2, ..ResolveOptions::default() }, ResolveLimits::default(), &mut Continue).unwrap();
    let record: serde_json::Value = serde_json::from_str(&mention_record(&plan.mentions()[0], 2).unwrap()).unwrap();
    assert_eq!(record["before"], "h "); assert_eq!(record["after"], " t");
}
#[test]
fn hostile_records_decode_exactly_without_privileged_controls() {
    let (p, registry) = fixture(); let d = docs(); let plan = core(&d); let ticket = plan.pairs().next().unwrap();
    let record = mention_record(ticket.left(), plan.options().context_scalars).unwrap();
    let encoded = p.encoder.encode(&record, 8192, 8192).unwrap();
    assert!(encoded.token_ids().iter().all(|&id| !registry.template_controls().contains(id)));
    assert_eq!(EmbeddedTokenizer::pinned().unwrap().tokenizer().decode_bytes(encoded.token_ids()).unwrap(), record.as_bytes());
}
#[test]
fn whole_run_prompt_pair_and_work_bounds_refuse_before_model_admission() {
    let (p, _) = fixture(); let d = docs(); let plan = core(&d); let id = identity(&p);
    let good = p.prepare(&plan, &id, NativeResolveLimits::default(), &mut Continue).unwrap();
    for axis in 0..5 {
        let mut limits = NativeResolveLimits::default();
        match axis { 0 => limits.max_pairs = 2, 1 => limits.max_total_prompt_tokens = 1,
            2 => limits.max_work.forward_positions = good.work.forward_positions - 1,
            3 => limits.max_work.projected_logits = good.work.projected_logits - 1,
            _ => limits.max_context_tokens = 2 }
        assert!(p.prepare(&plan, &id, limits, &mut Continue).is_err());
    }
}
#[test]
fn profile_thinking_tool_and_template_drift_are_rejected() {
    let (p, _) = fixture(); let d = docs(); let plan = core(&d);
    for axis in 0..5 {
        let mut id = identity(&p);
        match axis { 0 => id.thinking_mode = ThinkingMode::Enabled, 1 => id.tool_mode = ToolMode::Json,
            2 => id.numerics_profile = NumericsProfile::DiagnosticF32,
            3 => id.template_digest = Sha256Digest::of_bytes(b"different"), _ => id.task_spec = "judge-v1".to_owned() }
        assert!(p.prepare(&plan, &id, NativeResolveLimits::default(), &mut Continue).is_err());
    }
}
fn receipt() -> CandidateScores {
    CandidateScores { score_space: ScoreSpace::FullVocabSequenceLogprob, normalization_scope: "fixture".to_owned(),
        length_rule: "none".to_owned(), eos_rule: "scored".to_owned(), eos_token_id: 0,
        full_vocab_denominators_computed: true,
        candidates: ["different", "same", "uncertain"].iter().map(|&id| CandidateScore { id: id.to_owned(),
            scored_tokens: 2, sequence_score: -3.0, candidate_weight: 1.0 / 3.0 }).collect(),
        work: ScoringWork { prefix_evaluations: 4, scored_edges: 6, projected_logits: HEAD_ROWS } }
}
#[test]
fn candidate_receipt_checks_all_labels_denominators_eos_and_work() {
    assert!(validate_scores(&receipt(), 0).is_ok());
    for mode in 0..8 {
        let mut r = receipt();
        match mode { 0 => r.full_vocab_denominators_computed = false, 1 => r.eos_token_id = 1,
            2 => { r.candidates.pop(); }, 3 => r.candidates[0].id = "invented".to_owned(),
            4 => r.candidates[0].scored_tokens = 1, 5 => r.candidates[0].sequence_score = f64::NAN,
            6 => r.work.projected_logits -= 1, _ => r.candidates[0].candidate_weight = f64::NAN }
        assert!(validate_scores(&r, 0).is_err());
    }
}
#[test]
fn actual_native_work_cannot_hide_an_extra_prefill_or_missing_branch() {
    let b = head_cost(10).unwrap(); let w = PrefixWork { prefix_evaluations: 4, forward_positions: 13,
        prompt_positions: 10, continuation_positions: 3, projected_logits: HEAD_ROWS, rewound_positions: 2 };
    assert!(check_work(w, 10, b).is_ok());
    for axis in 0..5 {
        let mut changed = w; match axis { 0 => changed.forward_positions += 1, 1 => changed.prompt_positions += 1,
            2 => changed.continuation_positions -= 1, 3 => changed.prefix_evaluations -= 1, _ => changed.projected_logits -= 1 }
        assert!(check_work(changed, 10, b).is_err());
    }
    assert!(plus(BatchWork { forward_positions: u64::MAX, projected_logits: 1 }, b).is_err());
}
#[test]
fn planning_cancellation_preserves_cause_and_never_returns_partial_pairs() {
    struct Cancel;
    impl DecodeStepControl for Cancel { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let (p, _) = fixture(); let d = docs(); let plan = core(&d);
    assert!(matches!(p.prepare(&plan, &identity(&p), NativeResolveLimits::default(), &mut Cancel),
        Err(NativeResolveError::Resolution(ResolveError::Cancelled(DecodeCancellationKind::Deadline)))));
}
