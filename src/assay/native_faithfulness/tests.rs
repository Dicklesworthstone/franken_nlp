//! Synthetic logits exercise real judge finalizers, never claim model quality.
use super::*;
use crate::{execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        lmhead::scoring::ProjectionRows, portable_int8::ProjectionWork,
        strict_int4::{Int4Work, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE}},
    receipt::CommitmentKey, tasks::{extract::SourceDocumentEncoder, ir::{PlanContext, PromptSegment, TaskBudget},
        judge::{FaithfulnessPolicy, FaithfulnessRelation, JudgePlanner, JudgeLimits, JudgeLogits, JudgeResult,
            grouped_int4::{Int4JudgeRun, INT4_JUDGE_EXECUTION}}}};
use replay::ReplayBuilder;
struct Go;
impl DecodeStepControl for Go { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn policy() -> FaithfulnessPolicy { FaithfulnessPolicy { minimum_candidate_weight_ppm: 500_000,
    minimum_margin_milli: 100, evidence_window_bytes: 128, max_evidence_windows: 31, max_evidence_spans: 31 } }
fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 2, max_output_bytes: 1_000_000,
    max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
fn wire(sources: usize, claims: usize) -> (SuiteManifest, Vec<serde_json::Value>) {
    let mut rows = vec![serde_json::json!({"schema_version":1,"id":"private-dev","split":"development",
        "source":"PRIVATE development source","claim":"private dev claim","relation":"unsupported"}),
        serde_json::json!({"schema_version":1,"id":"private-cal","split":"calibration",
        "source":"PRIVATE calibration source","claim":"private cal claim","relation":"contradicted"})];
    let mut ids = Vec::new();
    for source in 0..sources { for claim in 0..claims {
        let id = format!("private-test-{source:03}-{claim:03}"); ids.push(id.clone());
        rows.push(serde_json::json!({"schema_version":1,"id":id,"split":"locked_test",
            "source":format!("PRIVATE source {source} é上海😀"),"claim":format!("PRIVATE claim {claim}"),
            "relation":RELATIONS[(source + claim) % 3]}));
    } }
    (SuiteManifest { schema_version: 1, task: NATIVE_TASK.to_owned(), dataset_id: "owned".to_owned(),
        dataset_version: "1".to_owned(), named_population: "synthetic fixture only".to_owned(), policy: policy(),
        expected_input_sha256: Sha256Digest::of_bytes(b"unbound"), splits: SplitIds {
            development: vec!["private-dev".to_owned()], calibration: vec!["private-cal".to_owned()], locked_test: ids } }, rows)
}
fn encode(mut suite: SuiteManifest, rows: &[serde_json::Value]) -> (Vec<u8>, Vec<u8>) {
    let mut bytes = Vec::new();
    for row in rows { bytes.extend(serde_json::to_vec(row).unwrap()); bytes.push(b'\n'); }
    suite.expected_input_sha256 = Sha256Digest::of_bytes(&bytes);
    (serde_json::to_vec(&suite).unwrap(), bytes)
}
fn dataset(sources: usize, claims: usize) -> Dataset {
    let (suite, rows) = wire(sources, claims); let (suite, bytes) = encode(suite, &rows);
    Dataset::parse(&suite, &bytes, Limits::default()).unwrap()
}
fn options() -> EvalOptions { EvalOptions { minimum_samples: 5,
    bootstrap: crate::assay::BootstrapOptions { resamples: 40, confidence_level: 0.95, seed: 42 } } }
fn gates() -> Gates { Gates { schema_version: 1, max_accuracy_decrease: 0.0,
    max_macro_f1_decrease: 0.0, max_brier_increase: 0.0, max_coverage_decrease: 0.0 } }
fn identity(model: &str, planner: &JudgePlanner) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"synthetic-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: Sha256Digest::of_bytes(model.as_bytes()), artifact_format: "fixture".to_owned(),
        quant_recipe: model.to_owned(), packing_set_digest: Sha256Digest::of_bytes(model.as_bytes()),
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(), task_spec: "judge-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: digest,
        decision_policy_digest: digest, backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
struct Logits { labels: Vec<Option<usize>>, tokens: [u32; 3] }
impl JudgeLogits for Logits {
    type Error = &'static str;
    fn project(&mut self, head: usize, _: &[PromptSegment], prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("full denominator required"); };
        let mut logits = vec![0.0; vocabulary_size];
        if prefix.is_empty() { if let Some(label) = self.labels[head] { logits[self.tokens[label] as usize] = 12.0; } }
        Ok(logits)
    }
}
struct Fixture { planner: JudgePlanner, tokens: [u32; 3] }
impl Fixture {
    fn new() -> Self {
        let registry = crate::tokenizer::pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
        let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
        let planner = JudgePlanner::pinned(controls, eos).unwrap();
        let encoder = SourceDocumentEncoder::pinned(controls).unwrap();
        let tokens = ["C", "E", "U"].map(|text| encoder.encode(text, 8, 8).unwrap().token_ids()[0]);
        Self { planner, tokens }
    }
    fn run(&self, dataset: &Dataset, index: usize, model: &str, choices: &[Option<usize>]) -> (ExecutionIdentity, Int4JudgeRun) {
        let request = dataset.request(index, budget()).unwrap(); let base = identity(model, &self.planner);
        let prepared = self.planner.plan(&request, &PlanContext::new(&base, budget()).unwrap(), JudgeLimits::default()).unwrap();
        let result = prepared.execute(prepared.execution_identity(), &mut Logits { labels: choices.to_vec(), tokens: self.tokens }).unwrap();
        let JudgeResult::Faithfulness(ref result_value) = result else { panic!("faithfulness required"); };
        let heads = result_value.heads.len();
        let work = Int4Work { forward_positions: 100 * heads as u64, projected_logits: result_value.work.projected_logits,
            attention_pairs: 500 * heads as u64, projections: ProjectionWork { dot_products: 1000, multiply_accumulates: 10_000 } };
        let mut identity = prepared.execution_identity().clone();
        identity.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
        identity.backend_semantic_version = STRICT_INT4_EXECUTION.to_owned();
        (identity, Int4JudgeRun { schema_version: 1, execution: INT4_JUDGE_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: 128, result,
            head_count: heads, model_work: work, rewound_positions: 2 * heads as u64 })
    }
    fn replay(&self, dataset: &Dataset, model: &str, select: impl Fn(usize) -> Option<usize>) -> NativeReplay {
        let mut collector = ReplayBuilder::new(dataset, Limits::default()).unwrap();
        for index in 0..dataset.replay_len() {
            let (identity, run) = self.run(dataset, index, model, &[select(index)]);
            collector.accept_fixture(index, &identity, budget(), run.model_work, 128, run.head_count, run, &mut Go).unwrap();
        }
        collector.finish().unwrap()
    }
}

#[test]
fn source_clusters_allow_distinct_claims_but_never_cross_splits_or_duplicate_units() {
    let data = dataset(5, 3); assert_eq!(data.validated_rows(), 17); assert_eq!(data.replay_len(), 15);
    assert_eq!(data.locked_test_sources(), 5);
    for fault in 0..6 {
        let (mut suite, mut rows) = wire(5, 3);
        match fault {
            0 => rows[2]["source"] = rows[0]["source"].clone(),
            1 => rows[3]["claim"] = rows[2]["claim"].clone(),
            2 => rows[2]["relation"] = serde_json::json!("abstain"),
            3 => rows[0]["probability"] = serde_json::json!(0.9),
            4 => suite.splits.development.clear(),
            _ => rows[0]["split"] = serde_json::json!("locked_test"),
        }
        let (suite, bytes) = encode(suite, &rows);
        assert!(Dataset::parse(&suite, &bytes, Limits::default()).is_err(), "fault {fault}");
    }
    let (suite, rows) = wire(5, 1); let (suite, mut bytes) = encode(suite, &rows); bytes.push(b'\n');
    assert!(matches!(Dataset::parse(&suite, &bytes, Limits::default()), Err(AssayError::InputDigest)));
}

#[test]
fn gold_id_and_split_are_excluded_from_requests_while_source_claim_policy_and_budget_bind() {
    let (suite, mut rows) = wire(5, 1); let (first, bytes) = encode(suite.clone(), &rows);
    let first = Dataset::parse(&first, &bytes, Limits::default()).unwrap();
    rows[2]["relation"] = serde_json::json!("unsupported");
    let (second, bytes) = encode(suite.clone(), &rows); let second = Dataset::parse(&second, &bytes, Limits::default()).unwrap();
    assert_eq!(first.request_digest(0, budget()).unwrap(), second.request_digest(0, budget()).unwrap());
    for fault in 0..4 {
        let mut altered_suite = suite.clone(); let mut altered_rows = rows.clone(); let mut changed_budget = budget();
        match fault { 0 => altered_rows[2]["source"] = serde_json::json!("new exact source"),
            1 => altered_rows[2]["claim"] = serde_json::json!("new exact claim"),
            2 => altered_suite.policy.minimum_margin_milli += 1, _ => changed_budget.max_output_bytes -= 1 }
        let (manifest, bytes) = encode(altered_suite, &altered_rows);
        let changed = Dataset::parse(&manifest, &bytes, Limits::default()).unwrap();
        assert_ne!(first.request_digest(0, budget()).unwrap(), changed.request_digest(0, changed_budget).unwrap());
    }
    let request = serde_json::to_string(&first.request(0, budget()).unwrap()).unwrap();
    assert!(!request.contains("private-test")); assert!(!request.contains("locked_test")); assert!(!request.contains("relation"));
}

#[test]
fn full_result_validation_refuses_work_eos_quotes_head_loss_and_hidden_abstention() {
    let data = dataset(1, 1); let fixture = Fixture::new();
    let (identity, original) = fixture.run(&data, 0, "candidate", &[Some(1)]);
    for fault in 0..9 {
        let mut run = original.clone();
        match fault {
            0 => run.model_work.projections.dot_products += 1,
            1 => run.head_count += 1,
            2 => run.group_width += 1,
            3 => run.numerics_profile = "foreign".to_owned(),
            _ => { let JudgeResult::Faithfulness(result) = &mut run.result else { unreachable!() };
                match fault { 4 => result.evidence[0].quote = "PRIVATE foreign source".to_owned(),
                    5 => result.windows[0].span.scalar_end += 1,
                    6 => result.heads[0].eos_token_id += 1,
                    7 => result.relation = Some(FaithfulnessRelation::Unsupported),
                    _ => result.heads[0].candidates[0].candidate_weight = 1.0 } }
        }
        let mut collector = ReplayBuilder::new(&data, Limits::default()).unwrap();
        assert!(collector.accept_fixture(0, &identity, budget(), original.model_work, 128,
            original.head_count, run, &mut Go).is_err(), "fault {fault}");
        assert!(collector.finish().is_err());
    }
    let mut collector = ReplayBuilder::new(&data, Limits::default()).unwrap();
    assert!(collector.accept_fixture(1, &identity, budget(), original.model_work, 128, original.head_count,
        original.clone(), &mut Go).is_err());
    assert!(collector.accept_fixture(0, &identity, budget(), original.model_work, 128, original.head_count, original, &mut Go).is_err());
}

#[test]
fn source_count_controls_insufficient_data_and_paired_draw_allowance() {
    let data = dataset(2, 8); let replay = Fixture::new().replay(&data, "candidate", |_| Some(2));
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    let report = evaluate_with_checkpoint::<AssayError>(&replay, &key, "fixture", options(), Limits::default(), &mut || Ok(())).unwrap();
    assert!(matches!(report.body.measurement, Measurement::InsufficientData { claims: 16, source_clusters: 2, minimum_sources: 5 }));
    let mut limits = Limits::default(); limits.max_bootstrap_draws = 2 * options().bootstrap.resamples as u64;
    assert!(evaluate_with_checkpoint::<AssayError>(&replay, &key, "fixture", options(), limits, &mut || Ok(())).is_ok());
    assert!(matches!(compare_with_checkpoint::<AssayError>(&replay, &replay, &key, "fixture", "declared", gates(),
        options(), limits, &mut || Ok(())), Err(AssayError::InputLimit)));
}

#[test]
fn complete_paired_reports_bind_every_claim_and_reject_changed_requests() {
    let data = dataset(6, 2); let fixture = Fixture::new();
    let current = fixture.replay(&data, "candidate", |index| Some((index / 2 + index % 2) % 3));
    let prior = fixture.replay(&data, "baseline", |index| Some((index / 2 + index % 2) % 3));
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    let paired = compare_with_checkpoint::<AssayError>(&current, &prior, &key, "fixture", "declared", gates(),
        options(), Limits::default(), &mut || Ok(())).unwrap();
    let Decision::Passed { paired: changes, .. } = &paired.body.decision else { panic!("identical predictions must pass"); };
    assert_eq!(changes.accuracy_change, 0.0); assert_eq!(changes.brier_change, 0.0); assert_eq!(changes.coverage_change, 0.0);
    for interval in [changes.intervals.accuracy, changes.intervals.brier, changes.intervals.coverage] {
        let interval = interval.unwrap(); assert_eq!(interval.lower, 0.0); assert_eq!(interval.upper, 0.0);
    }
    assert_ne!(paired.body.candidate.body.bindings.execution_collection, paired.body.baseline.body.bindings.execution_collection);
    assert_eq!(paired.body.candidate.body.bindings.observation_collection, paired.body.baseline.body.bindings.observation_collection);
    assert_eq!(paired.body.candidate.body.executed_rows, 12); assert_eq!(paired.body.candidate.body.validated_rows, 14);
    let text = String::from_utf8(crate::assay::render(&paired, Limits::default()).unwrap()).unwrap();
    for private in ["PRIVATE", "private-test", "private-dev", "private-cal"] { assert!(!text.contains(private)); }
    for row in data.replay_rows() {
        assert!(!text.contains(&Sha256Digest::of_bytes(row.source.as_bytes()).to_hex()));
        assert!(!text.contains(&Sha256Digest::of_bytes(row.claim.as_bytes()).to_hex()));
    }
    assert!(!text.contains(&data.input_digest.to_hex()));
    let mut changed = fixture.replay(&data, "baseline", |_| Some(2)); changed.compatible_executions[0] ^= 1;
    assert!(matches!(compare_with_checkpoint::<AssayError>(&current, &changed, &key, "fixture", "declared", gates(),
        options(), Limits::default(), &mut || Ok(())), Err(AssayError::Scope)));
}

#[test]
fn policy_abstentions_earn_no_accuracy_and_undefined_macro_f1_cannot_pass() {
    let data = dataset(6, 1); let fixture = Fixture::new(); let replay = fixture.replay(&data, "candidate", |_| None);
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    let report = evaluate_with_checkpoint::<AssayError>(&replay, &key, "fixture", options(), Limits::default(), &mut || Ok(())).unwrap();
    let Measurement::Measured { quality, intervals } = report.body.measurement else { panic!("measured fixture"); };
    assert_eq!(quality.labels.exact_match_accuracy, 0.0); assert_eq!(quality.labels.abstained_rows, 6);
    assert_eq!(quality.abstentions.ambiguous_distribution, 6); assert_eq!(quality.labels.label_decision_coverage, 0.0);
    assert_eq!(quality.selective_risk, None); assert_eq!(intervals.selective_risk, None);
    assert!((quality.brier - 1.0 / 3.0).abs() < 1e-12);
    let (suite, mut rows) = wire(6, 1); for row in &mut rows { row["relation"] = serde_json::json!("entailed"); }
    let (suite, bytes) = encode(suite, &rows); let data = Dataset::parse(&suite, &bytes, Limits::default()).unwrap();
    let replay = fixture.replay(&data, "candidate", |_| Some(1));
    let compared = compare_with_checkpoint::<AssayError>(&replay, &replay, &key, "fixture", "declared", gates(),
        options(), Limits::default(), &mut || Ok(())).unwrap();
    assert!(matches!(compared.body.decision, Decision::Rejected { .. }));
}

#[test]
fn cancellation_during_independent_validation_cannot_finish_a_prefix() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let data = dataset(1, 1); let (identity, run) = Fixture::new().run(&data, 0, "candidate", &[Some(1)]);
    let mut collector = ReplayBuilder::new(&data, Limits::default()).unwrap();
    assert_eq!(collector.accept_fixture(0, &identity, budget(), run.model_work, 128, run.head_count, run, &mut Stop),
        Err(ReplayError::Cancelled(DecodeCancellationKind::Deadline)));
    assert!(collector.finish().is_err());
}

#[test]
fn repeating_correlated_claims_does_not_narrow_source_bootstrap_intervals() {
    let fixture = Fixture::new(); let mut measured = Vec::new();
    for claims in [1, 5] {
        let (suite, mut rows) = wire(6, claims);
        for source in 0..6 { for claim in 0..claims {
            rows[2 + source * claims + claim]["relation"] = serde_json::json!(RELATIONS[source % 3]);
        } }
        let (suite, bytes) = encode(suite, &rows); let data = Dataset::parse(&suite, &bytes, Limits::default()).unwrap();
        let replay = fixture.replay(&data, "candidate", |index| Some((index / claims + index / claims % 2) % 3));
        let (measurement, _, _) = metrics::measure::<AssayError>(&replay, None, options(), Limits::default(), &mut || Ok(())).unwrap();
        let Measurement::Measured { quality, intervals } = measurement else { panic!("enough distinct sources"); };
        assert_eq!(quality.source_clusters, 6); assert_eq!(quality.labels.rows, 6 * claims);
        measured.push((quality.labels.exact_match_accuracy, intervals.accuracy));
    }
    assert_eq!(measured[0], measured[1]);
    let interval = measured[0].1.unwrap(); assert!(interval.lower < interval.upper);
}

#[test]
fn complete_unicode_windows_and_conflicting_evidence_keep_the_native_abstention() {
    let (mut suite, mut rows) = wire(1, 1); suite.policy.evidence_window_bytes = 4;
    rows[2]["source"] = serde_json::json!("éA.上海😀tail");
    let (suite, bytes) = encode(suite, &rows); let data = Dataset::parse(&suite, &bytes, Limits::default()).unwrap();
    let windows = crate::tasks::judge::partition_evidence(&data.replay_rows()[0].source, data.manifest.policy).unwrap();
    assert!(windows.len() > 1);
    assert!(data.validate_head_limit(windows.len()).is_err());
    data.validate_head_limit(windows.len() + 1).unwrap();
    let mut choices = vec![Some(1); windows.len() + 1]; choices[1] = Some(0);
    let (identity, run) = Fixture::new().run(&data, 0, "candidate", &choices);
    let JudgeResult::Faithfulness(result) = &run.result else { unreachable!() };
    assert_eq!(result.abstention, Some(crate::tasks::judge::FaithfulnessAbstention::ConflictingEvidence));
    assert_eq!(result.windows.iter().map(|window| window.span).collect::<Vec<_>>(), windows);
    let mut collector = ReplayBuilder::new(&data, Limits::default()).unwrap();
    collector.accept_fixture(0, &identity, budget(), run.model_work, 128, run.head_count, run, &mut Go).unwrap();
    let replay = collector.finish().unwrap(); assert_eq!(replay.rows[0].predicted, None);
    assert_eq!(replay.rows[0].abstention, Some(crate::tasks::judge::FaithfulnessAbstention::ConflictingEvidence));
}
