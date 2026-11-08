//! Arithmetic and replay-integrity fixtures, never evidence of model quality.
use super::*;
use crate::{
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl}, lmhead::scoring::{ProjectionRows, ScoringMode},
        portable_int8::ProjectionWork, strict_int4::{Int4Work, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE}},
    receipt::CommitmentKey,
    tasks::{ir::{PlanContext, PromptSegment, TaskBudget}, sentiment::{SentimentLogits, SentimentOptions,
        SentimentPlanner, SentimentPolicy, SentimentLimits, grouped_int4::{Int4SentimentRun, INT4_SENTIMENT_EXECUTION}}},
};
use replay::{NativeReplay, ReplayBuilder, ScoredRow};

pub(super) fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4092, max_output_tokens: 5, max_output_bytes: 1_000_000,
        max_grammar_states: 17, max_kv_bytes: 1 << 30 }
}
pub(super) fn policy() -> SentimentPolicy {
    SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 }
}
pub(super) fn wire(documents: usize, axes: usize) -> (SuiteManifest, Vec<serde_json::Value>) {
    let axes = SentimentAxis::ALL[..axes].to_vec(); let mut rows = Vec::new(); let mut ids = Vec::new();
    for (split, count) in [(Split::Development, 1), (Split::Calibration, 1), (Split::LockedTest, documents)] {
        for index in 0..count {
            let name = match split { Split::Development => "dev", Split::Calibration => "cal", Split::LockedTest => "test" };
            let id = format!("private-{name}-{index:03}");
            if split == Split::LockedTest { ids.push(id.clone()); }
            rows.push(serde_json::json!({"schema_version":1,"id":id,"split":split,
                "text":format!("PRIVATE owned {name} document {index} é上海😀"),
                "gold":axes.iter().map(|&axis| Gold { axis, value_milli: [-1000,-500,0,500,1000][index % 5] }).collect::<Vec<_>>()}));
        }
    }
    (SuiteManifest { schema_version: 1, task: NATIVE_TASK.to_owned(), dataset_id: "owned".to_owned(),
        dataset_version: "1".to_owned(), named_population: "synthetic fixture only".to_owned(), axes, policy: policy(),
        expected_input_sha256: Sha256Digest::of_bytes(b"unbound"), splits: SplitIds {
            development: vec!["private-dev-000".to_owned()], calibration: vec!["private-cal-000".to_owned()], locked_test: ids } }, rows)
}
pub(super) fn encode(mut suite: SuiteManifest, rows: &[serde_json::Value]) -> (Vec<u8>, Vec<u8>) {
    let mut bytes = Vec::new();
    for row in rows { bytes.extend(serde_json::to_vec(row).unwrap()); bytes.push(b'\n'); }
    suite.expected_input_sha256 = Sha256Digest::of_bytes(&bytes);
    (serde_json::to_vec(&suite).unwrap(), bytes)
}
pub(super) fn dataset(documents: usize, axes: usize) -> Dataset {
    let (suite, rows) = wire(documents, axes); let (suite, bytes) = encode(suite, &rows);
    Dataset::parse(&suite, &bytes, Limits::default()).unwrap()
}
pub(super) fn options() -> EvalOptions { EvalOptions { minimum_samples: 5,
    bootstrap: crate::assay::BootstrapOptions { resamples: 40, confidence_level: 0.95, seed: 42 } } }
pub(super) fn gates() -> Gates { Gates { schema_version: 1, max_mae_increase: 0.0, max_rmse_increase: 0.0,
    max_coverage_decrease: 0.0, max_spearman_decrease: None, max_estimated_mae_increase: None } }
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"synthetic-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: digest, artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(),
        packing_set_digest: digest, tokenizer_digest: digest, template_digest: digest, task_spec: "sentiment-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: Sha256Digest::of_bytes(b"null"), decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
/// Direct compact arithmetic fixture; only the separate native-finalizer test
/// below exercises ReplayBuilder. Production has no caller-supplied replay API.
pub(super) fn arithmetic_replay(data: &Dataset) -> NativeReplay {
    NativeReplay { manifest: data.manifest.clone(), manifest_digest: data.manifest_digest,
        input_digest: data.input_digest, sizes: data.sizes,
        rows: data.replay_rows().iter().map(|row| ScoredRow { id: row.id.clone(),
            source: Sha256Digest::of_bytes(row.text.as_bytes()), gold: row.gold.clone(),
            means: row.gold.iter().map(|gold| f64::from(gold.value_milli) / 1000.0).collect(),
            estimated: vec![true; row.gold.len()] }).collect(),
        first_identity: identity(), executions: [1; 32], compatible_executions: [2; 32], observations: [3; 32],
        model_work: Int4Work::default(), group_width: 128, heads: data.replay_len() * data.manifest.axes.len() }
}

#[test]
fn suites_require_exact_source_disjoint_splits_and_complete_bounded_human_axes() {
    for mutation in 0..9 {
        let (mut suite, mut rows) = wire(5, 2);
        match mutation {
            0 => rows[2]["text"] = rows[0]["text"].clone(),
            1 => rows[3]["text"] = rows[2]["text"].clone(),
            2 => suite.splits.calibration.clear(),
            3 => suite.splits.development[0] = suite.splits.locked_test[0].clone(),
            4 => rows[2]["split"] = serde_json::json!("development"),
            5 => rows[2]["gold"][0]["value_milli"] = serde_json::json!(1001),
            6 => rows[2]["gold"][0]["value_milli"] = serde_json::json!(0.5),
            7 => rows[2]["gold"][1]["axis"] = rows[2]["gold"][0]["axis"].clone(),
            _ => rows[2]["execution_identity"] = serde_json::json!({"foreign":"caller"}),
        }
        let (suite, bytes) = encode(suite, &rows);
        assert!(Dataset::parse(&suite, &bytes, Limits::default()).is_err(), "mutation {mutation}");
    }
    let (suite, rows) = wire(5, 2); let (suite, bytes) = encode(suite, &rows);
    assert!(Dataset::parse_with_axis_limit(&suite, &bytes, Limits::default(), 1).is_err());
    let mut changed = bytes.clone(); changed.push(b'\n');
    assert!(matches!(Dataset::parse(&suite, &changed, Limits::default()), Err(AssayError::InputDigest)));
}

#[test]
fn human_milli_ratings_preserve_between_bin_values_and_never_enter_requests() {
    let (suite, mut rows) = wire(1, 2);
    rows[2]["gold"][0]["value_milli"] = serde_json::json!(133);
    let (manifest, bytes) = encode(suite.clone(), &rows); let first = Dataset::parse(&manifest, &bytes, Limits::default()).unwrap();
    rows[2]["gold"][0]["value_milli"] = serde_json::json!(-777);
    let (manifest, bytes) = encode(suite, &rows); let second = Dataset::parse(&manifest, &bytes, Limits::default()).unwrap();
    assert_eq!(first.replay_rows()[0].gold[0].value_milli, 133);
    assert_eq!(first.validated_rows(), 3); assert_eq!(first.replay_len(), 1);
    let a = first.request(0, budget()).unwrap(); let b = second.request(0, budget()).unwrap();
    assert!(a == b); assert_ne!(first.input_digest, second.input_digest);
    let value = serde_json::to_value(&a).unwrap();
    assert_eq!(value.as_object().unwrap().len(), 3);
    for private in ["id", "split", "gold", "value_milli", "execution_identity"] { assert!(value.get(private).is_none()); }
    assert_eq!(a.document, first.replay_rows()[0].text);
}

struct Go;
impl DecodeStepControl for Go { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Flat;
impl SentimentLogits for Flat {
    type Error = &'static str;
    fn project(&mut self, _: SentimentAxis, _: &[PromptSegment], _: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("full denominator required"); };
        Ok(vec![0.0; vocabulary_size])
    }
}
fn native_fixture(data: &Dataset) -> (ExecutionIdentity, Int4SentimentRun) {
    let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
    let eos_token_id = registry.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let planner = SentimentPlanner::pinned(registry.template_controls(), SentimentOptions {
        mode: ScoringMode::FullVocabulary, eos_token_id, policy: data.manifest.policy }).unwrap();
    let mut identity = identity(); identity.tokenizer_digest = planner.tokenizer_digest();
    identity.template_digest = *planner.template_digest(); identity.numerics_profile = NumericsProfile::HfBf16Eager;
    let request = data.request(0, budget()).unwrap();
    let plan = planner.plan(&request, &PlanContext::new(&identity, budget()).unwrap(), SentimentLimits::default()).unwrap();
    let result = plan.execute(&mut Flat).unwrap();
    identity.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
    identity.taskir_digest = *plan.binding_digest();
    identity.decision_policy_digest = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(
        &(INT4_SENTIMENT_EXECUTION, identity.taskir_digest, 128_usize)).unwrap());
    let heads = result.dimensions.len();
    let work = Int4Work { forward_positions: 100 * heads as u64, projected_logits: result.work.projected_logits,
        attention_pairs: 500 * heads as u64, projections: ProjectionWork { dot_products: 1000, multiply_accumulates: 10_000 } };
    (identity, Int4SentimentRun { schema_version: 1, execution: INT4_SENTIMENT_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: 128, result,
        head_count: heads, model_work: work, rewound_positions: 0 })
}

#[test]
fn complete_real_finalizer_output_passes_and_tampered_native_contracts_poison_replay() {
    let data = dataset(1, 2); let (identity, run) = native_fixture(&data);
    let accept = |identity: &ExecutionIdentity, run: Int4SentimentRun| {
        let mut collector = ReplayBuilder::new(&data, Limits::default()).unwrap();
        let result = collector.accept_fixture(0, identity, budget(), run.model_work, 128, run, &mut Go);
        (result, collector.finish())
    };
    let (accepted, replay) = accept(&identity, run.clone()); assert!(accepted.is_ok());
    let replay = replay.unwrap(); assert_eq!(replay.rows.len(), 1); assert_eq!(replay.heads, 2);
    for mutation in 0..13 {
        let mut changed = run.clone(); let mut bound = identity.clone();
        match mutation {
            0 => changed.result.dimensions[0].distribution_mean = 0.9,
            1 => changed.result.dimensions[0].scores.candidates[0].candidate_weight = 0.5,
            2 => changed.result.dimensions[0].scores.candidates[0].scored_tokens -= 1,
            3 => changed.result.dimensions[0].scores.full_vocab_denominators_computed = false,
            4 => changed.result.dimensions[0].scores.eos_token_id = 0,
            5 => changed.result.dimensions[0].anchors[0].value_milli = -999,
            6 => changed.result.dimensions[0].estimate = None,
            7 => changed.result.dimensions[0].normalized_entropy = f64::NAN,
            8 => changed.head_count -= 1,
            9 => changed.model_work.projected_logits -= 1,
            10 => bound.decision_policy_digest = Sha256Digest::of_bytes(b"other-policy"),
            11 => bound.calibration_digest = Sha256Digest::of_bytes(b"foreign-calibration"),
            _ => {
                // A common log-score offset preserves candidate weights and
                // moments, but positive full-vocabulary log probabilities are impossible.
                let scores = &mut changed.result.dimensions[0].scores.candidates;
                let offset = 1.0 - scores.iter().map(|candidate| candidate.sequence_score).fold(f64::INFINITY, f64::min);
                for candidate in scores { candidate.sequence_score += offset; }
            }
        }
        let (accepted, replay) = accept(&bound, changed); assert!(accepted.is_err(), "mutation {mutation}"); assert!(replay.is_err());
    }
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let mut collector = ReplayBuilder::new(&data, Limits::default()).unwrap();
    assert_eq!(collector.accept_fixture(0, &identity, budget(), run.model_work, 128, run, &mut Stop),
        Err(ReplayError::Cancelled(DecodeCancellationKind::Deadline)));
    assert!(collector.finish().is_err());
}

#[test]
fn every_axis_must_pass_and_aggregate_reports_bind_without_exposing_private_cases() {
    let data = dataset(5, 2); let prior = arithmetic_replay(&data); let mut current = arithmetic_replay(&data);
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    let compare = |a: &NativeReplay, b: &NativeReplay, gates: Gates| compare_with_checkpoint::<AssayError>(a, b, &key,
        "fixture", "declared", gates, options(), Limits::default(), &mut || Ok(()));
    let passed = compare(&current, &prior, gates()).unwrap();
    let Decision::Passed { paired, comparisons } = &passed.body.decision else { panic!("exact predictions must pass"); };
    assert_eq!(paired.documents, 5); assert_eq!(paired.axes.len(), 2); assert_eq!(comparisons.len(), 6);
    for axis in &paired.axes {
        assert_eq!(axis.mae_change, 0.0); assert_eq!(axis.rmse_change, 0.0);
        let interval = axis.intervals.mae.unwrap(); assert_eq!(interval.lower, 0.0); assert_eq!(interval.upper, 0.0);
    }
    current.rows[0].means[1] = 1.0;
    let rejected = compare(&current, &prior, gates()).unwrap();
    let Decision::Rejected { comparisons, .. } = &rejected.body.decision else { panic!("one degraded axis must reject"); };
    assert!(comparisons.iter().filter(|gate| gate.name.starts_with("valence.")).all(|gate| gate.passed));
    assert!(comparisons.iter().any(|gate| gate.name == "arousal.mae_increase" && !gate.passed));
    let text = String::from_utf8(crate::assay::render(&passed, Limits::default()).unwrap()).unwrap();
    for private in ["PRIVATE", "private-test", "private-dev", "private-cal"] { assert!(!text.contains(private)); }
    assert!(!text.contains(&data.input_digest.to_hex()));
    for row in data.replay_rows() { assert!(!text.contains(&Sha256Digest::of_bytes(row.text.as_bytes()).to_hex())); }
    current.compatible_executions[0] ^= 1;
    assert!(matches!(compare(&current, &prior, gates()), Err(AssayError::Scope)));
}

#[test]
fn optional_undefined_gates_fail_closed_and_small_document_populations_stay_insufficient() {
    let data = dataset(5, 1); let mut replay = arithmetic_replay(&data);
    for row in &mut replay.rows { row.means[0] = 0.0; row.estimated[0] = false; }
    let key = CommitmentKey::new("fixture", [0x42; 32]).unwrap();
    for rank in [false, true] {
        let mut gates = gates();
        if rank { gates.max_spearman_decrease = Some(0.0); } else { gates.max_estimated_mae_increase = Some(0.0); }
        let report = compare_with_checkpoint::<AssayError>(&replay, &replay, &key, "fixture", "declared", gates,
            options(), Limits::default(), &mut || Ok(())).unwrap();
        let Decision::Rejected { comparisons, .. } = report.body.decision else { panic!("undefined requested metric must reject"); };
        assert!(comparisons.iter().any(|gate| gate.observed.is_none() && !gate.passed));
    }
    let small = arithmetic_replay(&dataset(4, 4));
    let report = compare_with_checkpoint::<AssayError>(&small, &small, &key, "fixture", "declared", gates(),
        options(), Limits::default(), &mut || Ok(())).unwrap();
    assert!(matches!(report.body.decision, Decision::InsufficientData { documents: 4, minimum_documents: 5 }));
}
