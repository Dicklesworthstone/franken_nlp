//! Independent model-free input, collector and quality-report fixtures.
//! Synthetic runs enter only the private cfg(test) builder; these tests never
//! claim to execute a real model or expose a production supplied-score seam.
use super::*;
use crate::{assay::{BootstrapOptions, Measurement, qualify::Decision},
    calibration::CalibrationError,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{CandidateScore, CandidateScores, ScoringWork}},
        strict_int4::{Int4Work, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE}},
    receipt::{CommitmentDomain, CommitmentKey},
    tasks::{classify::{ClassificationCalibration, ClassificationDecision, ClassificationLabel,
        ClassificationPolicy, ClassificationResult, ClassificationTaskResult,
        grouped_int4::{Int4ClassificationRun, INT4_CLASSIFICATION_EXECUTION}},
        ir::{ScoreSpace, TaskBudget}}};
use replay::ReplayBuilder;

fn policy() -> ClassificationPolicy { ClassificationPolicy::default() }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 16, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn options() -> EvalOptions {
    EvalOptions { minimum_samples: 5, acceptance_threshold: 0.5,
        bootstrap: BootstrapOptions { resamples: 8, confidence_level: 0.95, seed: 17 } }
}
fn key() -> CommitmentKey { CommitmentKey::new("native-test-key", [73; 32]).unwrap() }
fn gates() -> Gates {
    Gates { schema_version: 1, reliability: crate::assay::qualify::Gates { schema_version: 1,
        max_brier_increase: 1.0, max_ece_increase: 1.0, max_selective_risk: 1.0,
        min_acceptance_rate: 0.0, minimum_accepted: 1 },
        max_accuracy_decrease: 1.0, max_macro_f1_decrease: 1.0 }
}
fn wire(gold: &[bool], policy: ClassificationPolicy) -> (SuiteManifest, Vec<Row>) {
    let mut rows = vec![Row { schema_version: 1, id: "private-development".to_owned(),
        split: Split::Development, text: "PRIVATE_DEV é <|im_start|>".to_owned(), positive: true },
        Row { schema_version: 1, id: "private-calibration".to_owned(), split: Split::Calibration,
            text: "PRIVATE_CAL 中文".to_owned(), positive: false }];
    let mut locked = Vec::new();
    for (index, &positive) in gold.iter().enumerate() {
        let id = format!("private-test-{index:03}"); locked.push(id.clone());
        rows.push(Row { schema_version: 1, id, split: Split::LockedTest,
            text: format!("PRIVATE_LOCKED_{index} e\u{301} <think>data</think>"), positive });
    }
    (SuiteManifest { schema_version: 1, task: NATIVE_TASK.to_owned(), dataset_id: "native-suite".to_owned(),
        dataset_version: "v1".to_owned(), named_population: "Owned fixture population".to_owned(),
        positive_label: ClassificationLabel { id: "positive".to_owned(), description: "The positive event".to_owned() },
        negative_label: ClassificationLabel { id: "negative".to_owned(), description: "The negative event".to_owned() },
        policy, expected_input_sha256: Sha256Digest::of_bytes(b"unbound"),
        splits: SplitIds { development: vec![rows[0].id.clone()], calibration: vec![rows[1].id.clone()], locked_test: locked } }, rows)
}
fn encode(mut manifest: SuiteManifest, rows: &[Row]) -> (Vec<u8>, Vec<u8>) {
    let mut input = Vec::new();
    for row in rows { input.extend(serde_json::to_vec(row).unwrap()); input.push(b'\n'); }
    manifest.expected_input_sha256 = Sha256Digest::of_bytes(&input);
    (serde_json::to_vec(&manifest).unwrap(), input)
}
fn dataset(gold: &[bool], policy: ClassificationPolicy) -> Dataset {
    let (manifest, rows) = wire(gold, policy); let (manifest, input) = encode(manifest, &rows);
    Dataset::parse(&manifest, &input, Limits::default()).unwrap()
}
fn identity(row: &Row, subject: &str) -> ExecutionIdentity {
    let public = Sha256Digest::of_bytes(b"public pinned fixture assets");
    let private = Sha256Digest::of_bytes(row.text.as_bytes());
    ExecutionIdentity { schema_version: 1, source_revision: "fixture-pinned-revision".to_owned(),
        logical_model_digest: Sha256Digest::of_bytes(subject.as_bytes()), artifact_format: "fixture-int4".to_owned(),
        quant_recipe: subject.to_owned(), packing_set_digest: Sha256Digest::of_bytes(subject.as_bytes()),
        tokenizer_digest: public, template_digest: public, task_spec: "classify-v1".to_owned(),
        taskir_digest: private, prompt_digest: private, grammar_compiler_version: "none".to_owned(),
        schema_digest: public, numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture-full-vocabulary".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: public, decision_policy_digest: public,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn work(width: usize) -> Int4Work { Int4Work::for_sequence(width, 0, 4, 3 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn eos() -> u32 {
    crate::tokenizer::pinned_controls::pinned().unwrap().template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id
}
fn run(manifest: &SuiteManifest, p: f64, width: usize) -> Int4ClassificationRun {
    let log = |value: f64| if value == 0.0 { -1000.0 } else { value.ln() };
    let mut candidates = vec![CandidateScore { id: manifest.positive_label.id.clone(), scored_tokens: 2,
        sequence_score: log(p), candidate_weight: p }, CandidateScore { id: manifest.negative_label.id.clone(),
            scored_tokens: 2, sequence_score: log(1.0 - p), candidate_weight: 1.0 - p }];
    candidates.sort_by(|a, b| a.id.cmp(&b.id));
    let mut ranking = candidates.iter().collect::<Vec<_>>();
    ranking.sort_by(|a, b| if a.sequence_score == b.sequence_score { a.id.cmp(&b.id) }
        else { b.sequence_score.total_cmp(&a.sequence_score) });
    let best = ranking[0]; let margin = (best.candidate_weight - ranking[1].candidate_weight).max(0.0);
    let accepted = best.candidate_weight * 1_000_000.0 >= f64::from(manifest.policy.minimum_candidate_weight_ppm)
        && margin * 1_000_000.0 >= f64::from(manifest.policy.minimum_margin_ppm);
    let selected_id = accepted.then(|| best.id.clone()); let best_candidate_weight = best.candidate_weight;
    let order = ranking.iter().map(|candidate| candidate.id.clone()).collect();
    drop(ranking);
    let scores = CandidateScores { score_space: ScoreSpace::FullVocabSequenceLogprob,
        normalization_scope: "full_vocabulary_per_position; candidate_weights_over_completed_candidates".to_owned(),
        length_rule: "sum_logprobs_including_eos".to_owned(), eos_rule: "append_and_score_exactly_one_eos".to_owned(),
        eos_token_id: eos(), full_vocab_denominators_computed: true, candidates,
        work: ScoringWork { prefix_evaluations: 3, scored_edges: 4, projected_logits: 3 * NANBEIGE_VOCAB_SIZE as u64 } };
    Int4ClassificationRun { schema_version: 1, execution: INT4_CLASSIFICATION_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: width,
        result: ClassificationTaskResult::Exclusive(ClassificationResult { schema_version: 1,
            decision: if accepted { ClassificationDecision::Classified } else { ClassificationDecision::Abstained },
            selected_id, ranking: order, best_candidate_weight, candidate_weight_margin: margin,
            calibration: ClassificationCalibration::Uncalibrated, policy: manifest.policy, scores }),
        head_count: 1, model_work: work(width), rewound_positions: 1 }
}
fn replay(dataset: &Dataset, probabilities: &[f64], subject: &str) -> NativeReplay {
    replay_changed(dataset, probabilities, subject, |_, _, _, _| {})
}
fn replay_changed(dataset: &Dataset, probabilities: &[f64], subject: &str,
    mut change: impl FnMut(usize, &mut ExecutionIdentity, &mut TaskBudget, &mut Int4ClassificationRun)) -> NativeReplay {
    let mut builder = ReplayBuilder::new(dataset, Limits::default()).unwrap();
    let mut locked = 0;
    for (index, row) in dataset.rows().iter().enumerate() {
        let p = if row.split == Split::LockedTest { let p = probabilities[locked]; locked += 1; p }
            else if row.positive { 0.0 } else { 1.0 }; // Deliberately wrong non-test scores.
        let mut identity = identity(row, subject); let mut budget = budget(); let mut output = run(dataset.manifest(), p, 128);
        change(index, &mut identity, &mut budget, &mut output);
        builder.accept_fixture(index, &identity, budget, work(128), 128, output).unwrap();
    }
    builder.finish().unwrap()
}

#[test]
fn source_pin_exact_text_units_and_complete_split_membership_are_checked_before_replay() {
    let (manifest, mut rows) = wire(&[true, false, true, false, true], policy());
    let (bytes, mut input) = encode(manifest.clone(), &rows); input.push(b' ');
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default()).err(), Some(AssayError::InputDigest));
    rows[2].text = rows[0].text.clone();
    let (bytes, input) = encode(manifest.clone(), &rows);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default()).err(), Some(AssayError::DuplicateUnit));
    rows[2].text.push(' '); // Exact bytes, not a hidden normalization policy.
    let (bytes, input) = encode(manifest.clone(), &rows);
    assert!(Dataset::parse(&bytes, &input, Limits::default()).is_ok());
    rows[2].split = Split::Calibration;
    let (bytes, input) = encode(manifest.clone(), &rows);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default()).err(), Some(AssayError::SplitMembership));
    let mut overlapping = manifest; overlapping.splits.calibration.push(overlapping.splits.locked_test[0].clone());
    let (bytes, input) = encode(overlapping, &rows);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default()).err(), Some(AssayError::SplitMembership));
}

#[test]
fn canonical_replay_order_preserves_exact_input_framing_as_a_separate_frozen_binding() {
    let (manifest, rows) = wire(&[true, false, true, false, true, false], policy());
    let (bytes, input) = encode(manifest.clone(), &rows);
    let forward = Dataset::parse(&bytes, &input, Limits::default()).unwrap();
    let mut reversed_input = Vec::new();
    for (index, row) in rows.iter().rev().enumerate() {
        if index != 0 { reversed_input.extend_from_slice(b"\r\n"); }
        reversed_input.extend(serde_json::to_vec(row).unwrap());
    }
    // CRLF records, reversed physical order and an unterminated final record
    // preserve the exact source strings while changing the frozen byte pin.
    let mut changed_manifest = manifest;
    changed_manifest.expected_input_sha256 = Sha256Digest::of_bytes(&reversed_input);
    let reversed = Dataset::parse(&serde_json::to_vec(&changed_manifest).unwrap(), &reversed_input,
        Limits::default()).unwrap();
    assert_eq!(serde_json::to_value(forward.rows()).unwrap(), serde_json::to_value(reversed.rows()).unwrap());
    let a = replay(&forward, &[0.8; 6], "native-a");
    let b = replay(&reversed, &[0.8; 6], "native-a");
    assert_eq!(a.executions, b.executions);
    assert_eq!(a.compatible_executions, b.compatible_executions);
    assert_eq!(a.observations, b.observations);
    assert_ne!(a.input_digest, b.input_digest);
    let key = key();
    let first = evaluate(&a, &key, "scope", options(), Limits::default()).unwrap();
    let second = evaluate(&b, &key, "scope", options(), Limits::default()).unwrap();
    assert_ne!(first.seal, second.seal);
    assert_eq!(compare(&a, &b, &key, "scope", "host", gates(), options(), Limits::default()).err(),
        Some(AssayError::Scope));
}

#[test]
fn native_wire_rejects_scores_identities_unknown_keys_and_duplicate_json_fields() {
    let (manifest, rows) = wire(&[true, false, true, false, true], policy());
    let (bytes, input) = encode(manifest.clone(), &rows);
    for forbidden in ["probability", "execution_identity", "unit_sha256", "task_args"] {
        let mut values: Vec<serde_json::Value> = rows.iter().map(|r| serde_json::to_value(r).unwrap()).collect();
        values[0].as_object_mut().unwrap().insert(forbidden.to_owned(), serde_json::json!(0.9));
        let changed = values.into_iter().map(|v| serde_json::to_string(&v).unwrap()).collect::<Vec<_>>().join("\n");
        let mut declared = manifest.clone(); declared.expected_input_sha256 = Sha256Digest::of_bytes(changed.as_bytes());
        assert_eq!(Dataset::parse(&serde_json::to_vec(&declared).unwrap(), changed.as_bytes(), Limits::default()).err(),
            Some(AssayError::Json));
    }
    let duplicated = String::from_utf8(bytes).unwrap().replacen("\"schema_version\":1", "\"schema_version\":1,\"schema_version\":1", 1);
    assert_eq!(Dataset::parse(duplicated.as_bytes(), &input, Limits::default()).err(), Some(AssayError::Json));
    let mut invalid = manifest; invalid.policy.minimum_margin_ppm = 1_000_001;
    let (bytes, input) = encode(invalid, &rows);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default()).err(), Some(AssayError::Manifest));
}

#[test]
fn request_uses_original_source_labels_and_policy_without_gold_or_split_in_the_prompt() {
    let task_policy = ClassificationPolicy { minimum_candidate_weight_ppm: 700_000, minimum_margin_ppm: 100_000 };
    let data = dataset(&[true, false, true, false, true], task_policy);
    assert!(data.rows().windows(2).all(|pair| pair[0].id < pair[1].id));
    for (index, row) in data.rows().iter().enumerate() {
        let request = data.classification_request(index, budget()).unwrap();
        assert_eq!(request.document, row.text); assert_eq!(request.policy, task_policy);
        assert_eq!(request.mode, crate::tasks::classify::ClassificationMode::Exclusive);
        assert_eq!(request.budget, budget()); assert_eq!(request.labels.len(), 2);
        let fields = serde_json::to_value(request).unwrap();
        assert!(fields.get("positive").is_none()); assert!(fields.get("split").is_none());
        assert!(fields.get("id").is_none());
    }
    assert!(data.classification_request(data.len(), budget()).is_err());
    let mut tight = Limits::default(); tight.max_rows = 3;
    assert_eq!(data.validate_limits(tight), Err(AssayError::InputLimit));
}

#[test]
fn collector_rejects_incomplete_wrong_order_or_bad_native_outputs_and_poison_is_permanent() {
    let data = dataset(&[true, false, true, false, true], policy());
    assert!(ReplayBuilder::new(&data, Limits::default()).unwrap().finish().is_err());
    for mutation in 0..11 {
        let mut builder = ReplayBuilder::new(&data, Limits::default()).unwrap();
        let id = identity(&data.rows()[0], "native-a");
        let mut output = run(data.manifest(), 0.8, 128);
        match mutation {
            0 => output.head_count = 2,
            1 => output.model_work.forward_positions += 1,
            2 => output.group_width = 64,
            3 => output.numerics_profile = "fake".to_owned(),
            _ => { let ClassificationTaskResult::Exclusive(result) = &mut output.result else { unreachable!() };
                match mutation {
                    4 => result.scores.candidates.pop().map(|_| ()).unwrap(),
                    5 => result.scores.candidates[0].id = "foreign".to_owned(),
                    6 => result.scores.eos_token_id += 1,
                    7 => result.scores.full_vocab_denominators_computed = false,
                    8 => result.ranking.reverse(),
                    9 => result.selected_id = Some("negative".to_owned()),
                    _ => result.scores.candidates[0].candidate_weight = f64::NAN,
                }
            }
        }
        assert!(builder.accept_fixture(0, &id, budget(), work(128), 128, output).is_err(), "mutation {mutation}");
        assert!(builder.accept_fixture(0, &id, budget(), work(128), 128, run(data.manifest(), 0.8, 128)).is_err());
        assert!(builder.finish().is_err());
    }
    let mut builder = ReplayBuilder::new(&data, Limits::default()).unwrap();
    let id = identity(&data.rows()[0], "native-a");
    assert!(builder.accept_fixture(1, &id, budget(), work(128), 128, run(data.manifest(), 0.8, 128)).is_err());
    assert!(builder.finish().is_err());
}

#[test]
fn held_out_quality_uses_actual_decisions_and_excludes_wrong_development_and_calibration_scores() {
    let data = dataset(&[true, false, true, false, true, false], policy());
    let replay = replay(&data, &[0.8, 0.3, 0.4, 0.8, 0.9, 0.2], "native-a");
    let result = evaluate(&replay, &key(), "scope", options(), Limits::default()).unwrap();
    let counts = result.body.classification.as_ref().unwrap();
    assert_eq!((counts.true_positive, counts.false_positive, counts.true_negative, counts.false_negative), (2, 1, 2, 1));
    assert_eq!((counts.rows, counts.decided, counts.abstained), (6, 6, 0));
    assert!((counts.accuracy - 2.0 / 3.0).abs() < 1e-12);
    assert!((counts.macro_f1.unwrap() - 2.0 / 3.0).abs() < 1e-12);
    let Measurement::Measured { metrics } = &result.body.measurement else { panic!("held-out metrics") };
    assert_eq!(metrics.rows, 6); assert_eq!(metrics.accepted, 3);
    assert!((metrics.brier - 1.18 / 6.0).abs() < 1e-12);
    assert!((metrics.selective_risk.unwrap() - 1.0 / 3.0).abs() < 1e-12);
    assert_eq!(result.body.executed_rows, data.len());
    assert_eq!(result.body.bindings.execution_collection_rows, data.len());
    assert_eq!(result.body.evidence, NATIVE_EVIDENCE); assert!(!result.body.activation_authority);
}

#[test]
fn native_lexical_ties_and_abstentions_are_not_replaced_by_positive_probability_thresholds() {
    let data = dataset(&[true, false, true, false, true, false], policy());
    let tied = evaluate(&replay(&data, &[0.5; 6], "native-a"), &key(), "scope", options(), Limits::default()).unwrap();
    let counts = tied.body.classification.unwrap();
    assert_eq!((counts.true_negative, counts.false_negative, counts.true_positive, counts.false_positive), (3, 3, 0, 0));
    let strict = ClassificationPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_ppm: 300_000 };
    let data = dataset(&[true, false, true, false, false, true], strict);
    let output = evaluate(&replay(&data, &[0.6, 0.4, 0.8, 0.2, 0.8, 0.2], "native-a"),
        &key(), "scope", options(), Limits::default()).unwrap();
    let counts = output.body.classification.unwrap();
    assert_eq!((counts.true_positive, counts.false_positive, counts.true_negative, counts.false_negative), (1, 1, 1, 1));
    assert_eq!((counts.abstained, counts.abstained_positive, counts.abstained_negative), (2, 1, 1));
    assert!((counts.accuracy - 1.0 / 3.0).abs() < 1e-12);
    assert_eq!(counts.decided_accuracy, Some(0.5));
    assert!((counts.positive.recall.unwrap() - 1.0 / 3.0).abs() < 1e-12);
    assert!((counts.positive.f1.unwrap() - 0.4).abs() < 1e-12);
    assert!((counts.macro_f1.unwrap() - 0.4).abs() < 1e-12);
}

#[test]
fn paired_quality_gates_refuse_regression_even_when_all_reliability_gates_allow_it() {
    let data = dataset(&[true, false, true, false, true, false], policy());
    let baseline = replay(&data, &[0.9, 0.1, 0.9, 0.1, 0.9, 0.1], "native-baseline");
    let candidate = replay(&data, &[0.1, 0.9, 0.1, 0.9, 0.1, 0.9], "native-candidate");
    let mut limits = gates(); limits.max_accuracy_decrease = 0.1; limits.max_macro_f1_decrease = 0.1;
    let report = compare(&candidate, &baseline, &key(), "scope", "declared-host", limits, options(), Limits::default()).unwrap();
    let Decision::Rejected { comparisons, paired } = &report.body.decision else { panic!("task regression must reject") };
    assert!(comparisons[..5].iter().all(|gate| gate.passed));
    assert!(comparisons[5..].iter().all(|gate| !gate.passed && gate.observed == Some(1.0)));
    assert_eq!(paired.candidate_worse, 6); assert_eq!(paired.rows, 6);
    assert_eq!(report.body.total_model_work, candidate.model_work.checked_add(baseline.model_work).unwrap());
    assert_eq!(report.body.host_scope_evidence, "caller_declared"); assert!(!report.body.activation_authority);
}

#[test]
fn undefined_macro_f1_is_a_failed_gate_and_tiny_samples_report_insufficient_data() {
    let data = dataset(&[false; 6], policy()); let replay = replay(&data, &[0.1; 6], "native-a");
    let mut options = options(); options.acceptance_threshold = 0.0;
    let report = compare(&replay, &replay, &key(), "scope", "host", gates(), options, Limits::default()).unwrap();
    let Decision::Rejected { comparisons, .. } = &report.body.decision else { panic!("absent class F1 cannot pass") };
    assert!(comparisons[..6].iter().all(|gate| gate.passed));
    assert_eq!(comparisons[6].observed, None); assert!(!comparisons[6].passed);
    options.minimum_samples = 20;
    let small = compare(&replay, &replay, &key(), "scope", "host", gates(), options, Limits::default()).unwrap();
    assert!(matches!(small.body.decision, Decision::InsufficientData { rows: 6, minimum_samples: 20 }));
    assert!(small.body.candidate.body.classification.is_none());
    assert!(matches!(small.body.candidate.body.measurement, Measurement::InsufficientData { .. }));
}

#[test]
fn full_identity_collection_and_task_budgets_bind_every_row_without_exporting_private_material() {
    let data = dataset(&[true, false, true, false, true, false], policy()); let probabilities = [0.8; 6];
    let original = replay(&data, &probabilities, "native-a");
    for field in 0..4 {
        let changed = replay_changed(&data, &probabilities, "native-a", |index, identity, budget, _| {
            if index == data.len() - 1 { match field {
                0 => identity.prompt_digest = Sha256Digest::of_bytes(b"changed final prompt"),
                1 => identity.taskir_digest = Sha256Digest::of_bytes(b"changed final task"),
                2 => identity.decision_policy_digest = Sha256Digest::of_bytes(b"changed policy"),
                _ => budget.max_output_bytes += 1,
            }}
        });
        assert_ne!(original.executions, changed.executions);
        assert_eq!(compare(&changed, &original, &key(), "scope", "host", gates(), options(), Limits::default()).err(),
            Some(AssayError::Scope));
    }
    let different_model = replay(&data, &probabilities, "native-b");
    assert_ne!(original.executions, different_model.executions);
    assert_eq!(original.compatible_executions, different_model.compatible_executions);
    assert!(compare(&different_model, &original, &key(), "scope", "host", gates(), options(), Limits::default()).is_ok());
    let key = key(); let report = evaluate(&original, &key, "scope", options(), Limits::default()).unwrap();
    let encoded = String::from_utf8(crate::assay::render(&report, Limits::default()).unwrap()).unwrap();
    assert!(!encoded.contains("private-test")); assert!(!encoded.contains("PRIVATE_"));
    assert!(!encoded.contains(&data.manifest().expected_input_sha256.to_hex()));
    for row in data.rows() { assert!(!encoded.contains(&Sha256Digest::of_bytes(row.text.as_bytes()).to_hex())); }
    let body = crate::assay::render(&report.body, Limits::default()).unwrap();
    assert_eq!(report.seal, key.commit(CommitmentDomain::Configuration, "scope", "native_binary_evaluation_v1", &body).unwrap());
    let renamed = evaluate(&original, &key, "other-scope", options(), Limits::default()).unwrap();
    assert_ne!(report.body.bindings.execution_collection, renamed.body.bindings.execution_collection);
    assert_ne!(report.body.bindings.observation_collection, renamed.body.bindings.observation_collection);
}

#[test]
fn final_row_score_and_native_decision_are_bound_separately_from_complete_execution_identities() {
    let data = dataset(&[true, false, true, false, true, false], policy());
    let original = replay(&data, &[0.8; 6], "native-a");
    let changed = replay_changed(&data, &[0.8; 6], "native-a", |index, _, _, output| {
        if index == data.len() - 1 { *output = run(data.manifest(), 0.2, 128); }
    });
    assert_eq!(original.executions, changed.executions);
    assert_eq!(original.compatible_executions, changed.compatible_executions);
    assert_ne!(original.observations, changed.observations);
    assert_ne!(original.rows.last().unwrap().predicted, changed.rows.last().unwrap().predicted);
    let key = key();
    let first = evaluate(&original, &key, "scope", options(), Limits::default()).unwrap();
    let second = evaluate(&changed, &key, "scope", options(), Limits::default()).unwrap();
    assert_eq!(first.body.bindings.execution_collection, second.body.bindings.execution_collection);
    assert_ne!(first.body.bindings.observation_collection, second.body.bindings.observation_collection);
    assert_eq!(first.body.bindings.observation_collection_rows, data.len());
    assert_eq!(first.body.bindings.observation_collection_format, OBSERVATION_COLLECTION_FORMAT);
    assert_eq!(first.body.bindings.observation_collection,
        key.commit(CommitmentDomain::Output, "scope", "native_observation_collection_sha256_binary_v1",
            &original.observations).unwrap());
    let mut incomplete = ReplayBuilder::new(&data, Limits::default()).unwrap();
    for (index, row) in data.rows().iter().enumerate().take(data.len() - 1) {
        incomplete.accept_fixture(index, &identity(row, "native-a"), budget(), work(128), 128,
            run(data.manifest(), 0.8, 128)).unwrap();
    }
    assert_eq!(incomplete.finish().err(), Some(AssayError::Observation));
}

#[test]
fn paired_replay_refuses_changed_gold_or_input_population_and_combines_bootstrap_budget() {
    let data = dataset(&[true, false, true, false, true, false], policy());
    let changed = dataset(&[false, false, true, false, true, false], policy());
    let a = replay(&data, &[0.8; 6], "native-a"); let b = replay(&changed, &[0.8; 6], "native-b");
    assert_eq!(compare(&a, &b, &key(), "scope", "host", gates(), options(), Limits::default()).err(), Some(AssayError::Scope));
    let mut limit = Limits::default(); limit.max_bootstrap_draws = 6 * options().bootstrap.resamples as u64;
    assert!(evaluate(&a, &key(), "scope", options(), limit).is_ok());
    assert_eq!(compare(&a, &a, &key(), "scope", "host", gates(), options(), limit).err(), Some(AssayError::InputLimit));
}

#[derive(Debug, Eq, PartialEq)]
enum ControlledError { Stopped, Assay, Calibration }
impl From<AssayError> for ControlledError { fn from(_: AssayError) -> Self { Self::Assay } }
impl From<CalibrationError> for ControlledError { fn from(_: CalibrationError) -> Self { Self::Calibration } }
#[test]
fn controlled_report_retains_cancellation_during_early_middle_and_late_finalization() {
    let data = dataset(&[true, false, true, false, true, false], policy()); let a = replay(&data, &[0.8; 6], "native-a");
    let key = key(); let mut calls = 0;
    compare_with_checkpoint::<ControlledError>(&a, &a, &key, "scope", "host", gates(), options(), Limits::default(),
        &mut || { calls += 1; Ok(()) }).unwrap();
    assert!(calls > 10);
    for at in [0, calls / 2, calls - 1] {
        let mut current = 0;
        let result = compare_with_checkpoint::<ControlledError>(&a, &a, &key, "scope", "host", gates(), options(), Limits::default(),
            &mut || { let turn = current; current += 1; if turn == at { Err(ControlledError::Stopped) } else { Ok(()) } });
        assert_eq!(result.err(), Some(ControlledError::Stopped));
        assert_eq!(current, at + 1);
    }
}

#[test]
fn strict_native_gates_output_bounds_and_replay_memory_quote_are_enforced() {
    let gates = gates(); let bytes = serde_json::to_vec(&gates).unwrap();
    assert_eq!(parse_gates(&bytes, Limits::default()).unwrap(), gates);
    let mut value = serde_json::to_value(gates).unwrap(); value.as_object_mut().unwrap().remove("max_macro_f1_decrease");
    assert!(parse_gates(&serde_json::to_vec(&value).unwrap(), Limits::default()).is_err());
    let mut invalid = gates; invalid.max_accuracy_decrease = f64::NAN;
    assert_eq!(invalid.validate(Limits::default()), Err(AssayError::Qualification));
    let one = required_replay_bytes(Limits::default(), 1).unwrap();
    let two = required_replay_bytes(Limits::default(), 2).unwrap();
    assert!(two > one); assert!(two < one * 2);
    for arms in [0, 3, usize::MAX] { assert_eq!(required_replay_bytes(Limits::default(), arms), Err(AssayError::Limits)); }
    let data = dataset(&[true, false, true, false, true, false], policy()); let a = replay(&data, &[0.8; 6], "native-a");
    let mut tight = Limits::default(); tight.max_output_bytes = 64;
    assert_eq!(evaluate(&a, &key(), "scope", options(), tight).err(), Some(AssayError::Output));
    assert_eq!(evaluate(&a, &key(), "bad scope", options(), Limits::default()).err(), Some(AssayError::Commitment));
}
