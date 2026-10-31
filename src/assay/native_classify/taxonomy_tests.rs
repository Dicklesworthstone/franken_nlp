//! Closed-collector fixtures for exclusive multiclass and independent labels.
use super::*;
use super::tests::{budget, encode, identity, work};
use crate::{execution_identity::Sha256Digest,
    native_engine::{lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{CandidateScore, CandidateScores, ScoringWork}},
        strict_int4::{Int4Work, STRICT_INT4_PROFILE}},
    tasks::{classify::{ClassificationCalibration, ClassificationDecision, ClassificationLabel,
        ClassificationMode, ClassificationPolicy, ClassificationResult, ClassificationTaskResult,
        MultiLabelDecision, MultiLabelLabelResult, MultiLabelResult,
        grouped_int4::{Int4ClassificationRun, INT4_CLASSIFICATION_EXECUTION}}, ir::ScoreSpace}};

pub(super) fn vector_wire(kind: TaskKind, gold: &[Vec<usize>]) -> (SuiteManifest, Vec<Row>) {
    assert_ne!(kind, TaskKind::Binary);
    let ids = ["alpha", "beta", "gamma"];
    let make = |id: String, split, gold: &[usize]| Row { schema_version: 1,
        text: format!("PRIVATE_VECTOR_SOURCE_{id} é 中文"), id, split, positive: None,
        label: (kind == TaskKind::Multiclass).then(|| ids[gold[0]].to_owned()),
        labels: (kind == TaskKind::Multilabel).then(|| gold.iter().map(|&i| ids[i].to_owned()).collect()) };
    let mut rows = vec![make("vector-development".to_owned(), Split::Development, &[0]),
        make("vector-calibration".to_owned(), Split::Calibration, &[1])];
    let mut locked = Vec::new();
    for (i, gold) in gold.iter().enumerate() {
        let id = format!("vector-test-{i:03}"); locked.push(id.clone());
        rows.push(make(id, Split::LockedTest, gold));
    }
    let manifest = SuiteManifest { schema_version: 1,
        task: if kind == TaskKind::Multiclass { MULTICLASS_TASK } else { MULTILABEL_TASK }.to_owned(),
        dataset_id: "vector-suite".to_owned(), dataset_version: "v1".to_owned(),
        named_population: "Owned multilingual vector fixture".to_owned(), positive_label: None, negative_label: None,
        labels: ids.into_iter().map(|id| ClassificationLabel { id: id.to_owned(),
            description: format!("Description of {id}") }).collect(), policy: ClassificationPolicy::default(),
        expected_input_sha256: Sha256Digest::of_bytes(b"unbound"), splits: SplitIds {
            development: vec![rows[0].id.clone()], calibration: vec![rows[1].id.clone()], locked_test: locked } };
    (manifest, rows)
}

pub(super) fn vector_dataset(kind: TaskKind, gold: &[Vec<usize>]) -> Dataset {
    let (manifest, rows) = vector_wire(kind, gold); let (manifest, input) = encode(manifest, &rows);
    Dataset::parse(&manifest, &input, Limits::default()).unwrap()
}

fn classification(ids: &[&str], weights: &[f64], policy: ClassificationPolicy) -> ClassificationResult {
    let mut candidates: Vec<_> = ids.iter().zip(weights).map(|(id, &weight)| CandidateScore {
        id: (*id).to_owned(), scored_tokens: 2, candidate_weight: weight,
        sequence_score: if weight == 0.0 { -1000.0 } else { weight.ln() } }).collect();
    candidates.sort_by(|a, b| a.id.cmp(&b.id));
    let mut order: Vec<_> = candidates.iter().collect();
    order.sort_by(|a, b| if a.sequence_score == b.sequence_score { a.id.cmp(&b.id) }
        else { b.sequence_score.total_cmp(&a.sequence_score) });
    let best = order[0]; let margin = (best.candidate_weight - order[1].candidate_weight).max(0.0);
    let accepted = best.candidate_weight * 1_000_000.0 >= f64::from(policy.minimum_candidate_weight_ppm)
        && margin * 1_000_000.0 >= f64::from(policy.minimum_margin_ppm);
    let selected_id = accepted.then(|| best.id.clone()); let best_candidate_weight = best.candidate_weight;
    let ranking = order.iter().map(|candidate| candidate.id.clone()).collect();
    drop(order);
    let eos_token_id = crate::tokenizer::pinned_controls::pinned().unwrap().template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    ClassificationResult { schema_version: 1, decision: if accepted { ClassificationDecision::Classified }
        else { ClassificationDecision::Abstained }, selected_id, ranking, best_candidate_weight,
        candidate_weight_margin: margin, calibration: ClassificationCalibration::Uncalibrated, policy,
        scores: CandidateScores { score_space: ScoreSpace::FullVocabSequenceLogprob,
            normalization_scope: "full_vocabulary_per_position; candidate_weights_over_completed_candidates".to_owned(),
            length_rule: "sum_logprobs_including_eos".to_owned(), eos_rule: "append_and_score_exactly_one_eos".to_owned(),
            eos_token_id, full_vocab_denominators_computed: true, candidates,
            work: ScoringWork { prefix_evaluations: 3, scored_edges: 4, projected_logits: 3 * NANBEIGE_VOCAB_SIZE as u64 } } }
}

pub(super) fn vector_run(manifest: &SuiteManifest, weights: &[f64], width: usize) -> Int4ClassificationRun {
    let result = if manifest.task_kind().unwrap() == TaskKind::Multilabel {
        let mut labels = Vec::new(); let mut selected_ids = Vec::new(); let mut excluded_ids = Vec::new();
        let mut abstained_ids = Vec::new();
        for (label, &p) in manifest.labels().zip(weights) {
            let classification = classification(&["no", "yes"], &[1.0 - p, p], manifest.policy);
            let (decision, target) = match classification.selected_id.as_deref() {
                Some("yes") => (MultiLabelDecision::Included, &mut selected_ids),
                Some("no") => (MultiLabelDecision::Excluded, &mut excluded_ids),
                None => (MultiLabelDecision::Abstained, &mut abstained_ids), _ => unreachable!(),
            };
            target.push(label.id.clone());
            labels.push(MultiLabelLabelResult { label_id: label.id.clone(), decision,
                positive_candidate_weight: p, classification });
        }
        ClassificationTaskResult::MultiLabel(MultiLabelResult { schema_version: 1,
            decision_scope: "independent_binary_heads_not_a_distribution_over_labels".to_owned(),
            calibration: ClassificationCalibration::Uncalibrated, selected_ids, excluded_ids, abstained_ids, labels })
    } else {
        let ids: Vec<_> = manifest.labels().map(|label| label.id.as_str()).collect();
        ClassificationTaskResult::Exclusive(classification(&ids, weights, manifest.policy))
    };
    let mut model_work = Int4Work::default();
    for _ in 0..manifest.head_count() { model_work = model_work.checked_add(work(width)).unwrap(); }
    Int4ClassificationRun { schema_version: 1, execution: INT4_CLASSIFICATION_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: width, head_count: manifest.head_count(),
        model_work, rewound_positions: manifest.head_count() as u64, result }
}

pub(super) fn vector_replay(dataset: &Dataset, locked_weights: &[Vec<f64>], subject: &str) -> NativeReplay {
    let mut builder = replay::ReplayBuilder::new(dataset, Limits::default()).unwrap();
    let mut next = 0;
    for (index, row) in dataset.rows().iter().enumerate() {
        let weights = if row.split == Split::LockedTest {
            let weights = &locked_weights[next]; next += 1; weights.as_slice()
        } else { &[0.2, 0.3, 0.5] };
        let run = vector_run(dataset.manifest(), weights, 128);
        let planned = run.model_work;
        builder.accept_fixture(index, &identity(row, subject), budget(), planned, 128, run).unwrap();
    }
    builder.finish().unwrap()
}

#[test]
fn taxonomy_wire_is_strict_and_binary_serialization_keeps_its_existing_shape() {
    let (manifest, rows) = tests::wire(&[true], ClassificationPolicy::default());
    let declared = serde_json::to_value(&manifest).unwrap();
    assert!(declared.get("labels").is_none());
    let (bytes, input) = encode(manifest, &rows);
    let mut foreign: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    foreign["labels"] = serde_json::json!([]);
    assert_eq!(Dataset::parse(&serde_json::to_vec(&foreign).unwrap(), &input, Limits::default()).err(),
        Some(AssayError::Manifest));
    for row in &rows {
        let value = serde_json::to_value(row).unwrap();
        assert!(value.get("label").is_none() && value.get("labels").is_none());
    }
    for kind in [TaskKind::Multiclass, TaskKind::Multilabel] {
        let (manifest, rows) = vector_wire(kind, &[vec![0], vec![1], vec![2]]);
        for mutation in 0..5 {
            let mut altered = rows.clone();
            match mutation {
                0 => altered[0].positive = Some(false),
                1 => if kind == TaskKind::Multiclass { altered[0].label = Some("unknown".to_owned()); }
                    else { altered[0].labels = Some(vec!["unknown".to_owned()]); },
                2 => if kind == TaskKind::Multiclass { altered[0].labels = Some(vec![]); }
                    else { altered[0].label = Some("alpha".to_owned()); },
                3 => { altered[0].label = None; altered[0].labels = None; },
                _ => if kind == TaskKind::Multiclass { altered[0].label = Some(String::new()); }
                    else { altered[0].labels = Some(vec!["alpha".to_owned(), "alpha".to_owned()]); },
            }
            let (manifest, input) = encode(manifest.clone(), &altered);
            assert_eq!(Dataset::parse(&manifest, &input, Limits::default()).err(), Some(AssayError::Observation));
        }
        let mut values: Vec<_> = rows.iter().map(|row| serde_json::to_value(row).unwrap()).collect();
        values[0].as_object_mut().unwrap().insert("positive".to_owned(), serde_json::Value::Null);
        let input = values.iter().map(|row| serde_json::to_string(row).unwrap()).collect::<Vec<_>>().join("\n");
        let mut declared = manifest; declared.expected_input_sha256 = Sha256Digest::of_bytes(input.as_bytes());
        assert_eq!(Dataset::parse(&serde_json::to_vec(&declared).unwrap(), input.as_bytes(), Limits::default()).err(),
            Some(AssayError::Json));
    }
}

#[test]
fn label_sets_are_bounded_unique_canonical_and_never_leak_gold_into_native_requests() {
    for kind in [TaskKind::Multiclass, TaskKind::Multilabel] {
        let (mut manifest, rows) = vector_wire(kind, &[vec![0], vec![1], vec![2]]);
        manifest.labels.reverse();
        let (bytes, input) = encode(manifest.clone(), &rows);
        let dataset = Dataset::parse(&bytes, &input, Limits::default()).unwrap();
        assert_eq!(dataset.task_kind(), kind); assert_eq!(dataset.label_count(), 3);
        assert_eq!(dataset.head_count(), if kind == TaskKind::Multiclass { 1 } else { 3 });
        assert!(!dataset.is_binary());
        assert_eq!(dataset.manifest().labels().map(|label| label.id.as_str()).collect::<Vec<_>>(),
            ["alpha", "beta", "gamma"]);
        for (index, row) in dataset.rows().iter().enumerate() {
            let request = dataset.classification_request(index, budget()).unwrap();
            assert_eq!(request.document, row.text);
            assert_eq!(request.mode, if kind == TaskKind::Multiclass { ClassificationMode::Exclusive }
                else { ClassificationMode::MultiLabel });
            let value = serde_json::to_value(request).unwrap();
            for field in ["positive", "label", "split", "id"] { assert!(value.get(field).is_none()); }
        }
        manifest.labels[1].id = manifest.labels[0].id.clone();
        let (bytes, input) = encode(manifest, &rows);
        assert_eq!(Dataset::parse(&bytes, &input, Limits::default()).err(), Some(AssayError::Manifest));
    }
    let (mut manifest, rows) = vector_wire(TaskKind::Multilabel, &[vec![]]);
    let (bytes, input) = encode(manifest.clone(), &rows);
    assert!(Dataset::parse(&bytes, &input, Limits::default()).is_ok());
    manifest.labels = (0..=MAX_LABELS).map(|i| ClassificationLabel { id: format!("label-{i:03}"),
        description: String::new() }).collect();
    let (bytes, input) = encode(manifest, &rows);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default()).err(), Some(AssayError::Manifest));
}

#[test]
fn collector_retains_complete_exclusive_vectors_and_independent_binary_heads() {
    let multiclass = vector_dataset(TaskKind::Multiclass, &[vec![2], vec![0]]);
    let replay = vector_replay(&multiclass, &[vec![0.2, 0.3, 0.5], vec![0.6, 0.3, 0.1]], "model");
    let rows: Vec<_> = replay.rows.iter().filter(|row| row.split == Split::LockedTest).collect();
    assert_eq!(rows[0].positive, [false, false, true]);
    assert_eq!(rows[0].probability, [0.2, 0.3, 0.5]); assert_eq!(rows[0].top_label, Some(2));
    assert_eq!(rows[0].predicted, [Some(false), Some(false), Some(true)]);
    let multilabel = vector_dataset(TaskKind::Multilabel, &[vec![0, 1], vec![]]);
    let replay = vector_replay(&multilabel, &[vec![0.9, 0.9, 0.1], vec![0.1, 0.1, 0.1]], "model");
    let rows: Vec<_> = replay.rows.iter().filter(|row| row.split == Split::LockedTest).collect();
    assert_eq!(rows[0].positive, [true, true, false]);
    assert_eq!(rows[0].probability, [0.9, 0.9, 0.1]); // Sum 1.9 is valid across independent heads.
    assert_eq!(rows[0].predicted, [Some(true), Some(true), Some(false)]); assert_eq!(rows[0].top_label, None);
    assert!(replay.partition::<AssayError>(&mut || Ok(())).is_err());
}

#[test]
fn independent_head_truncation_reordering_false_decisions_and_work_mismatches_poison_the_collector() {
    let dataset = vector_dataset(TaskKind::Multilabel, &[vec![0, 1]]);
    for mutation in 0..8 {
        let mut builder = replay::ReplayBuilder::new(&dataset, Limits::default()).unwrap();
        let row = &dataset.rows()[0]; let id = identity(row, "model");
        let mut run = vector_run(dataset.manifest(), &[0.9, 0.9, 0.1], 128); let planned = run.model_work;
        let ClassificationTaskResult::MultiLabel(result) = &mut run.result else { unreachable!() };
        match mutation {
            0 => { result.labels.pop(); },
            1 => result.labels.swap(0, 1),
            2 => result.selected_ids.reverse(),
            3 => result.labels[0].positive_candidate_weight = 0.1,
            4 => result.labels[0].decision = MultiLabelDecision::Excluded,
            5 => result.labels[0].classification.scores.work.projected_logits += NANBEIGE_VOCAB_SIZE as u64,
            6 => run.head_count = 1,
            _ => result.labels[2].classification.scores.eos_token_id += 1,
        }
        assert!(builder.accept_fixture(0, &id, budget(), planned, 128, run).is_err(), "mutation {mutation}");
        let retry = vector_run(dataset.manifest(), &[0.9, 0.9, 0.1], 128);
        assert!(builder.accept_fixture(0, &id, budget(), planned, 128, retry).is_err());
        assert_eq!(builder.finish().err(), Some(AssayError::Observation));
    }
}

#[test]
fn exclusive_abstention_retains_raw_argmax_and_vector_bindings_cover_every_label() {
    let (mut manifest, rows) = vector_wire(TaskKind::Multiclass, &[vec![2], vec![0]]);
    manifest.policy.minimum_candidate_weight_ppm = 700_000;
    let (bytes, input) = encode(manifest, &rows);
    let dataset = Dataset::parse(&bytes, &input, Limits::default()).unwrap();
    let a = vector_replay(&dataset, &[vec![0.2, 0.3, 0.5], vec![0.6, 0.3, 0.1]], "model");
    let b = vector_replay(&dataset, &[vec![0.2, 0.3, 0.5], vec![0.6, 0.1, 0.3]], "model");
    assert_eq!(a.executions, b.executions); assert_ne!(a.observations, b.observations);
    for row in a.rows.iter().filter(|row| row.split == Split::LockedTest) {
        assert!(row.predicted.iter().all(Option::is_none)); assert!(row.top_label.is_some());
    }
    let small = required_replay_bytes_for_labels(Limits::default(), 1, 2).unwrap();
    let large = required_replay_bytes_for_labels(Limits::default(), 1, MAX_LABELS).unwrap();
    assert_eq!(large - small, Limits::default().max_rows as u64 * 32 * (MAX_LABELS - 2) as u64);
    for labels in [0, MAX_LABELS + 1, usize::MAX] {
        assert_eq!(required_replay_bytes_for_labels(Limits::default(), 1, labels), Err(AssayError::Limits));
    }
}
