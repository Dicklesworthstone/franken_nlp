//! Model-free parser/collector fixtures; native execution remains model-gated.
use super::*;
use crate::{execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{constrained_int4, strict_int4::{STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE}},
    tasks::{extract::ExtractionGrounding, ir::{ScoreSpace, TaskBudget},
        ner::{EntityType, NamedEntity, NerOptions, NerResult, NER_TASK_VERSION},
        source_planning::{SourceTaskRequest, SourceTaskResult,
            grouped_int4::{Int4SourceTaskRun, INT4_SOURCE_EXECUTION}}},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, scan_occurrences}};
use replay::ReplayBuilder;

pub(super) fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 128, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
pub(super) fn wire(locked: usize) -> (SuiteManifest, Vec<serde_json::Value>) {
    let mut rows = vec![
        serde_json::json!({"schema_version":1,"id":"private-development","split":"development",
            "text":"PRIVATE_DEVELOPMENT é","language":"und","spans":[]}),
        serde_json::json!({"schema_version":1,"id":"private-calibration","split":"calibration",
            "text":"PRIVATE_CALIBRATION 中文","language":"und","spans":[]})];
    let mut locked_ids = Vec::new();
    for index in 0..locked {
        let id = format!("private-test-{index:03}"); locked_ids.push(id.clone());
        rows.push(serde_json::json!({"schema_version":1,"id":id,"split":"locked_test",
            "text":format!("é上海 and Ana PRIVATE_{index}"),"language":if index % 2 == 0 { "zh-Hans" } else { "en" },
            "spans":[{"entity_type":"location","byte_start":2,"byte_end":8},
                {"entity_type":"person","byte_start":13,"byte_end":16}]}));
    }
    (SuiteManifest { schema_version: 1, task: NATIVE_TASK.to_owned(), dataset_id: "ner-suite".to_owned(),
        dataset_version: "v1".to_owned(), named_population: "Owned NER fixture population".to_owned(),
        options: NerOptions { types: vec![EntityType::Person, EntityType::Location], max_entities: 8,
            max_mention_scalars: 256 }, expected_input_sha256: Sha256Digest::of_bytes(b"unbound"),
        splits: SplitIds { development: vec!["private-development".to_owned()],
            calibration: vec!["private-calibration".to_owned()], locked_test: locked_ids } }, rows)
}
pub(super) fn encode(mut manifest: SuiteManifest, rows: &[serde_json::Value]) -> (Vec<u8>, Vec<u8>) {
    let mut input = Vec::new();
    for row in rows { input.extend(serde_json::to_vec(row).unwrap()); input.push(b'\n'); }
    manifest.expected_input_sha256 = Sha256Digest::of_bytes(&input);
    (serde_json::to_vec(&manifest).unwrap(), input)
}
pub(super) fn dataset(locked: usize) -> Dataset {
    let (manifest, rows) = wire(locked); let (manifest, input) = encode(manifest, &rows);
    Dataset::parse(&manifest, &input, Limits::default(), NerLimits::default()).unwrap()
}
pub(super) fn identity(row: &Row, subject: &str) -> ExecutionIdentity {
    let public = Sha256Digest::of_bytes(b"public pinned fixture assets");
    let private = Sha256Digest::of_bytes(row.text.as_bytes());
    ExecutionIdentity { schema_version: 1, source_revision: "fixture-pinned-revision".to_owned(),
        logical_model_digest: Sha256Digest::of_bytes(subject.as_bytes()), artifact_format: "fixture-int4".to_owned(),
        quant_recipe: subject.to_owned(), packing_set_digest: Sha256Digest::of_bytes(subject.as_bytes()),
        tokenizer_digest: public, template_digest: public, task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: private, prompt_digest: private, grammar_compiler_version: "source-fixture-v1".to_owned(),
        schema_digest: public, numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture-full-vocabulary".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: public, decision_policy_digest: public,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn eos() -> u32 {
    crate::tokenizer::pinned_controls::pinned().unwrap().template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id
}
pub(super) fn run(row: &Row) -> Int4SourceTaskRun {
    let generated_token_ids = vec![1, 2, eos()];
    let model_work = constrained_int4::planned_work(128, 8, generated_token_ids.len()).unwrap();
    let entities = row.spans.iter().map(|span| NamedEntity {
        text: row.text[span.byte_start..span.byte_end].to_owned(), entity_type: span.entity_type,
        occurrence: SourceOccurrence::Anchored, spans: vec![span.source_span()],
    }).collect();
    Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: 128,
        result: SourceTaskResult::Ner(NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), score_space: ScoreSpace::NotComputed,
            grounding: ExtractionGrounding::SourceMembership, entities, generated_token_ids,
            forward_positions: model_work.forward_positions, projected_logits: model_work.projected_logits,
            mask_node_visit_charge: 7 }), model_work }
}
pub(super) fn native_steps(row: &Row, run: &Int4SourceTaskRun) -> u64 {
    let SourceTaskResult::Ner(result) = &run.result else { panic!("fixture task"); };
    result.entities.iter().map(|entity| 16 * (row.text.len() as u64 + entity.text.len() as u64 + 1)).sum()
}
pub(super) fn verification() -> GroundingBudget {
    GroundingBudget { max_fields: 100_000, max_matches: 1_000_000, max_scan_steps: NerLimits::default().max_verification_steps }
}
pub(super) fn replay(dataset: &Dataset, subject: &str) -> NativeReplay {
    replay_changed(dataset, subject, |_, _, _| {})
}
pub(super) fn replay_changed(dataset: &Dataset, subject: &str,
    mut change: impl FnMut(usize, &mut ExecutionIdentity, &mut Int4SourceTaskRun)) -> NativeReplay {
    let mut builder = ReplayBuilder::new(dataset, Limits::default(), NerLimits::default()).unwrap();
    let mut verification = verification();
    for (index, row) in dataset.replay_rows().iter().enumerate() {
        let mut identity = identity(row, subject); let mut output = run(row); change(index, &mut identity, &mut output);
        let native_steps = native_steps(row, &output);
        verification.max_scan_steps -= native_steps;
        builder.accept_fixture(index, &identity, budget(), 8,
            constrained_int4::planned_work(128, 8, budget().max_output_tokens as usize).unwrap(), 128,
            output, native_steps, &mut verification, &mut || Ok::<(), AssayError>(())).unwrap();
    }
    builder.finish().unwrap()
}

#[test]
fn gold_coordinates_are_computed_once_and_only_locked_rows_reach_native_requests() {
    let data = dataset(5);
    assert_eq!(data.len(), 7); assert_eq!(data.replay_len(), 5);
    assert_eq!(data.manifest().options.types, vec![EntityType::Location, EntityType::Person]);
    assert!(data.replay_rows().windows(2).all(|pair| pair[0].id < pair[1].id));
    for (index, row) in data.replay_rows().iter().enumerate() {
        assert_eq!(row.split, Split::LockedTest);
        let location = row.spans[0].source_span();
        assert_eq!((location.byte_start, location.byte_end, location.scalar_start, location.scalar_end), (2, 8, 1, 3));
        let person = row.spans[1].source_span();
        assert_eq!((person.byte_start, person.byte_end, person.scalar_start, person.scalar_end), (13, 16, 8, 11));
        let request = data.source_request(index, budget()).unwrap();
        let encoded = crate::canonjson::canonical_bytes(&request).unwrap();
        assert_eq!(data.request_digest(index, budget()).unwrap(), Sha256Digest::of_bytes(&encoded));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 4);
        assert_eq!(value["task"], "ner"); assert_eq!(value["document"], row.text);
        for absent in ["id", "split", "language", "spans", "gold"] { assert!(value.get(absent).is_none()); }
        let SourceTaskRequest::Ner { document, options, budget: actual_budget } = request else { panic!("NER request"); };
        assert_eq!(document, row.text); assert_eq!(options, data.manifest().options); assert_eq!(actual_budget, budget());
    }
    assert!(data.source_request(data.replay_len(), budget()).is_err());
}

#[test]
fn gold_refuses_non_boundaries_duplicate_intervals_and_unknown_types_but_allows_overlap() {
    let (manifest, rows) = wire(1);
    for span in [
        serde_json::json!({"entity_type":"location","byte_start":3,"byte_end":8}),
        serde_json::json!({"entity_type":"location","byte_start":2,"byte_end":7}),
        serde_json::json!({"entity_type":"location","byte_start":2,"byte_end":2}),
        serde_json::json!({"entity_type":"location","byte_start":8,"byte_end":2}),
        serde_json::json!({"entity_type":"location","byte_start":2,"byte_end":999}),
        serde_json::json!({"entity_type":"money","byte_start":2,"byte_end":8}),
    ] {
        let mut changed = rows.clone(); changed[2]["spans"][0] = span;
        let (bytes, input) = encode(manifest.clone(), &changed);
        assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::Observation));
    }
    for kind in ["location", "person"] {
        let mut changed = rows.clone(); changed[2]["spans"].as_array_mut().unwrap().push(
            serde_json::json!({"entity_type":kind,"byte_start":2,"byte_end":8}));
        let (bytes, input) = encode(manifest.clone(), &changed);
        assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::Observation));
    }
    let mut nested = rows;
    nested[2]["spans"].as_array_mut().unwrap().push(serde_json::json!({"entity_type":"location","byte_start":2,"byte_end":5}));
    let (bytes, input) = encode(manifest, &nested);
    assert!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).is_ok());
}

#[test]
fn manifest_and_gold_caps_precede_typed_gold_allocation() {
    let (mut manifest, rows) = wire(1);
    let invalid = b"not-json\nnot-json\nnot-json\n";
    manifest.expected_input_sha256 = Sha256Digest::of_bytes(invalid);
    let cap = NerLimits { max_spans_per_row: 2, ..NerLimits::default() };
    assert_eq!(Dataset::parse(&serde_json::to_vec(&manifest).unwrap(), invalid, Limits::default(), cap).err(),
        Some(AssayError::InputLimit));
    manifest.options.max_entities = 2;
    let mut changed = rows; changed[2]["spans"] = serde_json::json!([null, null, null]);
    let (bytes, input) = encode(manifest.clone(), &changed);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), cap).err(), Some(AssayError::InputLimit));
    changed[2]["spans"] = serde_json::json!([null]);
    let (bytes, input) = encode(manifest, &changed);
    let language_cap = NerLimits { max_languages: 1, ..NerLimits::default() };
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), language_cap).err(), Some(AssayError::InputLimit));
}

#[test]
fn source_pins_global_units_and_complete_split_membership_are_enforced() {
    let (manifest, rows) = wire(2); let (bytes, mut input) = encode(manifest.clone(), &rows);
    input.push(b' ');
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::InputDigest));
    let mut changed = rows.clone(); changed[2]["text"] = changed[0]["text"].clone(); changed[2]["spans"] = serde_json::json!([]);
    let (bytes, input) = encode(manifest.clone(), &changed);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::DuplicateUnit));
    changed = rows.clone(); changed[2]["split"] = serde_json::json!("calibration");
    let (bytes, input) = encode(manifest.clone(), &changed);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::SplitMembership));
    let mut overlapping = manifest; overlapping.splits.development.push(overlapping.splits.locked_test[0].clone());
    let (bytes, input) = encode(overlapping, &rows);
    assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::SplitMembership));
}

#[test]
fn strict_ner_wire_refuses_supplied_scores_coordinates_and_duplicate_keys() {
    let (manifest, rows) = wire(1);
    for field in ["probability", "confidence", "execution_identity", "predictions", "unit_sha256"] {
        let mut changed = rows.clone(); changed[2].as_object_mut().unwrap().insert(field.to_owned(), serde_json::json!(0.9));
        let (bytes, input) = encode(manifest.clone(), &changed);
        assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::Json));
    }
    for field in ["scalar_start", "scalar_end", "text"] {
        let mut changed = rows.clone(); changed[2]["spans"][0].as_object_mut().unwrap().insert(field.to_owned(), serde_json::json!(1));
        let (bytes, input) = encode(manifest.clone(), &changed);
        assert_eq!(Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::Json));
    }
    let (bytes, input) = encode(manifest, &rows);
    let duplicated = String::from_utf8(bytes).unwrap().replacen("\"schema_version\":1", "\"schema_version\":1,\"schema_version\":1", 1);
    assert_eq!(Dataset::parse(duplicated.as_bytes(), &input, Limits::default(), NerLimits::default()).err(), Some(AssayError::Json));
}

#[test]
fn physical_row_order_and_framing_are_separate_from_canonical_replay_order() {
    let (manifest, rows) = wire(5); let (bytes, input) = encode(manifest.clone(), &rows);
    let forward = Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).unwrap();
    let input = rows.iter().rev().map(|row| serde_json::to_string(row).unwrap()).collect::<Vec<_>>().join("\r\n");
    let mut reverse_manifest = manifest; reverse_manifest.expected_input_sha256 = Sha256Digest::of_bytes(input.as_bytes());
    let reverse = Dataset::parse(&serde_json::to_vec(&reverse_manifest).unwrap(), input.as_bytes(), Limits::default(), NerLimits::default()).unwrap();
    assert_eq!(serde_json::to_value(forward.rows()).unwrap(), serde_json::to_value(reverse.rows()).unwrap());
    let a = replay(&forward, "fixture"); let b = replay(&reverse, "fixture");
    assert_eq!(a.executions, b.executions); assert_eq!(a.observations, b.observations);
    assert_ne!(a.input_digest, b.input_digest);
}

fn first_output(data: &Dataset, output: Int4SourceTaskRun) -> Result<NativeReplay, AssayError> {
    let mut builder = ReplayBuilder::new(data, Limits::default(), NerLimits::default())?;
    let row = &data.replay_rows()[0]; let native_steps = native_steps(row, &output);
    let mut verification = verification(); verification.max_scan_steps -= native_steps;
    builder.accept_fixture(0, &identity(row, "fixture"), budget(), 8,
        constrained_int4::planned_work(128, 8, budget().max_output_tokens as usize).unwrap(), 128,
        output, native_steps, &mut verification, &mut || Ok::<(), AssayError>(()))?;
    builder.finish()
}

#[test]
fn variable_native_work_uses_actual_generated_length_and_all_five_axes() {
    let data = dataset(1); let row = &data.replay_rows()[0];
    let perfect = first_output(&data, run(row)).unwrap();
    assert_eq!(perfect.model_work, constrained_int4::planned_work(128, 8, 3).unwrap());
    assert_ne!(perfect.model_work, constrained_int4::planned_work(128, 8, 128).unwrap());
    for axis in 0..5 {
        let mut output = run(row);
        match axis {
            0 => output.model_work.forward_positions += 1,
            1 => output.model_work.projected_logits += 1,
            2 => output.model_work.attention_pairs += 1,
            3 => output.model_work.projections.dot_products += 1,
            _ => output.model_work.projections.multiply_accumulates += 1,
        }
        assert_eq!(first_output(&data, output).err(), Some(AssayError::Observation));
    }
}

#[test]
fn native_result_requires_explicit_unscored_source_membership_and_terminal_eos() {
    let data = dataset(1); let row = &data.replay_rows()[0];
    for change in 0..6 {
        let mut output = run(row); let SourceTaskResult::Ner(result) = &mut output.result else { unreachable!(); };
        match change {
            0 => result.score_space = ScoreSpace::FullVocabSequenceLogprob,
            1 => result.task_spec_version = "extract-v1".to_owned(),
            2 => result.numerics_profile = "other-profile".to_owned(),
            3 => result.generated_token_ids[0] = eos(),
            4 => *result.generated_token_ids.last_mut().unwrap() = 2,
            _ => result.projected_logits += 1,
        }
        assert_eq!(first_output(&data, output).err(), Some(AssayError::Observation));
    }
}

#[test]
fn collector_rechecks_every_unicode_coordinate_and_complete_ambiguous_occurrence_set() {
    let data = dataset(1); let row = &data.replay_rows()[0];
    for coordinate in 0..4 {
        let mut output = run(row); let SourceTaskResult::Ner(result) = &mut output.result else { unreachable!(); };
        let span = &mut result.entities[0].spans[0];
        match coordinate { 0 => span.byte_start += 1, 1 => span.byte_end += 1,
            2 => span.scalar_start += 1, _ => span.scalar_end += 1 }
        assert_eq!(first_output(&data, output).err(), Some(AssayError::Observation));
    }
    let (manifest, mut rows) = wire(1); rows[2]["text"] = serde_json::json!("Ana Ana é PRIVATE_REPEAT");
    rows[2]["spans"] = serde_json::json!([
        {"entity_type":"person","byte_start":0,"byte_end":3},
        {"entity_type":"person","byte_start":4,"byte_end":7}]);
    let (bytes, input) = encode(manifest, &rows);
    let repeated = Dataset::parse(&bytes, &input, Limits::default(), NerLimits::default()).unwrap();
    let row = &repeated.replay_rows()[0]; let mut output = run(row);
    let SourceTaskResult::Ner(result) = &mut output.result else { unreachable!(); };
    let spans = scan_occurrences(&row.text, "Ana", &mut GroundingBudget::default()).unwrap();
    result.entities = vec![NamedEntity { text: "Ana".to_owned(), entity_type: EntityType::Person,
        occurrence: SourceOccurrence::Ambiguous, spans }];
    assert!(first_output(&repeated, output.clone()).is_ok());
    let SourceTaskResult::Ner(result) = &mut output.result else { unreachable!(); };
    result.entities[0].spans.pop(); result.entities[0].occurrence = SourceOccurrence::Anchored;
    assert_eq!(first_output(&repeated, output).err(), Some(AssayError::Observation));
}

#[test]
fn full_output_commitment_covers_duplicate_proposals_tokens_and_native_work() {
    let data = dataset(1); let row = &data.replay_rows()[0]; let a = first_output(&data, run(row)).unwrap();
    let mut duplicate = run(row); let SourceTaskResult::Ner(result) = &mut duplicate.result else { unreachable!(); };
    result.entities.push(result.entities[0].clone());
    let b = first_output(&data, duplicate).unwrap();
    assert_eq!(a.executions, b.executions); assert_ne!(a.observations, b.observations);
    assert_ne!(a.rows[0].counts, b.rows[0].counts);
    let mut token_change = run(row); let SourceTaskResult::Ner(result) = &mut token_change.result else { unreachable!(); };
    result.generated_token_ids[0] = 3;
    let c = first_output(&data, token_change).unwrap();
    assert_eq!(a.rows[0].counts, c.rows[0].counts); assert_ne!(a.observations, c.observations);
    assert_eq!(a.total_source_verification_steps, a.verification_steps * 3);
}

#[test]
fn verification_is_bounded_and_failed_collection_cannot_resume_or_finish() {
    let data = dataset(1); let row = &data.replay_rows()[0];
    let mut builder = ReplayBuilder::new(&data, Limits::default(), NerLimits::default()).unwrap();
    let output = run(row); let native_steps = native_steps(row, &output);
    let mut exhausted = GroundingBudget { max_scan_steps: 1, ..verification() };
    let maximum = constrained_int4::planned_work(128, 8, budget().max_output_tokens as usize).unwrap();
    assert_eq!(builder.accept_fixture(0, &identity(row, "fixture"), budget(), 8, maximum, 128,
        output, native_steps, &mut exhausted, &mut || Ok::<(), AssayError>(())).err(), Some(AssayError::InputLimit));
    assert_eq!(builder.accept_fixture(0, &identity(row, "fixture"), budget(), 8, maximum, 128,
        run(row), native_steps, &mut verification(), &mut || Ok::<(), AssayError>(())).err(), Some(AssayError::Observation));
    assert_eq!(builder.finish().err(), Some(AssayError::Observation));
}

#[test]
fn cancellation_survives_source_verification_and_poisons_the_collector() {
    #[derive(Debug, Eq, PartialEq)] enum Stop { Assay(AssayError), Cancelled }
    impl From<AssayError> for Stop { fn from(error: AssayError) -> Self { Self::Assay(error) } }
    let data = dataset(1); let row = &data.replay_rows()[0]; let output = run(row);
    let steps = native_steps(row, &output); let mut polls = 0;
    let mut builder = ReplayBuilder::new(&data, Limits::default(), NerLimits::default()).unwrap();
    let result = builder.accept_fixture(0, &identity(row, "fixture"), budget(), 8,
        constrained_int4::planned_work(128, 8, budget().max_output_tokens as usize).unwrap(), 128,
        output, steps, &mut verification(), &mut || { polls += 1;
            if polls == 3 { Err(Stop::Cancelled) } else { Ok(()) } });
    assert_eq!(result, Err(Stop::Cancelled)); assert_eq!(builder.finish().err(), Some(AssayError::Observation));
}

#[test]
fn native_scan_charge_is_checked_and_noncanonical_replay_indices_fail_closed() {
    let data = dataset(1); let row = &data.replay_rows()[0];
    let maximum = constrained_int4::planned_work(128, 8, budget().max_output_tokens as usize).unwrap();
    let mut builder = ReplayBuilder::new(&data, Limits::default(), NerLimits::default()).unwrap();
    assert_eq!(builder.accept_fixture(0, &identity(row, "fixture"), budget(), 8, maximum, 128,
        run(row), 0, &mut verification(), &mut || Ok::<(), AssayError>(())).err(), Some(AssayError::Observation));
    let mut out_of_order = ReplayBuilder::new(&data, Limits::default(), NerLimits::default()).unwrap();
    assert_eq!(out_of_order.accept_fixture(1, &identity(row, "fixture"), budget(), 8, maximum, 128,
        run(row), 0, &mut verification(), &mut || Ok::<(), AssayError>(())).err(), Some(AssayError::Scope));
    assert_eq!(out_of_order.finish().err(), Some(AssayError::Observation));
}
