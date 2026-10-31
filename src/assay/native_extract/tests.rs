//! Model-free contract/tamper fixtures. Actual native quality stays model-gated.
use super::*;
use crate::{batch::extract::ExtractionBatchGrounding,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::{CompileLimits, runtime::{JsonValidationSchema, SourceRuntimeLimits}},
    native_engine::{constrained::{JsonDecodeOutput}, constrained_int4,
        strict_int4::{STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE}},
    tasks::{extract::{ExtractResult, ExtractionGrounding, grouped_int4::{Int4ExtractRun, INT4_EXTRACT_VERSION}},
        ir::{ScoreSpace, TaskBudget}},
    validation::{parse_json, grounded_fields::{GroundingBudget, verify_source_fields}}};
use replay::ReplayBuilder;

pub(super) fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 128, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn compiler() -> CompileLimits { CompileLimits { max_output_bytes: 100_000, ..CompileLimits::default() } }
fn source(extract: ExtractLimits) -> SourceRuntimeLimits {
    let mut source = SourceRuntimeLimits::default();
    source.verification.max_fields = extract.max_fields_per_row;
    source.verification.max_matches = extract.max_occurrences_per_row;
    source.verification.max_scan_steps = extract.max_verification_steps;
    source
}
pub(super) fn wire(locked: usize) -> (SuiteManifest, Vec<serde_json::Value>) {
    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"amount":{"type":"number"}},"required":["name","amount"],"additionalProperties":false}"#;
    let mut rows = vec![
        serde_json::json!({"schema_version":1,"id":"private-development","split":"development",
            "text":"PRIVATE_DEVELOPMENT Ana é","schema_id":"record","language":"und",
            "expected_json":"{\"name\":\"Ana\",\"amount\":1}"}),
        serde_json::json!({"schema_version":1,"id":"private-calibration","split":"calibration",
            "text":"PRIVATE_CALIBRATION Ana 中文","schema_id":"record","language":"und",
            "expected_json":"{\"name\":\"Ana\",\"amount\":1}"})];
    let mut locked_ids = Vec::new();
    for index in 0..locked {
        let id = format!("private-test-{index:03}"); locked_ids.push(id.clone());
        rows.push(serde_json::json!({"schema_version":1,"id":id,"split":"locked_test",
            "text":format!("Ana and Ana PRIVATE_{index} é"),"schema_id":"record",
            "language":if index % 2 == 0 { "zh-Hans" } else { "en" },
            "expected_json":"{\"name\":\"Ana\",\"amount\":1.00}"}));
    }
    (SuiteManifest { schema_version: 1, task: NATIVE_TASK.to_owned(), dataset_id: "extraction-suite".to_owned(),
        dataset_version: "v1".to_owned(), named_population: "Owned extraction fixture population".to_owned(),
        schemas: vec![SchemaDefinition { id: "record".to_owned(), schema_json: schema.to_owned(),
            grounding: ExtractionBatchGrounding::Structural }], expected_input_sha256: Sha256Digest::of_bytes(b"unbound"),
        splits: SplitIds { development: vec!["private-development".to_owned()],
            calibration: vec!["private-calibration".to_owned()], locked_test: locked_ids } }, rows)
}
pub(super) fn encode(mut manifest: SuiteManifest, rows: &[serde_json::Value]) -> (Vec<u8>, Vec<u8>) {
    let mut input = Vec::new();
    for row in rows { input.extend(serde_json::to_vec(row).unwrap()); input.push(b'\n'); }
    manifest.expected_input_sha256 = Sha256Digest::of_bytes(&input);
    (serde_json::to_vec(&manifest).unwrap(), input)
}
fn parse(manifest: SuiteManifest, rows: &[serde_json::Value], extract: ExtractLimits) -> Result<Dataset, AssayError> {
    let (manifest, input) = encode(manifest, rows);
    Dataset::parse(&manifest, &input, Limits::default(), extract, compiler(), source(extract))
}
pub(super) fn dataset(locked: usize) -> Dataset {
    let (manifest, rows) = wire(locked); parse(manifest, &rows, ExtractLimits::default()).unwrap()
}
pub(super) fn identity(row: &Row, subject: &str) -> ExecutionIdentity {
    let public = Sha256Digest::of_bytes(b"public pinned extraction fixture assets");
    let private = Sha256Digest::of_bytes(row.text.as_bytes());
    ExecutionIdentity { schema_version: 1, source_revision: "fixture-pinned-revision".to_owned(),
        logical_model_digest: Sha256Digest::of_bytes(subject.as_bytes()), artifact_format: "fixture-int4".to_owned(),
        quant_recipe: subject.to_owned(), packing_set_digest: Sha256Digest::of_bytes(subject.as_bytes()),
        tokenizer_digest: public, template_digest: public, task_spec: "extract-v1".to_owned(),
        taskir_digest: private, prompt_digest: private, grammar_compiler_version: "extract-fixture-v1".to_owned(),
        schema_digest: public, numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture-full-vocabulary".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: public, decision_policy_digest: public,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn eos() -> u32 {
    crate::tokenizer::pinned_controls::pinned().unwrap().template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id
}
pub(super) fn run(dataset: &Dataset, index: usize) -> Int4ExtractRun {
    let row = &dataset.replay_rows()[index]; let schema = dataset.schema_for_row(index).unwrap();
    let token_ids = vec![1, 2, eos()];
    let model_work = constrained_int4::planned_work(128, 8, token_ids.len()).unwrap();
    let grounded = schema.grounding == ExtractionBatchGrounding::SourceMembership;
    let source_fields = if grounded {
        let compiled = JsonValidationSchema::compile(&schema.schema_json, compiler()).unwrap();
        let value = parse_json(&row.expected_json).unwrap(); compiled.validate_value(&value).unwrap();
        verify_source_fields(compiled.root(), &value, &row.text, GroundingBudget::default()).unwrap()
    } else { Vec::new() };
    Int4ExtractRun { schema_version: 1, execution: INT4_EXTRACT_VERSION.to_owned(), group_width: 128, model_work,
        result: ExtractResult { schema_version: if grounded { 2 } else { 1 }, task_spec_version: "extract-v1".to_owned(),
            score_space: ScoreSpace::NotComputed,
            grounding: if grounded { ExtractionGrounding::SourceMembership } else { ExtractionGrounding::NotRequested },
            source_fields, output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
                token_ids, json: row.expected_json.clone(), forward_positions: model_work.forward_positions,
                projected_logits: model_work.projected_logits, mask_node_visit_charge: 17 } } }
}
fn allowance(dataset: &Dataset) -> GroundingBudget {
    GroundingBudget { max_fields: 1_000_000, max_matches: 1_000_000,
        max_scan_steps: ExtractLimits::default().max_verification_steps - dataset.gold_verification_steps() }
}
fn accept(builder: &mut ReplayBuilder<'_>, dataset: &Dataset, index: usize, subject: &str,
    run: Int4ExtractRun, verification: &mut GroundingBudget) -> Result<(), AssayError> {
    let charge = dataset::verification_charge(&dataset.replay_rows()[index].text,
        &run.result.source_fields, ExtractLimits::default())?;
    let native_steps = charge.steps.checked_mul(2).unwrap();
    charge.add(charge)?.debit(verification)?;
    builder.accept_fixture(index, &identity(&dataset.replay_rows()[index], subject), budget(), 8,
        constrained_int4::planned_work(128, 8, budget().max_output_tokens as usize).unwrap(), 128, run,
        native_steps, verification, &mut || Ok::<(), AssayError>(()))
}
pub(super) fn replay(dataset: &Dataset, subject: &str) -> NativeReplay {
    let mut builder = ReplayBuilder::new(dataset, Limits::default(), ExtractLimits::default()).unwrap();
    let mut verification = allowance(dataset);
    for index in 0..dataset.replay_len() { accept(&mut builder, dataset, index, subject, run(dataset, index), &mut verification).unwrap(); }
    builder.finish().unwrap()
}
pub(super) fn replay_changed(dataset: &Dataset, subject: &str) -> NativeReplay {
    let mut builder = ReplayBuilder::new(dataset, Limits::default(), ExtractLimits::default()).unwrap();
    let mut verification = allowance(dataset);
    for index in 0..dataset.replay_len() {
        let mut result = run(dataset, index); result.result.output.json = "{\"name\":\"Other\",\"amount\":1}".to_owned();
        accept(&mut builder, dataset, index, subject, result, &mut verification).unwrap();
    }
    builder.finish().unwrap()
}

#[test]
fn parses_all_splits_but_forwards_only_locked_source_schema_and_budget() {
    let dataset = dataset(3);
    assert_eq!(dataset.len(), 5); assert_eq!(dataset.replay_len(), 3);
    assert!(dataset.replay_rows().iter().all(|row| row.split == Split::LockedTest));
    let request = dataset.extraction_request(0, budget()).unwrap();
    assert_eq!(request.text, dataset.replay_rows()[0].text);
    let args = request.task_args.unwrap();
    assert_eq!(args.schema, dataset.manifest().schemas[0].schema_json);
    assert_eq!(args.budget, budget());
    let wire = serde_json::to_string(&args).unwrap();
    for private in ["expected_json", "locked_test", "zh-Hans", "private-test"] { assert!(!wire.contains(private)); }
    assert_eq!(dataset.max_source_fields(0).unwrap(), 0);
}

#[test]
fn exact_decimal_gold_never_rounds_through_the_outer_json_wire() {
    let (manifest, mut rows) = wire(1);
    rows[2]["expected_json"] = serde_json::json!("{\"name\":\"Ana\",\"amount\":12345678901234567890123456789012345678}");
    let dataset = parse(manifest, &rows, ExtractLimits::default()).unwrap();
    let expected = parse_json(&dataset.replay_rows()[0].expected_json).unwrap();
    assert_eq!(dataset.gold(0).unwrap(), &expected);
    let replay = replay(&dataset, "exact"); assert!(replay.rows[0].counts.exact_match());
}

#[test]
fn gold_uses_native_unicode_scalar_limits_in_structural_and_grounded_modes() {
    for grounding in [ExtractionBatchGrounding::Structural, ExtractionBatchGrounding::SourceMembership] {
        let (mut manifest, mut rows) = wire(1);
        let annotation = if grounding == ExtractionBatchGrounding::SourceMembership { ",\"x-fnlp-source\":\"verbatim\"" } else { "" };
        manifest.schemas[0].schema_json = format!("{{\"type\":\"string\",\"maxLength\":2{annotation}}}");
        manifest.schemas[0].grounding = grounding;
        for row in &mut rows {
            row["text"] = serde_json::json!(format!("{} 上海", row["text"].as_str().unwrap()));
            row["expected_json"] = serde_json::json!("\"上海\"");
        }
        let dataset = parse(manifest.clone(), &rows, ExtractLimits::default()).unwrap();
        assert!(replay(&dataset, "unicode").rows[0].counts.exact_match());
        rows[2]["expected_json"] = serde_json::json!("\"é上海\"");
        assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::Observation)));
    }
}

#[test]
fn catalogue_cap_refuses_before_malformed_case_rows() {
    let (mut manifest, _) = wire(1);
    let mut extra = manifest.schemas[0].clone(); extra.id = "extra".to_owned(); manifest.schemas.push(extra);
    let input = b"malformed case"; manifest.expected_input_sha256 = Sha256Digest::of_bytes(input);
    let extract = ExtractLimits { max_schemas: 1, ..ExtractLimits::default() };
    assert!(matches!(Dataset::parse(&serde_json::to_vec(&manifest).unwrap(), input, Limits::default(),
        extract, compiler(), source(extract)), Err(AssayError::InputLimit)));
}

#[test]
fn all_schema_definitions_and_all_split_gold_are_validated() {
    let (mut manifest, rows) = wire(1);
    manifest.schemas.push(SchemaDefinition { id: "unused".to_owned(), schema_json: "{\"unsupported\":true}".to_owned(),
        grounding: ExtractionBatchGrounding::Structural });
    assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::Manifest)));
    let (manifest, mut rows) = wire(1); rows[1]["expected_json"] = serde_json::json!("{\"name\":null,\"amount\":1}");
    assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::Observation)));
}

#[test]
fn duplicate_inner_keys_and_cross_split_source_units_are_refused() {
    let (manifest, mut rows) = wire(1); rows[2]["expected_json"] = serde_json::json!("{\"name\":\"Ana\",\"name\":\"Other\",\"amount\":1}");
    assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::Json)));
    let (manifest, mut rows) = wire(1); rows[2]["text"] = rows[0]["text"].clone();
    assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::DuplicateUnit)));
}

#[test]
fn supplied_predictions_and_internal_gold_cache_are_not_wire_authority() {
    for key in ["gold", "prediction", "model_answer"] {
        let (manifest, mut rows) = wire(1); rows[2][key] = serde_json::json!({"name":"forged"});
        assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::Json)));
    }
}

#[test]
fn aggregate_node_cap_precedes_invalid_tail_and_field_caps_are_strict() {
    let (mut manifest, mut rows) = wire(1);
    manifest.schemas[0].schema_json = "{\"type\":\"array\",\"items\":{\"type\":\"number\"},\"maxItems\":8}".to_owned();
    for row in &mut rows { row["expected_json"] = serde_json::json!("[0]"); }
    rows[2]["expected_json"] = serde_json::json!("[0,0,INVALID]");
    let extract = ExtractLimits { max_json_nodes_per_row: 2, ..ExtractLimits::default() };
    assert!(matches!(parse(manifest, &rows, extract), Err(AssayError::InputLimit)));
    let (manifest, rows) = wire(1);
    let extract = ExtractLimits { max_fields_per_row: 1, ..ExtractLimits::default() };
    assert!(matches!(parse(manifest, &rows, extract), Err(AssayError::InputLimit)));
}

fn grounded_wire(locked: usize) -> (SuiteManifest, Vec<serde_json::Value>) {
    let (mut manifest, rows) = wire(locked);
    manifest.schemas[0].schema_json = r#"{"type":"object","properties":{"name":{"type":"string","x-fnlp-source":"verbatim"},"amount":{"type":"number"}},"required":["name","amount"],"additionalProperties":false}"#.to_owned();
    manifest.schemas[0].grounding = ExtractionBatchGrounding::SourceMembership; (manifest, rows)
}

#[test]
fn gold_source_verification_is_global_and_counts_all_splits_once() {
    let (manifest, rows) = grounded_wire(2); let dataset = parse(manifest, &rows, ExtractLimits::default()).unwrap();
    let expected: u64 = dataset.rows().iter().map(|row| 8 * (row.text.len() as u64 + 3 + 1)).sum();
    assert_eq!(dataset.gold_verification_steps(), expected);
    assert_eq!(dataset.gold_verification_fields(), 4); assert_eq!(dataset.gold_verification_matches(), 6);
    assert_eq!(dataset.max_source_fields(0).unwrap(), 1);
    let (manifest, rows) = grounded_wire(2);
    let extract = ExtractLimits { max_verification_steps: expected - 1, ..ExtractLimits::default() };
    assert!(matches!(parse(manifest, &rows, extract), Err(AssayError::InputLimit)));
}

#[test]
fn absent_grounded_gold_and_structural_source_annotations_are_refused() {
    let (manifest, mut rows) = grounded_wire(1);
    rows[0]["expected_json"] = serde_json::json!("{\"name\":\"absent\",\"amount\":1}");
    assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::Observation)));
    let (mut manifest, rows) = grounded_wire(1); manifest.schemas[0].grounding = ExtractionBatchGrounding::Structural;
    assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::Manifest)));
    let (mut manifest, rows) = wire(1); manifest.schemas[0].grounding = ExtractionBatchGrounding::SourceMembership;
    assert!(matches!(parse(manifest, &rows, ExtractLimits::default()), Err(AssayError::Manifest)));
}

#[test]
fn source_field_bounds_include_optional_keys_and_finite_arrays() {
    let (mut manifest, mut rows) = grounded_wire(1);
    manifest.schemas[0].schema_json = r#"{"type":"object","properties":{"optional":{"type":"string","x-fnlp-source":"verbatim"},"names":{"type":"array","items":{"type":"string","x-fnlp-source":"verbatim"},"maxItems":3}},"additionalProperties":false}"#.to_owned();
    for row in &mut rows { row["expected_json"] = serde_json::json!("{}"); }
    let dataset = parse(manifest.clone(), &rows, ExtractLimits::default()).unwrap();
    assert_eq!(dataset.max_source_fields(0).unwrap(), 4); assert_eq!(dataset.gold_verification_steps(), 0);
    let extract = ExtractLimits { max_fields_per_row: 2, ..ExtractLimits::default() };
    let dataset = parse(manifest, &rows, extract).unwrap(); assert_eq!(dataset.max_source_fields(0).unwrap(), 2);
}

#[test]
fn collector_binds_complete_output_while_decimal_spelling_preserves_quality() {
    let dataset = dataset(1); let a = replay(&dataset, "candidate"); let b = replay(&dataset, "baseline");
    assert_ne!(a.executions, b.executions); assert_eq!(a.compatible_executions, b.compatible_executions);
    let mut builder = ReplayBuilder::new(&dataset, Limits::default(), ExtractLimits::default()).unwrap();
    let mut changed = run(&dataset, 0); changed.result.output.json = "{\"name\":\"Ana\",\"amount\":1e0}".to_owned();
    changed.result.output.token_ids[0] = 3;
    accept(&mut builder, &dataset, 0, "candidate", changed, &mut allowance(&dataset)).unwrap();
    let changed = builder.finish().unwrap();
    assert_eq!(a.rows[0].counts, changed.rows[0].counts); assert_ne!(a.observations, changed.observations);
}

#[test]
fn native_and_collector_source_passes_are_charged_without_duplicating_gold() {
    let (manifest, rows) = grounded_wire(2); let dataset = parse(manifest, &rows, ExtractLimits::default()).unwrap();
    let replay = replay(&dataset, "candidate");
    let one_pass: u64 = dataset.replay_rows().iter().map(|row| 8 * (row.text.len() as u64 + 4)).sum();
    assert_eq!(replay.verification_steps, one_pass); assert_eq!(replay.total_source_verification_steps, 3 * one_pass);
    assert_eq!(replay.gold_verification_steps, dataset.gold_verification_steps());
}

#[test]
fn replay_poisoning_rejects_wrong_order_work_and_cancellation() {
    let dataset = dataset(2);
    let mut builder = ReplayBuilder::new(&dataset, Limits::default(), ExtractLimits::default()).unwrap();
    assert_eq!(accept(&mut builder, &dataset, 1, "candidate", run(&dataset, 1), &mut allowance(&dataset)), Err(AssayError::Scope));
    assert_eq!(accept(&mut builder, &dataset, 0, "candidate", run(&dataset, 0), &mut allowance(&dataset)), Err(AssayError::Observation));
    assert!(matches!(builder.finish(), Err(AssayError::Observation)));
    let mut builder = ReplayBuilder::new(&dataset, Limits::default(), ExtractLimits::default()).unwrap();
    let mut invalid = run(&dataset, 0); invalid.model_work.attention_pairs += 1;
    assert_eq!(accept(&mut builder, &dataset, 0, "candidate", invalid, &mut allowance(&dataset)), Err(AssayError::Observation));
    let mut builder = ReplayBuilder::new(&dataset, Limits::default(), ExtractLimits::default()).unwrap();
    let result = builder.accept_fixture(0, &identity(&dataset.replay_rows()[0], "candidate"), budget(), 8,
        constrained_int4::planned_work(128, 8, budget().max_output_tokens as usize).unwrap(), 128, run(&dataset, 0),
        0, &mut allowance(&dataset), &mut || Err::<(), AssayError>(AssayError::Commitment));
    assert_eq!(result, Err(AssayError::Commitment)); assert!(matches!(builder.finish(), Err(AssayError::Observation)));
}

#[test]
fn complete_source_coordinates_are_verified_before_a_replay_is_accepted() {
    let (manifest, rows) = grounded_wire(1); let dataset = parse(manifest, &rows, ExtractLimits::default()).unwrap();
    let mut builder = ReplayBuilder::new(&dataset, Limits::default(), ExtractLimits::default()).unwrap();
    let mut forged = run(&dataset, 0); forged.result.source_fields[0].spans[0].scalar_end += 1;
    assert_eq!(accept(&mut builder, &dataset, 0, "candidate", forged, &mut allowance(&dataset)), Err(AssayError::Observation));
    assert!(matches!(builder.finish(), Err(AssayError::Observation)));
}
