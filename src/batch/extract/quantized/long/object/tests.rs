//! Pinned native plans with private scripted completions; not model/parity evidence.
use super::*;
use std::{cell::Cell, rc::Rc};
use crate::{native_engine::{decode::DecodeCancellationKind, constrained::JsonDecodeOutput,
    portable_int8::ProjectionWork, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ir::ScoreSpace, mapreduce::{ChunkLimits, ExecutionLimits}},
    tokenizer::pinned_controls, grammar::runtime::JsonProgram};
const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"required":["quote"],"properties":{"quote":{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}}}"#;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn args(schema: &str, grounding: ExtractionBatchGrounding) -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: schema.to_owned(), grounding, budget: TaskBudget { max_input_tokens: 8192,
        max_output_tokens: 64, max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: 1 << 31 } }
}
fn source_args() -> ExtractionBatchArgs { args(SCHEMA, ExtractionBatchGrounding::SourceMembership) }
fn base_planner() -> Int8ExtractionBatchPlanner {
    let controls = pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter().find(|e| e.special && e.surface == IM_END).unwrap().id;
    let d = Sha256Digest::of_bytes(b"private extraction map fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 1 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None };
    Int8ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, source_args().budget,
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap()
}
fn base_limits() -> Int8ExtractionMapLimits {
    Int8ExtractionMapLimits { mapping: Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 6, max_chunk_tokens: 8192,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2, max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
            projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 256_000 },
        verification: GroundingBudget::default() }
}


const STRUCTURAL: &str = r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"number"},"b":{"type":"string","maxLength":16}},"required":["a","b"]}"#;
const SOURCE: &str = "éAéA      éAéA";

struct Control { polls: usize, stop: usize, cause: DecodeCancellationKind }
impl Default for Control {
    fn default() -> Self { Self { polls: 0, stop: usize::MAX, cause: DecodeCancellationKind::Deadline } }
}
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.polls == self.stop).then_some(self.cause)
    }
}
fn planner() -> Int8ObjectExtractionPlanner { Int8ObjectExtractionPlanner::new(base_planner()).unwrap() }
fn limits() -> Int8ObjectExtractionMapLimits {
    Int8ObjectExtractionMapLimits { mapping: base_limits(),
        reconciliation: RootObjectLimits { max_properties: 16, max_origins: 128,
            max_input_bytes: 1 << 20, max_json_bytes: 1 << 20,
            max_nodes: 65536, max_result_bytes: 1 << 20 } }
}
fn prepare<'s>(planner: &Int8ObjectExtractionPlanner, source: &'s str, args: &ExtractionBatchArgs,
    limits: Int8ObjectExtractionMapLimits) -> PreparedInt8ObjectExtractionMap<'s> {
    planner.plan_document_with_control(source, args, limits, &mut Continue).unwrap()
}
fn complementary(prepared: &PreparedInt8ObjectExtractionMap<'_>, number: &str, text: &str) -> Vec<String> {
    let count = prepared.preflight_metadata().chunk_count(); assert!(count >= 2);
    let mut values = vec!["{}".to_owned(); count];
    values[0] = format!("{{\"a\":{number}}}");
    values[count - 1] = format!("{{\"b\":{}}}", serde_json::to_string(text).unwrap());
    values
}
fn grounded_values(prepared: &PreparedInt8ObjectExtractionMap<'_>) -> Vec<String> {
    prepared.mapped.chunks.chunks().iter().map(|chunk| {
        if chunk.text().contains("éA") { r#"{"quote":"éA"}"#.to_owned() } else { "{}".to_owned() }
    }).collect()
}

/// Only private fixture completions enter this seam; the public wrapper always
/// consumes its sealed map through an actual native extraction engine.
struct Script { schema: String, grounded: bool, json: Vec<String>, calls: Rc<Cell<usize>> }
impl Driver for Script {
    fn capacity(&self) -> usize { 8192 }
    fn clean(&self) -> bool { true }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction,
        _: &ExecutionIdentity, _: Int8JsonBudget, _: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        let next = self.calls.get(); self.calls.set(next + 1);
        let json = self.json[next].clone();
        let program = if self.grounded {
            JsonProgram::compile_with_source(&self.schema, prepared.source().text(),
                CompileLimits::default(), SourceRuntimeLimits::default()).unwrap()
        } else { JsonProgram::compile(&self.schema, CompileLimits::default()).unwrap() };
        let fields = program.source_fields(&json).unwrap();
        let options = prepared.plan.options();
        let token = (0..NANBEIGE_VOCAB_SIZE as u32)
            .find(|id| !options.excluded_token_ids.contains(id) && *id != options.eos_token_id).unwrap();
        let work = constrained_int8::planned_work(prepared.plan.prompt_tokens(), 2).unwrap();
        Ok(Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.to_owned(), model_work: work,
            result: ExtractResult { schema_version: if self.grounded { 2 } else { 1 },
                task_spec_version: "extract-v1".to_owned(), score_space: ScoreSpace::NotComputed,
                grounding: if self.grounded { ExtractionGrounding::SourceMembership } else { ExtractionGrounding::NotRequested },
                source_fields: fields, output: JsonDecodeOutput { schema_version: 1,
                    numerics_profile: STRICT_INT8_PROFILE.to_owned(), token_ids: vec![token, options.eos_token_id],
                    json, forward_positions: work.forward_positions, projected_logits: work.projected_logits,
                    mask_node_visit_charge: 20 } } })
    }
}
fn run<C: DecodeStepControl>(prepared: PreparedInt8ObjectExtractionMap<'_>, json: Vec<String>,
    calls: Rc<Cell<usize>>, control: &mut C) -> Result<Int8ObjectExtractionMapRun, Int8ObjectExtractionMapError> {
    let PreparedInt8ObjectExtractionMap { source, schema, mapped, limits, expected } = prepared;
    assert_eq!(json.len(), expected.chunk_count());
    let script = Script { schema: schema.partial_schema().to_owned(),
        grounded: schema.requires_source(), json, calls };
    let identities: Vec<_> = mapped.execution_identities().cloned().collect();
    let partial = mapped.execute_with_driver(&identities, script, control)?;
    finish(source, schema, limits, expected, partial, control)
}

#[test]
fn different_chunks_supply_required_properties_without_rounding_or_rewriting_native_json() {
    let base = base_planner(); let previous = base.base_execution_identity().template_digest;
    let p = Int8ObjectExtractionPlanner::new(base).unwrap();
    assert_ne!(previous, p.base_execution_identity().template_digest);
    let a = args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let prepared = prepare(&p, SOURCE, &a, limits()); let expected = prepared.preflight_metadata();
    let mut work = Int8Work::default();
    for item in &prepared.mapped.plans {
        assert_eq!(item.execution_identity().schema_digest,
            Sha256Digest::of_bytes(prepared.schema.partial_schema().as_bytes()));
        assert_ne!(item.execution_identity().schema_digest, Sha256Digest::of_bytes(a.schema.as_bytes()));
        work = add_work(work, item.planned_work()).unwrap();
    }
    assert_eq!(work, expected.planned_work());
    let values = complementary(&prepared, "12345678901234567890123456789012345678", "last");
    let originals = values.clone(); let calls = Rc::new(Cell::new(0));
    let output = run(prepared, values, Rc::clone(&calls), &mut Continue).unwrap();
    expected.verify_completed(&output).unwrap();
    assert_eq!(calls.get(), expected.chunk_count());
    assert_eq!(output.object.json(), r#"{"a":1.2345678901234567890123456789012345678e37,"b":"last"}"#);
    assert_eq!(output.partial_map.mapped.root().value().chunks().map(|chunk|
        chunk.native.result.output.json.as_str()).collect::<Vec<_>>(),
        originals.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(output.object.properties()[0].origins[0].chunk_id, 0);
    assert_eq!(output.object.properties()[1].origins[0].chunk_id, expected.chunk_count() - 1);
    assert_eq!(output.partial_map.planned_model_work, expected.planned_work());
    assert_eq!(output.verification_used, output.partial_map.verification_used);
    assert!(output.object.original_source_fields().is_empty());
    let wire = serde_json::to_value(&output).unwrap();
    assert_eq!(wire["object"]["json"].as_str(), Some(output.object.json()));
    for hidden in ["binding", "completion_binding", "request_binding", "result_binding"] {
        assert!(!serde_json::to_string(&output).unwrap().contains(hidden));
    }
}

#[test]
fn original_required_set_source_and_policy_bind_even_when_partial_native_schemas_match() {
    let p = planner(); let a = args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let prepared = prepare(&p, SOURCE, &a, limits()); let expected = prepared.preflight_metadata();
    let ids: Vec<_> = prepared.execution_identities().cloned().collect();
    let values = complementary(&prepared, "1", "last");
    let output = run(prepared, values, Rc::new(Cell::new(0)), &mut Continue).unwrap();
    for schema in [format!(" {STRUCTURAL}\n"),
        STRUCTURAL.replace(r#""required":["a","b"]"#, r#""required":["a"]"#)] {
        let other = prepare(&p, SOURCE, &args(&schema, ExtractionBatchGrounding::Structural), limits());
        assert_eq!(other.execution_identities().cloned().collect::<Vec<_>>(), ids);
        assert_ne!(other.preflight_metadata(), expected);
        assert!(other.preflight_metadata().verify_completed(&output).is_err());
    }
    let mut changed = limits(); changed.reconciliation.max_origins += 1;
    assert!(prepare(&p, SOURCE, &a, changed).preflight_metadata().verify_completed(&output).is_err());
    assert!(prepare(&p, "éBéB      éBéB", &a, limits())
        .preflight_metadata().verify_completed(&output).is_err());
    let debug = format!("{expected:?}");
    assert!(!debug.contains(&expected.binding.to_hex()));
    assert!(!debug.contains("request_binding")); assert!(!debug.contains(SOURCE));
}

#[test]
fn conflicting_or_missing_required_properties_refuse_after_all_chunks_without_retry() {
    let p = planner(); let a = args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    for conflict in [false, true] {
        let prepared = prepare(&p, SOURCE, &a, limits()); let count = prepared.preflight_metadata().chunk_count();
        let mut values = vec![r#"{"a":1}"#.to_owned(); count];
        if conflict { values[count - 1] = r#"{"a":2,"b":"last"}"#.to_owned(); }
        let calls = Rc::new(Cell::new(0));
        let error = run(prepared, values, Rc::clone(&calls), &mut Continue).err().unwrap();
        assert_eq!(calls.get(), count);
        match error {
            Int8ObjectExtractionMapError::Reconciliation(actual) => assert_eq!(actual,
                if conflict { RootObjectError::Conflict } else { RootObjectError::MissingRequired }),
            _ => panic!("expected complete object reconciliation refusal"),
        }
    }
}

#[test]
fn global_grounding_keeps_all_unicode_occurrences_and_spends_only_remaining_map_allowances() {
    let p = planner(); let a = source_args();
    let prepared = prepare(&p, SOURCE, &a, limits()); let expected = prepared.preflight_metadata();
    let values = grounded_values(&prepared);
    let mut output = run(prepared, values, Rc::new(Cell::new(0)), &mut Continue).unwrap();
    let extra = output.object.verification_used(); let mapped = output.partial_map.verification_used;
    assert_eq!(extra.fields, 1); assert_eq!(extra.matches, 4);
    let field = &output.object.original_source_fields()[0];
    assert_eq!(field.occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(field.spans.iter().map(|s| (s.byte_start, s.byte_end, s.scalar_start, s.scalar_end))
        .collect::<Vec<_>>(), [(0, 3, 0, 2), (3, 6, 2, 4), (12, 15, 10, 12), (15, 18, 12, 14)]);
    for origin in &output.object.properties()[0].origins { assert_eq!(origin.source_field_indices, [0]); }
    assert_eq!(output.verification_used.fields, mapped.fields + extra.fields);
    assert_eq!(output.verification_used.matches, mapped.matches + extra.matches);
    assert_eq!(output.verification_used.scan_steps, mapped.scan_steps + extra.scan_steps);
    let mut exact = limits();
    exact.mapping.verification = GroundingBudget { max_fields: output.verification_used.fields,
        max_matches: output.verification_used.matches, max_scan_steps: output.verification_used.scan_steps };
    let prepared = prepare(&p, SOURCE, &a, exact); let values = grounded_values(&prepared);
    let exact_output = run(prepared, values, Rc::new(Cell::new(0)), &mut Continue).unwrap();
    assert_eq!(output.verification_used, exact_output.verification_used);
    for axis in 0..3 {
        let mut short = exact;
        match axis { 0 => short.mapping.verification.max_fields -= 1,
            1 => short.mapping.verification.max_matches -= 1,
            _ => short.mapping.verification.max_scan_steps -= 1 }
        let prepared = prepare(&p, SOURCE, &a, short); let count = prepared.preflight_metadata().chunk_count();
        let values = grounded_values(&prepared); let calls = Rc::new(Cell::new(0));
        let error = run(prepared, values, Rc::clone(&calls), &mut Continue).err().unwrap();
        assert_eq!(calls.get(), count);
        assert!(matches!(error, Int8ObjectExtractionMapError::Reconciliation(RootObjectError::Limit | RootObjectError::Source)));
    }
    output.partial_map.verification_used.scan_steps -= 1;
    output.verification_used.scan_steps -= 1;
    assert!(expected.verify_completed(&output).is_err());
}

#[test]
fn completed_object_substitution_envelope_edits_and_one_byte_output_shortfall_refuse() {
    let p = planner(); let a = args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let first = prepare(&p, SOURCE, &a, limits()); let expected = first.preflight_metadata();
    let values = complementary(&first, "1", "first");
    let mut first = run(first, values, Rc::new(Cell::new(0)), &mut Continue).unwrap();
    let second = prepare(&p, SOURCE, &a, limits()); let values = complementary(&second, "2", "second");
    let mut second = run(second, values, Rc::new(Cell::new(0)), &mut Continue).unwrap();
    expected.verify_completed(&first).unwrap(); expected.verify_completed(&second).unwrap();
    std::mem::swap(&mut first.object, &mut second.object);
    assert!(expected.verify_completed(&first).is_err()); assert!(expected.verify_completed(&second).is_err());
    std::mem::swap(&mut first.object, &mut second.object);
    expected.verify_completed(&first).unwrap();
    let mut exact = expected; exact.max_result_bytes = serde_json::to_vec(&first).unwrap().len() as u64;
    exact.verify_completed(&first).unwrap();
    exact.max_result_bytes -= 1; assert!(exact.verify_completed(&first).is_err());
    first.schema_version = 2; assert!(expected.verify_completed(&first).is_err()); first.schema_version = 1;
    first.execution = "other"; assert!(expected.verify_completed(&first).is_err()); first.execution = INT8_OBJECT_MAP_EXECUTION;
    first.map_prompt = "other"; assert!(expected.verify_completed(&first).is_err()); first.map_prompt = ROOT_OBJECT_MAP_PROMPT;
    first.verification_used.fields += 1; assert!(expected.verify_completed(&first).is_err()); first.verification_used.fields -= 1;
    first.untrusted_fields = ["object", "partial_map"]; assert!(expected.verify_completed(&first).is_err());
}

#[test]
fn explicit_object_mode_rejects_defaults_nonobjects_wrong_grounding_and_unfunded_compact_policies() {
    let mut base = base_planner(); base.compiler.defaults = Some(source_args());
    assert!(Int8ObjectExtractionPlanner::new(base).is_err());
    let p = planner();
    for (schema, grounding) in [(SCHEMA, ExtractionBatchGrounding::Structural),
        (STRUCTURAL, ExtractionBatchGrounding::SourceMembership),
        (r#"{"type":"string","maxLength":16}"#, ExtractionBatchGrounding::Structural)] {
        assert!(p.plan_document_with_control(SOURCE, &args(schema, grounding), limits(), &mut Continue).is_err());
    }
    let policy = crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 8192, max_output_tokens: 8192, max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 };
    assert!(Int8ObjectExtractionPlanner::new(base_planner()
        .with_document_encoding(UntrustedEncoding::SafeBpeV1(policy)).unwrap()).is_err());
    let mut invalid = limits(); invalid.reconciliation.max_result_bytes = invalid.mapping.mapping.reduction.max_result_bytes + 1;
    assert!(invalid.reconciliation_workspace_bytes().is_err());
}

#[test]
fn compact_preflight_and_reconstruction_share_one_nonrenewable_invocation_ledger() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 8192, max_output_tokens: 8192, max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 });
    let total = SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 };
    let p = planner().with_map_document_encoding(policy, total).unwrap();
    assert_eq!(p.map_document_encoding_limits(), Some(total));
    let a = args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let first = prepare(&p, SOURCE, &a, limits()); let expected = first.preflight_metadata();
    let used = p.map_document_encoding_work().unwrap().unwrap();
    assert!(used.input_bytes() > 0); drop(first);
    let second = prepare(&p, SOURCE, &a, limits());
    assert_eq!(second.preflight_metadata(), expected);
    assert_eq!(p.map_document_encoding_work().unwrap().unwrap(), used.checked_add(used).unwrap());
    let values = complementary(&second, "1", "last");
    let output = run(second, values, Rc::new(Cell::new(0)), &mut Continue).unwrap();
    expected.verify_completed(&output).unwrap();
}

#[test]
fn planning_and_late_complete_object_cancellation_preserve_original_terminal_causes() {
    let p = planner(); let a = args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let prepared = prepare(&p, SOURCE, &a, limits()); let count = prepared.preflight_metadata().chunk_count();
    let values = complementary(&prepared, "1", "last"); let mut completed = Control::default();
    run(prepared, values, Rc::new(Cell::new(0)), &mut completed).unwrap();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota] {
        let mut early = Control { polls: 0, stop: 1, cause };
        assert_eq!(p.plan_document_with_control(SOURCE, &a, limits(), &mut early).err().unwrap().cancellation(), Some(cause));
        let prepared = prepare(&p, SOURCE, &a, limits()); let values = complementary(&prepared, "1", "last");
        let calls = Rc::new(Cell::new(0)); let mut late = Control { polls: 0, stop: completed.polls, cause };
        let error = run(prepared, values, Rc::clone(&calls), &mut late).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause)); assert_eq!(late.polls, completed.polls);
        assert_eq!(calls.get(), count);
    }
}


#[test]
fn identical_public_objects_cannot_transplant_distinct_exact_partial_map_owners() {
    let p = planner(); let a = args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let prepared = prepare(&p, SOURCE, &a, limits()); let expected = prepared.preflight_metadata();
    let values = complementary(&prepared, "1.0", "same");
    let mut first = run(prepared, values, Rc::new(Cell::new(0)), &mut Continue).unwrap();
    let prepared = prepare(&p, SOURCE, &a, limits()); let values = complementary(&prepared, "1e0", "same");
    let mut second = run(prepared, values, Rc::new(Cell::new(0)), &mut Continue).unwrap();
    // Exact decimal canonicalization gives the same public object and counters;
    // ownership still names the independently completed partial native JSON.
    assert_eq!(serde_json::to_vec(&first.object).unwrap(), serde_json::to_vec(&second.object).unwrap());
    assert_ne!(first.object.map_binding(), second.object.map_binding());
    expected.verify_completed(&first).unwrap(); expected.verify_completed(&second).unwrap();
    std::mem::swap(&mut first.object, &mut second.object);
    assert!(expected.verify_completed(&first).is_err()); assert!(expected.verify_completed(&second).is_err());
}
