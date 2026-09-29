//! Pinned plans and private scripted corruption fixtures, NOT inference receipts.
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
fn planner() -> Int8ExtractionBatchPlanner {
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
fn limits() -> Int8ExtractionMapLimits {
    Int8ExtractionMapLimits { mapping: Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 6, max_chunk_tokens: 8192,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2, max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
            projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 256_000 },
        verification: GroundingBudget::default() }
}
fn prepare<'s>(planner: &Int8ExtractionBatchPlanner, source: &'s str, args: &ExtractionBatchArgs,
    limits: Int8ExtractionMapLimits) -> PreparedInt8ExtractionMap<'s> {
    planner.plan_document_with_control(source, args, limits, &mut Continue).unwrap()
}
struct Script {
    args: ExtractionBatchArgs, quote: Option<String>, json: Option<String>, calls: Rc<Cell<usize>>,
    corrupt: Option<usize>, fail_at: Option<usize>, capacity: usize,
}
impl Script {
    fn new(args: &ExtractionBatchArgs) -> Self {
        Self { args: args.clone(), quote: None, json: None, calls: Rc::new(Cell::new(0)), corrupt: None, fail_at: None, capacity: 8192 }
    }
}
impl Driver for Script {
    fn capacity(&self) -> usize { self.capacity }
    fn clean(&self) -> bool { true }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction, _: &ExecutionIdentity,
        _: Int8JsonBudget, _: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        let call = self.calls.get() + 1; self.calls.set(call);
        if self.fail_at == Some(call) { return Err(ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)).into()); }
        let source = prepared.source().text();
        let json = self.json.clone().unwrap_or_else(|| serde_json::to_string(&serde_json::json!({
            "quote": self.quote.as_deref().unwrap_or(source) })).unwrap());
        let grounded = self.args.grounding == ExtractionBatchGrounding::SourceMembership;
        let program = if grounded { JsonProgram::compile_with_source(&self.args.schema, source,
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap() }
            else { JsonProgram::compile(&self.args.schema, CompileLimits::default()).unwrap() };
        let fields = program.source_fields(&json).unwrap();
        let options = prepared.plan.options();
        let token = (0..NANBEIGE_VOCAB_SIZE as u32).find(|id| !options.excluded_token_ids.contains(id) && *id != options.eos_token_id).unwrap();
        let work = constrained_int8::planned_work(prepared.plan.prompt_tokens(), 2).unwrap();
        let mut run = Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.to_owned(), model_work: work,
            result: ExtractResult { schema_version: if grounded { 2 } else { 1 }, task_spec_version: "extract-v1".to_owned(),
                score_space: ScoreSpace::NotComputed, grounding: if grounded { ExtractionGrounding::SourceMembership }
                    else { ExtractionGrounding::NotRequested }, source_fields: fields,
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
                    token_ids: vec![token, options.eos_token_id], json, forward_positions: work.forward_positions,
                    projected_logits: work.projected_logits, mask_node_visit_charge: 20 } } };
        if call == 2 { if let Some(axis) = self.corrupt { match axis {
            0 => run.result.source_fields.clear(),
            1 => run.result.source_fields[0].json_pointer = "/wrong".to_owned(),
            2 => run.result.source_fields[0].spans[0].scalar_end += 1,
            3 => run.result.output.json = "{}".to_owned(),
            4 => { run.result.output.token_ids.pop(); },
            5 => run.model_work.attention_pairs += 1,
            6 => run.result.grounding = ExtractionGrounding::NotRequested,
            _ => run.result.output.mask_node_visit_charge = 1001,
        } } }
        Ok(run)
    }
}
fn run(prepared: PreparedInt8ExtractionMap<'_>, script: Script) -> Result<Int8ExtractionMapRun, Int8SourceMapError> {
    let identities: Vec<_> = prepared.execution_identities().cloned().collect();
    prepared.execute_with_driver(&identities, script, &mut Continue)
}
#[test]
fn exact_schema_and_source_plans_cover_unicode_crlf_whitespace_and_control_spellings() {
    let p = planner(); let a = source_args();
    for source in ["éAéA      éAéA", "Alice <tool_call> 上海\r\nBob"] {
        let prepared = prepare(&p, source, &a, limits());
        assert_eq!(prepared.chunks.chunks().iter().map(|c| c.text()).collect::<String>(), source);
        assert_eq!(prepared.expected.source_span.byte_end, source.len());
        assert_eq!(prepared.expected.source_span.scalar_end, source.chars().count());
        for (chunk, plan) in prepared.chunks.chunks().iter().zip(&prepared.plans) {
            assert_eq!(plan.source().text(), chunk.text());
            assert_eq!(plan.execution_identity().schema_digest, Sha256Digest::of_bytes(a.schema.as_bytes()));
            assert!(plan.plan.prompt_tokens() + a.budget.max_output_tokens as usize <= limits().mapping.chunks.context_tokens);
            assert!(!chunk.text().ends_with('\r'));
        }
        let expected = prepared.expected; let script = Script::new(&a); let calls = Rc::clone(&script.calls);
        let result = run(prepared, script).unwrap(); expected.verify_completed(&result).unwrap();
        assert_eq!(calls.get(), expected.chunk_count()); // Whitespace is a real extraction, not a fake empty object.
    }
}
#[test]
fn all_six_whole_run_allowances_are_precharged_and_one_unit_short_fails_before_execution() {
    let p = planner(); let a = source_args(); let source = "Alice Bob Carol";
    let expected = prepare(&p, source, &a, limits()).expected;
    let mut exact = limits(); exact.mapping.max_model_work = expected.planned_work(); exact.mapping.max_mask_visits = expected.reserved_mask_visits();
    prepare(&p, source, &a, exact);
    for axis in 0..6 {
        let mut bad = exact;
        match axis { 0 => bad.mapping.max_model_work.forward_positions -= 1, 1 => bad.mapping.max_model_work.projected_logits -= 1,
            2 => bad.mapping.max_model_work.attention_pairs -= 1, 3 => bad.mapping.max_model_work.projections.dot_products -= 1,
            4 => bad.mapping.max_model_work.projections.multiply_accumulates -= 1, _ => bad.mapping.max_mask_visits -= 1 }
        assert!(p.plan_document_with_control(source, &a, bad, &mut Continue).is_err(), "{axis}");
    }
}
#[test]
fn schema_changes_rebind_every_identity_and_invalid_or_unbound_source_schemas_refuse() {
    let p = planner(); let a = source_args(); let mut b = a.clone(); b.schema = a.schema.replace("64", "63");
    let left = prepare(&p, "Alice Bob", &a, limits()); let right = prepare(&p, "Alice Bob", &b, limits());
    for (left, right) in left.execution_identities().zip(right.execution_identities()) {
        assert_ne!(left.schema_digest, right.schema_digest); assert_ne!(left.prompt_digest, right.prompt_digest);
    }
    for schema in [r#"{"type":"string","type":"number"}"#, r#"{"$ref":"https://example.invalid/schema"}"#,
        r#"{"type":"string"}"#] {
        assert!(p.plan_document_with_control("Alice", &args(schema, ExtractionBatchGrounding::SourceMembership), limits(), &mut Continue).is_err());
    }
}
#[test]
fn every_final_identity_and_resident_kv_requirement_is_checked_before_the_first_forward() {
    let p = planner(); let a = source_args(); let prepared = prepare(&p, "Alice Bob", &a, limits());
    let mut ids: Vec<_> = prepared.execution_identities().cloned().collect();
    ids.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final prompt");
    let script = Script::new(&a); let calls = Rc::clone(&script.calls);
    assert!(prepared.execute_with_driver(&ids, script, &mut Continue).is_err()); assert_eq!(calls.get(), 0);
    let prepared = prepare(&p, "Alice Bob", &a, limits()); let mut script = Script::new(&a); script.capacity = 1;
    let calls = Rc::clone(&script.calls); assert!(run(prepared, script).is_err()); assert_eq!(calls.get(), 0);
}
#[test]
fn repeated_unicode_fields_lift_all_original_occurrences_without_merging_chunk_objects() {
    let p = planner(); let a = source_args(); let source = "éAéAéAéA";
    let prepared = prepare(&p, source, &a, limits()); let mut script = Script::new(&a); script.quote = Some("é".to_owned());
    let result = run(prepared, script).unwrap(); let chunks: Vec<_> = result.mapped.root().value().chunks().collect();
    assert_eq!(chunks.len(), 2); assert_eq!(result.verification_used.fields, 2); assert_eq!(result.verification_used.matches, 4);
    for chunk in chunks { let field = &chunk.original_fields[0]; assert_eq!(field.json_pointer, "/quote");
        assert_eq!(field.occurrence, SourceOccurrence::Ambiguous);
        for span in &field.spans { assert_eq!(&source[span.byte_start..span.byte_end], "é");
            assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start); }
    }
}
#[test]
fn exact_decimal_output_is_never_reparsed_through_f64_or_combined_across_chunks() {
    let p = planner(); let number = "12345678901234567890123456789012345678";
    let a = args(&format!(r#"{{"type":"number","const":{number}}}"#), ExtractionBatchGrounding::Structural);
    let prepared = prepare(&p, "Alice Bob", &a, limits()); let mut script = Script::new(&a); script.json = Some(number.to_owned());
    let result = run(prepared, script).unwrap(); assert_eq!(result.grounding, ExtractionGrounding::NotRequested);
    for chunk in result.mapped.root().value().chunks() { assert_eq!(chunk.native.result.output.json, number);
        assert!(chunk.original_fields.is_empty()); assert!(chunk.native.result.source_fields.is_empty()); }
    assert_eq!(result.verification_used.fields, 0); assert_eq!(result.verification_used.matches, 0);
    assert!(canonjson::canonical_string(&result).unwrap().contains(number));
}
#[test]
fn empty_verbatim_strings_keep_every_scalar_boundary_in_original_coordinates() {
    let p = planner(); let a = source_args(); let source = "éAéAéAéA";
    let prepared = prepare(&p, source, &a, limits()); let mut script = Script::new(&a); script.quote = Some(String::new());
    let result = run(prepared, script).unwrap();
    for chunk in result.mapped.root().value().chunks() { for span in &chunk.original_fields[0].spans {
        assert_eq!(span.byte_start, span.byte_end); assert_eq!(span.scalar_start, span.scalar_end);
        assert!(source.is_char_boundary(span.byte_start)); assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
    } }
}
#[test]
fn late_schema_field_receipt_and_grounding_corruption_never_publish_a_partial_success() {
    let p = planner(); let a = source_args();
    for axis in 0..8 { let prepared = prepare(&p, "Alice Bob", &a, limits()); let mut script = Script::new(&a);
        script.corrupt = Some(axis); let calls = Rc::clone(&script.calls);
        assert!(run(prepared, script).is_err(), "{axis}"); assert_eq!(calls.get(), 2); }
}
#[test]
fn verification_and_live_cumulative_final_output_limits_never_reset_at_chunk_boundaries() {
    let p = planner(); let a = source_args();
    for axis in 0..6 { let mut l = limits();
        match axis { 0 => l.verification.max_fields = 1, 1 => l.verification.max_matches = 1,
            2 => l.verification.max_scan_steps = 1, 3 => l.mapping.reduction.max_live_value_bytes = 1,
            4 => l.mapping.reduction.max_total_value_bytes = 1, _ => l.mapping.reduction.max_result_bytes = 1 }
        assert!(run(prepare(&p, "Alice Bob", &a, l), Script::new(&a)).is_err(), "{axis}");
    }
}
#[test]
fn source_chunk_and_schema_capacity_failures_do_not_return_truncated_plans() {
    let p = planner(); let a = source_args();
    assert!(p.plan_document_with_control("", &a, limits(), &mut Continue).is_err());
    let mut l = limits(); l.mapping.chunks.max_chunks = 1;
    assert!(p.plan_document_with_control("Alice Bob", &a, l, &mut Continue).is_err());
    let mut small = a; small.budget.max_input_tokens = 1;
    assert!(p.plan_document_with_control("Alice", &small, limits(), &mut Continue).is_err());
}
#[test]
fn native_cancellation_and_final_completion_checkpoint_preserve_the_exact_cause() {
    let p = planner(); let a = source_args(); let prepared = prepare(&p, "Alice Bob", &a, limits());
    let mut script = Script::new(&a); script.fail_at = Some(2);
    assert_eq!(run(prepared, script).err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
    struct Count { calls: usize, stop: Option<usize> }
    impl DecodeStepControl for Count { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.stop == Some(self.calls)).then_some(DecodeCancellationKind::Deadline)
    } }
    let prepared = prepare(&p, "Alice", &a, limits()); let ids: Vec<_> = prepared.execution_identities().cloned().collect();
    let mut count = Count { calls: 0, stop: None };
    prepared.execute_with_driver(&ids, Script::new(&a), &mut count).unwrap();
    let prepared = prepare(&p, "Alice", &a, limits()); let ids: Vec<_> = prepared.execution_identities().cloned().collect();
    let error = prepared.execute_with_driver(&ids, Script::new(&a), &mut Count { calls: 0, stop: Some(count.calls) }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert!(!format!("{error:?}").contains("Alice"));
}
