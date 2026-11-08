//! Pinned planning plus private scripted map results; not native parity evidence.
use super::*;
use super::super::tests as map_fixture;
use std::{cell::Cell, rc::Rc};
use crate::{native_engine::decode::DecodeCancellationKind,
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};

const STRUCTURAL: &str = r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"number"},"b":{"type":"string","maxLength":16}},"required":["a","b"]}"#;
const GROUNDED: &str = r#"{"type":"object","additionalProperties":false,"properties":{"quote":{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}},"required":["quote"]}"#;
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
fn planner() -> Int4ObjectExtractionPlanner {
    Int4ObjectExtractionPlanner::new(map_fixture::planner()).unwrap()
}
fn limits() -> Int4ObjectExtractionMapLimits {
    let mut mapping = map_fixture::limits(); mapping.mapping.reduction.map_batch_chunks = 1;
    Int4ObjectExtractionMapLimits { mapping,
        reconciliation: RootObjectLimits { max_properties: 16, max_origins: 128,
            max_input_bytes: 1 << 20, max_json_bytes: 1 << 20,
            max_nodes: 65536, max_result_bytes: 1 << 20 } }
}
fn prepare<'s>(planner: &Int4ObjectExtractionPlanner, source: &'s str, args: &ExtractionBatchArgs,
    limits: Int4ObjectExtractionMapLimits) -> PreparedInt4ObjectExtractionMap<'s> {
    planner.plan_document_with_control(source, args, limits, &mut Control::default()).unwrap()
}
fn complementary(prepared: &PreparedInt4ObjectExtractionMap<'_>, number: &str, text: &str) -> Vec<String> {
    let count = prepared.preflight_metadata().chunk_count(); assert!(count >= 2);
    let mut values = vec!["{}".to_owned(); count];
    values[0] = format!("{{\"a\":{number}}}");
    values[count - 1] = format!("{{\"b\":{}}}", serde_json::to_string(text).unwrap());
    values
}

struct Script { inner: map_fixture::Script, json: Vec<String>, next: usize }
impl Driver for Script {
    fn capacity(&self) -> usize { self.inner.capacity() }
    fn group_width(&self) -> usize { self.inner.group_width() }
    fn clean(&self) -> bool { self.inner.clean() }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt4BatchExtraction,
        admitted: &ExecutionIdentity, budget: Int4JsonBudget, control: &mut C)
        -> Result<Int4ExtractRun, Int4ExtractError> {
        self.inner.json = Some(self.json[self.next].clone()); self.next += 1;
        self.inner.execute(prepared, admitted, budget, control)
    }
}
fn run(prepared: PreparedInt4ObjectExtractionMap<'_>, args: &ExtractionBatchArgs, json: Vec<String>,
    calls: Rc<Cell<usize>>, control: &mut Control) -> Result<Int4ObjectExtractionMapRun, Int4ObjectExtractionMapError> {
    let PreparedInt4ObjectExtractionMap { source, schema, mapped, limits, expected } = prepared;
    assert_eq!(json.len(), expected.chunk_count());
    let partial_args = ExtractionBatchArgs { schema: schema.partial_schema().to_owned(),
        grounding: args.grounding, budget: args.budget };
    let mut inner = map_fixture::Script::new(&partial_args); inner.calls = calls;
    let identities: Vec<_> = mapped.execution_identities().cloned().collect();
    let partial_map = mapped.execute_with_driver(&identities, Script { inner, json, next: 0 }, control)?;
    finish(source, schema, limits, expected, partial_map, control)
}

#[test]
fn complete_partial_maps_reconcile_exact_values_and_retain_every_whitespace_and_source_chunk() {
    let p = planner(); let args = map_fixture::args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let prepared = prepare(&p, SOURCE, &args, limits()); let expected = prepared.preflight_metadata();
    assert_eq!(prepared.mapped.chunks.chunks().iter().map(|chunk| chunk.text()).collect::<String>(), SOURCE);
    assert!(prepared.mapped.chunks.chunks().iter().any(|chunk| chunk.text().chars().all(char::is_whitespace)));
    assert_eq!(prepared.group_width(), 32); assert_eq!(prepared.max_kv_bytes(), args.budget.max_kv_bytes);
    let mut work = Int4Work::default();
    for item in &prepared.mapped.plans {
        assert_eq!(item.execution_identity().schema_digest, Sha256Digest::of_bytes(prepared.schema.partial_schema().as_bytes()));
        assert_ne!(item.execution_identity().schema_digest, Sha256Digest::of_bytes(args.schema.as_bytes()));
        work = work.checked_add(item.planned_work()).unwrap();
    }
    assert_eq!(expected.planned_work(), work);
    let values = complementary(&prepared, "12345678901234567890123456789012345678", "last");
    let expected_values = values.clone(); let calls = Rc::new(Cell::new(0));
    let output = run(prepared, &args, values, Rc::clone(&calls), &mut Control::default()).unwrap();
    expected.verify_completed(&output).unwrap();
    assert_eq!(calls.get(), expected.chunk_count());
    assert_eq!(output.object.json(), r#"{"a":1.2345678901234567890123456789012345678e37,"b":"last"}"#);
    assert_eq!(output.partial_map.mapped.root().value().chunks().map(|chunk|
        chunk.native.result.output.json.as_str()).collect::<Vec<_>>(), expected_values.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(output.object.properties()[0].origins[0].chunk_id, 0);
    assert_eq!(output.object.properties()[1].origins[0].chunk_id, expected.chunk_count() - 1);
    assert_eq!(output.partial_map.planned_model_work, expected.planned_work());
    assert_eq!(output.verification_used, output.partial_map.verification_used);
    assert!(output.object.original_source_fields().is_empty());
    assert_eq!(output.untrusted_fields, ["partial_map", "object"]);
}

#[test]
fn partial_object_prompt_and_complete_original_schema_have_distinct_admission_bindings() {
    let base = map_fixture::planner(); let original_template = base.base_execution_identity().template_digest;
    let p = Int4ObjectExtractionPlanner::new(base).unwrap();
    assert_ne!(p.base_execution_identity().template_digest, original_template);
    let args = map_fixture::args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let prepared = prepare(&p, SOURCE, &args, limits()); let expected = prepared.preflight_metadata();
    let identities: Vec<_> = prepared.execution_identities().cloned().collect();
    let values = complementary(&prepared, "1", "last");
    let output = run(prepared, &args, values, Rc::new(Cell::new(0)), &mut Control::default()).unwrap();
    for changed_schema in [format!(" {STRUCTURAL}\n"), STRUCTURAL.replace(r#""required":["a","b"]"#, r#""required":["a"]"#)] {
        let changed = map_fixture::args(&changed_schema, ExtractionBatchGrounding::Structural);
        let other = prepare(&p, SOURCE, &changed, limits());
        assert_eq!(other.execution_identities().cloned().collect::<Vec<_>>(), identities);
        assert_ne!(other.preflight_metadata(), expected);
        assert!(other.preflight_metadata().verify_completed(&output).is_err());
    }
    let mut changed = limits(); changed.reconciliation.max_origins += 1;
    let other = prepare(&p, SOURCE, &args, changed);
    assert_eq!(other.execution_identities().cloned().collect::<Vec<_>>(), identities);
    assert!(other.preflight_metadata().verify_completed(&output).is_err());
    let other = prepare(&p, "éBéB      éBéB", &args, limits());
    assert!(other.preflight_metadata().verify_completed(&output).is_err());
    let mut repartitioned = limits(); repartitioned.mapping.mapping.chunks.max_chunk_bytes = 7;
    let other = prepare(&p, SOURCE, &args, repartitioned);
    assert!(other.preflight_metadata().verify_completed(&output).is_err());
}

#[test]
fn conflicts_and_missing_required_properties_refuse_after_the_complete_native_map_without_retries() {
    let p = planner(); let args = map_fixture::args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    for conflict in [false, true] {
        let prepared = prepare(&p, SOURCE, &args, limits()); let count = prepared.preflight_metadata().chunk_count();
        let mut values = vec![r#"{"a":1}"#.to_owned(); count];
        if conflict { values[count - 1] = r#"{"a":2,"b":"last"}"#.to_owned(); }
        let calls = Rc::new(Cell::new(0));
        let error = run(prepared, &args, values, Rc::clone(&calls), &mut Control::default()).err().unwrap();
        assert_eq!(calls.get(), count);
        match error {
            Int4ObjectExtractionMapError::Reconciliation(actual) => assert_eq!(actual,
                if conflict { RootObjectError::Conflict } else { RootObjectError::MissingRequired }),
            _ => panic!("expected final object reconciliation refusal"),
        }
    }
}

fn grounded_values(prepared: &PreparedInt4ObjectExtractionMap<'_>) -> Vec<String> {
    prepared.mapped.chunks.chunks().iter().map(|chunk| {
        if chunk.text().contains("éA") { r#"{"quote":"éA"}"#.to_owned() } else { "{}".to_owned() }
    }).collect()
}

#[test]
fn final_source_matching_spends_only_the_remaining_original_document_ledger() {
    let p = planner(); let args = map_fixture::args(GROUNDED, ExtractionBatchGrounding::SourceMembership);
    let prepared = prepare(&p, SOURCE, &args, limits()); let expected = prepared.preflight_metadata();
    let values = grounded_values(&prepared);
    let mut output = run(prepared, &args, values, Rc::new(Cell::new(0)), &mut Control::default()).unwrap();
    let additional = output.object.verification_used(); let mapped = output.partial_map.verification_used;
    assert_eq!(additional.fields, 1); assert_eq!(additional.matches, 4);
    assert_eq!(additional.scan_steps, 176);
    assert_eq!(output.object.original_source_fields()[0].occurrence, SourceOccurrence::Ambiguous);
    let spans = &output.object.original_source_fields()[0].spans;
    assert_eq!(spans.iter().map(|span| (span.byte_start, span.byte_end, span.scalar_start, span.scalar_end)).collect::<Vec<_>>(),
        [(0, 3, 0, 2), (3, 6, 2, 4), (12, 15, 10, 12), (15, 18, 12, 14)]);
    assert_eq!(output.verification_used.fields, mapped.fields + additional.fields);
    assert_eq!(output.verification_used.matches, mapped.matches + additional.matches);
    assert_eq!(output.verification_used.scan_steps, mapped.scan_steps + additional.scan_steps);
    let mut exact = limits();
    exact.mapping.verification = GroundingBudget { max_fields: output.verification_used.fields,
        max_matches: output.verification_used.matches, max_scan_steps: output.verification_used.scan_steps };
    let prepared = prepare(&p, SOURCE, &args, exact); let values = grounded_values(&prepared);
    let exact_output = run(prepared, &args, values, Rc::new(Cell::new(0)), &mut Control::default()).unwrap();
    assert_eq!(exact_output.verification_used, output.verification_used);
    for axis in 0..3 {
        let mut short = exact;
        match axis { 0 => short.mapping.verification.max_fields -= 1,
            1 => short.mapping.verification.max_matches -= 1,
            _ => short.mapping.verification.max_scan_steps -= 1 }
        let prepared = prepare(&p, SOURCE, &args, short); let count = prepared.preflight_metadata().chunk_count();
        let values = grounded_values(&prepared); let calls = Rc::new(Cell::new(0));
        let error = run(prepared, &args, values, Rc::clone(&calls), &mut Control::default()).err().unwrap();
        assert_eq!(calls.get(), count);
        assert!(matches!(error, Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Limit | RootObjectError::Source)));
    }
    // A caller cannot erase already spent map scans by lowering both public
    // aggregates while retaining the genuine sealed reconciliation value.
    assert!(output.partial_map.verification_used.scan_steps > 0);
    output.partial_map.verification_used.scan_steps -= 1;
    output.verification_used.scan_steps -= 1;
    assert!(expected.verify_completed(&output).is_err());
}

#[test]
fn sealed_object_cannot_be_substituted_for_another_complete_map_and_public_summary_mutations_refuse() {
    let p = planner(); let args = map_fixture::args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let first = prepare(&p, SOURCE, &args, limits()); let expected = first.preflight_metadata();
    let values = complementary(&first, "1", "first");
    let mut first = run(first, &args, values, Rc::new(Cell::new(0)), &mut Control::default()).unwrap();
    let second = prepare(&p, SOURCE, &args, limits()); let values = complementary(&second, "2", "second");
    let mut second = run(second, &args, values, Rc::new(Cell::new(0)), &mut Control::default()).unwrap();
    expected.verify_completed(&first).unwrap(); expected.verify_completed(&second).unwrap();
    std::mem::swap(&mut first.object, &mut second.object);
    assert!(expected.verify_completed(&first).is_err()); assert!(expected.verify_completed(&second).is_err());
    std::mem::swap(&mut first.object, &mut second.object);
    expected.verify_completed(&first).unwrap();
    first.schema_version = 2; assert!(expected.verify_completed(&first).is_err()); first.schema_version = 1;
    first.execution = "other"; assert!(expected.verify_completed(&first).is_err()); first.execution = INT4_OBJECT_MAP_EXECUTION;
    first.schema_derivation = "other"; assert!(expected.verify_completed(&first).is_err()); first.schema_derivation = ROOT_OBJECT_DERIVATION;
    first.map_prompt = "other"; assert!(expected.verify_completed(&first).is_err()); first.map_prompt = ROOT_OBJECT_MAP_PROMPT;
    first.verification_used.fields += 1; assert!(expected.verify_completed(&first).is_err()); first.verification_used.fields -= 1;
    first.untrusted_fields = ["object", "partial_map"]; assert!(expected.verify_completed(&first).is_err());
}

#[test]
fn wrapper_refuses_unadmitted_policies_wrong_grounding_and_unsupported_schedule_combinations() {
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 });
    assert!(Int4ObjectExtractionPlanner::new(map_fixture::planner().with_document_encoding(encoding).unwrap()).is_err());
    assert!(Int4ObjectExtractionPlanner::new(map_fixture::planner()
        .with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: NANBEIGE_VOCAB_SIZE }).unwrap()).is_err());
    let mut defaults = map_fixture::planner();
    defaults.compiler.defaults = Some(map_fixture::args(STRUCTURAL, ExtractionBatchGrounding::Structural));
    assert!(Int4ObjectExtractionPlanner::new(defaults).is_err());
    let p = planner();
    for (schema, grounding) in [(GROUNDED, ExtractionBatchGrounding::Structural),
        (STRUCTURAL, ExtractionBatchGrounding::SourceMembership)] {
        let args = map_fixture::args(schema, grounding);
        assert!(p.plan_document_with_control(SOURCE, &args, limits(), &mut Control::default()).is_err());
    }
    for axis in 0..4 {
        let mut invalid = limits();
        match axis { 0 => invalid.mapping.mapping.reduction.map_batch_chunks = 2,
            1 => invalid.mapping.root_array_union = Some(map_fixture::union_limits()),
            2 => invalid.reconciliation.max_json_bytes = invalid.mapping.mapping.reduction.max_result_bytes + 1,
            _ => invalid.reconciliation.max_result_bytes = invalid.mapping.mapping.reduction.max_result_bytes + 1 }
        assert!(invalid.reconciliation_workspace_bytes().is_err());
    }
}

#[test]
fn original_control_covers_derivation_all_maps_reconciliation_and_final_publication() {
    let p = planner(); let args = map_fixture::args(STRUCTURAL, ExtractionBatchGrounding::Structural);
    let mut planning = Control::default();
    p.plan_document_with_control(SOURCE, &args, limits(), &mut planning).unwrap();
    let prepared = prepare(&p, SOURCE, &args, limits()); let values = complementary(&prepared, "1", "last");
    let mut execution = Control::default();
    run(prepared, &args, values, Rc::new(Cell::new(0)), &mut execution).unwrap();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota] {
        for stop in [1, planning.polls / 2, planning.polls] {
            let mut control = Control { polls: 0, stop, cause };
            let error = p.plan_document_with_control(SOURCE, &args, limits(), &mut control).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause)); assert_eq!(control.polls, stop);
        }
        for stop in [1, execution.polls / 2, execution.polls] {
            let prepared = prepare(&p, SOURCE, &args, limits()); let values = complementary(&prepared, "1", "last");
            let mut control = Control { polls: 0, stop, cause };
            let error = run(prepared, &args, values, Rc::new(Cell::new(0)), &mut control).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause)); assert_eq!(control.polls, stop);
        }
    }
}
