//! Closed scripted population mechanics; no native parity or quality claim.
use super::*;

const OBJECT_SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"required":["a","b"],"properties":{"a":{"type":"number"},"b":{"type":"string","maxLength":16}}}"#;

fn object_config() -> ExtractionDocumentBatchConfig {
    let mut c = config();
    c.request.schema = OBJECT_SCHEMA.to_owned();
    c.request.grounding = ExtractionBatchGrounding::Structural;
    c
}
fn reconciliation() -> RootObjectLimits {
    RootObjectLimits { max_properties: 16, max_origins: 128, max_input_bytes: 1 << 20,
        max_json_bytes: 1 << 20, max_nodes: 65536, max_result_bytes: 1 << 20 }
}
fn compiler(c: &ExtractionDocumentBatchConfig) -> DocumentCompiler {
    DocumentCompiler::Map(planner(c))
        .with_root_object_reconciliation(&c.request, c.limits, reconciliation()).unwrap()
}
fn prepare_object(p: &DocumentCompiler, c: &ExtractionDocumentBatchConfig,
    d: &ObjectScript, l: &mut Ledger, source: &str) -> PreparedExtractionDocument {
    prepare_document(p, c, d, l, BatchDocument {
        id: "opaque".to_owned(), text: source.to_owned(), task_args: None,
    }, &mut Control::default()).unwrap()
}
fn complementary(chunks: usize) -> Vec<String> {
    assert!(chunks >= 2);
    let mut values = vec!["{}".to_owned(); chunks];
    values[0] = r#"{"a":12345678901234567890123456789012345678}"#.to_owned();
    values[chunks - 1] = r#"{"b":"last"}"#.to_owned();
    values
}
struct ObjectScript { schema: String, grounded: bool, values: Vec<String>, calls: usize }
impl ObjectScript {
    fn new(c: &ExtractionDocumentBatchConfig) -> Self {
        let schema = RootObjectSchema::derive(&c.request.schema, CompileLimits::default(),
            reconciliation(), &mut Control::default()).unwrap();
        Self { schema: schema.partial_schema().to_owned(), grounded: schema.requires_source(),
            values: Vec::new(), calls: 0 }
    }
}
impl Driver for ObjectScript {
    fn capacity(&self) -> usize { 8192 }
    fn clean(&self) -> bool { true }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction,
        _: &ExecutionIdentity, _: Int8JsonBudget, _: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        let source = prepared.source().text();
        let json = if self.grounded {
            if source.contains("éA") { r#"{"quote":"éA"}"#.to_owned() } else { "{}".to_owned() }
        } else { self.values[self.calls].clone() };
        self.calls += 1;
        let program = if self.grounded {
            JsonProgram::compile_with_source(&self.schema, source, CompileLimits::default(),
                SourceRuntimeLimits::default()).unwrap()
        } else { JsonProgram::compile(&self.schema, CompileLimits::default()).unwrap() };
        let options = prepared.extraction_plan().options();
        let token = (0..NANBEIGE_VOCAB_SIZE as u32)
            .find(|id| !options.excluded_token_ids.contains(id) && *id != options.eos_token_id).unwrap();
        let work = constrained_int8::planned_work(prepared.extraction_plan().prompt_tokens(), 2).unwrap();
        Ok(Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.to_owned(), model_work: work,
            result: ExtractResult { schema_version: if self.grounded { 2 } else { 1 },
                task_spec_version: "extract-v1".to_owned(), score_space: ScoreSpace::NotComputed,
                grounding: if self.grounded { ExtractionGrounding::SourceMembership } else { ExtractionGrounding::NotRequested },
                source_fields: program.source_fields(&json).unwrap(),
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
                    token_ids: vec![token, options.eos_token_id], json, forward_positions: work.forward_positions,
                    projected_logits: work.projected_logits, mask_node_visit_charge: 20 } } })
    }
}

#[test]
fn ordinary_document_enum_preserves_exact_map_serialization() {
    let c = config(); let p = planner(&c);
    let (mut d, mut a, mut l) = (Script::default(), Admission::default(), Ledger::default());
    let prepared = prepare(&p, &c, &d, &mut l, "Alice Bob");
    let out = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared,
        context(1, 1), &mut Control::default()).unwrap();
    assert!(matches!(out.result(), Int8ExtractionDocumentRun::Map(_)));
    assert_eq!(serde_json::to_vec(out.result()).unwrap(), serde_json::to_vec(out.result().mapped()).unwrap());
    assert_eq!(out.result().verification_used(), out.result().mapped().verification_used);
}

#[test]
fn only_an_empty_object_can_be_a_record_policy() {
    assert!(serde_json::from_str::<ExtractionDocumentArgs>("{}").is_ok());
    for rejected in ["[]", "null", "false", r#"{"reconcile_object":true}"#,
        r#"{"root_object_reconciliation":{}}"#, r#"{"schema":"private"}"#,
        r#"{"max_origins":1}"#, r#"{"document_encoding":"byte-fallback"}"#] {
        assert!(serde_json::from_str::<ExtractionDocumentArgs>(rejected).is_err());
    }
}

#[test]
fn complementary_objects_complete_multiple_documents_without_renewing_any_run_budget() {
    let c = object_config(); let p = compiler(&c);
    let (mut d, mut a, mut l) = (ObjectScript::new(&c), Admission::default(), Ledger::default());
    let mut work = Int8Work::default(); let mut chunks = 0;
    for (index, source) in ["first second", "third fourth"].into_iter().enumerate() {
        let prepared = prepare_object(&p, &c, &d, &mut l, source);
        let expected = prepared.expected;
        d.values.extend(complementary(expected.chunk_count()));
        work = add_work(work, expected.planned_work()).unwrap(); chunks += expected.chunk_count();
        let out = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared,
            context(index as u64 + 1, index as u64 + 1), &mut Control::default()).unwrap();
        expected.verify_completed_with_control(out.result(), &mut Control::default()).unwrap();
        let Int8ExtractionDocumentRun::Object(run) = out.result() else { panic!("object mode was lost") };
        assert_eq!(run.object.json(), r#"{"a":1.2345678901234567890123456789012345678e37,"b":"last"}"#);
        assert_eq!(run.partial_map.planned_model_work, expected.planned_work());
        assert_eq!(run.object.properties()[0].origins[0].chunk_id, 0);
        assert_eq!(run.object.properties()[1].origins[0].chunk_id, expected.chunk_count() - 1);
        assert_eq!(a.live.get(), 1);
        let encoded = serde_json::to_string(out.result()).unwrap();
        assert!(!encoded.contains("completion_binding")); assert!(!encoded.contains("request_binding"));
        drop(out); assert_eq!(a.live.get(), 0);
    }
    assert_eq!(d.calls, chunks); assert_eq!(a.calls, 2); assert_eq!(l.work, work);
    assert_eq!(l.masks, chunks as u64 * 1000);
    assert_eq!(l.verification, add_verification(c.limits.verification, c.limits.verification).unwrap());
}

#[test]
fn conflict_or_missing_required_property_discards_all_chunks_and_never_refunds_the_attempt() {
    let c = object_config(); let p = compiler(&c);
    for conflict in [false, true] {
        let (mut d, mut a, mut l) = (ObjectScript::new(&c), Admission::default(), Ledger::default());
        let prepared = prepare_object(&p, &c, &d, &mut l, "first second");
        let expected = prepared.expected; let count = expected.chunk_count();
        d.values = vec![r#"{"a":1}"#.to_owned(); count];
        if conflict { d.values[count - 1] = r#"{"a":2,"b":"last"}"#.to_owned(); }
        let error = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared,
            context(1, 1), &mut Control::default()).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
        assert_eq!(d.calls, count); assert_eq!(a.live.get(), 0); assert!(l.failed);
        assert_eq!(l.work, expected.planned_work()); assert_eq!(l.masks, expected.reserved_mask_visits());
        assert_eq!(l.verification, c.limits.verification);
        assert!(prepare_document(&p, &c, &d, &mut l, BatchDocument {
            id: "retry".to_owned(), text: "first second".to_owned(), task_args: None,
        }, &mut Control::default()).is_err());
        assert_eq!(d.calls, count);
    }
}

#[test]
fn original_required_set_and_reconciliation_caps_bind_even_when_partial_native_identities_match() {
    let c = object_config(); let p = compiler(&c);
    for changed_required in [false, true] {
        let (mut d, mut a, mut l) = (ObjectScript::new(&c), Admission::default(), Ledger::default());
        let prepared = prepare_object(&p, &c, &d, &mut l, "first second");
        let expected = prepared.expected;
        let mut other = c.clone(); let mut limits = reconciliation();
        if changed_required {
            other.request.schema = other.request.schema.replace(r#""required":["a","b"]"#, r#""required":["a"]"#);
        } else { limits.max_origins += 1; }
        let foreign = DocumentCompiler::Map(planner(&other))
            .with_root_object_reconciliation(&other.request, other.limits, limits).unwrap();
        let other_plan = foreign.plan(&prepared.source, &other.request, other.limits, &mut Control::default()).unwrap();
        assert_eq!(other_plan.execution_identities().cloned().collect::<Vec<_>>(), prepared.identities);
        assert_ne!(other_plan.preflight_metadata(), expected); drop(other_plan);
        let error = execute_document(&foreign, &other, &mut d, &mut a, &mut l, prepared,
            context(1, 1), &mut Control::default()).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
        assert_eq!(d.calls, 0); assert_eq!(a.calls, 0); assert_eq!(l.work, expected.planned_work());
    }
}

#[test]
fn compact_object_configuration_is_idempotent_without_restoring_spent_work() {
    let c = object_config();
    let p = compiler(&c).with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
    let identity = p.base_planner().base_execution_identity().clone();
    let prepared = prepare_object(&p, &c, &ObjectScript::new(&c), &mut Ledger::default(), "first second");
    let spent = p.base_planner().map_document_encoding_work().unwrap().unwrap();
    assert!(spent.input_bytes() > prepared.source.len());
    let p = p.with_root_object_reconciliation(&c.request, c.limits, reconciliation()).unwrap()
        .with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
    assert_eq!(p.base_planner().base_execution_identity(), &identity);
    assert_eq!(p.base_planner().map_document_encoding_work().unwrap(), Some(spent));
    let mut changed = reconciliation(); changed.max_origins += 1;
    assert!(p.with_root_object_reconciliation(&c.request, c.limits, changed).is_err());
    let p = compiler(&c).with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
    assert!(p.with_document_encoding(UntrustedEncoding::ByteFallback, None).is_err());
}

#[test]
fn a_prepared_compact_object_cannot_transfer_its_work_debt_to_another_adapter() {
    let c = object_config();
    let p = compiler(&c).with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
    let other = compiler(&c).with_document_encoding(compact_policy(), Some(encoding_limits())).unwrap();
    let (mut d, mut a, mut first, mut second) = (
        ObjectScript::new(&c), Admission::default(), Ledger::default(), Ledger::default(),
    );
    let prepared = prepare_object(&p, &c, &d, &mut first, "first second");
    assert!(p.base_planner().map_document_encoding_work().unwrap().unwrap().input_bytes() > 0);
    let error = execute_document(&other, &c, &mut d, &mut a, &mut second, prepared,
        context(1, 1), &mut Control::default()).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(d.calls, 0); assert_eq!(a.calls, 0); assert!(second.failed);
    assert_eq!(second.work, Int8Work::default());
    assert_eq!(other.base_planner().map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
}

#[test]
fn original_source_verification_is_included_in_complete_object_population_accounting() {
    let c = config(); let p = compiler(&c); let source = "éAéA      éAéA";
    let (mut d, mut a, mut l) = (ObjectScript::new(&c), Admission::default(), Ledger::default());
    let prepared = prepare_object(&p, &c, &d, &mut l, source); let expected = prepared.expected;
    let out = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared,
        context(1, 1), &mut Control::default()).unwrap();
    let Int8ExtractionDocumentRun::Object(run) = out.result() else { panic!("object mode was lost") };
    let extra = run.object.verification_used(); let mapped = run.partial_map.verification_used;
    assert_eq!(extra.fields, 1); assert_eq!(extra.matches, 4);
    assert_eq!(out.result().verification_used(), ExtractionMapVerification {
        fields: mapped.fields + extra.fields, matches: mapped.matches + extra.matches,
        scan_steps: mapped.scan_steps + extra.scan_steps,
    });
    assert_eq!(l.verification, c.limits.verification);
    expected.verify_completed_with_control(out.result(), &mut Control::default()).unwrap();
    drop(out); assert_eq!(a.live.get(), 0);
}

#[test]
fn cancellation_after_all_object_chunks_preserves_cause_and_prevents_publication() {
    let c = object_config(); let p = compiler(&c); let source = "first second";
    let (mut d, mut a, mut l) = (ObjectScript::new(&c), Admission::default(), Ledger::default());
    let prepared = prepare_object(&p, &c, &d, &mut l, source);
    let expected = prepared.expected; d.values = complementary(expected.chunk_count());
    let mut counter = Control::default();
    drop(execute_document(&p, &c, &mut d, &mut a, &mut l, prepared,
        context(1, 1), &mut counter).unwrap());
    for stop in counter.calls.saturating_sub(3)..=counter.calls {
        let (mut d, mut a, mut l) = (ObjectScript::new(&c), Admission::default(), Ledger::default());
        let prepared = prepare_object(&p, &c, &d, &mut l, source);
        d.values = complementary(expected.chunk_count());
        let error = execute_document(&p, &c, &mut d, &mut a, &mut l, prepared,
            context(1, 1), &mut Control { calls: 0, stop: Some(stop) }).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
        assert_eq!(d.calls, expected.chunk_count()); assert_eq!(a.live.get(), 0); assert!(l.failed);
        assert_eq!(l.work, expected.planned_work()); assert_eq!(l.verification, c.limits.verification);
    }
}
