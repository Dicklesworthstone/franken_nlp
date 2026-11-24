//! Actual original/partial schema and chunk preflight; no model success claims.
use super::*;
use crate::{candidate_cli::map::tests::command,
    tasks::extract::object_reconcile::RootObjectError};

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-unit-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn fixture(compact: bool) -> (Int8ObjectExtractionPlanner, ExtractionBatchArgs, Int8ObjectExtractionMapLimits) {
    let mut flags = vec!["--schema", "s", "--reconcile-object", "--max-chunk-bytes", "16", "--max-new-tokens", "16"];
    if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
    let mut command = command("extract", &flags);
    command.host.memory_mib = 16384;
    let (_, limits) = command.validate().unwrap();
    let mut base = compiler::planner(&facts(), &command.host, limits, None).unwrap();
    if let Some(encoding) = command.encoding.limits(&command).unwrap() {
        base = base.with_map_document_encoding(encoding.policy(), encoding.complete).unwrap();
    }
    let request = schema::arguments(
        r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}"#.to_owned(),
        false, command.host.task_budget(limits)).unwrap();
    let mapping = Int8ObjectExtractionMapLimits { mapping: command.extraction_mapping().unwrap(),
        reconciliation: command.extraction.object_limits(&command).unwrap().unwrap() };
    (Int8ObjectExtractionPlanner::new(base).unwrap(), request, mapping)
}
struct Control { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop_at).then_some(self.cause)
    }
}
fn continuing() -> Control { Control { calls: 0, stop_at: usize::MAX, cause: DecodeCancellationKind::PollQuota } }
const SOURCE: &str = "é\r\n上海 source <tool_call> and another paragraph\r\n";

#[test]
fn object_preflight_retains_complete_unicode_source_and_exact_original_schema_authority() {
    for compact in [false, true] {
        let (planner, request, mapping) = fixture(compact);
        let original = request.schema.clone();
        let expected = preflight_object(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
        assert!(expected.chunk_count() > 1);
        assert_eq!(expected.source_span().byte_end, SOURCE.len());
        assert_eq!(expected.source_span().scalar_end, SOURCE.chars().count());
        let prepared = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
        assert_eq!(expected, prepared.preflight_metadata());
        assert_eq!(prepared.execution_identities().len(), expected.chunk_count());
        for identity in prepared.execution_identities() {
            // Executable chunks bind the separately derived partial schema.
            assert_ne!(identity.schema_digest, Sha256Digest::of_bytes(original.as_bytes()));
            assert_eq!(identity.task_spec, "extract-v1");
        }
        assert_eq!(request.schema.as_bytes(), original.as_bytes());
        assert_eq!(expected.reserved_mask_visits(),
            mapping.mapping.mapping.mask_visits_per_chunk * expected.chunk_count() as u64);
    }
}

#[test]
fn object_preflight_refuses_every_underfunded_work_axis_and_mask_before_weights() {
    let (planner, request, mapping) = fixture(false);
    let expected = preflight_object(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    for axis in 0..6 {
        let mut cap = mapping;
        cap.mapping.mapping.max_model_work = expected.planned_work();
        cap.mapping.mapping.max_mask_visits = expected.reserved_mask_visits();
        match axis {
            0 => cap.mapping.mapping.max_model_work.forward_positions -= 1,
            1 => cap.mapping.mapping.max_model_work.projected_logits -= 1,
            2 => cap.mapping.mapping.max_model_work.attention_pairs -= 1,
            3 => cap.mapping.mapping.max_model_work.projections.dot_products -= 1,
            4 => cap.mapping.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => cap.mapping.mapping.max_mask_visits -= 1,
        }
        assert_eq!(preflight_object(SOURCE, &request, &planner, cap, &mut continuing()),
            Err(CandidateError::Planning), "{axis}");
    }
}

#[test]
fn unsupported_nonobject_duplicate_and_oversized_schemas_never_mint_object_metadata() {
    let (planner, mut request, mapping) = fixture(false);
    for raw in ["{", r#"{"type":"string","maxLength":4}"#,
        r#"{"type":"object","type":"object","properties":{},"additionalProperties":false}"#,
        r#"{"$ref":"https://example.invalid/schema"}"#] {
        request.schema = raw.to_owned();
        assert_eq!(preflight_object(SOURCE, &request, &planner, mapping, &mut continuing()),
            Err(CandidateError::Planning));
    }
    request.schema = format!(r#"{{"type":"object","additionalProperties":false,"properties":{{"v":{{"type":"string","const":"{}"}}}}}}"#,
        "x".repeat(request.budget.max_input_tokens as usize));
    assert!(preflight_object(SOURCE, &request, &planner, mapping, &mut continuing()).is_err());
}

#[test]
fn final_original_required_set_remains_bound_even_with_identical_partial_chunk_schemas() {
    let (planner, mut request, mapping) = fixture(false);
    request.schema = r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a"]}"#.to_owned();
    let first = preflight_object(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    request.schema = r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["b"]}"#.to_owned();
    let second = preflight_object(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    assert_eq!(first.source_span(), second.source_span());
    assert_eq!(first.planned_work(), second.planned_work());
    assert_ne!(first, second);
}

#[test]
fn original_source_annotations_and_blank_chunks_keep_one_remaining_verification_budget() {
    let (planner, mut request, mapping) = fixture(false);
    request.schema = r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}},"required":["a"]}"#.to_owned();
    request.grounding = crate::batch::extract::ExtractionBatchGrounding::SourceMembership;
    let source = "é\r\n                 上海";
    let expected = preflight_object(source, &request, &planner, mapping, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1);
    assert_eq!(expected.source_span().byte_end, source.len());
    request.grounding = crate::batch::extract::ExtractionBatchGrounding::Structural;
    assert_eq!(preflight_object(source, &request, &planner, mapping, &mut continuing()),
        Err(CandidateError::Planning));
}

#[test]
fn compact_object_preflight_and_host_reconstruction_spend_the_same_operation_ledger() {
    let (planner, request, mapping) = fixture(true);
    let expected = preflight_object(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    let first = planner.map_document_encoding_work().unwrap().unwrap();
    let planner = Arc::new(planner);
    let prepared = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
    assert_eq!(expected, prepared.preflight_metadata());
    let complete = planner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(complete, first.checked_add(first).unwrap());
    let report = serde_json::to_value(encoding::extraction_report(planner.base_planner()).unwrap().unwrap()).unwrap();
    assert_eq!(report["work_used"], serde_json::to_value(complete).unwrap());
    assert_eq!(report["complete_operation_limits"],
        serde_json::to_value(planner.map_document_encoding_limits().unwrap()).unwrap());
}

#[test]
fn cancellation_at_first_schema_and_final_preflight_checkpoint_returns_no_metadata() {
    let (planner, request, mapping) = fixture(false);
    let mut complete = continuing();
    preflight_object(SOURCE, &request, &planner, mapping, &mut complete).unwrap();
    for stop_at in [1, 2, complete.calls] {
        let mut stop = Control { stop_at, ..continuing() };
        assert_eq!(preflight_object(SOURCE, &request, &planner, mapping, &mut stop), Err(CandidateError::Execution));
    }
    assert_eq!(preflight_object("", &request, &planner, mapping, &mut continuing()), Err(CandidateError::Input));
    let mut truncated = mapping;
    truncated.mapping.mapping.chunks.max_chunks = 1;
    assert_eq!(preflight_object(SOURCE, &request, &planner, truncated, &mut continuing()), Err(CandidateError::Planning));
}

#[test]
fn object_cancellation_cause_survives_map_reconciliation_and_nested_physical_scope_wrappers() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for scopes in [0, 1, 2] {
            let mut error = HostedError::Int8ObjectExtractionMap(
                Int8ObjectExtractionMapError::Reconciliation(RootObjectError::Cancelled(cause)));
            for _ in 0..scopes {
                error = HostedError::Scope {
                    source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                    physical_error: Some(Box::new(error)),
                };
            }
            assert_eq!(object_execution_error(error), cancelled(cause));
        }
        let error = Int8ObjectExtractionMapError::Map(Int8SourceMapError::Source(
            crate::tasks::source_planning::quantized::Int8SourceError::Cancelled(cause)));
        assert_eq!(object_planning_error(error), cancelled(cause));
    }
}

#[test]
fn object_wire_keeps_exact_partial_and_final_json_strings_with_explicit_mode_metadata() {
    // Serialization only. This fixture is never accepted as native success.
    let exact = [
        r#" {"a":0.12345678901234567890123456789012345678,"text":"\u00e9\r\n上海"} "#,
        r#"{"a":0.12345678901234567890123456789012345678,"text":"é\r\n上海"}"#,
    ];
    let output = serde_json::json!({"partial_map":{"json":exact[0]},"object":{"json":exact[1]}});
    let (_, _, mapping) = fixture(false);
    let facts = facts();
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &output };
    let response = ObjectResponse { response, root_object: ObjectSelection::new(mapping.reconciliation) };
    let receipt = crate::hosted::Int8ComputeReceipt { schedule: "int8-output-row-team-v1", compute_threads: 4,
        completed_projections: 11, completed_rows: 123, completed_worker_rows: 91 };
    for compute in [None, Some(&receipt)] {
        let response = compute::Response { response: &response, compute };
        let mut wire = Vec::new();
        publish_checked(&response, 65536, &mut wire, &|| Ok(())).unwrap();
        let decoded: serde_json::Value = serde_json::from_slice(&wire).unwrap();
        assert_eq!(decoded["output"]["partial_map"]["json"].as_str().unwrap().as_bytes(), exact[0].as_bytes());
        assert_eq!(decoded["output"]["object"]["json"].as_str().unwrap().as_bytes(), exact[1].as_bytes());
        assert_eq!(decoded["root_object"]["policy"], ROOT_OBJECT_RECONCILIATION);
        assert_eq!(decoded["root_object"]["schema_derivation"], ROOT_OBJECT_DERIVATION);
        assert_eq!(decoded["root_object"]["map_prompt"], ROOT_OBJECT_MAP_PROMPT);
        assert_eq!(decoded.get("compute"), compute.map(|value| serde_json::to_value(value).unwrap()).as_ref());
    }
}
