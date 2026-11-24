//! Actual pinned schema/chunk preflight only; no successful model execution.
use super::*;
use crate::candidate_cli::map::tests::command;

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-unit-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn fixture() -> (Int8ExtractionBatchPlanner, ExtractionBatchArgs, Int8ExtractionMapLimits) {
    let cmd = command("extract", &["--schema", "s", "--max-chunk-bytes", "16", "--max-new-tokens", "16"]);
    let (_, limits) = cmd.validate().unwrap();
    let planner = compiler::planner(&facts(), &cmd.host, limits, None).unwrap();
    let request = schema::arguments(r#"{"type":"string","maxLength":32}"#.to_owned(), false,
        cmd.host.task_budget(limits)).unwrap();
    (planner, request, cmd.extraction_mapping().unwrap())
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
fn complete_unicode_source_and_compiler_metadata_survive_preflight_and_replanning() {
    let (planner, request, mapping) = fixture();
    let expected = preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1);
    let span = expected.source_span();
    assert_eq!((span.byte_start, span.byte_end), (0, SOURCE.len()));
    assert_eq!((span.scalar_start, span.scalar_end), (0, SOURCE.chars().count()));
    let second = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
    assert_eq!(expected, second.preflight_metadata());
    assert_eq!(second.execution_identities().len(), expected.chunk_count());
    assert_eq!(expected.reserved_mask_visits(), mapping.mapping.mask_visits_per_chunk * expected.chunk_count() as u64);
}
#[test]
fn each_native_work_axis_and_mask_allowance_can_refuse_before_weights() {
    let (planner, request, mapping) = fixture();
    let expected = preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    for axis in 0..6 {
        let mut cap = mapping; cap.mapping.max_model_work = expected.planned_work();
        cap.mapping.max_mask_visits = expected.reserved_mask_visits();
        match axis {
            0 => cap.mapping.max_model_work.forward_positions -= 1,
            1 => cap.mapping.max_model_work.projected_logits -= 1,
            2 => cap.mapping.max_model_work.attention_pairs -= 1,
            3 => cap.mapping.max_model_work.projections.dot_products -= 1,
            4 => cap.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => cap.mapping.max_mask_visits -= 1,
        }
        assert_eq!(preflight(SOURCE, &request, &planner, cap, &mut continuing()), Err(CandidateError::Planning), "{axis}");
    }
    let mut exact = mapping; exact.mapping.max_model_work = expected.planned_work();
    exact.mapping.max_mask_visits = expected.reserved_mask_visits();
    let admitted = preflight(SOURCE, &request, &planner, exact, &mut continuing()).unwrap();
    assert_eq!(admitted.source_span(), expected.source_span());
    assert_eq!(admitted.chunk_count(), expected.chunk_count());
    assert_eq!(admitted.planned_work(), expected.planned_work());
    assert_eq!(admitted.reserved_mask_visits(), expected.reserved_mask_visits());
    // Tightening the declared ceilings preserves geometry but changes the
    // admitted request identity, even when the native work is unchanged.
    assert_ne!(admitted, expected);
    let rebuilt = planner.plan_document_with_control(SOURCE, &request, exact, &mut continuing()).unwrap();
    assert_eq!(admitted, rebuilt.preflight_metadata());
}
#[test]
fn malformed_unsupported_duplicate_and_unfunded_schemas_never_mint_metadata() {
    let (planner, mut request, mapping) = fixture();
    for raw in ["{", r#"{"type":"string","type":"number"}"#, r#"{"$ref":"https://example.invalid/schema"}"#] {
        request.schema = raw.to_owned();
        assert!(preflight(SOURCE, &request, &planner, mapping, &mut continuing()).is_err());
    }
    request.schema = format!(r#"{{"type":"string","const":"{}"}}"#, "x".repeat(request.budget.max_input_tokens as usize));
    assert!(preflight(SOURCE, &request, &planner, mapping, &mut continuing()).is_err());
}
#[test]
fn every_chunk_binds_original_exact_decimal_schema_bytes() {
    let (planner, mut request, mapping) = fixture();
    for raw in [r#"{"type":"number","const":0.12345678901234567890123456789012345678}"#,
        " {\"type\":\"number\",\"const\":0.12345678901234567890123456789012345679}\n"] {
        request.schema = raw.to_owned();
        let prepared = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
        for identity in prepared.execution_identities() {
            assert_eq!(identity.schema_digest, Sha256Digest::of_bytes(raw.as_bytes()));
            assert_eq!(identity.task_spec, "extract-v1");
        }
        assert_eq!(request.schema.as_bytes(), raw.as_bytes());
    }
}
#[test]
fn cancellation_including_final_preflight_checkpoint_never_returns_metadata() {
    let (planner, request, mapping) = fixture();
    let mut complete = continuing();
    preflight(SOURCE, &request, &planner, mapping, &mut complete).unwrap();
    assert!(complete.calls > 2);
    for stop_at in [1, 2, complete.calls] {
        let mut control = Control { stop_at, ..continuing() };
        assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut control), Err(CandidateError::Execution));
    }
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        let mut control = Control { calls: 0, stop_at: 1, cause };
        assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut control), Err(CandidateError::Timeout));
    }
}
#[test]
fn no_source_truncation_or_fresh_budget_when_replanning() {
    let (planner, request, mut mapping) = fixture();
    assert_eq!(preflight("", &request, &planner, mapping, &mut continuing()), Err(CandidateError::Input));
    mapping.mapping.chunks.max_chunks = 1;
    assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut continuing()), Err(CandidateError::Planning));
    mapping.mapping.chunks.max_chunks = 64;
    let mut first = continuing(); preflight(SOURCE, &request, &planner, mapping, &mut first).unwrap();
    first.stop_at = first.calls + 1;
    assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut first), Err(CandidateError::Execution));
}
#[test]
fn source_membership_is_explicit_and_blank_chunks_are_not_skipped() {
    let (planner, mut request, mapping) = fixture();
    request.grounding = crate::batch::extract::ExtractionBatchGrounding::SourceMembership;
    request.schema = r#"{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}"#.to_owned();
    let text = "é\r\n                 上海";
    let expected = preflight(text, &request, &planner, mapping, &mut continuing()).unwrap();
    let prepared = planner.plan_document_with_control(text, &request, mapping, &mut continuing()).unwrap();
    assert_eq!(prepared.execution_identities().len(), expected.chunk_count());
    assert_eq!(expected.source_span().byte_end, text.len());
    request.grounding = crate::batch::extract::ExtractionBatchGrounding::Structural;
    assert!(preflight(text, &request, &planner, mapping, &mut continuing()).is_err());
}

fn compact_fixture(cmd: &MapCommand) -> (Int8ExtractionBatchPlanner, ExtractionBatchArgs, Int8ExtractionMapLimits) {
    let (_, limits) = cmd.validate().unwrap();
    let encoding = cmd.encoding.limits(cmd).unwrap().unwrap();
    let planner = compiler::planner(&facts(), &cmd.host, limits, None).unwrap()
        .with_map_document_encoding(encoding.policy(), encoding.complete).unwrap();
    let request = schema::arguments(r#"{"type":"string","maxLength":32}"#.to_owned(), false,
        cmd.host.task_budget(limits)).unwrap();
    (planner, request, cmd.extraction_mapping().unwrap())
}

#[test]
fn compact_cli_extraction_preflight_host_plans_and_report_keep_one_operation_ledger() {
    let cmd = command("extract", &["--schema", "s", "--input-encoding", "safe-bpe-v1",
        "--max-chunk-bytes", "128", "--max-new-tokens", "16"]);
    let (planner, request, mapping) = compact_fixture(&cmd);
    let source = SOURCE.repeat(32);
    let expected = preflight(&source, &request, &planner, mapping, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1);
    let first = planner.map_document_encoding_work().unwrap().unwrap();
    let planner = Arc::new(planner);
    let prepared = planner.plan_document_with_control(&source, &request, mapping, &mut continuing()).unwrap();
    assert_eq!(expected, prepared.preflight_metadata());
    assert_eq!(expected.source_span().byte_end, source.len());
    assert_eq!(expected.source_span().scalar_end, source.chars().count());
    for identity in prepared.execution_identities() {
        assert_eq!(identity.schema_digest, Sha256Digest::of_bytes(request.schema.as_bytes()));
        assert_eq!(identity.task_spec, "extract-v1");
    }
    let complete = planner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(complete, first.checked_add(first).unwrap());
    assert!(complete.input_bytes() > source.len() * 2);
    let report = serde_json::to_value(encoding::extraction_report(&planner).unwrap().unwrap()).unwrap();
    assert_eq!(report["policy"], serde_json::to_value(planner.document_encoding()).unwrap());
    assert_eq!(report["work_used"], serde_json::to_value(complete).unwrap());
    assert_eq!(report["complete_operation_limits"],
        serde_json::to_value(planner.map_document_encoding_limits().unwrap()).unwrap());
    let (byte, _, _) = fixture();
    assert!(encoding::extraction_report(&byte).unwrap().is_none());
}

#[test]
fn compact_cli_extraction_cannot_refund_preflight_before_hosted_reconstruction() {
    let cmd = command("extract", &["--schema", "s", "--input-encoding", "safe-bpe-v1",
        "--max-chunk-bytes", "128", "--max-new-tokens", "16"]);
    let (planner, request, mapping) = compact_fixture(&cmd);
    preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    let exact = planner.map_document_encoding_work().unwrap().unwrap().input_bytes().to_string();
    let cmd = command("extract", &["--schema", "s", "--input-encoding", "safe-bpe-v1",
        "--max-chunk-bytes", "128", "--max-new-tokens", "16", "--max-total-tokenizer-input-bytes", &exact]);
    let (planner, request, mapping) = compact_fixture(&cmd);
    preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    let spent = planner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(spent.input_bytes().to_string(), exact);
    assert!(planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).is_err());
    assert_eq!(planner.map_document_encoding_work().unwrap().unwrap(), spent);
    assert!(preflight(SOURCE, &request, &planner, mapping, &mut continuing()).is_err());
    assert_eq!(planner.map_document_encoding_work().unwrap().unwrap(), spent);
}

#[test]
fn compact_cli_extraction_preserves_cancellation_at_preflight_and_host_boundaries() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = crate::tasks::source_planning::quantized::Int8SourceError::Cancelled(cause);
        assert_eq!(planning_error(error.into()), cancelled(cause));
        let error = crate::tasks::source_planning::quantized::Int8SourceError::Cancelled(cause);
        assert_eq!(host_error(HostedError::SourceMap(error.into())), cancelled(cause));
    }
}

#[test]
fn document_schedule_envelope_preserves_serial_bytes_and_exact_nested_json_strings() {
    // Serialization fixture only: these strings are never accepted as native
    // task success. Decimal spelling, escapes and whitespace must survive the
    // new compute envelope without a parsed-object round trip or global merge.
    let exact = [
        concat!(r#" {"value":0.12345678901234567890123456789012345678,"label":"\u00e9","text":"<tool_call>\r\n上海"} "#, "\r\n"),
        r#"{"value":0.12345678901234567890123456789012345679,"label":"é"}"#,
    ];
    let output = serde_json::json!({"mapped":{"root":{"value":[
        {"native":{"result":{"output":{"json":exact[0]}}}},
        {"native":{"result":{"output":{"json":exact[1]}}}},
    ]}}});
    let facts = facts();
    let receipt = crate::hosted::Int8ComputeReceipt { schedule: "int8-output-row-team-v1",
        compute_threads: 4, completed_projections: 11, completed_rows: 123, completed_worker_rows: 91 };
    for compact in [false, true] {
        let cmd = command("extract", &["--schema", "s", "--input-encoding", "safe-bpe-v1",
            "--max-chunk-bytes", "128", "--max-new-tokens", "16"]);
        let (planner, request, mapping) = if compact { compact_fixture(&cmd) } else { fixture() };
        preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
        let report = encoding::extraction_report(&planner).unwrap();
        let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
            evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
            source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
            quant_recipe: &facts.recipe_id, output: &output };
        let response = encoding::EncodedResponse { response, input_encoding: report.as_ref() };
        let mut original = Vec::new();
        publish_checked(&response, 65536, &mut original, &|| Ok(())).unwrap();
        for compute in [None, Some(&receipt)] {
            let scheduled = compute::Response { response: &response, compute };
            let mut wire = Vec::new();
            publish_checked(&scheduled, 65536, &mut wire, &|| Ok(())).unwrap();
            if compute.is_none() { assert_eq!(wire, original); }
            let decoded: serde_json::Value = serde_json::from_slice(&wire).unwrap();
            assert_eq!(decoded["output"], output);
            assert_eq!(decoded.get("compute"), compute.map(|value| serde_json::to_value(value).unwrap()).as_ref());
            assert_eq!(decoded.get("input_encoding"), report.as_ref().map(|value| serde_json::to_value(value).unwrap()).as_ref());
            for (index, exact) in exact.iter().enumerate() {
                assert_eq!(decoded["output"]["mapped"]["root"]["value"][index]
                    ["native"]["result"]["output"]["json"].as_str().unwrap().as_bytes(), exact.as_bytes());
            }
        }
    }
}

#[test]
fn native_team_cancellation_preserves_timeout_classification_through_physical_scope_errors() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = match cause { Deadline | Timeout => CandidateError::Timeout,
            _ => CandidateError::Execution };
        for scopes in [0, 1, 2] {
            let mut error = HostedError::Native(crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause));
            for _ in 0..scopes {
                error = HostedError::Scope {
                    source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                    physical_error: Some(Box::new(error)),
                };
            }
            assert_eq!(host_error(error), expected);
        }
    }
}
