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
    assert_eq!(preflight(SOURCE, &request, &planner, exact, &mut continuing()).unwrap(), expected);
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
