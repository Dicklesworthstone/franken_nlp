//! Real pinned schema/chunk preparation; no fixture claims model execution.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

fn fixture(width: usize) -> (Int4ExtractionBatchPlanner, ExtractionBatchArgs, Int4ExtractionMapLimits) {
    let width_arg = width.to_string();
    let command = super::super::tests::command(&["--max-chunk-bytes", "16", "--max-new-tokens", "16",
        "--group-width", &width_arg]);
    let Int4Task::ExtractMap(options) = command.task else { panic!("wrong map task"); };
    let limits = options.validate(&command.args).unwrap();
    let budget = options.extraction.budget(&command.args, limits.common);
    let d = Sha256Digest::of_bytes(b"metadata-only-INT4-map-CLI-fixture");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
    )).unwrap());
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: packing, tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, width,
        budget, options.extraction.compiler(), options.extraction.source_limits(), None).unwrap();
    let request = options.request(r#"{"type":"string","maxLength":32}"#.to_owned(), budget).unwrap();
    (planner, request, limits.mapping)
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
fn complete_source_geometry_and_group_work_survive_preflight_and_host_replanning() {
    let (planner, request, mapping) = fixture(32);
    let expected = preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1);
    let span = expected.source_span();
    assert_eq!((span.byte_start, span.byte_end), (0, SOURCE.len()));
    assert_eq!((span.scalar_start, span.scalar_end), (0, SOURCE.chars().count()));
    assert_eq!(expected.group_width(), 32);
    let second = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
    assert_eq!(expected, second.preflight_metadata());
    assert_eq!(second.execution_identities().len(), expected.chunk_count());
    assert_eq!(expected.reserved_mask_visits(), mapping.mapping.mask_visits_per_chunk * expected.chunk_count() as u64);
    let (other, request, mapping) = fixture(128);
    let other = preflight(SOURCE, &request, &other, mapping, &mut continuing()).unwrap();
    assert_eq!(expected.planned_work().forward_positions, other.planned_work().forward_positions);
    assert_eq!(expected.planned_work().projections.multiply_accumulates, other.planned_work().projections.multiply_accumulates);
    assert!(expected.planned_work().projections.dot_products > other.planned_work().projections.dot_products);
}

#[test]
fn every_native_work_axis_and_mask_cap_refuses_the_complete_preflight() {
    let (planner, request, mapping) = fixture(32);
    let expected = preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    for axis in 0..6 {
        let mut cap = mapping;
        cap.mapping.max_model_work = expected.planned_work();
        cap.mapping.max_mask_visits = expected.reserved_mask_visits();
        match axis { 0 => cap.mapping.max_model_work.forward_positions -= 1,
            1 => cap.mapping.max_model_work.projected_logits -= 1,
            2 => cap.mapping.max_model_work.attention_pairs -= 1,
            3 => cap.mapping.max_model_work.projections.dot_products -= 1,
            4 => cap.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => cap.mapping.max_mask_visits -= 1 }
        assert_eq!(preflight(SOURCE, &request, &planner, cap, &mut continuing()), Err(CandidateError::Planning));
    }
    let mut exact = mapping;
    exact.mapping.max_model_work = expected.planned_work(); exact.mapping.max_mask_visits = expected.reserved_mask_visits();
    assert_eq!(preflight(SOURCE, &request, &planner, exact, &mut continuing()).unwrap(), expected);
}

#[test]
fn exact_schema_decimals_and_whitespace_bind_each_chunk_without_rewriting() {
    let (planner, mut request, mapping) = fixture(32);
    for raw in [r#"{"type":"number","const":0.12345678901234567890123456789012345678}"#,
        " {\"type\":\"number\",\"const\":0.12345678901234567890123456789012345679}\n"] {
        request.schema = raw.to_owned();
        let prepared = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
        for identity in prepared.execution_identities() {
            assert_eq!(identity.schema_digest, Sha256Digest::of_bytes(raw.as_bytes()));
            assert_eq!(identity.task_spec, "extract-v1");
            assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
        }
        assert_eq!(request.schema, raw);
    }
    for raw in ["{", r#"{"type":"boolean","type":"string"}"#, r#"{"$ref":"remote"}"#] {
        request.schema = raw.to_owned();
        assert!(preflight(SOURCE, &request, &planner, mapping, &mut continuing()).is_err());
    }
}

#[test]
fn whole_original_source_cannot_be_truncated_and_blank_chunks_are_retained() {
    let (planner, mut request, mut mapping) = fixture(32);
    assert_eq!(preflight("", &request, &planner, mapping, &mut continuing()), Err(CandidateError::Input));
    mapping.mapping.chunks.max_chunks = 1;
    assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut continuing()), Err(CandidateError::Planning));
    mapping.mapping.chunks.max_chunks = 64;
    request.grounding = crate::batch::extract::ExtractionBatchGrounding::SourceMembership;
    request.schema = r#"{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}"#.to_owned();
    let text = "é\r\n                 上海";
    let expected = preflight(text, &request, &planner, mapping, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1); assert_eq!(expected.source_span().byte_end, text.len());
    request.schema = r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#.to_owned();
    assert!(preflight(text, &request, &planner, mapping, &mut continuing()).is_err());
}

#[test]
fn caller_cancellation_survives_final_preflight_and_never_gets_a_fresh_budget() {
    let (planner, request, mapping) = fixture(32);
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
    complete.stop_at = complete.calls + 1;
    assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut complete), Err(CandidateError::Execution));
}

#[test]
fn cli_workspace_quote_matches_actual_host_for_complete_occurrence_vectors() {
    let command = super::super::tests::command(&[]);
    let Int4Task::ExtractMap(options) = command.task else { panic!("wrong map task"); };
    let limits = options.validate(&command.args).unwrap();
    let budget = options.extraction.budget(&command.args, limits.common);
    let config = Int4ExtractionMapConfig {
        request: options.request(r#"{"type":"boolean"}"#.to_owned(), budget).unwrap(), limits: limits.mapping,
        native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000, cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: limits.common.preparation_bytes, reduction_reserve_bytes: limits.reduction_reserve_bytes,
    };
    for chunks in [1, limits.mapping.mapping.chunks.max_chunks] {
        assert_eq!(config.workspace_bytes(chunks).unwrap(),
            workspace_bytes(budget, limits.mapping, chunks, limits.reduction_reserve_bytes).unwrap());
    }
    assert_eq!(config.workspace_bytes(limits.mapping.mapping.chunks.max_chunks).unwrap(), limits.workspace_bytes);
}

#[test]
fn publication_preserves_independent_json_strings_and_original_coordinates() {
    let exact = "{\"number\":12345678901234567890123456789012345678}";
    let output = serde_json::json!({"execution":"fixture-only", "chunks":[
        {"json":exact,"byte_start":0,"byte_end":10,"scalar_start":0,"scalar_end":6},
        {"json":"{}","byte_start":10,"byte_end":14,"scalar_start":6,"scalar_end":10}
    ]});
    let response = SourceResponse { schema_version: 1, scope: "real-source-current-candidate-int4",
        evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
        source_root_sha256: "fixture-source", model_content_sha256: "fixture-model",
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: "fixture:group=32", group_width: 32,
        integer_kernel: "s_scalar_i32", output: &output, snapshot: None };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["output"]["chunks"][0]["json"], exact);
    assert_eq!(value["output"]["chunks"][1]["byte_start"], 10);
    assert_eq!(value["output"]["chunks"].as_array().unwrap().len(), 2);
    assert_eq!(value["evidence"], "non_authoritative");
    for absent in ["source_dir", "source_manifest", "schema", "snapshot_path", "prompt_digest"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new(); assert!(publish(&response, 1, &mut refused).is_err());
    assert!(refused.is_empty());
}
