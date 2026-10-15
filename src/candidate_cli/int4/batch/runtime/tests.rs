//! Real pinned preparation and host-quote tests; no model inference is implied.
use super::*;
use crate::{artifact::converter::PINNED_REVISION,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls}};

fn fixture() -> (BatchArgs, Int4Args, BatchLimits, Int4ChatPlanner) {
    let matches = definition().try_get_matches_from(["text-batch", "--source-dir", "source", "--source-manifest", "manifest",
        "--memory-mib", "65536", "--preparation-mib", "4096", "--context-tokens", "256",
        "--max-new-tokens", "4", "--max-output-bytes", "64"]).unwrap();
    let batch = BatchArgs::from_arg_matches(&matches).unwrap();
    let args = Int4Args::from_arg_matches(&matches).unwrap();
    let limits = batch.validate(&args).unwrap();
    let d = Sha256Digest::of_bytes(b"synthetic-cli-int4-not-model-evidence");
    let packed = canonjson::canonical_bytes(&(INT4_SOURCE_FORMAT,
        "low-nibble-first-no-padding-f32le-scales-v1", d, args.group_width)).unwrap();
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={}", args.group_width),
        packing_set_digest: Sha256Digest::of_bytes(&packed), tokenizer_digest: d, template_digest: d,
        task_spec: "generate-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = Int4ChatPlanner::pinned(controls.template_controls(), eos, identity, args.group_width,
        task_budget(&args, limits.common).unwrap(), chat_limits(&args, limits.common)).unwrap();
    (batch, args, limits, planner)
}
fn records(order: &[&str]) -> Vec<Record> {
    order.iter().map(|id| Record { id: (*id).to_owned(), sample_index: 17,
        content: wire::Content::Generate(format!("Say hello to {id}.")) }).collect()
}
#[test]
fn record_identity_and_shared_policy_survive_real_pinned_batch_planning() {
    let (_, mut args, limits, planner) = fixture(); args.seed = Some("ab".repeat(32));
    let (a, ids) = prepare(records(&["first", "second"]), &planner, &args, limits.common, || Ok(())).unwrap();
    let (b, reversed) = prepare(records(&["second", "first"]), &planner, &args, limits.common, || Ok(())).unwrap();
    assert_eq!(ids[0].id, "first"); assert_eq!(ids[0].sample_index, 17); assert_eq!(ids[0].request_seq, 1);
    assert_eq!(reversed[1].id, "first"); assert_eq!(reversed[1].request_seq, 2);
    assert_eq!(a[0].prepared.execution_identity(), b[1].prepared.execution_identity());
    assert!(a[0].prepared.native_plan().options() == b[1].prepared.native_plan().options());
    assert_eq!(a[0].prepared.planned_work(), b[1].prepared.planned_work());
    assert!(matches!(a[0].prepared.native_plan().options().sampling, GenerationSampling::Seeded { .. }));
}
#[test]
fn declared_cli_budget_covers_real_host_quote_without_borrowing_per_member_capacity() {
    let (_, args, limits, planner) = fixture();
    let (members, _) = prepare(records(&["a", "b"]), &planner, &args, limits.common, || Ok(())).unwrap();
    let host = Int4BatchHostLimits { task: Int4ChatBatchLimits { generation: limits.generation,
        max_result_bytes: limits.result_bytes }, allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: stage_run(Duration::from_secs(1), 1_000_000) };
    let quote = host.requirements(&members).unwrap();
    assert!(quote.generation.native_payload_bytes <= limits.generation.max_native_payload_bytes);
    assert!(quote.generation.sampler_payload_bytes <= limits.generation.max_sampler_payload_bytes);
    assert!(quote.generation.output_payload_upper_bytes <= limits.generation.max_output_payload_bytes);
    assert_eq!(quote.max_output_tokens, 2 * args.max_new_tokens as u64);
    let actual = sum(&[quote.generation.native.kv_bytes, quote.workspace_bytes,
        limits.result_bytes * 4, quote.max_output_tokens * 8, 4096]).unwrap();
    let always = limits.common.preparation_bytes + limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES + CLEANUP_BYTES;
    assert!(always + actual <= limits.common.minimum_memory_bytes);
    let mut short = host; short.task.generation.max_sampler_payload_bytes = quote.generation.sampler_payload_bytes - 1;
    assert!(short.requirements(&members).is_err());
    let mut short = host; short.task.max_result_bytes = 2 * limits.common.result_bytes as u64 + 16_384 - 1;
    assert!(short.requirements(&members).is_err());
}
#[test]
fn a_late_planning_or_deadline_failure_cannot_return_a_prepared_prefix() {
    let (_, args, limits, planner) = fixture();
    let mut input = records(&["a", "b"]);
    input[1].content = wire::Content::Generate("hello ".repeat(400));
    assert!(matches!(prepare(input, &planner, &args, limits.common, || Ok(())), Err(CandidateError::Planning)));
    let mut calls = 0;
    assert!(matches!(prepare(records(&["a", "b"]), &planner, &args, limits.common, || {
        calls += 1; if calls == 2 { Err(CandidateError::Timeout) } else { Ok(()) }
    }), Err(CandidateError::Timeout)));
}
#[test]
fn aggregate_publication_keeps_mapping_and_native_output_and_refuses_partial_serialization() {
    let items = [Item { id: "first".to_owned(), sample_index: 7, request_seq: 1 }];
    let output = serde_json::json!({"results":[{"result":{"content":"hello"}}],"group_steps":3});
    let response = Response { schema_version: 1, protocol: PROTOCOL, scope: "real-source-current-candidate-int4",
        evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: "pinned", source_root_sha256: "public-source",
        model_content_sha256: "public-model", candidate_format: INT4_SOURCE_FORMAT, quant_recipe: "fixture",
        group_width: 128, integer_kernel: "s_scalar_i32", items: &items, output: &output, snapshot: None };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], PROTOCOL); assert_eq!(value["output"], output);
    assert_eq!(value["items"][0]["id"], "first"); assert_eq!(value["items"][0]["sample_index"], 7);
    assert!(value.get("snapshot").is_none()); assert!(value.get("prompt_digest").is_none());
    let mut bytes = Vec::new(); assert!(publish(&response, 10, &mut bytes).is_err()); assert!(bytes.is_empty());
    struct FlushFailure;
    impl Write for FlushFailure {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> { Ok(b.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture")) }
    }
    assert!(publish(&response, 4096, &mut FlushFailure).is_err());
}
