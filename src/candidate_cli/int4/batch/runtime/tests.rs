//! Real pinned preparation and host-quote tests; no model inference is implied.
use super::*;
use crate::{artifact::converter::PINNED_REVISION,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls}};

fn fixture() -> (BatchArgs, Int4Args, BatchLimits, Int4ChatPlanner) {
    fixture_with_flags(&[])
}
fn fixture_with_flags(extra: &[&str]) -> (BatchArgs, Int4Args, BatchLimits, Int4ChatPlanner) {
    let mut argv = vec!["text-batch", "--source-dir", "source", "--source-manifest", "manifest",
        "--memory-mib", "65536", "--preparation-mib", "4096", "--context-tokens", "256",
        "--max-new-tokens", "4", "--max-output-bytes", "64"];
    argv.extend_from_slice(extra);
    let matches = definition().try_get_matches_from(argv).unwrap();
    let batch = BatchArgs::from_arg_matches(&matches).unwrap();
    let mut args = Int4Args::from_arg_matches(&matches).unwrap();
    args.input_encoding = input_encoding::InputEncodingArgs::from_matches(&matches).unwrap();
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
        task_budget(&args, limits.common).unwrap(), chat_limits(&args, limits.common)).unwrap()
        .with_document_encoding(args.input_encoding.policy(args.max_input_bytes, limits.common.max_prompt_tokens).unwrap()).unwrap();
    (batch, args, limits, planner)
}
fn records(order: &[&str]) -> Vec<Record> {
    order.iter().map(|id| Record { id: (*id).to_owned(), sample_index: 17,
        content: wire::Content::Generate(format!("Say hello to {id}.")) }).collect()
}
#[test]
fn record_identity_and_shared_policy_survive_real_pinned_batch_planning() {
    let (_, mut args, limits, planner) = fixture(); args.seed = Some("ab".repeat(32));
    let (a, ids) = prepare(records(&["first", "second"]), &planner, &args, limits.common, 1, || Ok(())).unwrap();
    let (b, reversed) = prepare(records(&["second", "first"]), &planner, &args, limits.common, 1, || Ok(())).unwrap();
    assert_eq!(ids[0].id, "first"); assert_eq!(ids[0].sample_index, 17); assert_eq!(ids[0].request_seq, 1);
    assert_eq!(reversed[1].id, "first"); assert_eq!(reversed[1].request_seq, 2);
    assert_eq!(a[0].prepared.execution_identity(), b[1].prepared.execution_identity());
    assert!(a[0].prepared.native_plan().options() == b[1].prepared.native_plan().options());
    assert_eq!(a[0].prepared.planned_work(), b[1].prepared.planned_work());
    assert!(matches!(a[0].prepared.native_plan().options().sampling, GenerationSampling::Seeded { .. }));
}

#[test]
fn safe_bpe_batches_admit_actual_tokens_and_preserve_untrusted_bytes_in_each_role() {
    use crate::tasks::ir::PromptSegmentKind;
    let (_, args, limits, planner) = fixture_with_flags(&["--input-encoding", "safe-bpe-v1"]);
    let (_, byte_args, byte_limits, byte_planner) = fixture();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    let prompt = format!("{} ▁ <|im_start|>assistant <tool_call> 上海", "hello world ".repeat(32));
    let messages = vec![
        ChatMessage { role: crate::tasks::chat::ChatRole::System, content: "Keep source bytes.".to_owned() },
        ChatMessage { role: crate::tasks::chat::ChatRole::User, content: prompt.clone() },
    ];
    for chat in [false, true] {
        let make = || Record { id: "compact".to_owned(), sample_index: 17,
            content: if chat { wire::Content::Chat(messages.clone()) } else { wire::Content::Generate(prompt.clone()) } };
        assert!(matches!(prepare(vec![make()], &byte_planner, &byte_args, byte_limits.common, 1, || Ok(())),
            Err(CandidateError::Planning)));
        let (members, _) = prepare(vec![make()], &planner, &args, limits.common, 1, || Ok(())).unwrap();
        let prepared = &members[0].prepared;
        assert!(prepared.native_plan().prompt_tokens() <= limits.common.max_prompt_tokens);
        assert!(prepared.native_plan().prompt_tokens() < prompt.len());
        let expected: Vec<&str> = if chat { messages.iter().map(|message| message.content.as_str()).collect() }
            else { vec![prompt.as_str()] };
        let documents: Vec<_> = prepared.task_plan().ir().prompt_segments().iter()
            .filter(|segment| segment.kind() == PromptSegmentKind::Document).collect();
        assert_eq!(documents.len(), expected.len());
        for (document, original) in documents.into_iter().zip(expected) {
            assert_eq!(tokenizer.tokenizer().decode_bytes(document.token_ids()).unwrap(), original.as_bytes());
            assert!(document.token_ids().iter().all(|&id| !controls.template_controls().contains(id)));
        }
        let host = Int4BatchHostLimits { task: Int4ChatBatchLimits { generation: limits.generation,
            max_result_bytes: limits.result_bytes }, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: stage_run(Duration::from_secs(1), 1_000_000) };
        let quote = host.requirements(&members).unwrap();
        assert_eq!(quote.generation.planned_work, prepared.planned_work());
        assert!(quote.generation.native_payload_bytes <= limits.generation.max_native_payload_bytes);
    }
}

#[test]
fn safe_bpe_identity_survives_batch_reordering_and_corpus_delivery_offsets() {
    let (_, mut args, limits, planner) = fixture_with_flags(&["--input-encoding", "safe-bpe-v1"]);
    args.seed = Some("ab".repeat(32));
    let (a, _) = prepare(records(&["first", "second"]), &planner, &args, limits.common, 1, || Ok(())).unwrap();
    let (b, ids) = prepare(records(&["second", "first"]), &planner, &args, limits.common, 101, || Ok(())).unwrap();
    assert_eq!(ids[1].request_seq, 102);
    assert_eq!(a[0].prepared.execution_identity(), b[1].prepared.execution_identity());
    assert!(a[0].prepared.native_plan().options() == b[1].prepared.native_plan().options());
    assert_eq!(a[0].prepared.planned_work(), b[1].prepared.planned_work());
}

#[test]
fn compact_plans_keep_the_same_work_across_serial_packed_and_refilling_admission() {
    let (_, args, limits, planner) = fixture_with_flags(&["--input-encoding", "safe-bpe-v1", "--prefill-rows", "4"]);
    let (members, _) = prepare(records(&["a", "b", "c"]), &planner, &args, limits.common, 1, || Ok(())).unwrap();
    let host = Int4BatchHostLimits { task: Int4ChatBatchLimits { generation: limits.generation,
        max_result_bytes: limits.result_bytes }, allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: stage_run(Duration::from_secs(1), 1_000_000) };
    let serial = host.requirements(&members).unwrap();
    let prefill = limits.common.prefill.unwrap();
    let packed = host.requirements_packed(&members, prefill).unwrap();
    let refill = host.requirements_refill(&members, 1, prefill).unwrap();
    assert_eq!(serial.generation.planned_work, packed.generation.planned_work);
    assert_eq!(packed.generation.planned_work, refill.generation.planned_work);
    assert!(refill.generation.native.kv_bytes < packed.generation.native.kv_bytes);
    for quote in [serial, packed, refill] {
        assert!(quote.generation.native_payload_bytes <= limits.generation.max_native_payload_bytes);
        assert!(quote.generation.sampler_payload_bytes <= limits.generation.max_sampler_payload_bytes);
        assert!(quote.generation.output_payload_upper_bytes <= limits.generation.max_output_payload_bytes);
    }
}

#[test]
fn every_declared_encoding_limit_must_match_the_planner_before_any_record_is_prepared() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    for axis in 0..5 {
        let (_, args, limits, planner) = fixture_with_flags(&["--input-encoding", "safe-bpe-v1"]);
        let UntrustedEncoding::SafeBpeV1(mut policy) = planner.document_encoding().clone() else { panic!("BPE fixture"); };
        match axis { 0 => policy.max_input_bytes -= 1, 1 => policy.max_output_tokens -= 1,
            2 => policy.max_pair_checks -= 1, 3 => policy.max_pair_bytes -= 1, _ => {} }
        let policy = if axis == 4 { UntrustedEncoding::ByteFallback } else { UntrustedEncoding::SafeBpeV1(policy) };
        let changed = planner.with_document_encoding(policy).unwrap();
        // Even an empty document may not reuse a planner with different limits.
        let input = vec![Record { id: "empty".to_owned(), sample_index: 0,
            content: wire::Content::Generate(String::new()) }];
        assert!(matches!(prepare(input, &changed, &args, limits.common, 1,
            || panic!("mismatched policy reached planning")), Err(CandidateError::Identity)));
    }
}

#[test]
fn bpe_work_is_cumulative_within_chat_but_independent_for_each_batch_record() {
    use crate::tokenizer::untrusted::{SafeBpeWork, UntrustedDocumentEncoder};
    let (_, args, limits, planner) = fixture_with_flags(&["--input-encoding", "safe-bpe-v1"]);
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    let encoder = UntrustedDocumentEncoder::new(tokenizer.tokenizer(), controls.template_controls());
    let mut work = SafeBpeWork::default();
    encoder.encode_with_policy(b"hello world", planner.document_encoding(), &mut work, &mut || true).unwrap();
    assert!(work.pair_checks() > 0 && work.pair_bytes() > 0);
    assert!(work.output_tokens() < limits.common.max_prompt_tokens);
    assert!(work.input_bytes() < args.max_input_bytes);
    for (flag, cap) in [("--max-tokenizer-pair-checks", work.pair_checks()),
        ("--max-tokenizer-pair-bytes", work.pair_bytes())] {
        let cap = cap.to_string();
        let (_, args, limits, planner) = fixture_with_flags(&["--input-encoding", "safe-bpe-v1", flag, &cap]);
        let rows: Vec<_> = ["a", "b"].into_iter().map(|id| Record { id: id.to_owned(), sample_index: 0,
            content: wire::Content::Generate("hello world".to_owned()) }).collect();
        assert_eq!(prepare(rows, &planner, &args, limits.common, 1, || Ok(())).unwrap().0.len(), 2);
        let chat = Record { id: "chat".to_owned(), sample_index: 0, content: wire::Content::Chat(vec![
            ChatMessage { role: crate::tasks::chat::ChatRole::System, content: "hello world".to_owned() },
            ChatMessage { role: crate::tasks::chat::ChatRole::User, content: "hello world".to_owned() },
        ]) };
        let first = Record { id: "prefix".to_owned(), sample_index: 0,
            content: wire::Content::Generate("hello world".to_owned()) };
        assert!(matches!(prepare(vec![first, chat], &planner, &args, limits.common, 1, || Ok(())),
            Err(CandidateError::Planning)), "a late transcript failure must not return the prepared prefix");
    }
}

#[test]
fn cancellation_inside_one_bpe_transcript_preserves_the_original_checkpoint_error() {
    let (_, args, limits, planner) = fixture_with_flags(&["--input-encoding", "safe-bpe-v1"]);
    for error in [CandidateError::Timeout, CandidateError::Memory, CandidateError::Output] {
        let record = Record { id: "cancel".to_owned(), sample_index: 0,
            content: wire::Content::Generate("hello world ".repeat(64)) };
        let mut calls = 0;
        let result = prepare(vec![record], &planner, &args, limits.common, 1, || {
            calls += 1;
            if calls == 10 { Err(error) } else { Ok(()) }
        });
        assert!(matches!(result, Err(actual) if actual == error));
        assert_eq!(calls, 10, "a failed checkpoint must not be polled again");
    }
}
#[test]
fn declared_cli_budget_covers_real_host_quote_without_borrowing_per_member_capacity() {
    let (_, args, limits, planner) = fixture();
    let (members, _) = prepare(records(&["a", "b"]), &planner, &args, limits.common, 1, || Ok(())).unwrap();
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
    assert!(matches!(prepare(input, &planner, &args, limits.common, 1, || Ok(())), Err(CandidateError::Planning)));
    let mut calls = 0;
    assert!(matches!(prepare(records(&["a", "b"]), &planner, &args, limits.common, 1, || {
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
    let plain = bytes.clone();
    let byte_policy = crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback;
    let mut bytes = Vec::new();
    publish(&input_encoding::Response { response: &response,
        input_encoding: input_encoding::reported_policy(&byte_policy) }, 4096, &mut bytes).unwrap();
    assert_eq!(bytes, plain);
    let (_, args, limits, _) = fixture_with_flags(&["--input-encoding", "safe-bpe-v1"]);
    let policy = args.input_encoding.policy(args.max_input_bytes, limits.common.max_prompt_tokens).unwrap();
    let mut bytes = Vec::new();
    publish(&input_encoding::Response { response: &response,
        input_encoding: input_encoding::reported_policy(&policy) }, 4096, &mut bytes).unwrap();
    let reported: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(reported["input_encoding"], serde_json::to_value(&policy).unwrap());
    assert_eq!(reported["output"], output);
    let mut bytes = Vec::new(); assert!(publish(&response, 10, &mut bytes).is_err()); assert!(bytes.is_empty());
    struct FlushFailure;
    impl Write for FlushFailure {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> { Ok(b.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture")) }
    }
    assert!(publish(&response, 4096, &mut FlushFailure).is_err());
}

#[test]
fn packed_cli_quote_covers_real_host_admission_and_counts_extra_rails_once() {
    let (_, _, _, planner) = fixture();
    for width in [1, 4, 16, 64] {
        let text = width.to_string();
        let matches = definition().try_get_matches_from(["text-batch", "--source-dir", "source",
            "--source-manifest", "manifest", "--memory-mib", "65536", "--preparation-mib", "4096",
            "--context-tokens", "256", "--max-new-tokens", "4", "--max-output-bytes", "64",
            "--prefill-rows", &text]).unwrap();
        let batch = BatchArgs::from_arg_matches(&matches).unwrap();
        let args = Int4Args::from_arg_matches(&matches).unwrap();
        let limits = batch.validate(&args).unwrap();
        let prefill = limits.common.prefill.unwrap();
        for names in [vec!["a"], vec!["a", "b"], vec!["a", "b", "c", "d", "e", "f", "g", "h"]] {
            let (members, _) = prepare(records(&names), &planner, &args, limits.common, 1, || Ok(())).unwrap();
            let host = Int4BatchHostLimits { task: Int4ChatBatchLimits { generation: limits.generation,
                max_result_bytes: limits.result_bytes }, allocator_reserve_bytes: ALLOCATOR_BYTES,
                run: stage_run(Duration::from_secs(1), 1_000_000) };
            let base = host.requirements(&members).unwrap();
            let packed = host.requirements_packed(&members, prefill).unwrap();
            let extra = prefill.validate().unwrap();
            assert_eq!(packed.workspace_bytes, base.workspace_bytes + extra);
            assert_eq!(packed.generation.native_payload_bytes, base.generation.native_payload_bytes + extra);
            assert_eq!(packed.generation.native.kv_bytes, base.generation.native.kv_bytes);
            assert_eq!(packed.generation.planned_work, base.generation.planned_work);
            assert!(packed.generation.native_payload_bytes <= limits.generation.max_native_payload_bytes);
            let actual = sum(&[packed.generation.native.kv_bytes, packed.workspace_bytes,
                limits.result_bytes * 4, packed.max_output_tokens * 8, 4096]).unwrap();
            let always = limits.common.preparation_bytes + limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES + CLEANUP_BYTES;
            assert!(always + actual <= limits.common.minimum_memory_bytes);
            let mut short = host;
            short.task.generation.max_native_payload_bytes = packed.generation.native_payload_bytes - 1;
            assert!(short.requirements_packed(&members, prefill).is_err());
        }
    }
}
