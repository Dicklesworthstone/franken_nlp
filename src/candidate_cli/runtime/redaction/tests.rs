//! Actual pinned source preparation and arithmetic, not full-model inference.
use super::*;
use crate::candidate_cli::redact::tests::command;
use crate::native_engine::decode::DecodeCancellationKind;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "fixture-only".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
#[test]
fn verified_request_prices_two_independent_contexts_and_uses_ner_identity() {
    let cmd = command(&[]); let (_, limits) = cmd.validate().unwrap();
    let (p, _) = planner().unwrap(); let id = source_identity(&facts(), &p, BuiltInTask::Ner).unwrap();
    let config = detector("Alice <tool_call> é 上海", NerOptions::default(), &cmd.request(), &id, &p, &cmd, limits, &mut Continue).unwrap();
    assert_eq!(id.task_spec, "ner-v1"); assert_eq!(config.max_mask_visits, 2 * config.mask_visits_per_pass);
    assert_eq!(config.per_pass.max_kv_bytes, limits.kv_bytes);
    assert!(config.max_model_work.forward_positions > cmd.host.context_tokens as u64);
    assert!(config.planning.max_input_bytes >= cmd.host.max_result_bytes);
    Int8Redactor::new(&p, id, config).unwrap();
}
#[test]
fn no_verify_reserves_only_the_actual_original_plan() {
    let cmd = command(&["--no-verify"]); let (_, limits) = cmd.validate().unwrap();
    let (p, _) = planner().unwrap(); let id = source_identity(&facts(), &p, BuiltInTask::Ner).unwrap();
    let config = detector("Alice", NerOptions::default(), &cmd.request(), &id, &p, &cmd, limits, &mut Continue).unwrap();
    assert_eq!(config.max_mask_visits, config.mask_visits_per_pass);
    assert!(config.max_model_work.forward_positions < cmd.host.context_tokens as u64);
}
#[test]
fn verification_sums_every_work_axis_and_checks_overflow() {
    let first = constrained_int8::planned_work(20, 8).unwrap();
    let second = constrained_int8::planned_work(120, 8).unwrap();
    assert_eq!(work_ceiling(first, true, 128, 8).unwrap(), first.checked_add(second).unwrap());
    assert_eq!(work_ceiling(first, false, 128, 8).unwrap(), first);
    for axis in 0..5 {
        let mut w = first;
        match axis { 0 => w.forward_positions = u64::MAX, 1 => w.projected_logits = u64::MAX,
            2 => w.attention_pairs = u64::MAX, 3 => w.projections.dot_products = u64::MAX,
            _ => w.projections.multiply_accumulates = u64::MAX }
        assert!(work_ceiling(w, true, 128, 8).is_err());
    }
}
#[test]
fn preparation_cancellation_and_overlong_source_do_not_load_weights() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let cmd = command(&[]); let (_, limits) = cmd.validate().unwrap();
    let (p, _) = planner().unwrap(); let id = source_identity(&facts(), &p, BuiltInTask::Ner).unwrap();
    assert!(detector("Alice", NerOptions::default(), &cmd.request(), &id, &p, &cmd, limits, &mut Stop).is_err());
    assert!(detector(&"x".repeat(cmd.host.context_tokens), NerOptions::default(), &cmd.request(), &id, &p, &cmd, limits, &mut Continue).is_err());
}


fn compact(extra: &[&str]) -> RedactCommand {
    let mut args = vec!["--input-encoding", "safe-bpe-v1"];
    args.extend_from_slice(extra);
    let mut command = command(&args);
    command.host.memory_mib = 16_384;
    command
}
fn compact_planner(command: &RedactCommand, limits: Limits) -> Arc<SourceTaskPlanner> {
    let (policy, complete) = command.encoding_policy(limits).unwrap();
    let (planner, _) = planner().unwrap();
    Arc::new(planner.with_map_document_encoding(policy, complete.unwrap()).unwrap())
}
fn prepare_pass(planner: &SourceTaskPlanner, identity: &ExecutionIdentity,
    config: &Int8RedactionConfig, text: &str) -> Result<Int8Work, CandidateError> {
    let request = SourceTaskRequest::Ner { document: text.to_owned(),
        options: config.ner.clone(), budget: config.per_pass };
    let context = PlanContext::new(identity, config.per_pass).unwrap();
    planner.plan_int8_with_control(&request, &context, config.planning, &mut Continue)
        .map(|prepared| prepared.planned_work()).map_err(source_error)
}
const COMPACT_SOURCE: &str = "Alice met Bob in 上海. <think> <|im_start|> Contact alice@example.org.";

#[test]
fn compact_preflight_reconstruction_and_verification_spend_one_original_ledger() {
    for extra in [&[][..], &["--no-verify"][..]] {
        let command = compact(extra); let (_, limits) = command.validate().unwrap();
        let planner = compact_planner(&command, limits);
        let identity = source_identity(&facts(), &planner, BuiltInTask::Ner).unwrap();
        let request = command.request();
        let config = detector(COMPACT_SOURCE, NerOptions::default(), &request, &identity,
            &planner, &command, limits, &mut Continue).unwrap();
        let first = encoding_receipt(&planner).unwrap().unwrap().work;
        assert_eq!(first.input_bytes(), COMPACT_SOURCE.len());
        assert!(first.output_tokens() < COMPACT_SOURCE.len());
        assert!(first.pair_checks() > 0);
        let native_owner = Arc::clone(&planner);
        assert!(Arc::ptr_eq(&planner, &native_owner));
        prepare_pass(&native_owner, &identity, &config, COMPACT_SOURCE).unwrap();
        assert_eq!(encoding_receipt(&planner).unwrap().unwrap().work, first.checked_add(first).unwrap());
        let transformed = "Someone met someone in 上海. <think> <|im_start|> Contact [EMAIL].";
        if request.verify {
            prepare_pass(&native_owner, &identity, &config, transformed).unwrap();
        }
        drop(native_owner);
        let receipt = encoding_receipt(&planner).unwrap().unwrap();
        assert_eq!(receipt.work.input_bytes(),
            2 * COMPACT_SOURCE.len() + if request.verify { transformed.len() } else { 0 });
        assert_eq!(receipt.complete_limits, command.encoding_policy(limits).unwrap().1.unwrap());
        assert_eq!(receipt.policy, planner.document_encoding());
        assert!(receipt.work.input_bytes() as u64 <= receipt.complete_limits.max_input_bytes);
        assert!(receipt.work.output_tokens() as u64 <= receipt.complete_limits.max_output_tokens);
        assert!(receipt.work.pair_checks() <= receipt.complete_limits.max_pair_checks);
        assert!(receipt.work.pair_bytes() <= receipt.complete_limits.max_pair_bytes);
    }
}

#[test]
fn exact_pair_work_is_not_renewed_after_original_preflight() {
    let baseline = compact(&[]); let (_, limits) = baseline.validate().unwrap();
    let planner = compact_planner(&baseline, limits);
    let identity = source_identity(&facts(), &planner, BuiltInTask::Ner).unwrap();
    detector(COMPACT_SOURCE, NerOptions::default(), &baseline.request(), &identity,
        &planner, &baseline, limits, &mut Continue).unwrap();
    let first = encoding_receipt(&planner).unwrap().unwrap().work;
    for (flag, cap) in [("--max-tokenizer-pair-checks", first.pair_checks()),
        ("--max-tokenizer-pair-bytes", first.pair_bytes())] {
        assert!(cap > 0);
        let cap = cap.to_string();
        let command = compact(&[flag, &cap]); let (_, limits) = command.validate().unwrap();
        let planner = compact_planner(&command, limits);
        let identity = source_identity(&facts(), &planner, BuiltInTask::Ner).unwrap();
        let config = detector(COMPACT_SOURCE, NerOptions::default(), &command.request(), &identity,
            &planner, &command, limits, &mut Continue).unwrap();
        let spent = encoding_receipt(&planner).unwrap().unwrap().work;
        assert_eq!(spent, first);
        // Both execution reconstruction and a fresh transformed pass see the
        // exhausted original allowance. Neither retry refunds the preflight.
        for text in [COMPACT_SOURCE, "Someone in 上海."] {
            assert_eq!(prepare_pass(&planner, &identity, &config, text), Err(CandidateError::Planning));
            assert_eq!(encoding_receipt(&planner).unwrap().unwrap().work, spent);
        }
    }
}

#[test]
fn failed_compact_preflight_retains_work_and_refuses_a_fresh_allowance() {
    let command = compact(&["--max-tokenizer-pair-checks", "1"]);
    let (_, limits) = command.validate().unwrap();
    let planner = compact_planner(&command, limits);
    let identity = source_identity(&facts(), &planner, BuiltInTask::Ner).unwrap();
    assert!(matches!(detector(COMPACT_SOURCE, NerOptions::default(), &command.request(), &identity,
        &planner, &command, limits, &mut Continue), Err(CandidateError::Planning)));
    let spent = encoding_receipt(&planner).unwrap().unwrap().work;
    assert_eq!(spent.input_bytes(), COMPACT_SOURCE.len());
    assert_eq!(spent.pair_checks(), 1);
    assert!(matches!(detector(COMPACT_SOURCE, NerOptions::default(), &command.request(), &identity,
        &planner, &command, limits, &mut Continue), Err(CandidateError::Planning)));
    assert_eq!(encoding_receipt(&planner).unwrap().unwrap().work, spent);
}

#[test]
fn compact_cancellation_remains_typed_through_planning_and_host_errors() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    let command = compact(&[]); let (_, limits) = command.validate().unwrap();
    let planner = compact_planner(&command, limits);
    let identity = source_identity(&facts(), &planner, BuiltInTask::Ner).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
        RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = match cause { Timeout | Deadline => CandidateError::Timeout, _ => CandidateError::Execution };
        let error = detector(COMPACT_SOURCE, NerOptions::default(), &command.request(), &identity,
            &planner, &command, limits, &mut Stop(cause)).err().unwrap();
        assert_eq!(error, expected);
        let nested = || Int8SourceError::Planning(
            crate::tasks::source_planning::SourcePlanningError::Extraction(
                crate::tasks::extract::ExtractError::Decode(
                    crate::native_engine::constrained::JsonDecodeError::Cancelled(cause))));
        assert_eq!(source_error(nested()), expected);
        assert_eq!(planning_error(Int8RedactionError::Source(nested())), expected);
        assert_eq!(execution_error(HostedError::Redaction(Int8RedactionError::Source(nested()))), expected);
        assert_eq!(execution_error(HostedError::Redaction(Int8RedactionError::Cancelled(cause))), expected);
    }
    assert_eq!(encoding_receipt(&planner).unwrap().unwrap().work, SafeBpeWork::default());
    assert_eq!(planning_error(Int8RedactionError::Identity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
}

#[test]
fn encoding_receipt_is_public_policy_and_work_while_byte_response_is_unchanged() {
    let value = serde_json::json!({"fixture_kind":"publication-shape-only", "output":{"edited_text":"[EMAIL]"}});
    let (byte, _) = planner().unwrap();
    let response = EncodedResponse { response: &value, input_encoding: encoding_receipt(&byte).unwrap() };
    assert_eq!(serde_json::to_vec(&response).unwrap(), serde_json::to_vec(&value).unwrap());
    let command = compact(&[]); let (_, limits) = command.validate().unwrap();
    let planner = compact_planner(&command, limits);
    let identity = source_identity(&facts(), &planner, BuiltInTask::Ner).unwrap();
    detector(COMPACT_SOURCE, NerOptions::default(), &command.request(), &identity,
        &planner, &command, limits, &mut Continue).unwrap();
    let response = EncodedResponse { response: &value, input_encoding: encoding_receipt(&planner).unwrap() };
    let bytes = serde_json::to_vec(&response).unwrap();
    let mut reported: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(reported["input_encoding"]["policy"], serde_json::to_value(planner.document_encoding()).unwrap());
    assert_eq!(reported["input_encoding"]["complete_limits"],
        serde_json::to_value(planner.map_document_encoding_limits().unwrap()).unwrap());
    assert_eq!(reported["input_encoding"]["work"],
        serde_json::to_value(planner.map_document_encoding_work().unwrap().unwrap()).unwrap());
    reported.as_object_mut().unwrap().remove("input_encoding");
    assert_eq!(reported, value);
    let text = String::from_utf8(bytes).unwrap();
    for private in [COMPACT_SOURCE, "alice@example.org", "prompt_digest", "source_digest", "namespace", "key_commitment"] {
        assert!(!text.contains(private), "receipt must contain only public policy/work");
    }
}

#[test]
fn runtime_rejects_stale_or_invalid_admission_before_private_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("private read"); } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { panic!("publication"); }
        fn flush(&mut self) -> std::io::Result<()> { panic!("flush"); }
    }
    for axis in 0..7 {
        let command = compact(&[]); let (common, mut limits) = command.validate().unwrap();
        match axis {
            0 => limits.memory_bytes -= 1, 1 => limits.weight_bytes -= 1,
            2 => limits.preparation_bytes -= 1, 3 => limits.result_bytes -= 1,
            4 => limits.max_prompt_tokens -= 1, 5 => limits.kv_bytes -= 1,
            _ => limits.compute.threads += 1,
        }
        assert_eq!(execute(command, common, limits, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    for invalid in 0..5 {
        let mut command = compact(&[]); let (common, limits) = command.validate().unwrap();
        match invalid {
            0 => command.host.max_input_bytes = 0,
            1 => command.host.preparation_mib = 0,
            2 => command.host.context_tokens = command.host.max_new_tokens,
            3 => command.host.max_new_tokens = 0,
            _ => command.corpus.ndjson = true,
        }
        assert_eq!(execute(command, common, limits, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}
