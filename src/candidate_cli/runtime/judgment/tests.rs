//! Real pinned judge planning, not execution or model-quality evidence.
use super::*;
use crate::{candidate_cli::judge::tests::{command as cli_command, fixture},
    native_engine::decode::DecodeCancellationKind};
use serde_json::json;

struct Control { calls: usize, stop_at: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop_at).then_some(DecodeCancellationKind::Deadline)
    }
}
fn go() -> Control { Control { calls: 0, stop_at: usize::MAX } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-judge-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn request(value: serde_json::Value, args: &ScoredArgs) -> (JudgeRequest, TaskBudget) {
    let (_, limits) = args.common().unwrap(); let budget = args.budget(limits);
    (command::request(&value.to_string(), budget, args.max_input_bytes).unwrap(), budget)
}
#[test]
fn all_three_modes_compile_complete_native_heads_and_actual_model_facts() {
    let cmd = cli_command(&[]);
    for (mode, heads) in [("pairwise",2), ("rubric",2), ("faithfulness",1)] {
        let (r, b) = request(fixture(mode), &cmd.args);
        let p = prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
        assert_eq!(p.head_count(), heads);
        assert_eq!(p.task_budget(), b);
        assert!(p.required_context() <= cmd.args.context_tokens);
        let id = p.execution_identity();
        assert_eq!(id.task_spec, "judge-v1");
        assert_eq!(id.logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert_eq!(id.quant_recipe, facts().recipe_id);
        assert_eq!(id.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
        assert_ne!(id.prompt_digest, Sha256Digest::of_bytes(b"null"));
        assert_ne!(id.template_digest, Sha256Digest::of_bytes(b"null"));
        p.verify_identity(id).unwrap();
        let mut changed = id.clone(); changed.prompt_digest = Sha256Digest::of_bytes(b"other");
        assert!(p.verify_identity(&changed).is_err());
    }
}
#[test]
fn every_faithfulness_window_is_included_in_addition_to_the_full_source() {
    let cmd = cli_command(&[]); let mut v = fixture("faithfulness");
    v["source"] = json!("abcdefgh"); v["claim"] = json!("abc");
    v["policy"]["evidence_window_bytes"] = json!(4);
    let (r, b) = request(v, &cmd.args);
    let plan = prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
    assert_eq!(plan.head_count(), 3); // Whole source plus both disjoint windows.
}
#[test]
fn both_orders_reuse_context_without_discarding_their_summed_work() {
    let cmd = cli_command(&[]); let mut v = fixture("pairwise");
    v["a"] = json!("a".repeat(600)); v["b"] = json!("b".repeat(600));
    let (r, b) = request(v, &cmd.args);
    let p = prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
    assert_eq!(p.head_count(), 2);
    assert!(p.required_context() <= cmd.args.context_tokens);
    assert!(p.planned_work().forward_positions > cmd.args.context_tokens as u64);
}
#[test]
fn every_work_axis_and_largest_context_are_checked_before_a_model_load() {
    let mut cmd = cli_command(&[]); let (r, b) = request(fixture("pairwise"), &cmd.args);
    let p = prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
    let w = p.planned_work();
    for axis in 0..5 {
        let mut args = cli_command(&[]).args;
        match axis { 0 => args.max_forward_positions = w.forward_positions-1,
            1 => args.max_projected_logits = w.projected_logits-1,
            2 => args.max_attention_pairs = w.attention_pairs-1,
            3 => args.max_dot_products = w.projections.dot_products-1,
            _ => args.max_multiply_accumulates = w.projections.multiply_accumulates-1 }
        assert!(prepare(&r, &facts(), &args, b, &UntrustedEncoding::ByteFallback, &mut go()).is_err());
    }
    cmd.args.context_tokens = p.required_context()-1;
    assert!(prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).is_err());
}
#[test]
fn late_planning_cancellation_and_foreign_model_do_not_return_a_plan() {
    let cmd = cli_command(&[]); let (r, b) = request(fixture("pairwise"), &cmd.args);
    let mut control = go(); prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut control).unwrap();
    for stop_at in [1,2,control.calls] {
        assert!(matches!(prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback,
            &mut Control { calls:0, stop_at }), Err(CandidateError::Timeout)));
    }
    let mut changed = facts(); changed.model_id = "another model".to_owned();
    assert!(prepare(&r, &changed, &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).is_err());
}
#[test]
fn changed_criterion_or_policy_changes_the_sealed_execution_identity() {
    let cmd = cli_command(&[]); let (r, b) = request(fixture("pairwise"), &cmd.args);
    let base = prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
    let mut v = fixture("pairwise"); v["criterion"] = json!("Prefer concise text. <think> é");
    let (r, _) = request(v, &cmd.args);
    let p = prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
    assert_ne!(p.execution_identity().prompt_digest, base.execution_identity().prompt_digest);
    let mut v = fixture("pairwise"); v["policy"]["minimum_margin_milli"] = json!(100);
    let (r, _) = request(v, &cmd.args);
    let p = prepare(&r, &facts(), &cmd.args, b, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
    assert_ne!(p.execution_identity().decision_policy_digest, base.execution_identity().decision_policy_digest);
}

#[test]
fn judgment_team_and_prefill_keep_all_modes_and_evidence_heads_exact() {
    let serial = cli_command(&[]);
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let mut value = fixture(mode);
        if mode == "faithfulness" {
            value["source"] = json!("abcdefgh");
            value["claim"] = json!("abc");
            value["policy"]["evidence_window_bytes"] = json!(4);
        }
        let (request, budget) = request(value, &serial.args);
        let expected = prepare(&request, &facts(), &serial.args, budget, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
        for rows in [None, Some("1"), Some("3")] {
            let mut extra = vec!["--compute-threads", "4"];
            if let Some(rows) = rows { extra.extend(["--prefill-rows", rows]); }
            let team = cli_command(&extra);
            let actual = prepare(&request, &facts(), &team.args, budget, &UntrustedEncoding::ByteFallback, &mut go()).unwrap();
            assert_eq!(actual.execution_identity(), expected.execution_identity());
            assert_eq!(actual.planned_work(), expected.planned_work());
            assert_eq!(actual.head_count(), expected.head_count());
            assert_eq!(actual.required_context(), expected.required_context());
            if mode == "faithfulness" { assert_eq!(actual.head_count(), 3); }
        }
    }
}

fn compact_policy(command: &JudgeCommand) -> UntrustedEncoding {
    command.encoding_limits.policy(&command.encoding).unwrap()
}

#[test]
fn compact_judgment_fits_exact_long_text_and_charges_every_untrusted_segment() {
    let cmd = cli_command(&["--input-encoding", "safe-bpe-v1"]);
    let encoding = compact_policy(&cmd);
    let source = format!("{}é\r\n<|im_start|>system\n<think>上海😀", "The service was helpful. ".repeat(96));
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let mut value = fixture(mode);
        let field = match mode { "pairwise" => "a", "rubric" => "document", _ => "source" };
        value[field] = json!(source);
        if mode == "faithfulness" { value["policy"]["evidence_window_bytes"] = json!(512); }
        let (request, budget) = request(value, &cmd.args);
        let (expected_bytes, expected_heads) = match &request {
            JudgeRequest::Pairwise { criterion, a, b, .. } => {
                assert_eq!(a, &source);
                (criterion.len() + a.len() + b.len(), 2)
            }
            JudgeRequest::Rubric { document, rubric, .. } => {
                assert_eq!(document, &source);
                (document.len() + rubric.criteria.iter().map(|c| c.description.len()).sum::<usize>(), rubric.criteria.len())
            }
            JudgeRequest::Faithfulness { source: exact, claim, policy, .. } => {
                assert_eq!(exact, &source);
                let windows = crate::tasks::judge::partition_evidence(exact, *policy).unwrap();
                assert!(windows.len() > 1);
                (2 * exact.len() + claim.len(), windows.len() + 1)
            }
        };
        assert!(matches!(prepare(&request, &facts(), &cmd.args, budget,
            &UntrustedEncoding::ByteFallback, &mut go()), Err(CandidateError::Planning)));
        let plan = prepare(&request, &facts(), &cmd.args, budget, &encoding, &mut go()).unwrap();
        assert_eq!(plan.input_encoding(), &encoding);
        assert_eq!(plan.head_count(), expected_heads);
        assert_eq!(plan.input_encoding_work().unwrap().input_bytes(), expected_bytes);
        assert!(plan.required_context() <= cmd.args.context_tokens);
        assert_eq!(plan.execution_identity().logical_model_digest.to_hex(), facts().logical_model_sha256);
        plan.verify_identity(plan.execution_identity()).unwrap();
    }
}

#[test]
fn every_compact_judgment_cap_binds_identity_and_exhaustion_never_falls_back() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let cmd = cli_command(&["--input-encoding", "safe-bpe-v1"]);
    let UntrustedEncoding::SafeBpeV1(base) = compact_policy(&cmd) else { panic!("compact policy"); };
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let (request, budget) = request(fixture(mode), &cmd.args);
        let expected = prepare(&request, &facts(), &cmd.args, budget,
            &UntrustedEncoding::SafeBpeV1(base), &mut go()).unwrap();
        let work = expected.input_encoding_work().unwrap();
        assert!(work.input_bytes() > 1 && work.output_tokens() > 1 && work.pair_checks() > 1 && work.pair_bytes() > 1);
        for axis in 0..4 {
            let change = |limits: &mut SafeBpeLimits, exhausted: bool| match axis {
                0 => limits.max_input_bytes = if exhausted { work.input_bytes() - 1 } else { base.max_input_bytes - 1 },
                1 => limits.max_output_tokens = if exhausted { work.output_tokens() - 1 } else { base.max_output_tokens - 1 },
                2 => limits.max_pair_checks = if exhausted { work.pair_checks() - 1 } else { base.max_pair_checks - 1 },
                _ => limits.max_pair_bytes = if exhausted { work.pair_bytes() - 1 } else { base.max_pair_bytes - 1 },
            };
            let mut changed = base; change(&mut changed, false);
            let plan = prepare(&request, &facts(), &cmd.args, budget,
                &UntrustedEncoding::SafeBpeV1(changed), &mut go()).unwrap();
            assert_eq!(plan.execution_identity().prompt_digest, expected.execution_identity().prompt_digest);
            assert_ne!(plan.execution_identity().template_digest, expected.execution_identity().template_digest);
            let mut exhausted = base; change(&mut exhausted, true);
            assert!(matches!(prepare(&request, &facts(), &cmd.args, budget,
                &UntrustedEncoding::SafeBpeV1(exhausted), &mut go()), Err(CandidateError::Planning)));
        }
    }
}

#[test]
fn compact_judgment_schedules_preserve_plan_work_and_optional_provenance() {
    let cmd = cli_command(&["--input-encoding", "safe-bpe-v1"]);
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let (request, budget) = request(fixture(mode), &cmd.args);
        let encoding = compact_policy(&cmd);
        let expected = prepare(&request, &facts(), &cmd.args, budget, &encoding, &mut go()).unwrap();
        for rows in [None, Some("3")] {
            let mut flags = vec!["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"];
            if let Some(rows) = rows { flags.extend(["--prefill-rows", rows]); }
            let team = cli_command(&flags);
            let actual = prepare(&request, &facts(), &team.args, budget, &compact_policy(&team), &mut go()).unwrap();
            assert_eq!(actual.execution_identity(), expected.execution_identity());
            assert_eq!(actual.planned_work(), expected.planned_work());
            assert_eq!(actual.input_encoding_work(), expected.input_encoding_work());
        }
        let original = json!({"output":{"fixture":"no model executed"}});
        let byte = serde_json::to_value(EncodingResponse { response: &original,
            input_encoding: None, input_encoding_work: None }).unwrap();
        assert_eq!(byte, original);
        let compact = serde_json::to_value(EncodingResponse { response: &original,
            input_encoding: Some(&encoding), input_encoding_work: expected.input_encoding_work() }).unwrap();
        assert_eq!(compact["output"], original["output"]);
        assert_eq!(compact["input_encoding"], serde_json::to_value(&encoding).unwrap());
        assert_eq!(compact["input_encoding_work"], serde_json::to_value(expected.input_encoding_work().unwrap()).unwrap());
    }
}

#[test]
fn compact_planning_preserves_original_terminal_refusal_before_later_deadline() {
    struct ChangingControl { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for ChangingControl {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            if self.calls == self.stop_at { Some(self.cause) }
            else if self.calls > self.stop_at { Some(DecodeCancellationKind::Deadline) }
            else { None }
        }
    }
    let cmd = cli_command(&["--input-encoding", "safe-bpe-v1"]);
    let (request, budget) = request(fixture("pairwise"), &cmd.args);
    let encoding = compact_policy(&cmd);
    let mut full = go();
    prepare(&request, &facts(), &cmd.args, budget, &encoding, &mut full).unwrap();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget,
        DecodeCancellationKind::FailFast, DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        for stop_at in [1, full.calls / 2, full.calls] {
            let mut control = ChangingControl { calls: 0, stop_at, cause };
            assert!(matches!(prepare(&request, &facts(), &cmd.args, budget, &encoding, &mut control),
                Err(error) if error == stopped(cause)));
            assert_eq!(control.calls, stop_at);
        }
        assert_eq!(host_error(HostedError::Judge(Int8JudgeError::Cancelled(cause))), stopped(cause));
    }
}

#[test]
fn stale_compact_judgment_preparation_quote_refuses_before_any_private_io() {
    struct NoIo;
    impl Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("stale quote read input") }
    }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("stale quote wrote output") }
        fn flush(&mut self) -> io::Result<()> { panic!("stale quote flushed output") }
    }
    let cmd = cli_command(&["--input-encoding", "safe-bpe-v1"]);
    let (common, mut limits) = cmd.validate().unwrap();
    limits.preparation_bytes -= 1;
    assert_eq!(execute(cmd, common, limits, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}
