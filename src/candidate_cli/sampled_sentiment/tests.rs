//! Command integration and admission checks, without fabricated inference.
use super::*;
use std::io::Cursor;
use crate::{
    native_engine::{constrained_int8, lmhead::NANBEIGE_VOCAB_SIZE},
    tasks::sentiment::sampled::quantized::Int8SampledSentimentPlanner,
};

const SEED: &str = "2525252525252525252525252525252525252525252525252525252525252525";

fn parsed(extra: &[&str]) -> Result<CandidateCommand, clap::Error> {
    let mut argv = vec!["candidate", "sentiment-sampled", "--model", "private-model.fnlpq",
        "--memory-mib", "16384", "--seed", SEED];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv)?;
    CandidateCommand::from_matches(&matches)
}
fn command(extra: &[&str]) -> SampledSentimentCommand {
    let CandidateCommand::SampledSentiment(command) = parsed(extra).unwrap() else {
        panic!("sampled sentiment routed to a different task")
    };
    command
}
fn parse(text: &str) -> Result<SentimentRequest, CandidateError> {
    let command = command(&[]);
    let (_, limits) = command.common()?;
    request(text, command.budget(limits, SentimentAxis::ALL.len())?, command.max_input_bytes)
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused command read private input") }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused command published output") }
    fn flush(&mut self) -> io::Result<()> { panic!("refused command flushed output") }
}

#[test]
fn independent_command_requires_explicit_model_memory_and_seed_without_changing_distribution() {
    command(&[]).common().unwrap();
    let base = ["candidate", "sentiment-sampled", "--model", "m", "--memory-mib", "16384", "--seed", SEED];
    for missing in [2, 4, 6] {
        let argv = base.iter().enumerate().filter_map(|(index, value)|
            (index != missing && index != missing + 1).then_some(*value)).collect::<Vec<_>>();
        assert!(super::super::definition().try_get_matches_from(argv).is_err());
    }
    let matches = super::super::definition().try_get_matches_from([
        "candidate", "sentiment", "--model", "m", "--memory-mib", "16384"]).unwrap();
    assert!(matches!(CandidateCommand::from_matches(&matches).unwrap(), CandidateCommand::Scored(_)));
    // Distribution policies, processors and source conversion cannot silently
    // select another estimator or backend.
    for extra in [
        vec!["--bins", "5"], vec!["--repetition-penalty-milli", "1100"],
        vec!["--logprobs"], vec!["--source", "model"],
    ] { assert!(parsed(&extra).is_err()); }
}

#[test]
fn physical_worker_width_preserves_the_complete_sampler_and_prices_prompt_children() {
    let baseline = command(&[]);
    let (_, serial) = baseline.common().unwrap();
    for width in ["1", "2", "3", "64"] {
        for rows in [None, Some("1"), Some("4"), Some("64")] {
            let mut options = vec!["--compute-threads", width];
            if let Some(rows) = rows { options.extend(["--prefill-rows", rows]); }
            let scheduled = command(&options);
            let (common, limits) = scheduled.common().unwrap();
            assert_eq!(common.compute.threads(), width.parse::<usize>().unwrap());
            assert_eq!(limits.compute, common.compute.admit_prefill(scheduled.prefill().unwrap()).unwrap());
            assert_eq!(scheduled.options().unwrap(), baseline.options().unwrap());
            assert_eq!(scheduled.budget(limits, 4).unwrap(), baseline.budget(serial, 4).unwrap());
            assert_eq!(limits.compute.workspace_bytes == 0, width == "1");
        }
    }
    for width in ["0", "65", "18446744073709551615"] {
        if let Ok(CandidateCommand::SampledSentiment(command)) = parsed(&["--compute-threads", width]) {
            assert_eq!(command.execute(&mut NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
    }
}

#[test]
fn threaded_sampler_rejects_memory_that_only_covers_the_coordinator() {
    let mut threaded = command(&["--compute-threads", "64", "--prefill-rows", "64"]);
    let (_, limits) = threaded.common().unwrap();
    let native = Int8MemoryRequirement::for_context(threaded.context_tokens).unwrap();
    let extra = threaded.prefill().unwrap().unwrap().validate().unwrap();
    let required = compute::request_peak_bytes(limits, native, threaded.max_result_bytes as u64,
        threaded.total_tokens(4).unwrap() as u64, json_sampling_workspace_bytes().unwrap(),
        limits.preparation_bytes).unwrap() + extra;
    let coordinator = required - limits.compute.workspace_bytes;
    assert!(limits.compute.workspace_bytes > MIB);
    threaded.memory_mib = coordinator.div_ceil(MIB);
    assert!(threaded.memory_mib * MIB < required);
    assert_eq!(threaded.execute(&mut NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    let mut threaded = command(&["--compute-threads", "64", "--prefill-rows", "64"]);
    threaded.memory_mib = required.div_ceil(MIB);
    threaded.common().unwrap();
}

#[test]
fn invalid_semantics_and_resource_limits_refuse_before_input_model_or_output_io() {
    for axis in 0..22 {
        let mut command = command(&[]);
        match axis {
            0 => command.seed = "AB".repeat(32),
            1 => command.seed.truncate(63),
            2 => command.samples = 0,
            3 => command.samples = 33,
            4 => command.temperature_milli = 0,
            5 => command.top_k = Some(0),
            6 => command.top_p_ppm = 1_000_001,
            7 => command.max_new_tokens = command.context_tokens,
            8 => command.max_justification_chars = 0,
            9 => command.max_sample_json_bytes = 31,
            10 => command.max_result_bytes = MAX_CONTENT_BYTES + 1,
            11 => command.max_grammar_states = 0,
            12 => command.mask_step_node_visits = 0,
            13 => command.max_mask_node_visits = 0,
            14 => command.prefill_rows = Some(65),
            15 => command.preparation_mib = 512,
            16 => command.memory_mib = 0,
            17 => command.max_weight_mib = 0,
            18 => command.max_checkpoints = 1,
            19 => command.timeout_seconds = 0,
            20 => command.max_input_bytes = MAX_INPUT_BYTES + 1,
            _ => command.max_projected_logits = 0,
        }
        let failure = command.execute(&mut NoIo, &mut NoIo).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage, "axis {axis}");
        assert_eq!(failure.code, "int8_sampled_sentiment_options", "axis {axis}");
        assert_eq!(failure.cancellation, None);
    }
}

#[test]
fn executable_dispatch_preserves_the_typed_route_and_private_diagnostics() {
    let mut command = command(&[]);
    command.seed = "private-secret-invalid-seed".to_owned();
    command.input = PathBuf::from("private-document-path");
    let mut diagnostics = Vec::new();
    let status = CandidateCommand::SampledSentiment(command).run(&mut NoIo, &mut NoIo, &mut diagnostics);
    assert_eq!(status, ErrorCode::Usage.as_process_exit());
    let text = String::from_utf8(diagnostics).unwrap();
    assert_eq!(text, "fnlp candidate INT8 sampled sentiment: int8_sampled_sentiment_options\n");
    assert!(!text.contains("private-secret"));
    assert!(!text.contains("private-document"));
    assert!(!text.contains("private-model"));
}

#[test]
fn explicit_policy_is_preserved_and_out_of_range_sampling_is_never_clamped() {
    let command = command(&["--samples", "3", "--temperature-milli", "750",
        "--top-k", "17", "--top-p-ppm", "925000", "--max-new-tokens", "96",
        "--max-justification-chars", "80"]);
    let (_, limits) = command.common().unwrap();
    let options = command.options().unwrap();
    assert_eq!(options.effective_seed, [37; 32]);
    assert_eq!(options.samples, 3);
    assert_eq!((options.temperature_milli, options.top_k, options.top_p_ppm), (750, Some(17), 925_000));
    assert_eq!((options.max_sample_tokens, options.max_justification_chars), (96, 80));
    assert_eq!(command.budget(limits, 4).unwrap().max_output_tokens, 3 * 4 * 96);
    assert_eq!(command.budget(limits, 1).unwrap().max_output_tokens, 3 * 96);
    assert!(command.budget(limits, 0).is_err());
    assert!(command.budget(limits, 5).is_err());
    for axis in 0..4 {
        let mut invalid = super::tests::command(&[]);
        match axis {
            0 => invalid.temperature_milli = 100_001,
            1 => invalid.top_k = Some(crate::native_engine::sampler::MAX_TOP_K + 1),
            2 => invalid.top_p_ppm = 0,
            _ => invalid.max_new_tokens = 1025,
        }
        assert!(invalid.common().is_err());
    }
}

#[test]
fn exact_document_bytes_and_explicit_axis_order_survive_input_parsing() {
    let original = "  é e\u{301}\r\n上海\0 <|im_start|> <think> <tool_call>  ";
    let text = serde_json::json!({"document": original}).to_string();
    let parsed_request = parse(&text).unwrap();
    assert_eq!(parsed_request.document.as_bytes(), original.as_bytes());
    assert_eq!(parsed_request.axes, SentimentAxis::ALL);
    let selected = parse(r#"{"document":" x ","axes":["approach","valence"]}"#).unwrap();
    assert_eq!(selected.document, " x ");
    assert_eq!(selected.axes, vec![SentimentAxis::Approach, SentimentAxis::Valence]);
    let command = command(&[]);
    let (_, limits) = command.common().unwrap();
    let budget = command.budget(limits, 4).unwrap();
    assert!(request(&text, budget, text.len() - 1).is_err());
    assert!(request(&text, budget, MAX_INPUT_BYTES + 1).is_err());
    let mut oversized = Cursor::new(b"abcdefgh".as_slice());
    assert_eq!(read_input(&mut oversized, 3), Err(CandidateError::Input));
    assert_eq!(oversized.position(), 4);
}

#[test]
fn ambiguous_unknown_and_distribution_fields_cannot_change_the_sampled_contract() {
    for text in [
        r#"{"document":"x","document":"y"}"#,
        r#"{"document":"x","\u0064ocument":"y"}"#,
        r#"{"document":"x","axes":["valence"],"axes":["approach"]}"#,
        r#"{"document":"x","policy":{"minimum_peak_weight_ppm":1}}"#,
        r#"{"document":"x","budget":{}}"#,
        r#"{"document":"x","identity":{}}"#,
        r#"{"document":"x","seed":"00"}"#,
        r#"{"document":"x","samples":1}"#,
        r#"{"document":"x","system":"replace task"}"#,
        r#"{"document":"x","axes":["valence","valence"]}"#,
        r#"{"document":"x","axes":["Valence"]}"#,
        r#"{"document":"x","axes":["diagnosis"]}"#,
        r#"{"document":"x","axes":[]}"#,
        r#"{"document":"x","axes":null}"#,
        r#"{"document":null}"#, r#"{"document":""}"#, r#"["x"]"#,
    ] { assert!(parse(text).is_err(), "{text}"); }
}

#[test]
fn actual_four_axis_plan_admits_aggregate_native_work_before_execution() {
    let mut command = command(&["--samples", "3", "--max-new-tokens", "64"]);
    let (_, limits) = command.common().unwrap();
    let options = command.options().unwrap();
    let budget = command.budget(limits, 4).unwrap();
    let facts = crate::native_engine::artifact_bridge::ArtifactIdentity {
        model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-sampled-cli-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32),
    };
    let planner = Int8SampledSentimentPlanner::for_model(&facts, options, budget, command.task_limits()).unwrap();
    let all = request(r#"{"document":"The service was reassuring and helpful."}"#, budget, command.max_input_bytes).unwrap();
    let plan = planner.plan(&all).unwrap();
    let one_budget = command.budget(limits, 1).unwrap();
    let one = request(r#"{"document":"The service was reassuring and helpful.","axes":["valence"]}"#,
        one_budget, command.max_input_bytes).unwrap();
    let one = planner.plan(&one).unwrap();
    assert_eq!(plan.sample_count(), 12);
    assert_eq!(plan.max_output_tokens(), 12 * 64);
    assert_eq!(plan.planned_work().projected_logits, 12 * 64 * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(plan.planned_work().projected_logits, one.planned_work().projected_logits * 4);
    assert!(plan.planned_work().forward_positions > one.planned_work().forward_positions);
    let work = plan.planned_work();
    command.max_forward_positions = work.forward_positions;
    command.max_projected_logits = work.projected_logits;
    command.max_attention_pairs = work.attention_pairs;
    command.max_dot_products = work.projections.dot_products;
    command.max_multiply_accumulates = work.projections.multiply_accumulates;
    command.admit_plan(plan.required_context(), work).unwrap();
    for axis in 0..5 {
        match axis {
            0 => command.max_forward_positions -= 1,
            1 => command.max_projected_logits -= 1,
            2 => command.max_attention_pairs -= 1,
            3 => command.max_dot_products -= 1,
            _ => command.max_multiply_accumulates -= 1,
        }
        assert_eq!(command.admit_plan(plan.required_context(), work), Err(CandidateError::Planning));
        command.max_forward_positions = work.forward_positions;
        command.max_projected_logits = work.projected_logits;
        command.max_attention_pairs = work.attention_pairs;
        command.max_dot_products = work.projections.dot_products;
        command.max_multiply_accumulates = work.projections.multiply_accumulates;
    }
    assert_eq!(command.admit_plan(command.context_tokens + 1, work), Err(CandidateError::Planning));
    // A single-sample work allowance cannot execute an axis/sample bundle.
    let one_sample = constrained_int8::planned_work(16, 64).unwrap();
    assert!(work.projected_logits > one_sample.projected_logits);
}

#[test]
fn two_preparation_owners_full_sampler_and_complete_report_fit_the_same_memory_peak() {
    let mut command = command(&["--prefill-rows", "64"]);
    let (_, limits) = command.common().unwrap();
    let native = Int8MemoryRequirement::for_context(command.context_tokens).unwrap();
    let sampler = json_sampling_workspace_bytes().unwrap();
    let extra = command.prefill().unwrap().unwrap().max_extra_scratch_bytes;
    let tokens = command.total_tokens(4).unwrap() as u64;
    let one_owner = compute::request_peak_bytes(limits, native, command.max_result_bytes as u64,
        tokens, sampler, 0).unwrap() + extra;
    let complete = compute::request_peak_bytes(limits, native, command.max_result_bytes as u64,
        tokens, sampler, limits.preparation_bytes).unwrap() + extra;
    assert_eq!(complete - one_owner, limits.preparation_bytes);
    command.memory_mib = (complete - 1) / MIB;
    assert!(command.memory_mib * MIB >= one_owner);
    let failure = command.execute(&mut NoIo, &mut NoIo).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::Usage);
    let mut command = super::tests::command(&["--prefill-rows", "64"]);
    command.memory_mib = complete.div_ceil(MIB);
    command.common().unwrap();
    // Pricing the final task report uses all axis/sample tokens, even though
    // the native KV and sampler workspace are reused between samples.
    assert!(tokens > command.max_new_tokens as u64);
    assert!(sampler > 0 && extra > 0);
}

#[test]
fn prompt_schedule_does_not_change_sampling_or_task_budget_and_invalid_rows_refuse() {
    let serial = command(&[]);
    let (_, serial_limits) = serial.common().unwrap();
    for rows in ["1", "4", "64"] {
        let scheduled = command(&["--prefill-rows", rows]);
        let (_, limits) = scheduled.common().unwrap();
        assert_eq!(serial.options().unwrap(), scheduled.options().unwrap());
        assert_eq!(serial.budget(serial_limits, 4).unwrap(), scheduled.budget(limits, 4).unwrap());
        let prefill = scheduled.prefill().unwrap().unwrap();
        assert_eq!(prefill.max_batch_rows, rows.parse::<usize>().unwrap());
        assert_eq!(prefill.max_extra_scratch_bytes, prefill.validate().unwrap());
    }
    for rows in ["0", "65", "18446744073709551615"] {
        if let Ok(CandidateCommand::SampledSentiment(command)) = parsed(&["--prefill-rows", rows]) {
            assert_eq!(command.execute(&mut NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
    }
}

#[test]
fn compact_source_policy_is_explicit_and_its_work_ceiling_is_not_renewed_by_samples() {
    let compact = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-pair-checks", "1234", "--max-tokenizer-pair-bytes", "5678"]);
    let (common, limits) = compact.common().unwrap();
    let policy = common.input_encoding_policy(limits).unwrap();
    let crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(work) = policy else {
        panic!("compact policy lost")
    };
    assert_eq!((work.max_pair_checks, work.max_pair_bytes), (1234, 5678));
    assert_eq!(work.max_input_bytes, compact.max_input_bytes);
    assert_eq!(work.max_output_tokens, limits.max_prompt_tokens);
    assert!(compact.encoding.workspace_bytes(compact.max_input_bytes, limits.max_prompt_tokens).unwrap() > 0);
    assert!(parsed(&["--max-tokenizer-pair-checks", "1234"]).is_err());
    let invalid = command(&["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"]);
    assert_eq!(invalid.execute(&mut NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_valid_command_before_private_io() {
    let failure = command(&[]).execute(&mut NoIo, &mut NoIo).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::Usage);
    assert_eq!(failure.code, "int8_sampled_sentiment_profile_unavailable");
}

#[cfg(feature = "asupersync-runtime")]
#[test]
fn all_terminal_causes_retain_their_exit_category_and_exact_diagnostic_attribution() {
    use crate::{hosted::{HostedError, RunStop},
        native_engine::{constrained::JsonDecodeError, constrained_int8::Int8JsonError,
            decode::DecodeCancellationKind, strict_int8::StrictInt8Error},
        tasks::sentiment::sampled::{SampledSentimentError,
            quantized::Int8SampledSentimentError}};
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = match cause {
            User | ParentCancelled | Shutdown => ErrorCode::Cancelled,
            Timeout | Deadline | PollQuota | CostBudget => ErrorCode::BudgetOrTimeout,
            ResourceUnavailable => ErrorCode::AdmissionOrResourceLimit,
            FailFast | LinkedExit | RaceLost => ErrorCode::Generic,
        };
        let failures = [runtime::cancelled(cause),
            runtime::task_failure(SampledSentimentError::Cancelled(cause).into()),
            runtime::task_failure(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause)).into()),
            runtime::task_failure(StrictInt8Error::Cancelled(cause).into()),
            runtime::host_failure(HostedError::SampledSentiment(
                Int8SampledSentimentError::from(SampledSentimentError::Cancelled(cause)))),
            runtime::host_failure(HostedError::Native(StrictInt8Error::Cancelled(cause))),
            runtime::host_failure(HostedError::Stopped { stop: RunStop {
                kind: cause, runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }),
        ];
        for failure in failures {
            assert_eq!(failure.cancellation, Some(cause));
            assert_eq!(failure.exit, expected);
            let mut diagnostics = Vec::new();
            assert_eq!(report_failure(failure, &mut diagnostics), expected.as_process_exit());
            let text = String::from_utf8(diagnostics).unwrap();
            assert!(text.contains(&format!("({cause:?})")));
            assert!(text.starts_with("fnlp candidate INT8 sampled sentiment: int8_sampled_sentiment_"));
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
fn structured_failure_is_distinct_from_budget_identity_and_supervisor_attribution() {
    use crate::{hosted::{HostedError, RunStop},
        native_engine::{constrained::JsonDecodeError, constrained_int8::Int8JsonError,
            decode::DecodeCancellationKind},
        tasks::sentiment::sampled::{SampledSentimentError,
            quantized::Int8SampledSentimentError}};
    let cases = [
        (Int8SampledSentimentError::from(SampledSentimentError::NoResult), ErrorCode::StructuredTaskNoResult),
        (SampledSentimentError::OutputBudget.into(), ErrorCode::BudgetOrTimeout),
        (Int8JsonError::Decode(JsonDecodeError::BudgetExceeded("private-native-detail")).into(), ErrorCode::BudgetOrTimeout),
        (SampledSentimentError::WorkBudget.into(), ErrorCode::AdmissionOrResourceLimit),
        (SampledSentimentError::Identity.into(), ErrorCode::ArtifactIntegrityOrFormatOrVersion),
        (SampledSentimentError::Schema.into(), ErrorCode::SchemaOrRecipeCompile),
    ];
    for (error, expected) in cases {
        let failure = runtime::task_failure(error);
        assert_eq!(failure.exit, expected);
        assert_eq!(failure.cancellation, None);
        let mut diagnostics = Vec::new();
        report_failure(failure, &mut diagnostics);
        assert!(!String::from_utf8(diagnostics).unwrap().contains("private-native-detail"));
    }
    for cause in [DecodeCancellationKind::FailFast, DecodeCancellationKind::LinkedExit] {
        let failure = runtime::host_failure(HostedError::Stopped { stop: RunStop {
            kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::SampledSentiment(SampledSentimentError::NoResult.into()))) });
        assert_eq!(failure.exit, ErrorCode::StructuredTaskNoResult);
        assert_eq!(failure.cancellation, Some(cause));
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
fn original_deadline_wins_after_both_successful_and_failed_setup_stages() {
    use crate::native_engine::decode::DecodeCancellationKind;
    let classify = |_: CandidateError| Failure {
        exit: ErrorCode::InputDecodeOrParse, code: "int8_sampled_sentiment_input", cancellation: None };
    for failed in [false, true] {
        let stage = if failed { Err(CandidateError::Input) } else { Ok(17_u32) };
        let live = runtime::checked_stage(stage, || Ok(()), classify);
        if failed {
            let failure = live.unwrap_err();
            assert_eq!(failure.exit, ErrorCode::InputDecodeOrParse);
            assert_eq!(failure.cancellation, None);
        } else { assert_eq!(live.ok(), Some(17)); }
        let expired = runtime::checked_stage(stage, || Err(CandidateError::Timeout), classify).unwrap_err();
        assert_eq!(expired.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(expired.cancellation, Some(DecodeCancellationKind::Deadline));
    }
    // A native session may stop before physical cleanup crosses the outer
    // clock deadline. That later poll must retain the first terminal cause.
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let original = runtime::cancelled(cause);
        let expected_exit = original.exit;
        let stage: Result<(), Failure> = Err(original);
        let failure = runtime::checked_stage(stage, || Err(CandidateError::Timeout),
            |failure| failure).unwrap_err();
        assert_eq!(failure.cancellation, Some(cause));
        assert_eq!(failure.exit, expected_exit);
    }
}

mod reuse;
mod selected;
