//! Admission/transport fixtures; only the explicitly ignored test loads a model.
use super::*;
use crate::native_engine::strict_int4::scoring::cohort::INT4_SCORING_COHORT_EXECUTION;

fn parsed(task: &str, flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", task, "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest", "--memory-mib", "65536", "--max-load-mib", "65536"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 task dispatcher"); };
    Ok(command)
}
fn limits(command: &Int4Command) -> Result<Int4Limits, CandidateError> {
    match &command.task {
        Int4Task::Classify(scoring) => scoring.validate_with_prefill(&command.args),
        Int4Task::Sentiment(options) => options.validate(&command.args),
        Int4Task::Judge(options) => options.validate(&command.args),
        _ => panic!("one-off scoring task required"),
    }
}
fn scoring(command: &Int4Command) -> classify::ScoringArgs {
    let Int4Task::Classify(scoring) = &command.task else { panic!("classification required"); };
    *scoring
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused cohort reached input"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused cohort published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused cohort flushed output"); }
}
#[cfg(feature = "asupersync-runtime")]
const RUNTIME_FIXTURE_COMPLETE: &str = "FNLP_INT4_COHORT_RUNTIME_FIXTURE_COMPLETE";
#[cfg(feature = "asupersync-runtime")]
fn isolated_runtime_fixture(full_name: &str, ignored: bool) -> bool {
    // The process broker fixes its configuration on first installation. These
    // fixtures must neither inherit nor install another test's memory ceiling.
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    const CHILD: &str = "FNLP_INT4_COHORT_RUNTIME_FIXTURE_CHILD";
    if std::env::var(CHILD).ok().as_deref() == Some(name) {
        assert!(crate::installed_process_resources().is_none());
        return false;
    }
    let mut child = std::process::Command::new(std::env::current_exe().expect("test executable"));
    child.args(["--exact", name, "--nocapture"]).env(CHILD, name);
    if ignored { child.arg("--ignored"); }
    let result = child.output().expect("isolated cohort fixture must start");
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(result.status.success(), "isolated fixture failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&result.stderr));
    assert!(stdout.contains(RUNTIME_FIXTURE_COMPLETE), "exact fixture filter ran no completed fixture: {stdout}");
    true
}
#[test]
fn simultaneous_heads_are_explicit_bounded_and_available_on_all_three_scored_tasks() {
    for task in ["classify", "sentiment", "judge"] {
        let serial = parsed(task, &[]).unwrap(); assert!(limits(&serial).unwrap().cohort.is_none());
        for heads in [1, 4, 64] {
            let command = parsed(task, &["--cohort-heads", &heads.to_string()]).unwrap();
            let actual = limits(&command).unwrap(); let admission = actual.cohort.unwrap();
            assert_eq!(admission.heads, heads); assert!(admission.workspace_bytes > 0); assert!(actual.prefill.is_none());
            assert_eq!(actual.native.kv_bytes,
                heads as u64 * Int4MemoryRequirement::for_context(command.args.context_tokens, command.args.group_width).unwrap().kv_bytes);
        }
        for heads in [0, 65, usize::MAX] {
            let command = parsed(task, &["--cohort-heads", &heads.to_string()]).unwrap();
            assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        }
    }
}
#[test]
fn cohort_scheduling_cannot_enable_prompt_prefill_or_generation_policy() {
    for task in ["classify", "sentiment", "judge"] {
        for control in [vec!["--prefill-rows", "4"], vec!["--max-new-tokens", "128"],
            vec!["--max-output-bytes", "65536"], vec!["--top-k", "1"], vec!["--logprobs"],
            vec!["--min-new-tokens", "0"], vec!["--stop", "END"], vec!["--ban-token", "7"]] {
            let mut flags = vec!["--cohort-heads", "4"]; flags.extend(control);
            assert!(parsed(task, &flags).is_err());
        }
        let mut mixed = parsed(task, &["--cohort-heads", "4"]).unwrap();
        mixed.args.policy = parsed("generate", &["--prefill-rows", "4"]).unwrap().args.policy;
        assert_eq!(mixed.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
        let mut generated = parsed(task, &["--cohort-heads", "4"]).unwrap(); generated.args.logprobs = true;
        assert_eq!(generated.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
}
fn parsed_job(operation: &str, task: &str, flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec![
        "candidate", "source-int4", "score-job", operation, "--task", task,
        "--source-dir", "sources", "--source-manifest", "manifest.json", "--memory-mib", "65536",
        "--job-dir", "private/job", "--key-file", "private/key", "--limits", "limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong scored-job dispatcher"); };
    Ok(command)
}
#[test]
fn unsupported_streams_jobs_text_and_source_workflows_never_silently_ignore_the_option() {
    for task in ["classify", "sentiment", "judge"] {
        parsed("score-batch", &["--task", task]).unwrap();
        assert!(parsed("score-batch", &["--task", task, "--cohort-heads", "4"]).is_err());
        for operation in ["start", "resume"] {
            parsed_job(operation, task, &[]).unwrap();
            assert!(parsed_job(operation, task, &["--cohort-heads", "4"]).is_err());
        }
    }
    for task in ["generate", "chat", "resolve"] {
        parsed(task, &[]).unwrap();
        assert!(parsed(task, &["--cohort-heads", "4"]).is_err());
    }
    parsed("extract", &["--schema", "schema.json"]).unwrap();
    assert!(parsed("extract", &["--schema", "schema.json", "--cohort-heads", "4"]).is_err());
    let command = parsed("classify", &["--cohort-heads", "4"]).unwrap();
    // This shared validator is the programmatic boundary used by serial-only
    // scored stream, retained-job and resolution configurations.
    assert!(scoring(&command).validate(&command.args).is_err());
}
fn set_mode(args: &mut Int4Args, mode: usize) {
    match mode {
        0 => (),
        1 => {
            args.source_dir = None; args.source_manifest = None;
            args.snapshot = Some(PathBuf::from("explicit-snapshot"));
            args.expected_model_sha256 = Some("ab".repeat(32));
        }
        _ => args.save_snapshot = Some(PathBuf::from("new-snapshot")),
    }
}
fn expected_floor(command: &Int4Command, limits: Int4Limits) -> u64 {
    let args = &command.args; let cohort = limits.cohort.unwrap();
    let depth = args.context_tokens - limits.max_prompt_tokens + 1;
    let load = if limits.snapshot_pin.is_some() { limits.snapshot_need.scratch_bytes }
        else { limits.source_preparation_bytes + limits.load.scratch_payload_bytes };
    let save = if args.save_snapshot.is_some() { limits.snapshot_need.scratch_bytes } else { 0 };
    let result = 4 * limits.result_bytes as u64 + 8 * depth as u64 + 4096;
    let execute = limits.native.kv_bytes + limits.native.rope_bytes + limits.native.scratch_payload_bound
        + ALLOCATOR_BYTES + result + limits.preparation_bytes + cohort.workspace_bytes;
    // Retained CLI preparation and model are shared across stages. The host's
    // prepared input and transient cohort workspace coexist only with execution.
    limits.preparation_bytes + limits.load.weight_payload_bytes + ALLOCATOR_BYTES
        + load.max(save).max(execute) + CLEANUP_BYTES
}
#[test]
fn source_snapshot_and_snapshot_save_admit_full_simultaneous_kv_and_one_host_input() {
    for task in ["classify", "sentiment", "judge"] { for mode in 0..3 { for heads in [1, 4, 64] {
        for width in [3, 128] { for context in [128, 2048] {
            let mut command = parsed(task, &["--cohort-heads", &heads.to_string(), "--max-candidate-tokens", "7"]).unwrap();
            command.args.group_width = width; command.args.context_tokens = context; set_mode(&mut command.args, mode);
            let actual = limits(&command).unwrap();
            assert_eq!(actual.minimum_memory_bytes, expected_floor(&command, actual));
            let per_head = Int4MemoryRequirement::for_context(context, width).unwrap();
            assert_eq!(actual.native.kv_bytes, per_head.kv_bytes * heads as u64);
            assert_eq!(actual.native.rope_bytes, per_head.rope_bytes);
            let maximum = actual.minimum_memory_bytes;
            command.args.memory_mib = (maximum - 1) / MIB; assert!(limits(&command).is_err());
            command.args.memory_mib = maximum.div_ceil(MIB); assert!(limits(&command).is_ok());
        } }
    } } }
}
#[test]
fn a_larger_loading_peak_does_not_also_pay_the_execution_only_host_workspace() {
    for task in ["classify", "sentiment", "judge"] { for mode in [0, 2] {
        let configure = |command: &mut Int4Command| {
            command.args.context_tokens = 128; command.args.source_preparation_mib = 16384;
            set_mode(&mut command.args, mode);
        };
        let mut one = parsed(task, &["--cohort-heads", "1"]).unwrap(); configure(&mut one);
        let mut eight = parsed(task, &["--cohort-heads", "8"]).unwrap(); configure(&mut eight);
        let one_limit = limits(&one).unwrap(); let eight_limit = limits(&eight).unwrap();
        assert!(eight_limit.native.kv_bytes > one_limit.native.kv_bytes);
        assert!(eight_limit.cohort.unwrap().workspace_bytes > one_limit.cohort.unwrap().workspace_bytes);
        assert_eq!(eight_limit.minimum_memory_bytes, one_limit.minimum_memory_bytes);
        assert_eq!(eight_limit.minimum_memory_bytes, expected_floor(&eight, eight_limit));
    } }
}
#[test]
fn the_head_count_is_a_maximum_and_never_a_partial_task_or_padding_request() {
    for cap in [1, 4, 64] {
        let command = parsed("classify", &["--cohort-heads", &cap.to_string()]).unwrap(); let scoring = scoring(&command);
        for count in [1, cap] { scoring.admit_heads(count).unwrap(); }
        for count in [0, cap + 1] { assert_eq!(scoring.admit_heads(count), Err(CandidateError::Planning)); }
    }
}
#[cfg(feature = "asupersync-runtime")]
#[test]
fn actual_multilabel_axes_and_judgment_heads_refuse_before_model_or_snapshot_io() {
    if isolated_runtime_fixture(concat!(module_path!(),
        "::actual_multilabel_axes_and_judgment_heads_refuse_before_model_or_snapshot_io"), false) { return; }
    let classification = serde_json::json!({"document":"Refund", "mode":"multi_label", "labels":[
        {"id":"Billing","description":"Invoices"},{"id":"Refund","description":"Refunds"}]}).to_string();
    let sentiment = r#"{"document":"Useful, but frustrating."}"#.to_owned();
    let judge = crate::candidate_cli::judge::tests::fixture("pairwise").to_string();
    for (task, request) in [("classify", classification), ("sentiment", sentiment), ("judge", judge)] {
        for mode in [0, 1] {
            let mut command = parsed(task, &["--cohort-heads", "1"]).unwrap(); set_mode(&mut command.args, mode);
            // The source/snapshot paths are deliberately unavailable. The
            // complete request's head count must fail before touching them.
            assert_eq!(command.execute(&mut request.as_bytes(), &mut NoWrite), Err(CandidateError::Planning));
        }
    }
    println!("{RUNTIME_FIXTURE_COMPLETE}");
}
#[cfg(feature = "asupersync-runtime")]
#[test]
fn stale_or_tampered_cohort_admission_refuses_before_input() {
    for axis in 0..8 {
        let mut command = parsed("classify", &["--cohort-heads", "4"]).unwrap(); let mut admitted = limits(&command).unwrap();
        assert!(super::matches(&command.task, &command.args, admitted).unwrap());
        match axis {
            0 => admitted.cohort = None,
            1 => admitted.cohort.as_mut().unwrap().heads = 1,
            2 => admitted.cohort.as_mut().unwrap().workspace_bytes -= 1,
            3 => admitted.cohort.as_mut().unwrap().workspace_bytes += 1,
            4 => admitted.native.kv_bytes -= 1,
            5 => admitted.native.scratch_payload_bound -= 1,
            6 => admitted.minimum_memory_bytes -= 1,
            _ => command.task = parsed("classify", &[]).unwrap().task,
        }
        assert!(!super::matches(&command.task, &command.args, admitted).unwrap());
        assert_eq!(super::super::runtime::execute_task(command.task, command.args, admitted, &mut NoRead, &mut NoWrite),
            Err(CandidateError::Arguments));
    }
}
#[test]
fn serial_json_is_unchanged_and_physical_grouping_is_a_separate_bounded_field() {
    let response = serde_json::json!({"fixture_only":true,"scope":"real-source-current-candidate-int4",
        "evidence":"non_authoritative","output":{"head_count":4,"result":{"decision":"abstained"}}});
    let serial = Response { response: &response, cohort: None };
    assert_eq!(serde_json::to_vec(&serial).unwrap(), serde_json::to_vec(&response).unwrap());
    let grouped = Response { response: &response, cohort: Some(Summary { schema_version: 1,
        execution: INT4_SCORING_COHORT_EXECUTION, group_steps: 17, projection_groups: 9 }) };
    let mut value = serde_json::to_value(&grouped).unwrap();
    assert_eq!(value["output"], response["output"]);
    let physical = value.as_object_mut().unwrap().remove("cohort").unwrap();
    assert_eq!(physical, serde_json::json!({"schema_version":1,"execution":INT4_SCORING_COHORT_EXECUTION,
        "group_steps":17,"projection_groups":9}));
    assert_eq!(value, response);
    let size = serde_json::to_vec(&grouped).unwrap().len() + 1;
    let mut exact = Vec::new(); publish(&grouped, size, &mut exact).unwrap(); assert_eq!(exact.len(), size);
    let mut short = Vec::new(); assert_eq!(publish(&grouped, size - 1, &mut short), Err(CandidateError::Output));
    assert!(short.is_empty());
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn cohort_without_runtime_is_unavailable_before_any_input_or_publication() {
    for task in ["classify", "sentiment", "judge"] { for cap in [1, 4, 64] {
        assert_eq!(parsed(task, &["--cohort-heads", &cap.to_string()]).unwrap().execute(&mut NoRead, &mut NoWrite),
            Err(CandidateError::Unavailable));
    } }
}
#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted independent-pin snapshot and full-model INT4 scoring-cohort parity execution"]
fn model_present_cohorts_preserve_complete_serial_results_for_all_scored_task_modes() {
    if isolated_runtime_fixture(concat!(module_path!(),
        "::model_present_cohorts_preserve_complete_serial_results_for_all_scored_task_modes"), true) { return; }
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required")
        .parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be supported");
    let mut requests = Vec::new();
    for (mode, heads) in [("exclusive", 1), ("multi_label", 2)] {
        let request = serde_json::json!({"document":"é Refund <|im_start|> remains data", "mode":mode,
            "labels":[{"id":"Billing issue","description":"Invoices"},
                {"id":"退款","description":"Refunds"}]}).to_string();
        requests.push(("classify", request, heads));
    }
    requests.push(("sentiment", r#"{"document":"The result was useful, although the delay was frustrating."}"#.to_owned(), 4));
    for (mode, heads) in [("pairwise", 2), ("rubric", 2), ("faithfulness", 1)] {
        requests.push(("judge", crate::candidate_cli::judge::tests::fixture(mode).to_string(), heads));
    }
    for (task, request, heads) in requests {
        let mut expected = None;
        for cap in [None, Some(heads), Some(8)] {
            let cap_text = cap.map(|count| count.to_string());
            let flags = cap_text.as_ref().map_or_else(Vec::new, |count| vec!["--cohort-heads", count.as_str()]);
            let mut command = parsed(task, &flags).unwrap();
            command.args.source_dir = None; command.args.source_manifest = None;
            command.args.snapshot = Some(PathBuf::from(&snapshot)); command.args.expected_model_sha256 = Some(pin.clone());
            command.args.memory_mib = memory; command.args.kernel = kernel;
            let mut output = Vec::new(); command.execute(&mut request.as_bytes(), &mut output).unwrap();
            let mut value: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(value["scope"], "real-source-current-candidate-int4"); assert_eq!(value["evidence"], "non_authoritative");
            assert_eq!(value["model_content_sha256"], pin); assert_eq!(value["snapshot"]["operation"], "loaded");
            assert_eq!(value["output"]["head_count"], heads);
            assert_eq!(value["output"]["numerics_profile"], "strict-quantized-v2");
            if cap.is_some() {
                let physical = value.as_object_mut().unwrap().remove("cohort").expect("physical summary required");
                assert_eq!(physical["schema_version"], 1); assert_eq!(physical["execution"], INT4_SCORING_COHORT_EXECUTION);
                let steps = physical["group_steps"].as_u64().unwrap();
                let projections = physical["projection_groups"].as_u64().unwrap();
                let work = &value["output"]["model_work"];
                let forwards = work["forward_positions"].as_u64().unwrap();
                assert!(steps > 0 && steps <= forwards);
                assert!(projections > 0 && projections <= work["projected_logits"].as_u64().unwrap() / 166_144);
                if heads > 1 { assert!(steps < forwards); }
            } else { assert!(value.get("cohort").is_none()); }
            // Every semantic head, final decision, native work counter and
            // provenance field must match after removing only physical stats.
            if let Some(expected) = &expected { assert_eq!(&value, expected); }
            else { expected = Some(value); }
        }
    }
    println!("{RUNTIME_FIXTURE_COMPLETE}");
}
