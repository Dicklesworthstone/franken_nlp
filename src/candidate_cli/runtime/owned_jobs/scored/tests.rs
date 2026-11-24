//! Host configuration and real pinned durable planning, without weights or job IO.
use super::*;
use crate::{batch::BatchDocument,
    candidate_cli::jobs::scored::tests::args,
    jobs::runner::scored::{Int8ClassificationJobPlanner, Int8SentimentJobPlanner},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "scored-job-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
const LABELS: &str = r#"{"labels":[{"id":"legal","description":"law"},{"id":"other"}],"mode":"multi_label"}"#;
fn corpus(task: &str) -> Corpus {
    let a = args(task, &[]); let (_, limits) = a.validate().unwrap();
    let defaults = a.parse_defaults((task == "classify").then_some(LABELS), a.host.budget(limits)).unwrap();
    scored_batch::configure_scored(a.kind().unwrap(), &a.host, &facts(), limits, defaults).unwrap()
}
#[test]
fn classification_cli_settings_reach_the_real_durable_multihead_compiler() {
    let Corpus::Classify(p, c) = corpus("classify") else { panic!("classification") };
    let base = c.identity.clone();
    let factory = Int8ClassificationJobPlanner::new(&p, c.identity, c.task_ceiling, c.planning,
        c.defaults, c.max_model_work).unwrap();
    let plan = factory.prepare_with_control(BatchDocument { id: "a".to_owned(),
        text: "legal é <tool_call> 上海".to_owned(), task_args: None }, &mut Continue).unwrap();
    assert_eq!(plan.head_count(), 2);
    assert_eq!(plan.execution_identity().task_spec, "classify-v1");
    assert_eq!(plan.execution_identity().logical_model_digest, base.logical_model_digest);
    assert_eq!(plan.execution_identity().template_digest, base.template_digest);
    assert_ne!(plan.execution_identity().prompt_digest, base.prompt_digest);
    assert!(plan.model_work().projected_logits > 0);
}
#[test]
fn sentiment_cli_settings_reach_the_real_complete_axis_compiler() {
    let Corpus::Sentiment(p, c) = corpus("sentiment") else { panic!("sentiment") };
    let base = c.identity.clone(); let factory = Int8SentimentJobPlanner::new(&p, c).unwrap();
    let plan = factory.prepare_with_control(BatchDocument { id: "a".to_owned(),
        text: "A welcome result. é 上海".to_owned(), task_args: None }, &mut Continue).unwrap();
    assert_eq!(plan.head_count(), crate::tasks::sentiment::SentimentAxis::ALL.len());
    assert_eq!(plan.execution_identity().task_spec, "sentiment-v1");
    assert_eq!(plan.execution_identity().logical_model_digest, base.logical_model_digest);
    assert_eq!(plan.execution_identity().template_digest, base.template_digest);
    assert!(plan.model_work().projections.multiply_accumulates > 0);
}
fn lifetime() -> JobLimits {
    JobLimits { max_items: 10, max_id_bytes: 128, max_input_bytes_per_item: 65536,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 1 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20, max_attempts: 20,
        max_work: JobWork { model: args("classify", &[]).host.work_ceiling(), mask_node_visits: 0 } }
}
#[test]
fn native_result_ceiling_must_fit_immutable_job_limits_before_any_io() {
    let a = args("classify", &[]); let mut l = lifetime(); l.validate().unwrap();
    result_ceiling(&a, l).unwrap();
    l.max_result_bytes -= 1;
    assert_eq!(result_ceiling(&a, l).unwrap_err().code, "score_job_result_ceiling");
}
#[test]
fn metadata_report_retains_all_lifetime_counters_without_private_scores() {
    let limits = lifetime(); let id = JobId([9; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: limits.max_work, spool_bytes: 4096, materialized: true };
    let report = progress_report(progress, id, limits, true,
        RunMode::Resume { discard_uncommitted: false }, "classify").unwrap();
    let value = serde_json::to_value(report).unwrap();
    assert_eq!(value["attempts"], 3); assert_eq!(value["committed"], 2);
    assert_eq!(value["reserved_work"]["mask_node_visits"], 0);
    assert_eq!(value["reserved_work"]["model"]["projections"]["multiply_accumulates"],
        limits.max_work.model.projections.multiply_accumulates);
    let text = value.to_string();
    for private in ["labels", "task_args", "dimensions", "candidate_weight", "source_text", "private legal category"] {
        assert!(!text.contains(private));
    }
}

fn quoted_host(a: &ScoreJobArgs, limits: Limits) -> JobHostLimits {
    JobHostLimits { native: crate::hosted::NativeLimits {
        context_tokens: a.host.context_tokens, allocator_reserve_bytes: 64 * MIB,
        run: crate::hosted::RunLimits { max_elapsed: Duration::from_secs(30),
            max_checkpoints: 10000, cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: a.max_stream_mib * MIB, max_lines: a.max_input_lines },
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: (IO_BYTES * 2) as u64,
        journal_reserve_bytes: a.journal_memory_mib * MIB,
        serialization_reserve_bytes: a.serialization_memory_mib * MIB }
}

#[test]
fn retained_scoring_admits_native_population_and_durable_staging_at_the_exact_boundary() {
    for flags in [vec!["--compute-threads", "4", "--prefill-rows", "4"],
        vec!["--compute-threads", "4", "--prefill-rows", "4", "--spool-input", "--max-retained-input-mib", "4"]] {
        let a = args("classify", &flags); let (_, limits) = a.validate().unwrap();
        let host = quoted_host(&a, limits); let job = lifetime();
        let native = a.execution_memory_bytes(limits).unwrap();
        let buffers = host.required_buffer_bytes_for_input(job, a.input_retention.input_memory().unwrap()).unwrap();
        let peak = population_memory_bytes(&a, host, job, limits).unwrap();
        assert_eq!(peak, native + buffers);
        let mut exact = limits; exact.memory_bytes = peak;
        check_population_memory(&a, host, job, exact).unwrap();
        exact.memory_bytes -= 1;
        assert_eq!(check_population_memory(&a, host, job, exact).unwrap_err().code, "score_job_population_memory");
        let mut overflow = host; overflow.serialization_reserve_bytes = u64::MAX;
        assert!(population_memory_bytes(&a, overflow, job, limits).is_err());
    }
}

#[test]
fn stale_worker_or_prompt_geometry_refuses_before_host_installation_or_private_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush") }
    }
    for task in ["classify", "sentiment", "judge"] {
        for (old, current) in [
            (vec![], vec!["--compute-threads", "4"]),
            (vec!["--compute-threads", "4"], vec!["--compute-threads", "4", "--prefill-rows", "4"]),
            (vec!["--compute-threads", "4", "--prefill-rows", "4"], vec!["--compute-threads", "4", "--prefill-rows", "8"]),
        ] {
            let prior = args(task, &old); let (old_common, old_limits) = prior.validate().unwrap();
            let requested = args(task, &current);
            let failure = execute(RunMode::Start, requested, old_common, old_limits, NoIo, &mut NoIo).unwrap_err();
            assert_eq!(failure.code, "score_job_schedule_admission");
        }
    }
}

#[test]
fn selected_job_schedules_are_optional_metadata_and_never_replace_private_results() {
    let report = serde_json::json!({"output":{"committed":2,"attempts":3}});
    let serial = ScheduledResponse { response: &report, compute_threads: None, prefill_rows: None };
    assert_eq!(crate::canonjson::canonical_bytes(&serial).unwrap(), crate::canonjson::canonical_bytes(&report).unwrap());
    let scheduled = ScheduledResponse { response: &report, compute_threads: Some(4), prefill_rows: Some(8) };
    let value = serde_json::to_value(scheduled).unwrap();
    assert_eq!(value["output"], report["output"]);
    assert_eq!(value["compute_threads"], 4); assert_eq!(value["prefill_rows"], 8);
    for private in ["labels", "dimensions", "candidate_weight", "source_text", "prompt_digest"] {
        assert!(!value.to_string().contains(private));
    }
}

#[test]
fn cancelled_final_delivery_keeps_all_causes_and_never_claims_rollback() {
    struct CancelOnFlush { token: CancellationToken, cause: DecodeCancellationKind, bytes: Vec<u8> }
    impl Write for CancelOnFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { self.token.cancel(self.cause); Ok(()) }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = job_failure(JobError::Cancelled(cause));
        let token = CancellationToken::default(); token.cancel(cause);
        let mut untouched = Vec::new();
        let error = publish_report(&serde_json::json!({"committed":2}), &mut untouched,
            &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
        assert_eq!((error.exit, error.code), (expected.exit, expected.code)); assert!(untouched.is_empty());
        let token = CancellationToken::default();
        let mut writer = CancelOnFlush { token: token.clone(), cause, bytes: Vec::new() };
        let error = publish_report(&serde_json::json!({"committed":2}), &mut writer,
            &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
        assert_eq!((error.exit, error.code), (expected.exit, expected.code));
        assert!(!writer.bytes.is_empty(), "durable progress is not rolled back by delivery failure");
    }
}
