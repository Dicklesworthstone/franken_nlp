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

fn compact_corpus(task: &str, flags: &[&str]) -> (ScoreJobArgs, Limits, Corpus) {
    let a = args(task, flags); let (_, limits) = a.validate().unwrap();
    let defaults = a.parse_defaults((task == "classify").then_some(LABELS), a.host.budget(limits)).unwrap();
    let configured = configure(&a, &facts(), limits, defaults).unwrap();
    (a, limits, configured)
}

fn compact_document<A>(id: &str) -> BatchDocument<A> {
    BatchDocument { id: id.to_owned(), text: "A welcome legal result. é 上海 <tool_call>".to_owned(), task_args: None }
}

fn compact_recipe(corpus: &Corpus, total: Option<SafeBpeWorkLimits>) -> serde_json::Value {
    match corpus {
        Corpus::Classify(p, c) => {
            let factory = Int8ClassificationJobPlanner::new_with_document_encoding(p, c.identity.clone(),
                c.task_ceiling, c.planning, c.defaults.clone(), c.max_model_work, total).unwrap();
            serde_json::to_value(factory.job_recipe()).unwrap()
        }
        Corpus::Sentiment(p, c) => {
            let config = crate::hosted::corpus::SentimentCorpusConfig {
                identity: c.identity.clone(), task_ceiling: c.task_ceiling, planning: c.planning,
                defaults: c.defaults.clone(), max_item_work: c.max_item_work, max_model_work: c.max_model_work,
            };
            let factory = Int8SentimentJobPlanner::new_with_document_encoding(p, config, total).unwrap();
            serde_json::to_value(factory.job_recipe()).unwrap()
        }
        Corpus::Judge(_, _) => panic!("judgment recipe is checked by the hosted job contract tests"),
    }
}

#[test]
fn compact_cli_configuration_reaches_the_real_durable_compilers_in_every_schedule() {
    for task in ["classify", "sentiment"] {
        for schedule in [vec![], vec!["--prefill-rows", "4"],
            vec!["--compute-threads", "4"], vec!["--compute-threads", "4", "--prefill-rows", "4"]] {
            let mut flags = vec!["--input-encoding", "safe-bpe-v1"]; flags.extend(schedule);
            let (a, _, corpus) = compact_corpus(task, &flags);
            let policy = a.encoding_policy().unwrap(); let total = a.encoding_work_limits().unwrap();
            let workspace = a.encoding_workspace_bytes().unwrap();
            check_encoding_quote(&corpus, &policy, total, workspace).unwrap();
            assert_eq!(check_encoding_quote(&corpus, &policy, total, workspace - 1).unwrap_err().code,
                "score_job_encoding_quote_mismatch");
            let recipe = compact_recipe(&corpus, total);
            assert_eq!(recipe["input_encoding"], serde_json::to_value(&policy).unwrap());
            assert_eq!(recipe["encoding_work_limits"], serde_json::to_value(total.unwrap()).unwrap());
            match corpus {
                Corpus::Classify(p, c) => {
                    let factory = Int8ClassificationJobPlanner::new_with_document_encoding(&p, c.identity,
                        c.task_ceiling, c.planning, c.defaults, c.max_model_work, total).unwrap();
                    let plan = factory.prepare_with_control(compact_document("one"), &mut Continue).unwrap();
                    assert_eq!(plan.head_count(), 2);
                    let once = factory.encoding_work().unwrap(); assert!(once.input_bytes() > 0);
                    factory.prepare_with_control(compact_document("two"), &mut Continue).unwrap();
                    assert_eq!(factory.encoding_work().unwrap(), once.checked_add(once).unwrap());
                }
                Corpus::Sentiment(p, c) => {
                    let factory = Int8SentimentJobPlanner::new_with_document_encoding(&p, c, total).unwrap();
                    let plan = factory.prepare_with_control(compact_document("one"), &mut Continue).unwrap();
                    assert_eq!(plan.head_count(), crate::tasks::sentiment::SentimentAxis::ALL.len());
                    let once = factory.encoding_work().unwrap(); assert!(once.input_bytes() > 0);
                    factory.prepare_with_control(compact_document("two"), &mut Continue).unwrap();
                    assert_eq!(factory.encoding_work().unwrap(), once.checked_add(once).unwrap());
                }
                Corpus::Judge(_, _) => panic!("wrong task"),
            }
        }
    }
}

#[test]
fn every_cli_tokenizer_cap_changes_the_authenticated_recipe_without_changing_native_limits() {
    for task in ["classify", "sentiment"] {
        let (a, _, corpus) = compact_corpus(task, &["--input-encoding", "safe-bpe-v1"]);
        let original = compact_recipe(&corpus, a.encoding_work_limits().unwrap());
        for (flag, value) in [
            ("--max-tokenizer-pair-checks", "4194305"), ("--max-tokenizer-pair-bytes", "67108865"),
            ("--max-total-tokenizer-input-bytes", "67108865"),
            ("--max-total-tokenizer-output-tokens", "4194305"),
            ("--max-total-tokenizer-pair-checks", "268435457"),
            ("--max-total-tokenizer-pair-bytes", "4294967297"),
        ] {
            let (changed, _, corpus) = compact_corpus(task, &["--input-encoding", "safe-bpe-v1", flag, value]);
            let recipe = compact_recipe(&corpus, changed.encoding_work_limits().unwrap());
            assert_ne!(recipe, original, "{task} {flag}");
            assert_eq!(changed.host.work_ceiling(), a.host.work_ceiling());
            assert_eq!(recipe["task_ceiling"], original["task_ceiling"]);
            assert_eq!(recipe["max_model_work"], original["max_model_work"]);
            assert_eq!(changed.encoding_workspace_bytes().unwrap(), a.encoding_workspace_bytes().unwrap());
        }
        let (byte, _, corpus) = compact_corpus(task, &[]);
        let legacy = compact_recipe(&corpus, byte.encoding_work_limits().unwrap());
        assert!(legacy.get("input_encoding").is_none());
        assert!(legacy.get("encoding_work_limits").is_none());
    }
}

#[test]
fn compact_quote_checks_exact_policy_even_when_workspace_size_is_identical() {
    for task in ["classify", "sentiment", "judge"] {
        let (a, _, corpus) = compact_corpus(task, &["--input-encoding", "safe-bpe-v1"]);
        let policy = a.encoding_policy().unwrap(); let total = a.encoding_work_limits().unwrap();
        let workspace = a.encoding_workspace_bytes().unwrap();
        let UntrustedEncoding::SafeBpeV1(mut changed) = policy else { panic!("compact") };
        changed.max_pair_checks += 1;
        assert_eq!(changed.requirements().unwrap().workspace_bytes + 1024, workspace);
        assert_eq!(check_encoding_quote(&corpus, &UntrustedEncoding::SafeBpeV1(changed),
            total, workspace).unwrap_err().code, "score_job_encoding_quote_mismatch");
        assert!(check_encoding_quote(&corpus, &UntrustedEncoding::ByteFallback, None, 0).is_err());
    }
}

#[test]
fn compact_population_quote_includes_encoder_with_original_or_retained_record_buffers() {
    for task in ["classify", "sentiment", "judge"] { for retained in [false, true] {
        let mut flags = vec!["--input-encoding", "safe-bpe-v1", "--compute-threads", "4", "--prefill-rows", "4"];
        if retained { flags.extend(["--spool-input", "--max-retained-input-mib", "4"]); }
        let a = args(task, &flags); let (_, limits) = a.validate().unwrap();
        let host = quoted_host(&a, limits); let job = lifetime();
        let peak = population_memory_bytes(&a, host, job, limits).unwrap();
        let workspace = a.encoding_workspace_bytes().unwrap();
        let buffers = host.required_buffer_bytes_for_input(job, a.input_retention.input_memory().unwrap()).unwrap();
        assert_eq!(peak, a.execution_memory_bytes(limits).unwrap() + buffers);
        let mut exact = limits; exact.memory_bytes = peak;
        check_population_memory(&a, host, job, exact).unwrap();
        exact.memory_bytes -= workspace;
        assert_eq!(check_population_memory(&a, host, job, exact).unwrap_err().code, "score_job_population_memory");
    } }
}

#[test]
fn stale_compact_policy_is_refused_before_runtime_private_io_or_output() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush") }
    }
    for task in ["classify", "sentiment", "judge"] {
        for old in [vec![], vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "4194305"]] {
            let original = args(task, &old); let (common, limits) = original.validate().unwrap();
            let current = args(task, &["--input-encoding", "safe-bpe-v1"]);
            let error = execute(RunMode::Start, current, common, limits, NoIo, &mut NoIo).unwrap_err();
            assert_eq!(error.code, "score_job_encoding_admission");
        }
    }
}

#[test]
fn compact_job_metadata_reports_only_policy_and_invocation_limits() {
    let a = args("classify", &["--input-encoding", "safe-bpe-v1"]);
    let policy = a.encoding_policy().unwrap(); let total = a.encoding_work_limits().unwrap();
    let report = serde_json::json!({"output":{"committed":2,"attempts":3}});
    let byte = EncodingReport { response: &report, input_encoding: None,
        invocation_encoding_work_limits: None, root_object_reconciliation: None };
    assert_eq!(crate::canonjson::canonical_bytes(&byte).unwrap(), crate::canonjson::canonical_bytes(&report).unwrap());
    let compact = EncodingReport { response: &report, input_encoding: Some(&policy),
        invocation_encoding_work_limits: total, root_object_reconciliation: None };
    let value = serde_json::to_value(compact).unwrap();
    assert_eq!(value["output"], report["output"]);
    assert_eq!(value["input_encoding"], serde_json::to_value(policy).unwrap());
    assert_eq!(value["invocation_encoding_work_limits"], serde_json::to_value(total.unwrap()).unwrap());
    for private in ["labels", "dimensions", "source_text", "candidate_weight", "prompt_digest"] {
        assert!(!value.to_string().contains(private));
    }
}

#[test]
fn compact_judgment_job_configuration_reaches_all_complete_native_modes_and_schedules() {
    for schedule in [vec![], vec!["--prefill-rows", "4"],
        vec!["--compute-threads", "4"], vec!["--compute-threads", "4", "--prefill-rows", "4"]] {
        let mut flags = vec!["--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "8192"];
        flags.extend(schedule);
        let a = args("judge", &flags); let (_, limits) = a.validate().unwrap();
        for (mode, source_key, heads) in [("pairwise", "a", 2), ("rubric", "document", 2), ("faithfulness", "source", 1)] {
            let mut data = crate::candidate_cli::judge::tests::fixture(mode);
            let text = data.as_object_mut().unwrap().remove(source_key).unwrap().as_str().unwrap().to_owned();
            let defaults = a.parse_defaults(Some(&data.to_string()), a.host.budget(limits)).unwrap();
            let corpus = configure(&a, &facts(), limits, defaults).unwrap();
            let policy = a.encoding_policy().unwrap();
            let totals = a.encoding_work_limits().unwrap();
            let workspace = a.encoding_workspace_bytes().unwrap();
            check_encoding_quote(&corpus, &policy, totals, workspace).unwrap();
            assert!(check_encoding_quote(&corpus, &policy, None, workspace).is_err());
            assert_eq!(check_encoding_quote(&corpus, &policy, totals, workspace - 1).unwrap_err().code,
                "score_job_encoding_quote_mismatch");
            let Corpus::Judge(planner, config) = corpus else { panic!("judgment") };
            assert_eq!(planner.document_encoding(), &policy);
            assert_eq!(config.identity.template_digest, *planner.template_digest());
            let request = config.defaults.clone().unwrap().into_request(text);
            let context = crate::tasks::ir::PlanContext::new(&config.identity, config.task_ceiling).unwrap();
            let plan = planner.plan_int8_with_control(&request, &context, config.planning, &mut Continue).unwrap();
            assert_eq!(plan.head_count(), heads);
            assert_eq!(plan.input_encoding(), &policy);
            assert!(plan.input_encoding_work().unwrap().input_bytes() > 0);
            assert_eq!(plan.execution_identity().template_digest, config.identity.template_digest);
        }
    }
}

#[test]
fn compact_judgment_job_quote_checks_all_invocation_caps_without_renewing_native_limits() {
    let (a, _, corpus) = compact_corpus("judge", &["--input-encoding", "safe-bpe-v1"]);
    let policy = a.encoding_policy().unwrap();
    let totals = a.encoding_work_limits().unwrap().unwrap();
    let workspace = a.encoding_workspace_bytes().unwrap();
    let Corpus::Judge(planner, config) = &corpus else { panic!("judgment") };
    for axis in 0..4 {
        let mut changed = totals;
        match axis {
            0 => changed.max_input_bytes += 1,
            1 => changed.max_output_tokens += 1,
            2 => changed.max_pair_checks += 1,
            _ => changed.max_pair_bytes += 1,
        }
        let (actual_policy, actual_totals, actual_workspace) =
            config.encoding_requirements(planner, Some(changed)).unwrap();
        assert_eq!(actual_policy, policy);
        assert_eq!(actual_totals, Some(changed));
        assert_eq!(actual_workspace, workspace);
        assert_eq!(config.max_model_work, a.host.work_ceiling());
    }
    let (byte, _, corpus) = compact_corpus("judge", &[]);
    check_encoding_quote(&corpus, &UntrustedEncoding::ByteFallback, None, 0).unwrap();
    assert!(check_encoding_quote(&corpus, &UntrustedEncoding::ByteFallback, Some(totals), 0).is_err());
    assert_eq!(byte.encoding_work_limits().unwrap(), None);
}
