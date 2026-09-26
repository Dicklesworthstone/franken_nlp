//! Pinned factory/admission and metadata fixtures, not native-success evidence.
use super::*;
use crate::candidate_cli::jobs::tests::{args, lifetime};
use crate::batch::{BatchDocument, source::quantized::Int8SourceBatchPlanner};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-test".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
#[test]
fn actual_source_factories_bind_model_and_task_and_compile_original_documents() {
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let args = args(task, &[]); let (_, limits) = args.validate().unwrap();
        let ceiling = args.host.task_budget(limits);
        let defaults = command::load_defaults(&args, ceiling, None, None).unwrap();
        let (prepared, _vocabulary) = prepare(&args, &facts(), ceiling, defaults,
            command::native_limits(&args, lifetime()).unwrap(), &mut Continue).unwrap();
        let PreparedJob::Source { planner, config } = prepared else { panic!("wrong native job family") };
        assert_eq!(config.identity.logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert_eq!(config.identity.task_spec, args.task_kind().unwrap().spec().identity());
        assert_eq!(config.identity.template_digest, *planner.template_digest());
        let overrides = (task == "answer").then(|| crate::batch::source::SourceBatchArgs::Answer {
            options: crate::tasks::answer::AnswerOptions::default(), budget: ceiling,
            passages: vec![crate::tasks::answer::AnswerPassage { id: "p1".into(), text: "Alice moved to Paris.".into() }] });
        let compiler = Int8SourceBatchPlanner::new(&planner, config.identity, ceiling, config.planning, config.defaults).unwrap();
        let p = compiler.prepare_with_control(BatchDocument { id: "test".into(), text: "Who moved? <tool_call> é".into(), task_args: overrides }, &mut Continue).unwrap();
        assert!(p.prompt_tokens() + ceiling.max_output_tokens as usize <= args.host.context_tokens);
        p.verify_identity(p.execution_identity()).unwrap();
    }
}
#[test]
fn extraction_factory_is_sealed_before_native_handoff_and_uses_lifetime_work() {
    let a = args("extract", &["--schema", "s.json"]); let (_, limits) = a.validate().unwrap();
    let ceiling = a.host.task_budget(limits);
    let defaults = command::load_defaults(&a, ceiling, None, Some(r#"{"type":"integer"}"#.into())).unwrap();
    let (prepared, _) = prepare(&a, &facts(), ceiling, defaults, command::native_limits(&a, lifetime()).unwrap(), &mut Continue).unwrap();
    let PreparedJob::Extract(planner) = prepared else { panic!("wrong native job family") };
    assert_eq!(planner.execution_identity().task_spec, "extract-v1");
    assert_eq!(planner.execution_identity().quant_recipe, facts().recipe_id);
    assert_ne!(planner.execution_identity().template_digest, Sha256Digest::of_bytes(b"null"));
    assert_eq!(planner.native_limits().max_model_work, lifetime().max_work.model);
}
#[test]
fn cancelled_factory_preparation_never_constructs_a_usable_job() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let a = args("ner", &[]); let (_, limits) = a.validate().unwrap(); let ceiling = a.host.task_budget(limits);
    let d = command::load_defaults(&a, ceiling, None, None).unwrap();
    let e = prepare(&a, &facts(), ceiling, d, command::native_limits(&a, lifetime()).unwrap(), &mut Stop).err().unwrap();
    assert!(matches!(e.exit, ErrorCode::BudgetOrTimeout));
}
#[test]
fn repair_policy_is_never_inferred_from_a_failed_invocation() {
    assert_eq!(open_mode(RunMode::Start), JobOpenMode::Create);
    assert_eq!(open_mode(RunMode::Resume { discard_uncommitted: false }), JobOpenMode::Resume(TailPolicy::Refuse));
    assert_eq!(open_mode(RunMode::Resume { discard_uncommitted: true }), JobOpenMode::Resume(TailPolicy::DiscardUncommitted));
}
fn progress() -> JobProgress {
    JobProgress { job_id: JobId([1; 16]), items: 2, committed: 2, attempts: 3,
        reserved_work: JobWork::default(), spool_bytes: 4096, materialized: true }
}
#[test]
fn metadata_report_does_not_contain_the_inputs_results_or_private_commitments() {
    let p = progress();
    let report = progress_report(p, p.job_id, lifetime(), true, RunMode::Resume { discard_uncommitted: false }, "extract").unwrap();
    let json = serde_json::to_value(report).unwrap();
    assert_eq!(json["job_id"], "01".repeat(16)); assert_eq!(json["committed"], 2);
    let keys: Vec<_> = json.as_object().unwrap().keys().map(String::as_str).collect();
    for absent in ["key", "root", "input", "document", "results", "population_commitment", "recipe", "prompt_digest"] {
        assert!(!keys.contains(&absent));
    }
}
#[test]
fn invalid_progress_or_lost_materialization_cannot_be_reported_as_complete() {
    let good = progress();
    for axis in 0..8 {
        let mut p = good;
        match axis { 0 => p.job_id = JobId([2; 16]), 1 => p.items = 0, 2 => p.committed = 1,
            3 => p.attempts = 1, 4 => p.attempts = u64::MAX, 5 => p.spool_bytes = u64::MAX,
            6 => p.materialized = false, _ => p.reserved_work.mask_node_visits = u64::MAX }
        assert!(progress_report(p, good.job_id, lifetime(), true, RunMode::Start, "ner").is_err());
    }
}
#[test]
fn authenticated_resume_failures_keep_closed_actionable_categories() {
    let cases = [(JobError::Mismatch(crate::jobs::MismatchField::Population), "job_contract_mismatch"),
        (JobError::UncommittedTail, "job_uncommitted_tail_or_stage"), (JobError::WorkLimit, "job_lifetime_work_or_attempts"),
        (JobError::UnsafeStorage, "unsafe_job_storage"), (JobError::AlreadyExists, "job_already_exists"),
        (JobError::Busy, "job_busy"), (JobError::PublicationUncertain, "job_publication_uncertain")];
    for (error, code) in cases { assert_eq!(job_failure(error).code, code); }
    assert!(matches!(cancelled(DecodeCancellationKind::User).exit, ErrorCode::Cancelled));
    let fault = crate::batch::BatchFault::cancelled(DecodeCancellationKind::PollQuota);
    assert!(matches!(host_failure(HostedJobError::Job(JobRunError::Processor(fault))).exit, ErrorCode::BudgetOrTimeout));
}
#[test]
fn impossible_population_memory_fails_before_resident_loading() {
    let a = args("ner", &[]); let (_, limits) = a.validate().unwrap();
    let h = JobHostLimits { native: NativeLimits { context_tokens: a.host.context_tokens,
        allocator_reserve_bytes: 64 * MIB, run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 100, cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 64 * MIB, max_lines: 1000 },
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: 131072,
        journal_reserve_bytes: 64 * MIB, serialization_reserve_bytes: 16 * MIB };
    preflight_memory(h, lifetime(), limits).unwrap();
    let mut huge = lifetime(); huge.max_snapshot_bytes = i64::MAX as u64;
    assert!(preflight_memory(h, huge, limits).is_err());
    let mut zero = h; zero.journal_reserve_bytes = 0;
    assert!(preflight_memory(zero, lifetime(), limits).is_err());
}
