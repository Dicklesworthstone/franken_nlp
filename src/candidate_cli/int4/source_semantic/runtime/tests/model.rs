//! Controller-only serial/team composition comparison; source is not run evidence.
use super::*;
use crate::{
    hosted::Int4JudgeHostLimits,
    native_engine::portable_int8::ProjectionWork,
    tasks::{ir::TaskBudget, judge::{FaithfulnessPolicy, JudgeRequest, JudgeResult},
        source_planning::{SourceTaskRequest, grouped_int4::{PreparedInt4SourceTask,
            semantic::{SourceSemanticLimits, SourceSemanticStatus}}}},
};

fn command(name: &str, snapshot: &str, pin: &str, memory: &str, kernel: &str)
    -> (Kind, SemanticSourceArgs, Int4Args, Int4Limits) {
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", name, "--snapshot", snapshot,
        "--expected-model-sha256", pin, "--memory-mib", memory, "--kernel", kernel,
        "--semantic-options", "fixture-options-not-read.json", "--context-tokens", "2048",
        "--max-new-tokens", "512", "--timeout-seconds", "86400", "--compute-threads", "4",
    ]).unwrap();
    let CandidateCommand::SourceInt4(Int4Command {
        task: Int4Task::SourceSemantic(kind, command), args, batch: None, job: None,
    }) = CandidateCommand::from_matches(&matches).unwrap() else { panic!("semantic source route required"); };
    let common = command.validate(&args).unwrap();
    (kind, command, args, common)
}

fn options(common: Int4Limits) -> Int4SourceSemanticOptions {
    Int4SourceSemanticOptions { schema_version: 1, experimental_opt_in: true,
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
            evidence_window_bytes: 8192, max_evidence_windows: 8, max_evidence_spans: 8 },
        judge_budget: TaskBudget { max_input_tokens: 1985, max_output_tokens: 64, max_output_bytes: 65536,
            max_grammar_states: 4096, max_kv_bytes: common.native.kv_bytes },
        limits: SourceSemanticLimits { max_statements: 2, max_source_bytes: 4096, max_claim_bytes: 4096,
            max_total_claim_bytes: 8192, max_total_prompt_tokens: 16384,
            max_total_projected_logits: 100_000_000, max_output_bytes: 4 * MIB },
        max_model_work: Int4Work { forward_positions: 100_000, projected_logits: 200_000_000,
            attention_pairs: 1_000_000_000_000, projections: ProjectionWork {
                dot_products: 10_000_000_000_000, multiply_accumulates: 1_000_000_000_000_000 } } }
}

fn prepare(kind: Kind, command: &SemanticSourceArgs, args: &Int4Args,
    model: &crate::hosted::ResidentInt4, request: &SourceTaskRequest,
    control: &mut DeadlineControl) -> (PreparedInt4SourceTask, Arc<crate::tasks::extract::ExtractionVocabulary>) {
    let (compiler, vocabulary) = source::planner().unwrap();
    let identity = source::identity(model.model_identity(), &compiler, source::task(kind)).unwrap();
    let context = PlanContext::new(&identity, request.budget()).unwrap();
    let prepared = compiler.plan_int4_with_control(request, &context, command.source.planning(args),
        model.model_identity().group_width(), control).unwrap();
    (prepared, Arc::new(vocabulary))
}

#[test]
#[ignore = "run alone with a controller-admitted independently pinned snapshot for serial/team source/review parity"]
fn snapshot_source_review_matches_individual_judges_and_persistent_teams_then_releases_owners() {
    assert!(crate::installed_process_resources().is_none(), "fixture requires a fresh process");
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required");
    let (kind, base_command, base_args, common) = command("summarize-semantic", &snapshot, &pin, &memory, &kernel);
    let policy = options(common);
    let common = base_command.admit_options(&base_args, common, &policy).unwrap();
    let host = host_limits(&base_args, &base_command, common, Duration::from_secs(86400));
    // Quote the larger QA layout before model access; the same process owns
    // baseline captures, source assets and diagnostic comparisons throughout.
    let ceiling = input_ceiling(Kind::Answer, &base_args, &policy).unwrap();
    let quote = host.requirements(base_args.group_width, base_command.source.budget(&base_args, common),
        policy.judge_budget, policy.limits, judge_limits(&policy), ceiling).unwrap();
    let common = admit_memory(&base_args, common, quote).unwrap();
    assert_eq!(kind, Kind::Summarize);
    // Admit the widest tested numerical team before model IO. Direct serial
    // calls below retain their existing schedule on this same resource host.
    assert_eq!(common.compute.threads, 4);
    let engine = NlpEngine::builder().resource_config(
        crate::candidate_cli::int4::compute::resource_config(common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes).unwrap().commit().unwrap();
    {
        let model = crate::candidate_cli::int4::runtime::load_model(&engine, &base_args, common,
            Duration::from_secs(3600), CancellationToken::default()).unwrap();
        let resident = engine.resources().memory_snapshot();
        let mut attempted = 0;
        for name in ["summarize-semantic", "answer-semantic"] {
            let (kind, command, args, common) = command(name, &snapshot, &pin, &memory, &kernel);
            let policy = options(common);
            let common = command.admit_options(&args, common, &policy).unwrap();
            let budget = command.source.budget(&args, common);
            let (text, task_options, evidence) = match kind {
                Kind::Summarize => (
                    "Alice works in Shanghai. Bob works in Beijing.",
                    r#"{"max_bullets":2,"max_bullet_scalars":128,"max_citations_per_bullet":1,"max_quote_scalars":64}"#,
                    "Alice works in Shanghai. Bob works in Beijing.",
                ),
                Kind::Answer => (
                    r#"{"question":"Where does Alice work?","passages":[{"id":"alice","text":"Alice works in Shanghai."},{"id":"bob","text":"Bob works in Beijing."}]}"#,
                    r#"{"max_answer_scalars":128,"max_citations":2,"max_quote_scalars":64}"#,
                    "Alice works in Shanghai.\n\nBob works in Beijing.",
                ),
                _ => unreachable!(),
            };
            let request = crate::candidate_cli::source::request(kind, text.to_owned(), Some(task_options),
                budget, args.max_input_bytes).unwrap();
            let host = host_limits(&args, &command, common, Duration::from_secs(86400));
            let quote = host.requirements(args.group_width, budget, policy.judge_budget, policy.limits,
                judge_limits(&policy), policy.input_facts(&request).unwrap()).unwrap();
            admit_memory(&args, common, quote).unwrap();
            let mut control = DeadlineControl { started: Instant::now(), deadline: Duration::from_secs(86400) };
            let (baseline, vocabulary) = prepare(kind, &command, &args, &model, &request, &mut control);
            let source_reserved = baseline.planned_work();
            let baseline = engine.execute_int4_source_task(&model, baseline, vocabulary, host.native,
                host.masks, host.max_mask_node_visits, common.kernel, CancellationToken::default()).unwrap();
            let original = crate::canonjson::canonical_bytes(baseline.result()).unwrap();
            drop(baseline);
            assert_eq!(engine.resources().memory_snapshot(), resident);

            let (source, vocabulary) = prepare(kind, &command, &args, &model, &request, &mut control);
            let judge = Int4JudgePlanner::for_model(model.model_identity(), policy.judge_budget, judge_limits(&policy)).unwrap();
            let prepared = PreparedInt4SourceSemantic::new(source, request.clone(), judge, policy.clone(), &mut control).unwrap();
            let result = engine.execute_int4_semantic_source(&model, prepared, vocabulary, host,
                common.kernel, CancellationToken::default()).unwrap();
            let run = result.result();
            assert_eq!(crate::canonjson::canonical_bytes(&run.source_task).unwrap(), original);
            check_result(run, args.group_width, source_reserved, policy.max_model_work, policy.policy_digest().unwrap()).unwrap();
            assert_eq!(run.source_reserved_work, source_reserved);
            assert_eq!(run.reserved_model_work, source_reserved.checked_add(run.semantic_model_work).unwrap());
            assert_eq!(run.model_work, run.source_task.model_work.checked_add(run.semantic_model_work).unwrap());
            attempted += run.semantic.attempted_statements;
            let mut review_work = Int4Work::default();
            let mut heads = 0;
            let mut rewound = 0;
            for statement in &run.semantic.statements {
                let Some(claim) = &statement.claim else {
                    assert_eq!(statement.status, SourceSemanticStatus::NotChecked);
                    assert!(statement.judge.is_none() && statement.not_checked.is_some());
                    continue;
                };
                let judge = Int4JudgePlanner::for_model(model.model_identity(), policy.judge_budget, judge_limits(&policy)).unwrap();
                let plan = judge.plan_with_control(&JudgeRequest::Faithfulness { source: evidence.to_owned(),
                    claim: claim.clone(), policy: policy.judge_policy, budget: policy.judge_budget }, &mut control).unwrap();
                let judgment = engine.execute_int4_judge(&model, plan, Int4JudgeHostLimits {
                    native: host.native, preparation_reserve_bytes: common.preparation_bytes,
                    max_model_work: policy.max_model_work,
                }, common.kernel, CancellationToken::default()).unwrap();
                let standalone = judgment.result();
                let JudgeResult::Faithfulness(faithfulness) = &standalone.result else { panic!("faithfulness required"); };
                assert_eq!(crate::canonjson::canonical_bytes(statement.judge.as_ref().unwrap()).unwrap(),
                    crate::canonjson::canonical_bytes(faithfulness).unwrap());
                review_work = review_work.checked_add(standalone.model_work).unwrap();
                heads += standalone.head_count; rewound += standalone.rewound_positions;
            }
            assert_eq!(run.semantic_model_work, review_work);
            assert_eq!(run.judge_head_count, heads);
            assert_eq!(run.rewound_positions, rewound);
            let serial = crate::canonjson::canonical_bytes(run).unwrap();
            let serial_work = run.model_work;
            drop(result);
            assert_eq!(engine.resources().memory_snapshot(), resident);
            let closures = engine.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);

            let mut projection_counts = None;
            for threads in [2, 4] {
                let (source, vocabulary) = prepare(kind, &command, &args, &model, &request, &mut control);
                let judge = Int4JudgePlanner::for_model(model.model_identity(),
                    policy.judge_budget, judge_limits(&policy)).unwrap();
                let prepared = PreparedInt4SourceSemantic::new(source, request.clone(), judge,
                    policy.clone(), &mut control).unwrap();
                let team = engine.execute_int4_semantic_source_with_threads(&model, prepared,
                    vocabulary, host, common.kernel, threads, CancellationToken::default()).unwrap();
                // Compare the COMPLETE composition: original generation,
                // every review verdict/evidence item and all semantic work.
                assert_eq!(crate::canonjson::canonical_bytes(team.result()).unwrap(), serial);
                assert_eq!(team.result().model_work, serial_work);
                assert_eq!(crate::canonjson::canonical_bytes(&team.result().source_task).unwrap(), original);
                let receipt = team.compute();
                assert_eq!(receipt.schedule, "int4-output-row-team-v1");
                assert_eq!(receipt.compute_threads, threads);
                assert!(receipt.completed_projections > 0 && receipt.completed_worker_rows > 0);
                assert!(receipt.completed_worker_rows <= receipt.completed_rows);
                let counts = (receipt.completed_projections, receipt.completed_rows);
                if let Some(expected) = projection_counts { assert_eq!(counts, expected); }
                else { projection_counts = Some(counts); }
                // A returned result owns output storage, while physical work
                // and all scoped children have already drained.
                assert_ne!(engine.resources().memory_snapshot(), resident);
                let closures = engine.resources().outstanding_closure_snapshot();
                assert_eq!(closures.active_closures, 0);
                assert_eq!(closures.wrapper_cancelled_closures, 0);
                drop(team);
                assert_eq!(engine.resources().memory_snapshot(), resident);
            }

            let (source, vocabulary) = prepare(kind, &command, &args, &model, &request, &mut control);
            let judge = Int4JudgePlanner::for_model(model.model_identity(),
                policy.judge_budget, judge_limits(&policy)).unwrap();
            let prepared = PreparedInt4SourceSemantic::new(source, request, judge,
                policy.clone(), &mut control).unwrap();
            let cancelled = CancellationToken::default();
            cancelled.cancel(DecodeCancellationKind::User);
            let error = engine.execute_int4_semantic_source_with_threads(&model, prepared,
                vocabulary, host, common.kernel, 4, cancelled).err().expect("cancelled request has no result");
            assert!(matches!(error, HostedError::Stopped { ref stop, .. }
                if stop.kind == DecodeCancellationKind::User));
            assert_eq!(engine.resources().memory_snapshot(), resident);
            let closures = engine.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0);
            assert_eq!(closures.wrapper_cancelled_closures, 0);
        }
        // This fixture must exercise actual review inference. No particular
        // relation is asserted, and no accuracy or calibration claim follows.
        assert!(attempted > 0, "fixture population produced no claim to review");
        drop(model);
    }
    drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}
