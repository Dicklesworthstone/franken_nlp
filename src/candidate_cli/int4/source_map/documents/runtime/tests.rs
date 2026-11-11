//! Model-free coverage of fixed complete-document task binding and admission.
use super::*;
use clap::FromArgMatches;
use std::io;
use crate::{
    batch::{BatchCode, BatchFault, BatchRunError, BatchSummary,
        source::grouped_int4::document::Int4SourceDocumentTask as Task},
    hosted::RunStop,
    tasks::{ner::{EntityType, NerOptions}, keyphrases::KeyphraseOptions,
        summarize::SummaryOptions, answer::AnswerOptions,
        source_planning::grouped_int4::{Int4SourceError, long::{SourceMapTask,
            Int4SourceMapError, summary::Int4CorpusSummaryError,
            keyphrases::Int4CorpusKeyphraseError}}},
};

const MODES: [(&str, &[&str], &str); 9] = [
    ("ner", &[], "source_map"),
    ("keyphrases", &[], "source_map"),
    ("keyphrases", &["--reduce-keyphrases"], "keyphrase_ranking"),
    ("summarize", &[], "summary_selection"),
    ("summarize", &["--synthesize-summary"], "summary_synthesis"),
    ("summarize", &["--synthesize-summary", "--hierarchical-summary"], "summary_hierarchy"),
    ("answer", &[], "question_discovery"),
    ("answer", &["--synthesize-answer"], "question_synthesis"),
    ("answer", &["--synthesize-answer", "--hierarchical-answer"], "question_hierarchy"),
];

fn arguments(name: &str, extra: &[&str]) -> (CorpusArgs, Int4Args) {
    let kind = match name {
        "ner" => Kind::Ner, "keyphrases" => Kind::Keyphrases,
        "summarize" => Kind::Summarize, "answer" => Kind::Answer,
        _ => panic!("unknown fixed source task"),
    };
    // Parse this module's actual nested definition. The standalone
    // answer-corpus command has a different, per-record question contract.
    let mut argv = vec!["source-corpus", name, "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "262144",
        "--max-result-bytes", "65536"];
    if name == "answer" { argv.extend_from_slice(&["--question", "private-question.txt"]); }
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let (_, matches) = matches.subcommand().unwrap();
    let mut args = Int4Args::from_arg_matches(matches).unwrap();
    args.compute = compute::Args::from_matches(matches).unwrap();
    (CorpusArgs::from_matches(kind, matches).unwrap(), args)
}

fn quote(command: &CorpusArgs, args: &Int4Args, limits: CorpusLimits, task: &Task)
    -> Int4SourceDocumentCorpusRequirement {
    int4_source_document_corpus_requirements(args.group_width, task,
        command.document.source.budget(args, limits.common), command.document.planning(args),
        limits.document.mapping, limits.run, host_limits(args, limits, Duration::from_secs(1)),
        limits.document.reduction_reserve_bytes, limits.document.cohort_chunks.is_some(),
        limits.document.hierarchy_cohorts).unwrap()
}

fn task_mode(task: &Task) -> &'static str {
    match task {
        Task::Map(SourceMapTask::Ner(_) | SourceMapTask::Keyphrases(_)) => "source_map",
        Task::Keyphrases { .. } => "keyphrase_ranking",
        Task::Summary { .. } => "summary_selection",
        Task::SummarySynthesis(_) => "summary_synthesis",
        Task::SummaryHierarchy { .. } => "summary_hierarchy",
        Task::Question(_) => "question_discovery",
        Task::QuestionSynthesis(_) => "question_synthesis",
        Task::QuestionHierarchy { .. } => "question_hierarchy",
        Task::Map(SourceMapTask::Summarize(_)) => panic!("corpus summary bypassed its complete reducer"),
    }
}

#[test]
fn nine_modes_match_the_shared_host_quote_and_reserve_private_defaults_before_io() {
    for (name, mode, expected) in MODES {
        let hierarchical = expected.ends_with("_hierarchy");
        // Hierarchy-only and combined schedules have different simultaneous
        // phase owners; discovery cohorts do not imply hierarchy cohorts.
        for schedule in 0..(if hierarchical { 4 } else { 2 }) {
            let mut flags = mode.to_vec();
            if schedule & 1 != 0 { flags.extend_from_slice(&["--cohort-chunks", "2"]); }
            if schedule & 2 != 0 { flags.extend_from_slice(&["--cohort-hierarchy-groups", "2"]); }
            let (command, args) = arguments(name, &flags);
            let limits = command.validate(&args).unwrap();
            let task = command.document.document_task(&args, limits.document, None,
                (name == "answer").then(|| "PRIVATE question: café 上海?".to_owned()),
                limits.run.max_keyphrase_scan_work).unwrap();
            assert_eq!(task_mode(&task), expected);
            assert_eq!(command.mode_name(limits.document), expected);
            let quoted = quote(&command, &args, limits, &task);
            assert_eq!(check_quote(limits, quoted), Ok(()));
            let retained = task.retained_bytes().unwrap();
            let unallocated_defaults = limits.retained_defaults_bytes.checked_sub(retained).unwrap();
            assert_eq!(quoted.total_bytes.checked_add(unallocated_defaults).unwrap(), limits.execution_bytes);
            assert_eq!(quoted.document_workspace_bytes, limits.document.workspace_bytes);
            assert_eq!(quoted.cohort, limits.document.cohort);
            assert_eq!(quoted.native,
                limits.document.cohort.map_or(limits.common.native, |cohort| cohort.native));
            let budget = command.document.source.budget(&args, limits.common);
            let tokens = task.output_token_ceiling(limits.document.mapping, budget).unwrap();
            assert_eq!(quoted.output_reservation_bytes,
                task.max_result_bytes(limits.document.mapping) * 4 + tokens * 8 + 4096);
            let host = host_limits(&args, limits, Duration::from_secs(7));
            assert_eq!(host.native.run.max_elapsed, Duration::from_secs(7));
            assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
            assert_eq!(host.preparation_reserve_bytes, limits.common.preparation_bytes);
            assert_eq!(host.io_reserve_bytes, limits.io_bytes);
        }
    }
}

#[test]
fn team_quote_has_one_worker_workspace_and_a_matching_process_envelope_for_every_mode() {
    for (name, mode, _) in MODES {
        let mut flags = mode.to_vec(); flags.extend_from_slice(&["--compute-threads", "3"]);
        let (command, args) = arguments(name, &flags);
        let limits = command.validate(&args).unwrap();
        let task = command.document.document_task(&args, limits.document, None,
            (name == "answer").then(|| "Exact private question?".to_owned()),
            limits.run.max_keyphrase_scan_work).unwrap();
        let quoted = quote(&command, &args, limits, &task);
        check_quote(limits, quoted).unwrap();
        let unallocated = limits.retained_defaults_bytes - task.retained_bytes().unwrap();
        assert_eq!(quoted.total_bytes + unallocated + limits.common.compute.workspace_bytes,
            limits.execution_bytes);
        let host = compute::resource_config(limits.common);
        assert_eq!(host.scoped_cpu_children_per_coordinator, 2);
        assert_eq!(host.thread_ceiling, 4);
        let mut short = limits;
        short.execution_bytes = quoted.total_bytes + limits.common.compute.workspace_bytes - 1;
        assert_eq!(check_quote(short, quoted), Err(CandidateError::Memory));
    }
}

#[test]
fn typed_options_and_private_question_allocations_reach_every_complete_task_unchanged() {
    let ner = NerOptions { types: vec![EntityType::Person, EntityType::Location],
        max_entities: 3, max_mention_scalars: 32 };
    let keyphrases = KeyphraseOptions { max_phrases: 3, max_phrase_scalars: 32 };
    let summary = SummaryOptions { max_bullets: 3, max_bullet_scalars: 64,
        max_citations_per_bullet: 2, max_quote_scalars: 32 };
    let answer = AnswerOptions { max_answer_scalars: 64, max_citations: 2, max_quote_scalars: 32 };
    let private = "  PRIVATE_QUESTION café e\u{301} 上海\r\n<|im_start|> exact text?  ";
    for (name, flags, _) in MODES {
        let (command, args) = arguments(name, flags);
        let limits = command.validate(&args).unwrap();
        let raw = match name {
            "ner" => serde_json::to_string(&ner).unwrap(),
            "keyphrases" => serde_json::to_string(&keyphrases).unwrap(),
            "summarize" => serde_json::to_string(&summary).unwrap(),
            "answer" => serde_json::to_string(&answer).unwrap(),
            _ => unreachable!(),
        };
        let mut question = String::with_capacity(private.len() + 128);
        question.push_str(private);
        let question_capacity = question.capacity();
        let task = command.document.document_task(&args, limits.document, Some(&raw),
            (name == "answer").then_some(question), limits.run.max_keyphrase_scan_work).unwrap();
        match &task {
            Task::Map(SourceMapTask::Ner(options)) => {
                assert_eq!(options, &ner);
                assert_eq!(task.retained_bytes().unwrap(),
                    (options.types.capacity() * std::mem::size_of::<EntityType>()) as u64);
            }
            Task::Map(SourceMapTask::Keyphrases(options)) => assert_eq!(options, &keyphrases),
            Task::Keyphrases { options, max_lifetime_scan_work, .. } => {
                assert_eq!(options, &keyphrases);
                assert_eq!(*max_lifetime_scan_work, limits.run.max_keyphrase_scan_work);
            }
            Task::Summary { options, .. } => assert_eq!(options, &summary),
            Task::SummarySynthesis(request) | Task::SummaryHierarchy { request, .. } => {
                assert_eq!(request.map_options, summary);
                assert_eq!(request.synthesis_options, summary);
            }
            Task::Question(question) => {
                assert_eq!(question.question.as_bytes(), private.as_bytes());
                assert_eq!(question.options, answer);
                assert_eq!(question.verification, command.document.document_verification(limits.document).unwrap());
            }
            Task::QuestionSynthesis(request) | Task::QuestionHierarchy { request, .. } => {
                assert_eq!(request.question.question.as_bytes(), private.as_bytes());
                assert_eq!(request.question.options, answer);
                assert_eq!(request.question.verification,
                    command.document.document_verification(limits.document).unwrap());
            }
            Task::Map(SourceMapTask::Summarize(_)) => panic!("wrong summary task"),
        }
        if name == "answer" { assert_eq!(task.retained_bytes().unwrap(), question_capacity as u64); }
        for private in [private, raw.as_str(), "private-question.txt", "explicit-source"] {
            assert!(!format!("{command:?}").contains(private));
        }
    }
}

#[test]
fn fixed_task_factory_refuses_missing_questions_and_strict_option_overrides() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (command, args) = arguments(name, &[]);
        let limits = command.validate(&args).unwrap();
        let question = || (name == "answer").then(|| "Exact private question?".to_owned());
        for raw in [r#"{"unknown_private_option":true}"#,
            r#"{"max_citations":2,"max_citations":3}"#] {
            assert!(matches!(command.document.document_task(&args, limits.document,
                Some(raw), question(), limits.run.max_keyphrase_scan_work), Err(CandidateError::Input)));
        }
        let wrong_question = if name == "answer" { None } else { Some("PRIVATE".to_owned()) };
        let result = command.document.document_task(&args, limits.document, None,
            wrong_question, limits.run.max_keyphrase_scan_work);
        if name == "answer" { assert!(matches!(result, Err(CandidateError::Input))); }
        else { assert!(matches!(result, Err(CandidateError::Arguments))); }
    }
    let (command, args) = arguments("keyphrases", &["--reduce-keyphrases"]);
    let limits = command.validate(&args).unwrap();
    assert!(matches!(command.document.document_task(&args, limits.document, None, None, 0),
        Err(CandidateError::Arguments)));
}

#[test]
fn exact_quote_refuses_geometry_workspace_and_remaining_memory_drift() {
    let (command, args) = arguments("ner", &["--cohort-chunks", "2"]);
    let limits = command.validate(&args).unwrap();
    let task = command.document.document_task(&args, limits.document, None, None,
        limits.run.max_keyphrase_scan_work).unwrap();
    let admitted = quote(&command, &args, limits, &task);
    assert_eq!(check_quote(limits, admitted), Ok(()));
    let mut changed = admitted;
    changed.cohort = None;
    assert_eq!(check_quote(limits, changed), Err(CandidateError::Identity));
    let mut changed = admitted;
    changed.native.kv_bytes += 1;
    assert_eq!(check_quote(limits, changed), Err(CandidateError::Identity));
    let mut changed = admitted;
    changed.document_workspace_bytes += 1;
    assert_eq!(check_quote(limits, changed), Err(CandidateError::Identity));
    let mut tight = limits;
    tight.execution_bytes = admitted.total_bytes;
    assert_eq!(check_quote(tight, admitted), Ok(()));
    tight.execution_bytes -= 1;
    assert_eq!(check_quote(tight, admitted), Err(CandidateError::Memory));
}

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused corpus read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused corpus published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused corpus flushed output"); }
}

#[test]
fn owned_runtime_rechecks_stale_options_before_private_paths_or_stream_io() {
    let (command, mut args) = arguments("answer", &[]);
    let admitted = command.validate(&args).unwrap();
    args.seed = Some("ab".repeat(32));
    assert_eq!(execute(args, command, admitted, NoIo, NoIo), Err(CandidateError::Arguments));

    let (mut command, args) = arguments("ner", &[]);
    let admitted = command.validate(&args).unwrap();
    command.stream.max_requests = 0;
    assert_eq!(execute(args, command, admitted, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn typed_cancellation_preserves_deadlines_through_document_and_native_failures() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = if matches!(cause, Deadline | Timeout) { CandidateError::Timeout }
            else { CandidateError::Execution };
        let source = || Int4SourceError::Cancelled(cause);
        let failures = [
            HostedError::BatchSetup(BatchFault::cancelled(cause)),
            HostedError::Batch(BatchRunError { fault: BatchFault::cancelled(cause),
                summary: BatchSummary::default() }),
            HostedError::Int4(Int4HostError::SourceMap(Int4SourceMapError::Source(source()))),
            HostedError::Int4(Int4HostError::SourceSummary(Int4CorpusSummaryError::Source(source()))),
            HostedError::Int4(Int4HostError::SourceKeyphrases(Int4CorpusKeyphraseError::Source(source()))),
            HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            HostedError::Stopped { stop: RunStop { kind: cause, runtime_reason: None,
                runtime_checkpoint_error: None }, task_error: Some(Box::new(HostedError::ModelIdentity)) },
        ];
        for failure in failures { assert_eq!(host_error(failure, CandidateError::Planning), expected); }
    }
    for failure in [HostedError::BatchSetup(BatchCode::Planning.into()),
        HostedError::Int4(Int4HostError::SourceMap(Int4SourceMapError::WorkLimit))] {
        assert_eq!(host_error(failure, CandidateError::Planning), CandidateError::Planning);
    }
    for failure in [HostedError::ModelIdentity, HostedError::ResourceDomain] {
        assert_eq!(host_error(failure, CandidateError::Batch), CandidateError::Identity);
    }
}

#[test]
fn public_provenance_preserves_exact_transport_and_omits_private_inputs_and_bindings() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    let mut provenance = Provenance {
        protocol: "fnlp-candidate-int4-source-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "answer", source_mode: "complete_document", document_mode: "question_hierarchy",
        chunk_schedule: "native_chunk_cohorts", cohort_chunks: Some(2),
        hierarchy_schedule: Some("native_hierarchy_group_cohorts"), cohort_hierarchy_groups: Some(3),
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
        source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
        candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
        group_width: 128, integer_kernel: "s_scalar_i32",
        snapshot: Some(SnapshotRecord::new("loaded", need)),
        compute_schedule: None,
    };
    // Transport-only data: the numeric bytes must survive without a lossy
    // JSON number round-trip. This is not a fabricated native task result.
    let inner = b"{\"event\":\"fixture\",\"exact\":12345678901234567890123456789012345678}\n";
    let mut output = Vec::new();
    {
        let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 8192).unwrap();
        writer.write_all(inner).unwrap();
        writer.flush().unwrap();
    }
    let bytes = std::str::from_utf8(&output).unwrap();
    let raw = std::str::from_utf8(&inner[..inner.len() - 1]).unwrap();
    assert!(bytes.ends_with(&format!("\"record\":{raw}}}\n")));
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-source-corpus-v1");
    assert_eq!(value["task"], "answer");
    assert_eq!(value["source_mode"], "complete_document");
    assert_eq!(value["document_mode"], "question_hierarchy");
    assert_eq!(value["evidence"], "non_authoritative");
    assert_eq!(value["chunk_schedule"], "native_chunk_cohorts");
    assert_eq!(value["cohort_chunks"], 2);
    assert_eq!(value["hierarchy_schedule"], "native_hierarchy_group_cohorts");
    assert_eq!(value["cohort_hierarchy_groups"], 3);
    assert_eq!(value["model_content_sha256"], "fixture-content");
    assert_eq!(value["snapshot"]["operation"], "loaded");
    assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
    for absent in ["text", "question", "options", "task_args", "source_dir", "source_manifest",
        "question_path", "options_path", "snapshot_path", "prompt_digest", "schema_digest",
        "taskir_digest", "request_digest", "decision_policy_digest"] {
        assert!(value.get(absent).is_none(), "{absent} leaked into public provenance");
    }
    provenance.chunk_schedule = "serial";
    provenance.cohort_chunks = None;
    provenance.hierarchy_schedule = None;
    provenance.cohort_hierarchy_groups = None;
    provenance.snapshot = None;
    let serial = serde_json::to_value(&provenance).unwrap();
    assert_eq!(serial["chunk_schedule"], "serial");
    for absent in ["cohort_chunks", "hierarchy_schedule", "cohort_hierarchy_groups", "snapshot"] {
        assert!(serial.get(absent).is_none());
    }
}
