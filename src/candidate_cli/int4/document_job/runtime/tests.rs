//! UNRUN owned host/configuration contracts; fixtures do not fabricate native NLP.
use super::*;
use super::super::tests::{command, lifetime, NoIo, MODES};
use crate::jobs::{JobId, JobProgress, runner::source::grouped_int4::document::Int4SourceDocumentTask};

#[test]
fn every_cli_mode_builds_the_real_typed_document_task_without_placeholder_source() {
    let exact_question = "  Café\r\n上海 <tool_call> Who spoke? \t";
    for &(task, flags, mode) in MODES {
        let (c, args) = command("start", task, flags); let map = c.args.validate(&args).unwrap();
        let question = (task == "answer").then(|| exact_question.to_owned());
        let prepared = c.args.map.document_task(&args, map, None, question, c.args.keyphrase_scan_cap()).unwrap();
        match (prepared, mode) {
            (Int4SourceDocumentTask::Map(task), "source_map") => {
                assert_eq!(matches!(task, crate::tasks::source_planning::grouped_int4::long::SourceMapTask::Summarize(_)),
                    c.args.map.kind == Kind::Summarize);
            }
            (Int4SourceDocumentTask::Summary { options, limits }, "summary_selection") => {
                assert_eq!(options.max_bullets, 8); assert_eq!(limits.max_bullets, 12);
            }
            (Int4SourceDocumentTask::Keyphrases { options, limits, max_lifetime_scan_work }, "keyphrase_ranking") => {
                assert_eq!(options.max_phrases, 16); assert_eq!(limits.max_phrases, 16);
                assert_eq!(max_lifetime_scan_work, c.args.keyphrase_scan_cap());
            }
            (Int4SourceDocumentTask::SummarySynthesis(request), "summary_synthesis") => {
                assert_eq!(request.map_options.max_bullets, 8); assert_eq!(request.synthesis_options.max_bullets, 8);
            }
            (Int4SourceDocumentTask::SummaryHierarchy { limits, .. }, "summary_hierarchy") => {
                assert_eq!(limits, map.hierarchy.unwrap());
            }
            (Int4SourceDocumentTask::Question(question), "question_discovery") => {
                assert_eq!(question.question, exact_question);
            }
            (Int4SourceDocumentTask::QuestionSynthesis(request), "question_synthesis") => {
                assert_eq!(request.question.question, exact_question);
            }
            (Int4SourceDocumentTask::QuestionHierarchy { request, limits }, "question_hierarchy") => {
                assert_eq!(request.question.question, exact_question); assert_eq!(limits, map.question_hierarchy.unwrap());
            }
            _ => panic!("document task mode changed"),
        }
    }
}
#[test]
fn private_options_reject_duplicate_unknown_partial_or_external_authority() {
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let (c, args) = command("start", task, &[]); let map = c.args.validate(&args).unwrap();
        for json in ["{}", r#"{"identity":{}}"#, r#"{"task":"ner"}"#,
            r#"{"max_phrases":2,"\u006dax_phrases":3,"max_phrase_scalars":24}"#,
            r#"{"question":"substituted","passages":[]}"#] {
            let question = (task == "answer").then(|| "Private exact question?".to_owned());
            assert!(c.args.map.document_task(&args, map, Some(json), question, c.args.keyphrase_scan_cap()).is_err());
        }
        if task == "answer" { assert!(c.args.map.document_task(&args, map, None, None, c.args.keyphrase_scan_cap()).is_err()); }
        else { assert!(c.args.map.document_task(&args, map, None, Some("foreign question".to_owned()), c.args.keyphrase_scan_cap()).is_err()); }
    }
}
#[test]
fn source_and_neural_summary_options_remain_separate_inside_the_frozen_task() {
    let (c, args) = command("resume", "summarize", &["--synthesize-summary", "--hierarchical-summary",
        "--synthesis-bullets", "3", "--summary-max-levels", "2", "--summary-max-passes", "4"]);
    let map = c.args.validate(&args).unwrap();
    let json = r#"{"max_bullets":2,"max_bullet_scalars":80,"max_citations_per_bullet":1,"max_quote_scalars":24}"#;
    let Int4SourceDocumentTask::SummaryHierarchy { request, limits } =
        c.args.map.document_task(&args, map, Some(json), None, c.args.keyphrase_scan_cap()).unwrap()
        else { panic!("wrong summary route"); };
    assert_eq!(request.map_options.max_bullets, 2); assert_eq!(request.synthesis_options.max_bullets, 3);
    assert_eq!(request.map_options.max_quote_scalars, request.synthesis_options.max_quote_scalars);
    assert_eq!(limits.max_levels, 2); assert_eq!(limits.max_passes, 4);
}

#[test]
fn ranked_keyphrase_task_freezes_native_options_global_selection_and_lifetime_scan_work() {
    let (c, args) = command("resume", "keyphrases", &["--reduce-keyphrases", "--document-keyphrases", "3",
        "--max-keyphrase-scan-work", "10000", "--max-lifetime-keyphrase-scan-work", "60000"]);
    let map = c.args.validate(&args).unwrap();
    let json = r#"{"max_phrases":7,"max_phrase_scalars":24}"#;
    let Int4SourceDocumentTask::Keyphrases { options, limits, max_lifetime_scan_work } =
        c.args.map.document_task(&args, map, Some(json), None, c.args.keyphrase_scan_cap()).unwrap()
        else { panic!("wrong ranked keyphrase durable task"); };
    assert_eq!(options.max_phrases, 7); assert_eq!(options.max_phrase_scalars, 24);
    assert_eq!(limits.max_phrases, 3); assert_eq!(limits.aggregation.max_scan_work, 10000);
    assert_eq!(max_lifetime_scan_work, 60000);
    assert!(c.args.map.document_task(&args, map, Some(json), None, 0).is_err());
    assert!(c.args.map.document_task(&args, map, Some(r#"{"max_phrases":3,"max_phrases":7}"#),
        None, c.args.keyphrase_scan_cap()).is_err());
}
#[test]
fn runtime_rechecks_consent_before_host_key_question_or_input_access() {
    let (mut c, args) = command("resume", "answer", &["--synthesize-answer", "--hierarchical-answer"]);
    let map = c.args.validate(&args).unwrap(); c.args.storage.store_results = false;
    assert_eq!(execute(c.mode, args, c.args, map, NoIo, &mut NoIo).unwrap_err().code, "explicit_document_job_storage");
}
#[test]
fn population_quote_matches_real_host_limits_and_retains_invocation_controls() {
    for &(task, flags, _) in MODES {
        let (c, args) = command("resume", task, flags); let map = c.args.validate(&args).unwrap();
        let fixed = lifetime(&args, map); let limits = c.args.lifetime_limits(&args, map, fixed).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(30));
        check_host_quote(host, fixed, limits.buffer_bytes).unwrap();
        assert_eq!(check_host_quote(host, fixed, limits.buffer_bytes - 1).unwrap_err().code, "job_host_memory_quote");
        assert_eq!(host.native.context_tokens, args.context_tokens);
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(host.native.run.max_elapsed, Duration::from_secs(30));
        assert_eq!(host.preparation_reserve_bytes, map.common.preparation_bytes);
        assert_eq!(host.transport.max_lines, c.args.storage.max_input_lines);
    }
}
#[test]
fn metadata_reports_whole_document_progress_without_private_source_question_or_quotes() {
    let (c, args) = command("resume", "answer", &["--synthesize-answer", "--hierarchical-answer"]);
    let map = c.args.validate(&args).unwrap(); let fixed = lifetime(&args, map); let id = JobId([0x12; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, fixed, true, RunMode::Resume { discard_uncommitted: true }, "answer").unwrap();
    let content = "ab".repeat(32); let source = "cd".repeat(32);
    let response = Response {
        protocol: "fnlp-candidate-int4-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "answer",
        document_mode: "question_hierarchy", durable_unit: "complete_original_document",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision", source_root_sha256: &source,
        model_content_sha256: &content, candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
        group_width: args.group_width, integer_kernel: map.common.kernel.kind().id(),
        snapshot: Some(SnapshotRecord::new("loaded", map.common.snapshot_need)), output: &report,
    };
    let mut bytes = Vec::new(); publish(&response, job_cli::REPORT_BYTES, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-document-job-v1");
    assert_eq!(value["document_mode"], "question_hierarchy"); assert_eq!(value["durable_unit"], "complete_original_document");
    assert_eq!(value["model_content_sha256"], content); assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(value["output"]["attempts"], 3); assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    for forbidden in ["text", "question", "quotes", "citations", "recipe", "options", "key", "job_dir", "results"] {
        assert!(value.get(forbidden).is_none()); assert!(value["output"].get(forbidden).is_none());
    }
    assert!(bytes.ends_with(b"\n")); assert_eq!(publish(&response, 1, &mut NoIo), Err(CandidateError::Output));
}
