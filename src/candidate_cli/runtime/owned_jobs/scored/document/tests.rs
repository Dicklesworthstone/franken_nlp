use super::*;
use crate::{
    candidate_cli::jobs::scored::document::tests::{document_args, lifetime, SETTINGS},
    jobs::runner::scored::document::{Int8ClassificationDocumentJobPlanner, Int8SentimentDocumentJobPlanner},
};

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "document-job-cli-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
const LABELS: &str = r#"{"labels":[{"id":"legal","description":"Private legal text"},{"id":"other"}],"mode":"multi_label"}"#;

fn prepared(task: &str, extra: &[&str]) -> (ScoreJobArgs, Limits, Mapping, DocumentCorpus) {
    let args = document_args(task, extra);
    let (_, limits) = args.validate().unwrap();
    let mapping = settings::load(SETTINGS, &args, limits, lifetime(&args)).unwrap();
    let defaults = args.parse_defaults((task == "classify").then_some(LABELS),
        args.task_budget(limits).unwrap()).unwrap();
    let corpus = configure(&args, &facts(), limits, mapping, defaults).unwrap();
    let mapping = settings::load(SETTINGS, &args, limits, lifetime(&args)).unwrap();
    (args, limits, mapping, corpus)
}

fn recipe(corpus: DocumentCorpus) -> serde_json::Value {
    match corpus {
        DocumentCorpus::Classify(planner, config) => {
            let factory = Int8ClassificationDocumentJobPlanner::new(planner, config.identity,
                config.task_ceiling, config.planning, config.mapping, config.defaults,
                config.max_model_work, config.max_tokenizer_work).unwrap();
            serde_json::to_value(factory.job_recipe()).unwrap()
        }
        DocumentCorpus::Sentiment(planner, config) => {
            let factory = Int8SentimentDocumentJobPlanner::new(planner, config).unwrap();
            serde_json::to_value(factory.job_recipe()).unwrap()
        }
    }
}

#[test]
fn cli_settings_reach_distinct_complete_document_factories_and_immutable_recipes() {
    for task in ["classify", "sentiment"] {
        for compact in [false, true] {
            let flags: &[&str] = if compact { &["--input-encoding", "safe-bpe-v1"] } else { &[] };
            let (args, limits, mapping, corpus) = prepared(task, flags);
            let wire = recipe(corpus);
            assert_eq!(wire["dependency_scope"], "item-local-complete-document");
            assert_eq!(wire["chunks"]["max_chunks"], 4);
            assert_eq!(wire["chunks"]["max_chunk_bytes"], 1024);
            assert_eq!(wire["max_result_bytes"], 4 << 20);
            assert_eq!(wire["per_document_model_work"], serde_json::to_value(mapping.work()).unwrap());
            assert_eq!(wire["scoring"]["task_ceiling"], serde_json::to_value(args.task_budget(limits).unwrap()).unwrap());
            assert_eq!(wire["scoring"]["input_encoding"], serde_json::to_value(args.encoding_policy().unwrap()).unwrap());
            if task == "classify" {
                assert_eq!(wire["scoring"]["defaults"]["labels"][0]["description"], "Private legal text");
            } else {
                assert_eq!(wire["scoring"]["defaults"]["axes"].as_array().unwrap().len(), 4);
            }
        }
    }
}

#[test]
fn physical_prefill_and_team_widths_cannot_change_authenticated_document_semantics() {
    for task in ["classify", "sentiment"] {
        let expected = recipe(prepared(task, &[]).3);
        for flags in [vec!["--prefill-rows", "4"], vec!["--compute-threads", "4"],
            vec!["--prefill-rows", "4", "--compute-threads", "4"]] {
            let actual = recipe(prepared(task, &flags).3);
            assert_eq!(actual, expected);
        }
    }
}

fn host(args: &ScoreJobArgs, limits: Limits) -> JobHostLimits {
    JobHostLimits {
        native: crate::hosted::NativeLimits {
            context_tokens: args.host.context_tokens, allocator_reserve_bytes: 64 * MIB,
            run: crate::hosted::RunLimits { max_elapsed: Duration::from_secs(30),
                max_checkpoints: args.host.max_checkpoints, cleanup_reserve_bytes: 65_536 },
        },
        transport: PopulationReadLimits { max_stream_bytes: args.max_stream_mib * MIB,
            max_lines: args.max_input_lines },
        preparation_reserve_bytes: limits.preparation_bytes, io_reserve_bytes: (IO_BYTES * 2) as u64,
        journal_reserve_bytes: args.journal_memory_mib * MIB,
        serialization_reserve_bytes: args.serialization_memory_mib * MIB,
    }
}

#[test]
fn complete_map_population_prefill_and_workers_share_one_exact_memory_ceiling() {
    for task in ["classify", "sentiment"] {
        for flags in [vec![], vec!["--compute-threads", "4", "--prefill-rows", "4"],
            vec!["--input-encoding", "safe-bpe-v1", "--spool-input", "--max-retained-input-mib", "4"]] {
            let (args, mut limits, mapping, _) = prepared(task, &flags);
            let host = host(&args, limits);
            let job = lifetime(&args);
            let peak = memory_bytes(&args, limits, &mapping, host, job).unwrap();
            assert!(peak > args.execution_memory_bytes(limits).unwrap());
            limits.memory_bytes = peak;
            assert_eq!(admit(&args, limits, &mapping, host, job).unwrap(), peak);
            limits.memory_bytes -= 1;
            assert!(matches!(admit(&args, limits, &mapping, host, job),
                Err(Failure { code: "score_document_job_population_memory", .. })));
        }
    }
}

#[test]
fn metadata_marks_whole_document_completion_without_private_chunk_results() {
    let value = serde_json::json!({"output":{"committed":2,"attempts":3}});
    let report = serde_json::to_value(DocumentReport { response: &value, complete_document: true }).unwrap();
    assert_eq!(report["complete_document"], true);
    assert_eq!(report["output"]["committed"], 2);
    for field in ["labels", "text", "chunks", "scores", "task_args"] {
        assert!(report.get(field).is_none());
    }
}
