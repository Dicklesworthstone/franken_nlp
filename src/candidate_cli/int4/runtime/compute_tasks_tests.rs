//! Controller-only real-model comparison of the common structured/scored routes.
//! Definitions are not execution, quality, parity or performance evidence.
use super::*;
use crate::native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work};

#[derive(Clone, Copy, Debug)]
enum Case { MultiLabel, Structural, SelectedSource, Keyphrases }
const CASES: [Case; 4] = [Case::MultiLabel, Case::Structural, Case::SelectedSource, Case::Keyphrases];
const SOURCE: &str = "é Alice 上海 Alice";

fn command(case: Case, threads: usize, snapshot: &str, pin: &str, memory: u64) -> Int4Command {
    let memory = memory.to_string(); let threads = threads.to_string();
    let operation = match case {
        Case::MultiLabel => "classify", Case::Structural | Case::SelectedSource => "extract",
        Case::Keyphrases => "keyphrases",
    };
    let mut argv = vec!["candidate", "source-int4", operation,
        "--snapshot", snapshot, "--expected-model-sha256", pin, "--memory-mib", &memory,
        "--context-tokens", "1024", "--max-input-bytes", "4096", "--preparation-mib", "512",
        "--max-result-bytes", "65536", "--compute-threads", &threads];
    match case {
        Case::MultiLabel => argv.extend(["--max-candidate-tokens", "4", "--input-encoding", "safe-bpe-v1"]),
        Case::Structural => argv.extend(["--schema", "controller-inline-schema.json", "--max-new-tokens", "16"]),
        Case::SelectedSource => argv.extend(["--schema", "controller-inline-schema.json",
            "--max-new-tokens", "16", "--source-membership", "--input-encoding", "safe-bpe-v1",
            // Every legal row fits. This exercises exact selected-row execution
            // without assuming a small, unmeasured vocabulary-mask cardinality.
            "--max-selected-rows", "166144"]),
        Case::Keyphrases => argv.extend(["--max-new-tokens", "16", "--input-encoding", "safe-bpe-v1"]),
    }
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("expected the explicit snapshot task dispatcher"); };
    command
}

fn admission(command: &Int4Command) -> Int4Limits {
    match &command.task {
        Int4Task::Classify(scoring) => scoring.validate_one_off_with_prefill(&command.args),
        Int4Task::Extract(options) => options.validate_one_off_with_prefill(&command.args),
        Int4Task::Source(_, options) => options.validate_with_prefill(&command.args),
        _ => panic!("unexpected controller fixture route"),
    }.unwrap()
}

fn request(command: Int4Command, limits: Int4Limits, case: Case) -> (Int4Args, Input) {
    let Int4Command { task, args, batch, job } = command;
    assert!(batch.is_none() && job.is_none());
    let input = match task {
        Int4Task::Classify(scoring) => {
            let json = r#"{"document":"Solar energy powers local homes.","mode":"multi_label","labels":[{"id":"energy","description":"Energy sources and supply."},{"id":"homes","description":"Residential buildings."}]}"#;
            let request = classify::parse_request(json, scoring.budget(limits), args.max_input_bytes).unwrap();
            scoring.admit_heads(request.labels.len()).unwrap();
            Input::Classify(request, scoring)
        }
        Int4Task::Extract(options) => {
            let (document, schema) = match case {
                Case::Structural => ("A finite structural result.", r#"{"type":"boolean","const":true}"#),
                Case::SelectedSource => (SOURCE, r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#),
                _ => panic!("extraction fixture kind differs"),
            };
            // Exercise the actual raw-schema validator and the same Input
            // representation as file IO, without making a fake model/schema file.
            let request = options.check_schema(schema.to_owned(), document, options.budget(&args, limits)).unwrap();
            Input::Extract { document: document.to_owned(), request, options }
        }
        Int4Task::Source(kind, options) => {
            // Positive finite options are required. A zero phrase limit is an
            // invalid request, and would test refusal instead of native teams.
            let request = crate::candidate_cli::source::request(kind, "Ada Ada".to_owned(),
                Some(r#"{"max_phrases":1,"max_phrase_scalars":3}"#), options.budget(&args, limits),
                args.max_input_bytes).unwrap();
            Input::Source { request, options }
        }
        _ => panic!("unexpected controller fixture input"),
    };
    (args, input)
}

fn planned_work(prepared: &Prepared) -> Int4Work {
    match prepared {
        Prepared::Classify(plan) => plan.planned_work(),
        Prepared::Extract { plan, .. } => plan.planned_work(),
        Prepared::Source { plan, .. } => plan.planned_work(),
        Prepared::Text(_) => panic!("text execution has its separate controller fixture"),
    }
}

fn completed_work(completed: &Completed, case: Case) -> Int4Work {
    match (completed, case) {
        (Completed::Classify(output), Case::MultiLabel) => {
            assert_eq!(output.result().head_count, 2);
            output.result().model_work
        }
        (Completed::Extract(output), Case::Structural) => {
            assert_eq!(output.result().result.output.json, "true");
            output.result().model_work
        }
        (Completed::Extract(output), Case::SelectedSource) => {
            let run = output.result();
            assert_eq!(run.execution, crate::tasks::extract::grouped_int4::INT4_SPARSE_EXTRACT_VERSION);
            assert_eq!(run.result.output.json, "\"Alice\"");
            assert_eq!(run.result.source_fields.len(), 1);
            let expected: Vec<_> = SOURCE.match_indices("Alice").map(|(start, _)| start).collect();
            let spans = &run.result.source_fields[0].spans;
            assert_eq!(spans.len(), expected.len());
            for (span, start) in spans.iter().zip(expected) {
                assert_eq!((span.byte_start, span.byte_end), (start, start + "Alice".len()));
                assert_eq!(span.scalar_start, SOURCE[..start].chars().count());
                assert_eq!(span.scalar_end, SOURCE[..span.byte_end].chars().count());
            }
            // The physical all-legal head really projected fewer rows; merely
            // returning a selected-row version label is not sufficient.
            assert!(run.model_work.projected_logits < run.result.output.token_ids.len() as u64 * NANBEIGE_VOCAB_SIZE as u64);
            run.model_work
        }
        (Completed::Source(output), Case::Keyphrases) => {
            let run = output.result();
            let crate::tasks::source_planning::SourceTaskResult::Keyphrases(result) = &run.result
                else { panic!("source finalizer changed task"); };
            assert!(result.phrases.len() <= 1);
            for phrase in &result.phrases {
                assert!(!phrase.text.is_empty() && phrase.text.chars().count() <= 3);
                assert!(!phrase.spans.is_empty());
                for span in &phrase.spans { assert_eq!(&"Ada Ada"[span.byte_start..span.byte_end], phrase.text); }
            }
            run.model_work
        }
        _ => panic!("native completion differs from the requested task"),
    }
}

#[test]
#[ignore = "run alone in a fresh process with controller-admitted INT4 snapshot, independent pin and real structured/scored multi-thread inference"]
fn snapshot_task_teams_match_full_serial_results_identities_work_and_release_every_charge() {
    let snapshot = std::env::var("FNLP_INT4_STREAM_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_STREAM_MODEL_SHA256").expect("independent content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().unwrap();
    // Admit the largest tested preparation before any model/planner allocation.
    // The same charge covers private inputs and serialized comparison values.
    let preparation_bytes = CASES.into_iter().map(|case|
        admission(&command(case, 4, &snapshot, &pin, memory)).preparation_bytes).max().unwrap();
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory.checked_mul(MIB).unwrap(),
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, preparation_bytes).unwrap().commit().unwrap();
    let base = command(Case::MultiLabel, 1, &snapshot, &pin, memory);
    let model = load_model(&engine, &base.args, admission(&base), Duration::from_secs(3600),
        CancellationToken::default()).unwrap();
    let resident = engine.resources().memory_snapshot();
    for case in CASES {
        let mut serial_identity = None;
        let mut serial_planned = None;
        let mut serial_actual = None;
        let mut serial_result = None;
        for width in [1, 2, 4] {
            let command = command(case, width, &snapshot, &pin, memory);
            let limits = admission(&command);
            let (args, input) = request(command, limits, case);
            let prepared = prepare(input, &model, &args, limits, &mut DeadlineControl {
                started: Instant::now(), deadline: Duration::from_secs(3600),
            }).unwrap();
            let identity = prepared.identity().clone();
            let planned = planned_work(&prepared);
            if width == 1 { serial_identity = Some(identity.clone()); serial_planned = Some(planned); }
            else {
                assert_eq!(&identity, serial_identity.as_ref().unwrap(), "identity changed for {case:?} width={width}");
                assert_eq!(planned, serial_planned.unwrap(), "planned work changed for {case:?} width={width}");
            }
            let completed = prepared.execute(&engine, &model, NativeLimits {
                context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: args.max_checkpoints,
                    cleanup_reserve_bytes: CLEANUP_BYTES },
            }, limits, CancellationToken::default()).unwrap();
            let actual = completed_work(&completed, case);
            assert!(actual.forward_positions <= planned.forward_positions);
            assert!(actual.projected_logits <= planned.projected_logits);
            assert!(actual.attention_pairs <= planned.attention_pairs);
            assert!(actual.projections.fits(planned.projections));
            let value = serde_json::to_value(&completed).unwrap();
            if width == 1 {
                assert!(completed.compute().is_none());
                serial_actual = Some(actual); serial_result = Some(value);
            } else {
                assert_eq!(actual, serial_actual.unwrap(), "completed work changed for {case:?} width={width}");
                assert_eq!(&value, serial_result.as_ref().unwrap(), "full result changed for {case:?} width={width}");
                let receipt = completed.compute().unwrap();
                assert_eq!(receipt.schedule, "int4-output-row-team-v1");
                assert_eq!(receipt.compute_threads, width);
                assert!(receipt.completed_projections > 0);
                assert!(receipt.completed_worker_rows > 0);
                assert!(receipt.completed_worker_rows < receipt.completed_rows);
            }
            drop(completed);
            assert_eq!(engine.resources().memory_snapshot(), resident, "retained charge for {case:?} width={width}");
            let closures = engine.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0);
            assert_eq!(closures.wrapper_cancelled_closures, 0);
        }
    }
    drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}
