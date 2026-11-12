//! Explicit model gate for one team spanning both complete redaction maps.
//! This definition is not native parity, NER quality or runtime evidence.
use super::*;
use crate::{hosted::Int4ComputeReceipt, tasks::redact::pseudonym::PseudonymKey};

#[test]
#[ignore = "requires controller-admitted independently pinned INT4 snapshot, kernel and memory; compares complete serial/2/4-thread redaction"]
fn model_present_document_team_preserves_edits_verification_and_all_owned_memory() {
    const CHILD: &str = "FNLP_INT4_REDACTION_DOCUMENT_TEAM_CHILD";
    const COMPLETE: &str = "FNLP_INT4_REDACTION_DOCUMENT_TEAM_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_document_team_preserves_edits_verification_and_all_owned_memory");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated document-redaction team gate must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated document-redaction team gate failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact filter ran no completed model gate");
        return;
    }
    assert!(crate::installed_process_resources().is_none(), "gate needs a fresh process");
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit supported kernel required");
    let group = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH").unwrap_or_else(|_| "128".to_owned());
    let make = |action, verify| {
        let mut flags = vec!["private-document", "--compute-threads", "4",
            "--context-tokens", "2048", "--max-new-tokens", "512",
            "--max-ner-chunk-bytes", "32", "--max-ner-chunks", "64",
            "--group-width", &group, "--kernel", &kernel, "--timeout-seconds", "86400",
            "--rules", "email", "--action", action, "--include-map"];
        if action == "pseudonymize" {
            flags.extend(["--key-stdin", "--key-id", "fixture", "--namespace", "fixture"]);
        }
        if !verify { flags.push("--no-verify"); }
        let mut selected = command(&flags);
        selected.args.source_dir = None; selected.args.source_manifest = None;
        selected.args.snapshot = Some(PathBuf::from(&snapshot));
        selected.args.expected_model_sha256 = Some(pin.clone());
        selected.args.memory_mib = memory;
        selected
    };
    let selected = make("placeholder", true);
    let common = options(&selected).validate(&selected.args).unwrap().common;
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .unwrap().commit().unwrap();
    let model = crate::candidate_cli::int4::runtime::load_model(&engine, &selected.args, common,
        Duration::from_secs(3600), CancellationToken::default()).unwrap();
    let (planner, vocabulary) = source::planner().unwrap();
    let planner = Arc::new(planner); let vocabulary = Arc::new(vocabulary);
    let resident = engine.resources().memory_snapshot();
    let source = "é Contact a@example.org near 上海. Another a@example.org appears.\r\n";
    let key = Arc::new(PseudonymKey::from_bytes(&[0x80; 32], "fixture").unwrap());

    for (action, verify) in [("mask", true), ("placeholder", true), ("pseudonymize", true), ("placeholder", false)] {
        let selected = make(action, verify);
        let settings = options(&selected);
        let limits = settings.validate(&selected.args).unwrap();
        let request = settings.request(&selected.args, limits);
        let detector = settings.detector(&selected.args, limits, NerOptions::default());
        let identity = source::identity(model.model_identity(), &planner, BuiltInTask::Ner).unwrap();
        let config = || RedactConfig {
            ner_identity: identity.clone(), request: request.clone(), detector: detector.clone(),
            native: NativeLimits { context_tokens: selected.args.context_tokens,
                allocator_reserve_bytes: ALLOCATOR_BYTES, run: RunLimits {
                    max_elapsed: Duration::from_secs(86_400), max_checkpoints: selected.args.max_checkpoints,
                    cleanup_reserve_bytes: CLEANUP_BYTES } },
            preparation_reserve_bytes: limits.common.preparation_bytes, edit_reserve_bytes: limits.edit_bytes,
        };
        let pseudonyms = || (action == "pseudonymize").then(|| RedactionPseudonyms {
            key: Arc::clone(&key), namespace: "fixture".to_owned() });
        let redactor = Int4DocumentRedactor::new(&planner, identity.clone(), detector.clone(), model.model_identity()).unwrap();
        let expected = redactor.preflight(source, &mut DeadlineControl {
            started: Instant::now(), deadline: Duration::from_secs(3600) }).unwrap();
        assert!(expected.chunks > 1);
        drop(redactor);
        let mut serial = None;
        for threads in [1, 2, 4] {
            let result = if threads == 1 {
                compute::Completion::serial(engine.redact_int4_document(&model, source.to_owned(),
                    Arc::clone(&planner), Arc::clone(&vocabulary), config(), pseudonyms(),
                    common.kernel, CancellationToken::default()).unwrap())
            } else {
                compute::Completion::Team(engine.redact_int4_document_with_threads(&model, source.to_owned(),
                    Arc::clone(&planner), Arc::clone(&vocabulary), config(), pseudonyms(),
                    common.kernel, threads, CancellationToken::default()).unwrap())
            };
            check_completed(expected, result.result(), &request, &detector).unwrap();
            let output = result.result();
            assert!(!output.result.text().contains("a@example.org"));
            assert_eq!(output.verification.is_some(), verify);
            if let Some(second) = &output.verification {
                assert!(second.preflight.chunks > 1, "team gate must execute both complete maps");
                assert_eq!(second.preflight.source_bytes, output.result.text().len());
                assert_eq!(second.preflight.source_scalars, output.result.text().chars().count());
                assert!(output.model_work.forward_positions > output.original.model_work.forward_positions);
                if action != "mask" {
                    assert_ne!(second.preflight.source_bytes, expected.source_bytes,
                        "fresh verification must repartition actual expanded output");
                }
            }
            let held = engine.resources().memory_snapshot();
            assert!(held.committed_bytes > resident.committed_bytes, "complete output retains its owner");
            let bytes = crate::canonjson::canonical_bytes(output).unwrap();
            if let Some(original) = &serial { assert_eq!(&bytes, original, "action {action}, threads {threads}"); }
            else { serial = Some(bytes); }
            if threads == 1 { assert!(result.compute().is_none()); }
            else { inspect_receipt(result.compute().unwrap(), threads, output); }
            assert_eq!(engine.resources().memory_snapshot(), held);
            drop(result);
            assert_eq!(engine.resources().memory_snapshot(), resident);
            assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
        }
        if verify {
            // The first complete map exhausts this reservation. The edited
            // source must fail before inference, never obtain a second budget.
            let mut short = config();
            short.detector.mapping.max_model_work = expected.planned_model_work;
            let error = engine.redact_int4_document_with_threads(&model, source.to_owned(),
                Arc::clone(&planner), Arc::clone(&vocabulary), short, pseudonyms(),
                common.kernel, 4, CancellationToken::default()).err().expect("verification must not renew work");
            assert!(matches!(error, HostedError::Int4(Int4HostError::SourceMap(Int4SourceMapError::WorkLimit))));
            assert_eq!(engine.resources().memory_snapshot(), resident);
            assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
        }
        for cause in [DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::Deadline,
            DecodeCancellationKind::PollQuota] {
            let cancellation = CancellationToken::default();
            let mut stopped = config();
            if cause == DecodeCancellationKind::PollQuota { stopped.native.run.max_checkpoints = 2; }
            else { cancellation.cancel(cause); }
            let error = engine.redact_int4_document_with_threads(&model, source.to_owned(),
                Arc::clone(&planner), Arc::clone(&vocabulary), stopped, pseudonyms(),
                common.kernel, 4, cancellation).err().expect("stopped edit must not return output");
            assert_eq!(stopped_kind(error), cause);
            assert_eq!(engine.resources().memory_snapshot(), resident);
            assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
        }
    }
    assert_eq!(Arc::strong_count(&key), 1);
    drop(key); drop(vocabulary); drop(planner); drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
    println!("{COMPLETE}");
}

fn inspect_receipt(receipt: &Int4ComputeReceipt, threads: usize, output: &Int4LongRedactionRun) {
    assert_eq!(receipt.schedule, "int4-output-row-team-v1");
    assert_eq!(receipt.compute_threads, threads);
    assert!(receipt.completed_projections > output.model_work.forward_positions);
    assert!(receipt.completed_worker_rows > 0);
    assert!(receipt.completed_rows > receipt.completed_worker_rows);
}

fn stopped_kind(error: HostedError) -> DecodeCancellationKind {
    match error {
        HostedError::Stopped { stop, .. } => stop.kind,
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => stopped_kind(*error),
        _ => panic!("original hosted cancellation cause required"),
    }
}
