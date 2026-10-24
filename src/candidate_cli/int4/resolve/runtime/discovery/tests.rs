//! Completion-policy fixtures only; model execution remains an explicit gate.
use super::*;
use crate::{corpus::{entities::EntityVerificationWork,
        entities_int4::{Int4EntityChunkReceipt, Int4EntityDocumentReceipt}},
    native_engine::{portable_int8::ProjectionWork, strict_int4::StrictInt4Error},
    validation::grounded_fields::VerifiedSourceSpan};

fn fixture(chunked: bool) -> (Int4EntityRun, DiscoveryExpected) {
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    let input = vec![ResolutionDocument { id: "a".to_owned(), text: "é".to_owned(), mentions: Vec::new() }];
    let graph = ResolutionPlan::prepare(&input, ResolveOptions::default(), ResolveLimits::default(), &mut Continue).unwrap();
    let resolution = Int4ResolutionRun { schema_version: 1, execution: INT4_RESOLUTION_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, scoring_execution: INT4_SCORING_EXECUTION, group_width: 128,
        model_evaluated: false, head_count: 0, planned_model_work: Int4Work::default(),
        model_work: Int4Work::default(), rewound_positions: 0, result: graph.finalize(Vec::new(), &mut Continue).unwrap() };
    // Synthetic bounded counters exercise the consumer's accounting checks,
    // not the private native executor and not model-inference correctness.
    let work = Int4Work { forward_positions: 1, projected_logits: 1, attention_pairs: 1,
        projections: ProjectionWork { dot_products: 1, multiply_accumulates: 1 } };
    let reserved = work.checked_add(work).unwrap();
    let profile = chunked.then_some(CHUNK_PROFILE);
    let output = Int4EntityRun { schema_version: 1, execution: INT4_ENTITY_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: 128, extraction_task: NER_TASK_VERSION,
        chunk_profile: profile, chunk_count: 1,
        documents: vec![Int4EntityDocumentReceipt { document_id: "a".to_owned(), proposed_entities: 0,
            anchored_mentions: 0, model_work: work, mask_node_visits: 3,
            chunks: vec![Int4EntityChunkReceipt { chunk_id: 0, source_span: VerifiedSourceSpan {
                byte_start: 0, byte_end: 2, scalar_start: 0, scalar_end: 1 },
                proposed_entities: 0, anchored_mentions: 0, model_work: work, mask_node_visits: 3 }] }],
        ner_reserved_work: reserved, ner_model_work: work, reserved_model_work: reserved, model_work: work,
        reserved_mask_node_visits: 10, mask_node_visit_charge: 3,
        verification_used: EntityVerificationWork::default(), resolution, warnings: ["fixture_only", "not_native_execution"] };
    let expected = DiscoveryExpected { documents: 1, chunks: 1, chunk_profile: profile, group_width: 128,
        ner_work: reserved, whole_work: reserved, masks: 10,
        verification: GroundingBudget { max_fields: 8, max_matches: 8, max_scan_steps: 1024 } };
    (output, expected)
}

#[test]
fn output_checker_requires_complete_chunk_document_pair_and_global_accounting() {
    for chunked in [false, true] {
        let (valid, expected) = fixture(chunked); check_result(&valid, expected).unwrap();
        for mutation in 0..20 {
            let (mut result, expected) = fixture(chunked);
            match mutation {
                0 => result.chunk_count = 0,
                1 => result.documents.clear(),
                2 => result.documents[0].chunks.clear(),
                3 => result.documents[0].chunks[0].chunk_id = 1,
                4 => result.documents[0].chunks[0].mask_node_visits += 1,
                5 => result.documents[0].chunks[0].model_work.attention_pairs += 1,
                6 => result.documents[0].proposed_entities += 1,
                7 => result.documents[0].anchored_mentions += 1,
                8 => result.ner_reserved_work.forward_positions += 1,
                9 => result.reserved_model_work.projections.dot_products += 1,
                10 => result.model_work.projections.multiply_accumulates += 1,
                11 => result.mask_node_visit_charge = 11,
                12 => result.verification_used.fields = 9,
                13 => result.verification_used.matches = 9,
                14 => result.verification_used.scan_steps = 1025,
                15 => result.group_width = 32,
                16 => result.extraction_task = "not-ner",
                17 => result.resolution.head_count = 2,
                18 => result.resolution.result.document_count = 0,
                _ => result.chunk_profile = if chunked { None } else { Some(CHUNK_PROFILE) },
            }
            assert_eq!(check_result(&result, expected), Err(CandidateError::Execution));
        }
    }
}

#[test]
fn early_eos_capacity_cannot_be_reassigned_or_erased_from_the_global_reservation() {
    let (mut result, expected) = fixture(true);
    // An early stop used one unit but the admitted NER reservation was two.
    // Claiming the actual debit as its reservation is not an allowed refund.
    result.reserved_model_work = result.ner_model_work;
    assert_eq!(check_result(&result, expected), Err(CandidateError::Execution));
    let (mut result, expected) = fixture(false);
    result.ner_model_work = Int4Work::default(); result.model_work = Int4Work::default();
    result.documents[0].model_work = Int4Work::default();
    result.documents[0].chunks[0].model_work = Int4Work::default();
    assert_eq!(check_result(&result, expected), Err(CandidateError::Execution));
    for axis in 0..5 {
        let (result, mut expected) = fixture(true);
        match axis { 0 => expected.whole_work.forward_positions = 1,
            1 => expected.whole_work.projected_logits = 1, 2 => expected.whole_work.attention_pairs = 1,
            3 => expected.whole_work.projections.dot_products = 1,
            _ => expected.whole_work.projections.multiply_accumulates = 1 }
        assert_eq!(check_result(&result, expected), Err(CandidateError::Execution));
    }
}

#[test]
fn native_discovery_cancellation_and_physical_identity_errors_survive_the_cli_boundary() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let expected = if matches!(cause, DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) {
            CandidateError::Timeout
        } else { CandidateError::Execution };
        assert_eq!(planning_error(Int4EntityError::Graph(ResolveError::Cancelled(cause))), expected);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::Entities(
            Int4EntityError::Native(StrictInt4Error::Cancelled(cause))))), expected);
    }
    assert_eq!(planning_error(Int4EntityError::Identity), CandidateError::Identity);
    assert_eq!(planning_error(Int4EntityError::InvalidInput), CandidateError::Input);
    assert_eq!(planning_error(Int4EntityError::WorkBudget), CandidateError::Planning);
    for kind in [DecodeCancellationKind::FailFast, DecodeCancellationKind::LinkedExit] {
        let error = HostedError::Stopped { stop: crate::hosted::RunStop { kind,
            runtime_reason: None, runtime_checkpoint_error: None }, task_error: Some(Box::new(
                HostedError::Int4(Int4HostError::Entities(Int4EntityError::Identity)))) };
        assert_eq!(execution_error(error), CandidateError::Identity);
    }
}

#[test]
fn discovery_publication_keeps_original_offsets_without_document_or_witness_provenance() {
    let (result, expected) = fixture(true); check_result(&result, expected).unwrap();
    let provenance = Provenance { protocol: "fnlp-candidate-int4-entities-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "resolve",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "model-source-root",
        model_content_sha256: "model-content", candidate_format: "candidate-format", quant_recipe: "candidate:group=128",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &result };
    let mut output = Vec::new(); publish(&response, 16384, &mut output).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-entities-v1");
    assert_eq!(value["output"]["documents"][0]["chunks"][0]["source_span"]["byte_end"], 2);
    assert_eq!(value["output"]["documents"][0]["chunks"][0]["source_span"]["scalar_end"], 1);
    for field in ["text", "prompt", "witness", "private_source_hash", "binding", "source_dir", "snapshot_path"] {
        assert!(value.get(field).is_none()); assert!(value["output"]["documents"][0].get(field).is_none());
    }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

fn command(flags: &[&str]) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", "resolve", "--discover-entities", "--memory-mib", "16384",
        "--max-ner-tokens", "512", "--max-ner-mask-node-visits", "2000000000"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("INT4 discovery required"); }; command
}

#[test]
#[ignore = "requires controller-admitted pinned sources, new snapshot and explicit available SIMD kernel"]
fn raw_and_chunked_discovery_match_between_source_scalar_and_pinned_snapshot_simd() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let snapshot = std::env::var("FNLP_INT4_TEST_ENTITIES_SNAPSHOT").expect("explicit new snapshot required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fresh caller-owned path required; fixture deletes nothing");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required")
        .parse().expect("canonical kernel ID required");
    assert_ne!(kernel, Int8KernelKind::Scalar); PackedInt4Kernel::detect(kernel).unwrap();
    let input = serde_json::json!({"documents":[{"id":"a","text":"Élan met Ada. Ada waved."},
        {"id":"b","text":"Élan works here. Ada works there."}],
        "options":{"blocking":"exact_surface","context_scalars":64,"minimum_margin_milli":1000},
        "ner":{"types":["person"],"max_entities":8,"max_mention_scalars":32}}).to_string();
    let mut pin = String::new();
    for chunked in [false, true] {
        let flags = if chunked { vec!["--snapshot", snapshot.as_str(), "--expected-model-sha256", pin.as_str(),
                "--chunked", "--max-ner-chunk-bytes", "16"] }
            else { vec!["--source-dir", source.as_str(), "--source-manifest", manifest.as_str(),
                "--save-snapshot", snapshot.as_str()] };
        let mut scalar = command(&flags); scalar.args.memory_mib = memory;
        let mut output = Vec::new(); scalar.execute(&mut input.as_bytes(), &mut output).unwrap();
        let baseline: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(baseline["protocol"], "fnlp-candidate-int4-entities-v1");
        assert_eq!(baseline["output"]["documents"].as_array().unwrap().len(), 2);
        if chunked { assert!(baseline["output"]["chunk_count"].as_u64().unwrap() > 2); }
        else { assert_eq!(baseline["output"]["chunk_count"], 2); }
        drop(flags);
        let found = baseline["model_content_sha256"].as_str().unwrap().to_owned();
        if pin.is_empty() { pin = found; } else { assert_eq!(pin, found); }
        let mut reload_flags = vec!["--snapshot", snapshot.as_str(), "--expected-model-sha256", pin.as_str(), "--kernel", kernel.id()];
        if chunked { reload_flags.extend(["--chunked", "--max-ner-chunk-bytes", "16"]); }
        let mut selected = command(&reload_flags); selected.args.memory_mib = memory;
        let mut output = Vec::new(); selected.execute(&mut input.as_bytes(), &mut output).unwrap();
        let selected: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(selected["snapshot"]["operation"], "loaded");
        assert_eq!(selected["output"], baseline["output"]);
    }
    // The explicit snapshot remains available on success and failure.
}
