//! Fixed-team admission and physically drained ownership; no model parity claim.
use super::*;
use std::time::Duration;

fn config(width: usize) -> RedactConfig<Int4LongRedactionConfig> {
    let mut config = super::super::tests::config(width);
    config.detector.mapping.reduction.map_batch_chunks = 1;
    config
}

#[test]
fn one_worker_workspace_is_reused_across_chunks_and_fresh_verification() {
    for width in [32, 127, 128] {
        let mut config = config(width);
        for threads in [2, 3, 4, 64] {
            let expected = int4_task_team_workspace_bytes(threads, width).unwrap();
            assert!(expected > 0);
            for verify in [false, true] {
                config.request.verify = verify;
                for chunks in [1, 3, 256] {
                    config.detector.mapping.chunks.max_chunks = chunks;
                    assert_eq!(team_requirements(&config, width, threads).unwrap(), expected);
                }
            }
            // The final edited document has its own charged result envelope;
            // neither a larger output nor a second detector stage grows the team.
            config.detector.max_result_bytes += 1;
            assert_eq!(team_requirements(&config, width, threads).unwrap(), expected);
        }
    }
}

#[test]
fn invalid_team_widths_cohort_mixtures_and_combined_scratch_overflow_refuse() {
    let mut config = config(32);
    for threads in [0, 1, 65, usize::MAX] {
        assert!(team_requirements(&config, 32, threads).is_err());
    }
    for chunks in [0, 2, 64, usize::MAX] {
        config.detector.mapping.reduction.map_batch_chunks = chunks;
        assert!(matches!(team_requirements(&config, 32, 2), Err(HostedError::Limits(
            "INT4 document redaction teams require serial chunk scheduling"))));
    }
    config.detector.mapping.reduction.map_batch_chunks = 1;
    config.native.allocator_reserve_bytes = u64::MAX;
    assert!(team_requirements(&config, 32, 2).is_err());
    config.native.allocator_reserve_bytes = 65_536;
    for width in [0, crate::native_engine::quant_algebra::MAX_MODEL_K + 1, usize::MAX] {
        assert!(team_requirements(&config, width, 2).is_err());
    }
}

#[test]
fn complete_edits_and_their_physical_receipt_have_one_sendable_owner() {
    fn send<T: Send + 'static>() {}
    send::<RedactInput<Int4LongRedactionConfig>>();
    send::<HostedInt4TeamResult<Int4LongRedactionRun>>();
    let _entry = NlpEngine::redact_int4_document_with_threads;
}

/// Exercise the real installed runtime and team dispatcher with tiny test
/// weights. The impossible source-plan budget refuses before any forward.
#[test]
#[ignore = "run alone with controller-admitted runtime; synthetic cancellation/ownership, not model parity"]
fn installed_envelope_cancellation_and_post_entry_planning_refusal_fully_drain() {
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture}};

    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 2,
        helper_threads: 0, thread_ceiling: 4, memory_ceiling_bytes: 768 << 20,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let (weights, _) = projection_fixture(32);
    let lease = host.resources().acquire_lease();
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(),
        || Ok(weights)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls()).unwrap());
    let baseline = host.resources().memory_snapshot();
    let make = || {
        let planner = Arc::new(SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap());
        let mut config = config(32);
        config.ner_identity.tokenizer_digest = planner.tokenizer_digest();
        config.ner_identity.template_digest = *planner.template_digest();
        config.ner_identity = bind_candidate_model(config.ner_identity, model.model_identity()).unwrap();
        config.native.run.max_elapsed = Duration::from_secs(120);
        config.native.run.max_checkpoints = 1_000_000;
        (planner, config)
    };
    let drained = || {
        assert_eq!(host.resources().memory_snapshot(), baseline);
        assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
        assert_eq!(Arc::strong_count(&vocabulary), 1);
    };
    let source = "Alice visits Paris. Bob visits Tokyo.";

    let (planner, config) = make();
    let error = host.redact_int4_document_with_threads(&model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, None, PackedInt4Kernel::scalar(), 4, CancellationToken::default())
        .err().expect("unadmitted team must fail");
    assert!(matches!(error, HostedError::Limits("INT4 team exceeds installed child envelope")));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    let (planner, config) = make();
    let cancelled = CancellationToken::default();
    cancelled.cancel(DecodeCancellationKind::ParentCancelled);
    let error = host.redact_int4_document_with_threads(&model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, None, PackedInt4Kernel::scalar(), 3, cancelled)
        .err().expect("pre-cancelled document must fail");
    assert!(matches!(error, HostedError::Stopped { stop, .. }
        if stop.kind == DecodeCancellationKind::ParentCancelled));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    let (planner, mut config) = make();
    config.detector.mapping.reduction.map_batch_chunks = 2;
    let error = host.redact_int4_document_with_threads(&model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, None, PackedInt4Kernel::scalar(), 3, CancellationToken::default())
        .err().expect("independent-sequence scheduling cannot share this numerical team");
    assert!(matches!(error, HostedError::Limits(
        "INT4 document redaction teams require serial chunk scheduling")));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    let (planner, mut config) = make();
    // A positive cap passes host geometry admission, but cannot fit the
    // trusted NER scaffold. The owned engine and team must already be live
    // when exact source planning refuses, without using fixture projections.
    config.detector.per_chunk.max_input_tokens = 1;
    document_requirements(&config, source.len(), 32,
        planner.tokenizer_digest(), *planner.template_digest()).unwrap();
    let error = host.redact_int4_document_with_threads(&model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, None, PackedInt4Kernel::scalar(), 3, CancellationToken::default())
        .err().expect("impossible complete source plan must fail before forward");
    assert!(matches!(error, HostedError::Int4(Int4HostError::SourceMap(_))));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();
}
