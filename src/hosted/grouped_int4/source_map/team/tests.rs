//! Admission and ownership regressions. Executable/model gates remain separate.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};

fn fixture(task: SourceMapTask) -> (SourceTaskPlanner, Int4SourceMapConfig) {
    super::super::tests::fixture(task, 32)
}

#[test]
fn one_worker_workspace_is_independent_of_chunks_and_reducer_caps() {
    for task in [SourceMapTask::Ner(Default::default()), SourceMapTask::Keyphrases(Default::default()),
        SourceMapTask::Summarize(Default::default())] {
        let (_, mut config) = fixture(task);
        for threads in [2, 3, 4, 64] {
            let expected = int4_task_team_workspace_bytes(threads, 32).unwrap();
            for chunks in [1, 16, 256] {
                config.mapping.chunks.max_chunks = chunks;
                assert_eq!(team_requirements(&config, 32, threads).unwrap(), expected);
            }
            // Changing an exact reducer's output ceiling changes neither
            // physical worker count nor its single reusable scratch owner.
            config.mapping.reduction.max_result_bytes += 1;
            assert_eq!(team_requirements(&config, 32, threads).unwrap(), expected);
        }
    }
}

#[test]
fn invalid_team_cohort_mixtures_and_combined_scratch_overflow_refuse() {
    let (_, mut config) = fixture(SourceMapTask::Ner(Default::default()));
    for threads in [0, 1, 65, usize::MAX] {
        assert!(team_requirements(&config, 32, threads).is_err());
    }
    for chunks in [0, 2, 64, usize::MAX] {
        config.mapping.reduction.map_batch_chunks = chunks;
        assert!(team_requirements(&config, 32, 2).is_err());
    }
    config.mapping.reduction.map_batch_chunks = 1;
    config.native.allocator_reserve_bytes = u64::MAX;
    assert!(team_requirements(&config, 32, 2).is_err());
    config.native.allocator_reserve_bytes = 65_536;
    for width in [0, crate::native_engine::quant_algebra::MAX_MODEL_K + 1, usize::MAX] {
        assert!(team_requirements(&config, width, 2).is_err());
    }
}

#[test]
fn compact_encoder_memory_and_semantic_identity_do_not_absorb_worker_scheduling() {
    let (planner, mut config) = fixture(SourceMapTask::Ner(Default::default()));
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
    });
    let total = SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 };
    let planner = planner.with_map_document_encoding(policy.clone(), total).unwrap();
    config.identity.template_digest = *planner.template_digest();
    let original_identity = config.identity.clone();
    let encoding_bytes = planner.map_document_encoding_workspace_bytes().unwrap();
    config.preparation_reserve_bytes = encoding_bytes;
    for threads in [2, 4] {
        let team_bytes = team_requirements(&config, 32, threads).unwrap();
        assert!(team_bytes > 0);
        map_requirements(&config, 16, 32, &planner).unwrap();
        assert_eq!(config.identity, original_identity);
        assert_eq!(planner.document_encoding(), &policy);
        assert_eq!(planner.map_document_encoding_limits(), Some(total));
        assert_eq!(planner.map_document_encoding_workspace_bytes().unwrap(), encoding_bytes);
    }
    config.preparation_reserve_bytes = encoding_bytes - 1;
    assert!(map_requirements(&config, 16, 32, &planner).is_err());
}

#[test]
fn every_complete_result_and_its_physical_receipt_keep_one_sendable_owner() {
    fn send<T: Send + 'static>() {}
    send::<DocumentAdmission>();
    send::<HostedInt4TeamResult<Int4SourceMapRun>>();
    send::<HostedInt4TeamResult<Int4CorpusSummaryRun>>();
    send::<HostedInt4TeamResult<Int4CorpusKeyphraseRun>>();
    let _map = NlpEngine::map_int4_source_with_threads;
    let _summary = NlpEngine::summarize_int4_source_with_threads;
    let _phrases = NlpEngine::reduce_int4_keyphrases_document_with_threads;
}

/// This fixture exercises real dispatch and memory ownership with tiny
/// test-only projection weights. It never performs model inference. Its
/// private reducer seam deliberately cancels at the finalization boundary.
#[test]
#[ignore = "run alone with controller-admitted runtime; synthetic cancellation/ownership, not model parity"]
fn installed_envelope_planning_refusal_and_late_reducer_cancel_fully_drain() {
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture}};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct CancelAtReduction { cancellation: CancellationToken, entered: Arc<AtomicBool> }
    impl DocumentOperation for CancelAtReduction {
        type Output = ();
        fn validate(&self, _: &Int4SourceMapConfig) -> Result<(), HostedError> { Ok(()) }
        fn allocation(&self, config: &Int4SourceMapConfig, prepared: &PreparedInt4SourceMap<'_>)
            -> Result<DocumentAllocation, HostedError> { MapOperation.allocation(config, prepared) }
        fn execute(self, prepared: PreparedInt4SourceMap<'_>, _: &[ExecutionIdentity],
            engine: &mut StrictInt4Engine<'_>, _: &ExtractionVocabulary,
            _: &mut RunControl) -> Result<(), HostedError> {
            assert!(prepared.chunk_count() > 1);
            assert!(engine.kv_cache().all_slots_have_len(0));
            self.entered.store(true, Ordering::SeqCst);
            self.cancellation.cancel(DecodeCancellationKind::Deadline);
            Ok(())
        }
    }
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
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls()).unwrap());
    let baseline = host.resources().memory_snapshot();
    let make = || {
        let (planner, mut config) = fixture(SourceMapTask::Ner(crate::tasks::ner::NerOptions {
            types: vec![crate::tasks::ner::EntityType::Person], max_entities: 2, max_mention_scalars: 32,
        }));
        config.identity = bind_candidate_model(config.identity, model.model_identity()).unwrap();
        config.budget.max_output_tokens = 64;
        config.mapping.chunks.reserved_tokens = 64;
        config.mapping.chunks.max_chunk_bytes = 16;
        config.mapping.chunks.max_chunks = 8;
        config.mapping.max_model_work = crate::native_engine::strict_int4::Int4Work::for_sequence(
            32, 0, 16_384, 512 * NANBEIGE_VOCAB_SIZE).unwrap();
        config.preparation_reserve_bytes = 64 << 20;
        config.native.run.max_elapsed = Duration::from_secs(120);
        config.native.run.max_checkpoints = 1_000_000;
        (Arc::new(planner), config)
    };
    let drained = || {
        assert_eq!(host.resources().memory_snapshot(), baseline);
        assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
    };
    let source = "Alice visits Paris. Bob visits Tokyo.";
    let (planner, config) = make();
    let error = host.map_int4_source_with_threads(&model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, PackedInt4Kernel::scalar(), 4, CancellationToken::default())
        .err().expect("unadmitted team must fail");
    assert!(matches!(error, HostedError::Limits("INT4 team exceeds installed child envelope")));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    let (planner, config) = make();
    let cancelled = CancellationToken::default(); cancelled.cancel(DecodeCancellationKind::ParentCancelled);
    let error = host.map_int4_source_with_threads(&model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, PackedInt4Kernel::scalar(), 3, cancelled)
        .err().expect("pre-cancelled document must fail");
    assert!(matches!(error, HostedError::Stopped { stop, .. }
        if stop.kind == DecodeCancellationKind::ParentCancelled));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    let (planner, mut config) = make();
    // Work cannot reach the incomplete projection fixture: chunk planning
    // must refuse the invalid complete-document tokenizer-call allowance.
    config.mapping.chunks.max_tokenizer_calls = 0;
    let error = host.map_int4_source_with_threads(&model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, PackedInt4Kernel::scalar(), 3, CancellationToken::default())
        .err().expect("invalid complete plan must fail");
    assert!(matches!(error, HostedError::Int4(Int4HostError::SourceMap(_))));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    let (planner, config) = make();
    let cancellation = CancellationToken::default();
    let entered = Arc::new(AtomicBool::new(false));
    let operation = CancelAtReduction { cancellation: cancellation.clone(), entered: Arc::clone(&entered) };
    let error = execute_with_threads(&host, &model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, PackedInt4Kernel::scalar(), 3, cancellation, operation)
        .err().expect("late reduction cancellation must suppress complete output");
    assert!(entered.load(Ordering::SeqCst));
    assert!(matches!(error, HostedError::Stopped { stop, .. } if stop.kind == DecodeCancellationKind::Deadline));
    assert_eq!(Arc::strong_count(&planner), 1);
    assert_eq!(Arc::strong_count(&vocabulary), 1);
    drained();
}
