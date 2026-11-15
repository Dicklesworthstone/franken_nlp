use super::*;
use crate::native_engine::{constrained::JsonDecodeError,
    constrained_int4::{Int4JsonError, planned_work}, decode::DecodeCancellationKind};

const PREPARATION: u64 = 600 * 1024 * 1024;

fn fixture(samples: usize) -> (SampledSentimentHostLimits, TaskBudget, Int4Work) {
    let native = NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536 } };
    let one = planned_work(128, 32, 16).unwrap();
    let work = (0..samples).try_fold(Int4Work::default(), |work, _| work.checked_add(one)).unwrap();
    let task = TaskBudget { max_input_tokens: 32, max_output_tokens: 16 * samples as u32,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(128, 128).unwrap().kv_bytes };
    let limits = SampledSentimentHostLimits { native, preparation_reserve_bytes: PREPARATION,
        max_model_work: work, max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000,
            checkpoint_interval_nodes: 256 }, max_mask_node_visits: 32_000_000_000 };
    (limits, task, work)
}

#[test]
fn samples_share_one_native_capacity_and_sampler_reservation() {
    let mut previous = None;
    for samples in [1, 4, 32] {
        let (limits, task, work) = fixture(samples);
        let (native, scratch) = requirements(limits, task, 47, 128, work, PREPARATION).unwrap();
        assert_eq!(scratch, native.rope_bytes + native.scratch_payload_bound
            + limits.native.allocator_reserve_bytes + json_sampling_workspace_bytes().unwrap());
        assert_eq!(native.kv_bytes, task.max_kv_bytes);
        if let Some(expected) = previous { assert_eq!((native, scratch), expected); }
        previous = Some((native, scratch));
        assert_eq!(work, limits.max_model_work);
    }
}

#[test]
fn every_complete_model_work_axis_is_admitted_before_native_allocation() {
    let (baseline, task, work) = fixture(4);
    for axis in 0..5 {
        let mut limits = baseline;
        match axis {
            0 => limits.max_model_work.forward_positions -= 1,
            1 => limits.max_model_work.projected_logits -= 1,
            2 => limits.max_model_work.attention_pairs -= 1,
            3 => limits.max_model_work.projections.dot_products -= 1,
            _ => limits.max_model_work.projections.multiply_accumulates -= 1,
        }
        assert!(requirements(limits, task, 47, 128, work, PREPARATION).is_err());
    }
}

#[test]
fn preparation_sampler_kv_and_mask_refusals_precede_native_allocation() {
    let (baseline, task, work) = fixture(4);
    for axis in 0..9 {
        let mut limits = baseline;
        let mut task = task;
        match axis {
            0 => limits.preparation_reserve_bytes -= 1,
            1 => limits.max_sampler_bytes -= 1,
            2 => task.max_kv_bytes -= 1,
            3 => limits.native.context_tokens = 46,
            4 => limits.native.allocator_reserve_bytes = 0,
            5 => limits.mask_limits.max_trie_node_visits = 0,
            6 => limits.mask_limits.checkpoint_interval_nodes = 0,
            7 => limits.max_mask_node_visits = 0,
            _ => limits.native.run.max_elapsed = Duration::ZERO,
        }
        assert!(requirements(limits, task, 47, 128, work, PREPARATION).is_err());
    }
    assert!(requirements(baseline, task, 0, 128, work, PREPARATION).is_err());
    assert!(requirements(baseline, task, 47, 128, work, 0).is_err());
    assert!(requirements(SampledSentimentHostLimits { native: NativeLimits {
        allocator_reserve_bytes: u64::MAX, ..baseline.native }, ..baseline },
        task, 47, 128, work, PREPARATION).is_err());
}

#[test]
fn task_errors_retain_cancellation_without_disclosing_nested_text() {
    let error = task_error(SampledSentimentError::Native(Int4JsonError::Decode(
        JsonDecodeError::InvalidRequest("PRIVATE_DOCUMENT_OR_SEED"))));
    assert!(!format!("{error:?} {error}").contains("PRIVATE_DOCUMENT_OR_SEED"));
    assert!(error.source().is_some());
    let error = task_error(SampledSentimentError::Cancelled(DecodeCancellationKind::ParentCancelled));
    let HostedError::Int4(Int4HostError::SampledSentiment(error)) = error else {
        panic!("sampled task error was flattened");
    };
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::ParentCancelled));
}

#[test]
fn sampled_plans_and_guarded_results_cross_the_one_owned_team_boundary() {
    fn send<T: Send + 'static>() {}
    send::<PreparedSampledSentiment>();
    send::<Arc<SampledSentimentVocabulary>>();
    send::<HostedInt4TeamResult<SampledSentimentRun>>();
    let _serial = NlpEngine::execute_int4_sampled_sentiment;
    let _team = NlpEngine::execute_int4_sampled_sentiment_with_threads;
    let _prefill = NlpEngine::execute_int4_sampled_sentiment_layer_major;
}

#[test]
fn sampled_prefill_admits_declared_rows_once_even_for_a_short_prompt() {
    for samples in [1, 4, 32] {
        let (limits, task, work) = fixture(samples);
        let (required, base) = requirements(limits, task, 47, 128, work, PREPARATION).unwrap();
        for rows in [1, 4, 64] {
            let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let prefill = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
            assert_eq!(super::super::prefill::with_extra_workspace(base, Some(prefill)).unwrap(), base + extra);
            assert_eq!(required.kv_bytes, task.max_kv_bytes);
            assert!(super::super::prefill::with_extra_workspace(base, Some(Int4PrefillLimits {
                max_extra_scratch_bytes: extra - 1, ..prefill })).is_err());
        }
    }
}

#[test]
fn persistent_sample_team_prices_extra_workers_without_multiplying_kv_or_sampler() {
    let (limits, task, work) = fixture(32);
    let (required, scratch) = requirements(limits, task, 47, 128, work, PREPARATION).unwrap();
    for threads in [2, 4, 64] {
        let workers = int4_task_team_workspace_bytes(threads, 128).unwrap();
        assert!(workers > 0);
        assert_eq!(sum(&[scratch, workers]).unwrap(), scratch + workers);
        assert_eq!(required.kv_bytes, task.max_kv_bytes);
        assert_eq!(scratch, required.rope_bytes + required.scratch_payload_bound
            + limits.native.allocator_reserve_bytes + json_sampling_workspace_bytes().unwrap());
        assert!(sum(&[u64::MAX, workers]).is_err());
    }
    for threads in [0, 1, 65, usize::MAX] {
        assert!(int4_task_team_workspace_bytes(threads, 128).is_err());
    }
}

/// A full-model comparison, deliberately armed by the controller in a fresh
/// process. The tiny ownership fixtures elsewhere cannot establish seeded
/// task parity. This gate requires every sample and explanation to complete.
#[test]
#[ignore = "controller-only pinned model: sampled sentiment serial/team/prefill parity, bounded input and physical drain"]
fn native_sampled_bundles_keep_every_sample_across_physical_team_widths() {
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        tasks::sentiment::{SentimentAxis, SentimentRequest,
            sampled::{SampledSentimentLimits, SampledSentimentOptions, SampledSentimentPlanner}},
        tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};

    assert!(crate::installed_process_resources().is_none(), "requires a fresh process");
    let source = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source"));
    let manifest = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest"));
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_BYTES").expect("explicit memory cap").parse().unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let idle = host.resources().memory_snapshot();
    let run = RunLimits { max_elapsed: Duration::from_secs(86_400),
        max_checkpoints: 1_000_000_000, cleanup_reserve_bytes: 65_536 };
    let load = Int4ModelLoadRequirement::for_group_width(32).unwrap();
    let model = host.load_source_candidate_int4(source, manifest, 32, Int4LoadLimits {
        model: Int4ModelLoadBudget { max_payload_bytes: load.peak_payload_bytes().unwrap(),
            max_source_values: load.source_values, max_groups: load.groups,
            max_shard_revalidation_bytes: load.shard_revalidation_bytes },
        source_preparation_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run,
    }, CancellationToken::default()).unwrap();
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 1 << 30)
        .unwrap().commit().unwrap();
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
    let native = NativeLimits { context_tokens: 2048, allocator_reserve_bytes: 64 << 20, run };
    let budget = TaskBudget { max_input_tokens: 2048, max_output_tokens: 1024,
        max_output_bytes: 1 << 20, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(2048, 32).unwrap().kv_bytes };
    let request = SentimentRequest { document: "The service was helpful. I feel calm. 東京 🦀".to_owned(),
        axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget };
    let baseline = host.resources().memory_snapshot();
    let drained = || {
        assert_eq!(host.resources().memory_snapshot(), baseline);
        assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
    };

    // Compare schedules only within one exact encoding/model/sampler recipe.
    // Compact input intentionally has its own prompt identity and token work.
    for encoding in [UntrustedEncoding::ByteFallback,
        UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096,
            max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 16 << 20 })] {
        let options = SampledSentimentOptions { effective_seed: [37; 32], samples: 2,
            temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
            max_sample_tokens: 256, max_justification_chars: 8 };
        let planner = SampledSentimentPlanner::for_model(model.model_identity(), options,
            budget, SampledSentimentLimits::default()).unwrap().with_input_encoding(encoding).unwrap();
        let make = || planner.plan(&request).unwrap();
        let prepared = make();
        let limits = SampledSentimentHostLimits { native,
            preparation_reserve_bytes: prepared.preparation_requirement_bytes(),
            max_model_work: prepared.planned_work(),
            max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
            mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000,
                checkpoint_interval_nodes: 256 }, max_mask_node_visits: 32_000_000_000 };
        let scalar = host.execute_int4_sampled_sentiment(&model, prepared, Arc::clone(&vocabulary),
            limits, PackedInt4Kernel::scalar(), CancellationToken::default()).unwrap();
        assert_eq!(scalar.result().dimensions.len(), 2);
        assert!(scalar.result().dimensions.iter().all(|dimension| dimension.samples.len() == 2));
        let expected = crate::canonjson::canonical_bytes(scalar.result()).unwrap();
        assert!(host.resources().memory_snapshot().committed_bytes > baseline.committed_bytes);
        drop(scalar);
        drained();

        for threads in [2, 4] {
            let team = host.execute_int4_sampled_sentiment_with_threads(&model, make(),
                Arc::clone(&vocabulary), limits, PackedInt4Kernel::scalar(), threads,
                CancellationToken::default()).unwrap();
            assert_eq!(crate::canonjson::canonical_bytes(team.result()).unwrap(), expected);
            assert_eq!(team.compute().compute_threads, threads);
            assert!(team.compute().completed_projections > 0);
            assert!(team.compute().completed_worker_rows > 0);
            assert!(host.resources().memory_snapshot().committed_bytes > baseline.committed_bytes);
            assert_eq!(host.resources().memory_snapshot().reserved_bytes, 0);
            assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
            drop(team);
            drained();
        }
        for rows in [1, 4] {
            let prefill = Int4PrefillLimits { max_batch_rows: rows,
                max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
            let scheduled = host.execute_int4_sampled_sentiment_layer_major(&model, make(),
                Arc::clone(&vocabulary), limits, PackedInt4Kernel::scalar(), prefill,
                CancellationToken::default()).unwrap();
            assert_eq!(crate::canonjson::canonical_bytes(scheduled.result()).unwrap(), expected);
            assert!(host.resources().memory_snapshot().committed_bytes > baseline.committed_bytes);
            assert_eq!(host.resources().memory_snapshot().reserved_bytes, 0);
            assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
            drop(scheduled);
            drained();
        }
        for rows in [0, 65] {
            assert!(host.execute_int4_sampled_sentiment_layer_major(&model, make(),
                Arc::clone(&vocabulary), limits, PackedInt4Kernel::scalar(),
                Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX },
                CancellationToken::default()).is_err());
            drained();
        }
        for threads in [0, 1, 5, 65] {
            assert!(matches!(host.execute_int4_sampled_sentiment_with_threads(&model, make(),
                Arc::clone(&vocabulary), limits, PackedInt4Kernel::scalar(), threads,
                CancellationToken::default()), Err(HostedError::Limits(_))));
            drained();
        }
        let mut impossible = limits;
        impossible.preparation_reserve_bytes = memory;
        assert!(matches!(host.execute_int4_sampled_sentiment_with_threads(&model, make(),
            Arc::clone(&vocabulary), impossible, PackedInt4Kernel::scalar(), 2,
            CancellationToken::default()), Err(HostedError::Reservation(_))));
        drained();

        let cancel = CancellationToken::default();
        cancel.cancel(DecodeCancellationKind::ParentCancelled);
        assert!(matches!(host.execute_int4_sampled_sentiment_with_threads(&model, make(),
            Arc::clone(&vocabulary), limits, PackedInt4Kernel::scalar(), 2, cancel),
            Err(HostedError::Stopped { stop: RunStop {
                kind: DecodeCancellationKind::ParentCancelled, .. }, .. })));
        drained();

        let cancel = CancellationToken::default();
        cancel.cancel(DecodeCancellationKind::ParentCancelled);
        assert!(matches!(host.execute_int4_sampled_sentiment_layer_major(&model, make(),
            Arc::clone(&vocabulary), limits, PackedInt4Kernel::scalar(),
            Int4PrefillLimits { max_batch_rows: 4,
                max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() }, cancel),
            Err(HostedError::Stopped { stop: RunStop {
                kind: DecodeCancellationKind::ParentCancelled, .. }, .. })));
        drained();
    }
    drop(vocabulary);
    drop(preparation);
    drop(lease);
    drop(model);
    assert_eq!(host.resources().memory_snapshot(), idle);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}
