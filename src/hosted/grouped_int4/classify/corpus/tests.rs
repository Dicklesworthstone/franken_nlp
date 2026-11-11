//! Pure declared-memory/admission fixtures. Synthetic identities never grant
//! model execution or task-quality authority.
use super::*;
use crate::{artifact::converter::PINNED_REVISION, batch::BatchLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::{Int4Work, STRICT_INT4_EXECUTION}},
    tasks::{classify::ClassificationAggregationPolicy, mapreduce::ChunkLimits}};

mod cohorts;

fn fixture() -> (TaskBudget, ClassificationLimits, Int4ClassificationMapLimits, CorpusLimits) {
    let task = TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes };
    let planning = ClassificationLimits { max_input_bytes: 4096, max_labels: 4,
        max_total_label_bytes: 1024, ..Default::default() };
    let mapping = Int4ClassificationMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
        max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 2048, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: ClassificationAggregationPolicy::default() };
    let limits = CorpusLimits { native: NativeLimits { context_tokens: 1024, allocator_reserve_bytes: 65536,
        run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 128 * 1024 * 1024, io_reserve_bytes: 65536 };
    (task, planning, mapping, limits)
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"classification-document-host-admission-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 32_usize)).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "classify-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "complete-candidate-eos".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn request(identities: &[ExecutionIdentity]) -> Int4ClassificationDocumentAdmission<'_> {
    let (task, _, mapping, _) = fixture();
    Int4ClassificationDocumentAdmission { identities, model_work: mapping.max_model_work, group_width: 32,
        required_context: 768, task_budget: task, kv_reservation_bytes: task.max_kv_bytes,
        cohort_chunks: None, head_count: 1, max_result_bytes: mapping.max_result_bytes }
}
fn check(request: &Int4ClassificationDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    let (task, planning, mapping, limits) = fixture();
    check_request(32, DocumentCapacity { context: limits.native.context_tokens, kv_bytes: task.max_kv_bytes,
        mapping, max_heads: planning.max_labels, cohort_chunks: None, max_result_bytes: mapping.max_result_bytes }, request)
}

#[test]
fn corpus_quote_keeps_one_native_cache_and_adds_all_stream_and_frozen_identity_owners() {
    let (task, planning, mapping, limits) = fixture();
    for width in [1, 3, 32, 128, 10752] {
        let base = Int4ClassificationMapMemoryLimits { context_tokens: limits.native.context_tokens,
            preparation_reserve_bytes: limits.preparation_reserve_bytes,
            allocator_reserve_bytes: limits.native.allocator_reserve_bytes }
            .requirements(width, task, planning, mapping).unwrap();
        let actual = int4_classification_document_corpus_requirements(width, task, planning, mapping, limits).unwrap();
        assert_eq!(actual.native, base.native); assert_eq!(actual.workspace_bytes, base.workspace_bytes);
        assert_eq!(actual.max_output_bytes, mapping.max_result_bytes);
        assert_eq!(actual.output_reservation_bytes, base.output_reservation_bytes);
        assert_eq!(actual.input_bytes, base.input_bytes + limits.reservation_bytes().unwrap()
            + mapping.chunks.max_chunks as u64 * CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES);
    }
}

#[test]
fn missing_host_memory_axes_and_checked_overflows_fail_before_any_input_or_model_io() {
    for field in 0..10 {
        let (mut task, planning, mut mapping, mut limits) = fixture();
        match field { 0 => limits.preparation_reserve_bytes = 0, 1 => limits.io_reserve_bytes = 0,
            2 => limits.native.allocator_reserve_bytes = 0, 3 => limits.native.run.max_checkpoints = 1,
            4 => limits.preparation_reserve_bytes = u64::MAX, 5 => limits.io_reserve_bytes = u64::MAX,
            6 => limits.native.allocator_reserve_bytes = u64::MAX, 7 => task.max_kv_bytes -= 1,
            8 => mapping.max_result_bytes = limits.transport.max_output_line_bytes as u64 + 1,
            _ => limits.transport.max_json_depth = 65 }
        assert!(int4_classification_document_corpus_requirements(32, task, planning, mapping, limits).is_err(),
            "field {field}");
    }
    let (task, planning, mapping, limits) = fixture();
    for width in [0, 10753, usize::MAX] {
        assert!(int4_classification_document_corpus_requirements(width, task, planning, mapping, limits).is_err());
    }
}

#[test]
fn buffers_and_preparation_grow_exactly_without_allocating_another_native_cache() {
    let (task, planning, mapping, mut limits) = fixture();
    let base = int4_classification_document_corpus_requirements(32, task, planning, mapping, limits).unwrap();
    limits.io_reserve_bytes += 4096; limits.preparation_reserve_bytes += 8192;
    let larger = int4_classification_document_corpus_requirements(32, task, planning, mapping, limits).unwrap();
    // The explicitly conservative map and corpus preparation quotes each
    // retain this owner allowance; the public helper and CLI share that fact.
    assert_eq!(larger.input_bytes - base.input_bytes, 4096 + 2 * 8192);
    assert_eq!(larger.native, base.native); assert_eq!(larger.workspace_bytes, base.workspace_bytes);
}

#[test]
fn complete_map_output_is_not_mistaken_for_one_chunk_output_or_one_context_work() {
    let identities = [identity(), identity()]; let request = request(&identities);
    assert!(request.max_result_bytes > request.task_budget.max_output_bytes);
    assert!(request.model_work.forward_positions > fixture().3.native.context_tokens as u64);
    check(&request).unwrap();
}

#[test]
fn every_chunk_identity_must_keep_the_model_profile_and_classification_task() {
    for field in 0..6 {
        let mut identities = [identity(), identity()];
        match field { 0 => identities[1].task_spec = "judge-v1".to_owned(),
            1 => identities[1].thinking_mode = ThinkingMode::Enabled,
            2 => identities[1].tool_mode = ToolMode::Json,
            3 => identities[1].numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            4 => identities[1].kv_dtype = "int8".to_owned(),
            _ => identities[1].packing_set_digest = Sha256Digest::of_bytes(b"wrong grouping") }
        let error = check(&request(&identities)).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
    let empty = request(&[]); assert!(check(&empty).is_err());
    let too_many = vec![identity(); fixture().2.chunks.max_chunks + 1];
    assert!(check(&request(&too_many)).is_err());
}

#[test]
fn actual_full_kv_and_all_document_work_axes_are_independently_authorized() {
    let identities = [identity(), identity()];
    for axis in 0..5 {
        let mut item = request(&identities);
        match axis { 0 => item.model_work.forward_positions += 1, 1 => item.model_work.projected_logits += 1,
            2 => item.model_work.attention_pairs += 1, 3 => item.model_work.projections.dot_products += 1,
            _ => item.model_work.projections.multiply_accumulates += 1 }
        let failure = check(&item).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::InvalidExecution);
    }
    for field in 0..6 {
        let mut item = request(&identities);
        match field { 0 => item.kv_reservation_bytes -= 1, 1 => item.task_budget.max_kv_bytes -= 1,
            2 => item.required_context = 0, 3 => item.required_context = 1025,
            4 => item.group_width = 128, _ => item.max_result_bytes -= 1 }
        assert!(check(&item).unwrap_err().stop, "field {field}");
    }
}

#[test]
fn owned_stream_crosses_only_the_existing_blocking_host_surface() {
    fn send<T: Send + 'static>() {}
    send::<DeferredClassificationStream<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    let _host = NlpEngine::batch_int4_classification_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int4_classification_documents_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _quote = crate::hosted::int4_classification_document_corpus_requirements;
    let _cohorts = int4_classification_document_corpus_cohort_requirements;
}

#[test]
fn numerical_teams_reject_every_explicit_native_chunk_cohort() {
    for chunks in [1, 2, 8] {
        check_team_schedule(None, Some(chunks)).unwrap();
        for threads in [2, 3, 64] {
            check_team_schedule(Some(threads), None).unwrap();
            assert!(matches!(check_team_schedule(Some(threads), Some(chunks)), Err(HostedError::Limits(_))));
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
mod team {
    use super::*;
    use std::{io::{self, Cursor, Read}, sync::Mutex};
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        hosted::RunStop, native_engine::{decode::DecodeCancellationKind,
            strict_int4::weights::projection_fixture},
        tasks::classify::grouped_int4::Int4ClassificationPlanner,
        tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};

    const INPUT: &[u8] = b"invalid-json\n{\"flush\":true}\n";
    const PREPARATION: u64 = 128 << 20;
    #[derive(Clone, Copy)]
    enum Delivery { Complete, PartialTerminal, CancelAtEof, CancelAtTerminalFlush, PanicAtTerminal }
    #[derive(Default)]
    struct Observation { bytes: Vec<u8>, writes: usize, terminals: usize, drops: usize, input_at_drop: u64 }
    struct Writer { host: Arc<NlpEngine>, observed: Arc<Mutex<Observation>>, run: RunLimits,
        threads: usize, delivery: Delivery, stop: CancellationToken, terminal: bool, eof: bool }
    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            self.terminal = event["event"] == "run_complete";
            self.eof = event["event"] == "flush" && event["eof"] == true;
            assert_eq!(self.host.resources().outstanding_closure_snapshot().active_closures, 1);
            assert!(matches!(dispatch::preflight(&self.host, self.run), Err(HostedError::Reentrant)));
            if self.terminal {
                let snapshot = self.host.resources().memory_snapshot();
                let kv = snapshot.by_class.get(&MemoryClass::KvPages).map_or(0, |value| value.committed_bytes);
                if self.threads > 1 {
                    assert_eq!(kv, 0, "native state must drop before post-join completion");
                    assert_eq!(snapshot.by_class[&MemoryClass::ActivationScratch].committed_bytes,
                        int4_task_team_workspace_bytes(self.threads, 32).unwrap());
                } else { assert!(kv > 0, "preserve the serial terminal schedule"); }
                self.observed.lock().unwrap().terminals += 1;
            }
            self.observed.lock().unwrap().writes += 1;
            if self.terminal && matches!(self.delivery, Delivery::PanicAtTerminal) {
                panic!("injected classification terminal writer panic");
            }
            let mut observed = self.observed.lock().unwrap();
            if self.terminal && matches!(self.delivery, Delivery::PartialTerminal) {
                observed.bytes.extend_from_slice(&bytes[..bytes.len().min(7)]);
                return Err(io::Error::other("injected classification terminal partial write"));
            }
            observed.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.eof && matches!(self.delivery, Delivery::CancelAtEof) {
                self.stop.cancel(DecodeCancellationKind::Deadline);
            }
            if self.terminal && matches!(self.delivery, Delivery::CancelAtTerminalFlush) {
                self.stop.cancel(DecodeCancellationKind::Shutdown);
            }
            Ok(())
        }
    }
    impl Drop for Writer {
        fn drop(&mut self) {
            let snapshot = self.host.resources().memory_snapshot();
            let mut observed = self.observed.lock().unwrap();
            observed.input_at_drop = snapshot.by_class.get(&MemoryClass::JobBuffers)
                .map_or(0, |value| value.committed_bytes + value.reserved_bytes);
            observed.drops += 1;
        }
    }
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused classification read input"); }
    }
    impl BufRead for NoRead {
        fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("refused classification filled input"); }
        fn consume(&mut self, _: usize) { panic!("refused classification consumed input"); }
    }
    fn planner(model: &ResidentInt4, compact: bool, cohorts: Option<usize>)
        -> (Int4ClassificationDocumentBatchPlanner, CorpusLimits) {
        let (mut budget, planning, mapping, mut limits) = fixture();
        if cohorts.is_some() {
            budget.max_kv_bytes = budget.max_kv_bytes.checked_mul(planning.max_labels as u64).unwrap();
        }
        limits.native.run.max_elapsed = Duration::from_secs(60);
        limits.native.run.max_checkpoints = 100_000;
        let encoding = if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        }) } else { UntrustedEncoding::ByteFallback };
        let compiler = Int4ClassificationPlanner::for_model(model.model_identity(), budget, planning).unwrap()
            .with_document_encoding(encoding).unwrap();
        let planner = Int4ClassificationDocumentBatchPlanner::new(compiler, None, mapping, mapping.max_model_work).unwrap();
        let planner = if let Some(chunks) = cohorts {
            planner.with_chunk_cohorts(chunks, Int4ClassificationMapMemoryLimits {
                context_tokens: limits.native.context_tokens, preparation_reserve_bytes: limits.preparation_reserve_bytes,
                allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
            }).unwrap()
        } else { planner };
        (planner, limits)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead + Send + 'static>(host: &Arc<NlpEngine>, model: &ResidentInt4,
        threads: Option<usize>, compact: bool, cohorts: Option<usize>, delivery: Delivery,
        input: R, stop: CancellationToken) -> (Result<BatchSummary, HostedError>, Arc<Mutex<Observation>>) {
        let (planner, limits) = planner(model, compact, cohorts);
        let observed = Arc::new(Mutex::new(Observation::default()));
        let writer = Writer { host: Arc::clone(host), observed: Arc::clone(&observed), run: limits.native.run,
            threads: threads.unwrap_or(1), delivery, stop: stop.clone(), terminal: false, eof: false };
        let result = if let Some(threads) = threads {
            host.batch_int4_classification_documents_with_threads(model, planner, limits,
                PackedInt4Kernel::scalar(), threads, input, writer, stop)
        } else {
            host.batch_int4_classification_documents(model, planner, limits,
                PackedInt4Kernel::scalar(), input, writer, stop)
        };
        (result, observed)
    }
    fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
        assert_eq!(&host.resources().memory_snapshot(), baseline);
        let closures = host.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
    }
    #[test]
    #[ignore = "controller-only fresh runtime: tiny weights prove stream ownership, never native classification parity"]
    fn classification_stream_team_joins_before_terminal_delivery_and_drains_all_failure_paths() {
        assert!(crate::installed_process_resources().is_none());
        let host = Arc::new(NlpEngine::builder().resource_config(ResourceHostConfig {
            runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
            max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 2,
            helper_threads: 0, thread_ceiling: 4, memory_ceiling_bytes: 1024 << 20,
            leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
        }).build().unwrap());
        let idle = host.resources().memory_snapshot();
        let lease = host.resources().acquire_lease();
        let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, PREPARATION).unwrap().commit().unwrap();
        let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(),
            || Ok(projection_fixture(32).0)).unwrap();
        let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
        let baseline = host.resources().memory_snapshot();
        let (task, planning, mapping, limits) = fixture();
        let charge = int4_classification_document_corpus_requirements(32, task, planning, mapping, limits).unwrap().input_bytes;
        for compact in [false, true] {
            let mut expected = None;
            for threads in [None, Some(2), Some(3)] {
                let (result, observed) = execute(&host, &model, threads, compact, None,
                    Delivery::Complete, Cursor::new(INPUT.to_vec()), CancellationToken::default());
                let summary = result.unwrap();
                assert_eq!(summary.failed, 1); assert_eq!(summary.succeeded, 0); assert_eq!(summary.flushes, 2);
                assert_eq!(summary.reserved_work, crate::batch::BatchWork::default());
                let observed = observed.lock().unwrap();
                assert_eq!(observed.drops, 1); assert_eq!(observed.terminals, 1);
                assert!(observed.input_at_drop >= PREPARATION + charge);
                if let Some(expected) = &expected { assert_eq!(&observed.bytes, expected); }
                else { expected = Some(observed.bytes.clone()); }
                drained(&host, &baseline);
            }
        }
        for delivery in [Delivery::PartialTerminal, Delivery::CancelAtEof,
            Delivery::CancelAtTerminalFlush, Delivery::PanicAtTerminal] {
            let (result, observed) = execute(&host, &model, Some(3), true, None, delivery,
                Cursor::new(INPUT.to_vec()), CancellationToken::default());
            match delivery {
                Delivery::PartialTerminal => assert!(matches!(result, Err(HostedError::Batch(error))
                    if error.fault.code == BatchCode::OutputIo)),
                Delivery::CancelAtEof => assert!(matches!(result, Err(HostedError::Stopped {
                    stop: RunStop { kind: DecodeCancellationKind::Deadline, .. }, .. }))),
                Delivery::CancelAtTerminalFlush => assert!(matches!(result, Err(HostedError::Stopped {
                    stop: RunStop { kind: DecodeCancellationKind::Shutdown, .. }, .. }))),
                Delivery::PanicAtTerminal => assert!(matches!(result, Err(HostedError::Panicked))),
                Delivery::Complete => unreachable!(),
            }
            let observed = observed.lock().unwrap();
            assert_eq!(observed.drops, 1); assert!(observed.input_at_drop >= PREPARATION + charge);
            let text = std::str::from_utf8(&observed.bytes).unwrap();
            if matches!(delivery, Delivery::CancelAtEof) {
                assert_eq!(observed.terminals, 0); assert!(!text.contains("run_complete"));
                assert!(text.contains("run_error"));
            } else { assert_eq!(observed.terminals, 1); assert!(!text.contains("run_error")); }
            drained(&host, &baseline);
        }
        for (threads, cohorts) in [(0, None), (1, None), (4, None), (65, None), (usize::MAX, None),
            (2, Some(1)), (3, Some(2))] {
            let (result, observed) = execute(&host, &model, Some(threads), false, cohorts,
                Delivery::Complete, NoRead, CancellationToken::default());
            assert!(matches!(result, Err(HostedError::Limits(_))));
            let observed = observed.lock().unwrap();
            assert_eq!(observed.writes, 0); assert_eq!(observed.drops, 1);
            drained(&host, &baseline);
        }
        {
            let (task, planning, mapping, limits) = fixture();
            let required = int4_classification_document_corpus_requirements(32, task, planning, mapping, limits).unwrap();
            let room = sum(&[required.input_bytes, required.native.kv_bytes,
                required.workspace_bytes, int4_task_team_workspace_bytes(3, 32).unwrap() - 1]).unwrap();
            let bytes = host.resources().config().memory_ceiling_bytes
                .checked_sub(baseline.committed_bytes + baseline.reserved_bytes).unwrap().checked_sub(room).unwrap();
            let lease = host.resources().acquire_lease();
            let held = Pending::reserve(&lease, MemoryClass::JobBuffers, bytes).unwrap().commit().unwrap();
            let (result, observed) = execute(&host, &model, Some(3), false, None,
                Delivery::Complete, NoRead, CancellationToken::default());
            assert!(matches!(result, Err(HostedError::Reservation(_))));
            let observed = observed.lock().unwrap();
            assert_eq!(observed.writes, 0); assert_eq!(observed.drops, 1);
            drop(held); drop(lease);
            drained(&host, &baseline);
        }
        drop(model); drop(preparation);
        drained(&host, &idle);
    }
}
