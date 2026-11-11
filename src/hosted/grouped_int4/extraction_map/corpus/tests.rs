//! Pure admission/ownership fixtures. Synthetic identities confer no native
//! model authority and these tests make no extraction-quality claim.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::BatchLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
};

fn fixture() -> (TaskBudget, Int4ExtractionMapLimits, CorpusLimits, u64) {
    let mut base = super::super::tests::config();
    base.limits.mapping.reduction.map_batch_chunks = 3;
    let limits = CorpusLimits { native: base.native, transport: BatchLimits::default(),
        preparation_reserve_bytes: base.preparation_reserve_bytes, io_reserve_bytes: 65_536 };
    (base.request.budget, base.limits, limits, base.reduction_reserve_bytes)
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"model-free-extraction-document-corpus-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 32_usize)).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn capacity(cohorts: bool) -> DocumentCapacity {
    let (budget, document, limits, reserve) = fixture();
    let chunks = cohorts.then_some(document.mapping.reduction.map_batch_chunks);
    let quote = int4_extraction_document_corpus_requirements(32, budget, document, limits, reserve, chunks).unwrap();
    DocumentCapacity { document, budget, selected_rows: None, encoding: None,
        kv_bytes: quote.native.kv_bytes, max_result_bytes: quote.max_output_bytes }
}
fn request(identities: &[ExecutionIdentity], capacity: DocumentCapacity) -> Int4ExtractionDocumentAdmission<'_> {
    let chunks = identities.len();
    let rows = capacity.selected_rows.map_or(NANBEIGE_VOCAB_SIZE, |limits| limits.max_rows_per_step);
    Int4ExtractionDocumentAdmission { identities,
        document_encoding: capacity.encoding.map_or(UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1),
        model_work: Int4Work::for_sequence(32, 0, 128, capacity.budget.max_output_tokens as usize * chunks * rows).unwrap(),
        mask_node_visits: capacity.document.mapping.mask_visits_per_chunk * chunks as u64,
        mask_limits: capacity.document.mapping.mask_limits, kv_reservation_bytes: capacity.kv_bytes,
        max_result_bytes: capacity.max_result_bytes,
        max_output_tokens: u64::from(capacity.budget.max_output_tokens) * chunks as u64, chunk_count: chunks }
}

#[test]
fn compact_stream_quotes_share_one_workspace_and_preserve_native_authority() {
    let (budget, document, limits, reserve) = fixture();
    let policy = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 2048,
        max_pair_checks: 65_536, max_pair_bytes: 131_072 };
    let encoding = UntrustedEncoding::SafeBpeV1(policy);
    for chunks in [None, Some(3)] {
        let plain = int4_extraction_document_corpus_requirements(32, budget, document, limits, reserve, chunks).unwrap();
        let quote = int4_extraction_document_corpus_requirements_with_encoding(32, budget,
            document, limits, reserve, chunks, &encoding).unwrap();
        let core = memory_limits(limits, reserve, chunks)
            .requirements_with_encoding(32, budget, document, limits.transport, &encoding).unwrap();
        assert_eq!(quote, core);
        assert_eq!(quote.input_bytes - plain.input_bytes, policy.requirements().unwrap().workspace_bytes);
        assert_eq!(quote.native, plain.native);
        assert_eq!(quote.workspace_bytes, plain.workspace_bytes);
        assert_eq!(quote.document_workspace_bytes, plain.document_workspace_bytes);
        assert_eq!(quote.output_reservation_bytes, plain.output_reservation_bytes);
        assert_eq!(quote.max_output_bytes, plain.max_output_bytes);
        assert_eq!(int4_extraction_document_object_corpus_requirements_with_encoding(32, budget,
            document, limits, reserve, chunks, CompileLimits::default(), None, &encoding).unwrap(), quote);
        assert!(int4_extraction_document_object_corpus_requirements_with_encoding(32, budget,
            document, limits, reserve, chunks, CompileLimits::default(), Some(RootObjectLimits {
                max_properties: 16, max_origins: 64, max_input_bytes: 65_536,
                max_json_bytes: 1024, max_nodes: 4096, max_result_bytes: 4096,
            }), &encoding).is_err());
    }
}

#[test]
fn admission_binds_all_compact_policy_limits_for_serial_and_cohort_documents() {
    let base = identity();
    let identities = [identity(), identity()];
    let policy = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 2048,
        max_pair_checks: 65_536, max_pair_bytes: 131_072 };
    for cohorts in [false, true] {
        let mut compact = capacity(cohorts); compact.encoding = Some(policy);
        for selected in [None, Some(Int4JsonSparseLimits { max_rows_per_step: 65 })] {
            compact.selected_rows = selected;
            let claim = request(&identities, compact);
            check_request(32, &base, compact, &claim).unwrap();
            let mut plain = compact; plain.encoding = None;
            assert_eq!(check_request(32, &base, plain, &claim).unwrap_err().fault.code, BatchCode::Admission);
            assert_eq!(check_request(32, &base, compact, &request(&identities, plain)).unwrap_err().fault.code,
                BatchCode::Admission);
            for axis in 0..4 {
                let mut changed = policy;
                match axis { 0 => changed.max_input_bytes += 1, 1 => changed.max_output_tokens += 1,
                    2 => changed.max_pair_checks += 1, _ => changed.max_pair_bytes += 1 }
                let mut substituted = request(&identities, compact);
                substituted.document_encoding = UntrustedEncoding::SafeBpeV1(changed);
                assert_eq!(check_request(32, &base, compact, &substituted).unwrap_err().fault.code,
                    BatchCode::Admission, "axis {axis}");
            }
        }
    }
}

#[test]
fn owned_host_and_cli_share_the_complete_quote_for_both_physical_schedules() {
    let (budget, document, limits, reserve) = fixture();
    let mut serial = None;
    for chunks in [None, Some(3)] {
        let core = memory_limits(limits, reserve, chunks).requirements(32, budget, document, limits.transport).unwrap();
        let host = int4_extraction_document_corpus_requirements(32, budget, document, limits, reserve, chunks).unwrap();
        assert_eq!(host, core);
        assert_eq!(host.document_workspace_bytes,
            document_workspace_bytes(budget, document, document.mapping.chunks.max_chunks, reserve).unwrap()
                + memory_limits(limits, reserve, chunks).cohort_requirements(32, budget, document).unwrap()
                    .map_or(0, |quote| quote.workspace_bytes));
        if let Some(serial) = serial {
            let serial: Int4ExtractionDocumentMemoryRequirement = serial;
            assert_eq!(host.native.kv_bytes, serial.native.kv_bytes * 3);
            assert_eq!(host.input_bytes, serial.input_bytes);
            assert_eq!(host.output_reservation_bytes, serial.output_reservation_bytes);
            assert_eq!(host.max_output_bytes, serial.max_output_bytes);
        } else { serial = Some(host); }
    }
}

#[test]
fn object_stream_quote_retains_both_schema_parses_and_reducer_beside_complete_maps() {
    let (budget, document, limits, reserve) = fixture();
    let compiler = CompileLimits { max_schema_bytes: 8192, max_states: 256, ..CompileLimits::default() };
    let object = RootObjectLimits { max_properties: 16, max_origins: 64,
        max_input_bytes: 65_536, max_json_bytes: 1024, max_nodes: 4096, max_result_bytes: 4096 };
    let extra = object.workspace_bytes().unwrap()
        + crate::tasks::extract::object_reconcile::RootObjectSchema::workspace_bytes(compiler).unwrap();
    for chunks in [None, Some(3)] {
        let base = int4_extraction_document_corpus_requirements(32, budget, document, limits, reserve, chunks).unwrap();
        let omitted = int4_extraction_document_object_corpus_requirements(32, budget, document,
            limits, reserve, chunks, compiler, None).unwrap();
        assert_eq!(base, omitted);
        let quoted = int4_extraction_document_object_corpus_requirements(32, budget, document,
            limits, reserve, chunks, compiler, Some(object)).unwrap();
        let core = memory_limits(limits, reserve, chunks).requirements_with_root_object(32, budget,
            document, limits.transport, compiler, Some(object)).unwrap();
        assert_eq!(quoted, core);
        assert_eq!(quoted.document_workspace_bytes, base.document_workspace_bytes + extra);
        assert_eq!(quoted.native, base.native);
        assert_eq!(quoted.workspace_bytes, base.workspace_bytes);
        assert_eq!(quoted.input_bytes, base.input_bytes);
        assert_eq!(quoted.output_reservation_bytes, base.output_reservation_bytes);
        assert_eq!(quoted.max_output_bytes, base.max_output_bytes);
    }
}

#[test]
fn object_stream_refuses_conflicting_reducers_invalid_schema_quotes_and_oversized_results() {
    let (budget, document, limits, reserve) = fixture();
    let compiler = CompileLimits::default();
    let object = RootObjectLimits { max_properties: 16, max_origins: 64,
        max_input_bytes: 65_536, max_json_bytes: 1024, max_nodes: 4096, max_result_bytes: 4096 };
    for fault in 0..5 {
        let mut policy = object; let mut c = compiler; let mut d = document;
        match fault {
            0 => policy.max_result_bytes = d.mapping.reduction.max_result_bytes + 1,
            1 => policy.max_json_bytes = d.mapping.reduction.max_result_bytes + 1,
            2 => policy.max_properties = 0,
            3 => c.max_schema_bytes = 0,
            _ => d.root_array_union = Some(crate::tasks::extract::array_union::RootArrayUnionLimits {
                max_records: 16, max_origins: 64, max_input_bytes: 65_536,
                max_json_bytes: 1024, max_nodes: 4096, max_result_bytes: 4096 }),
        }
        assert!(int4_extraction_document_object_corpus_requirements(32, budget, d, limits,
            reserve, Some(3), c, Some(policy)).is_err(), "fault {fault}");
    }
}

#[test]
fn admission_checks_the_final_chunk_identity_and_preserves_the_complete_count() {
    let base = identity();
    for cohorts in [false, true] {
        let c = capacity(cohorts);
        let identities = [identity(), identity(), identity(), identity()];
        check_request(32, &base, c, &request(&identities, c)).unwrap();
        for fault in 0..6 {
            let mut changed = identities.clone();
            let last = changed.last_mut().unwrap();
            match fault { 0 => last.task_spec = "classify-v1".to_owned(),
                1 => last.tokenizer_digest = Sha256Digest::of_bytes(b"other-tokenizer"),
                2 => last.template_digest = Sha256Digest::of_bytes(b"other-template"),
                3 => last.kv_dtype = "int8".to_owned(),
                4 => last.numerics_profile = NumericsProfile::HfBf16Eager,
                _ => last.backend_semantic_version.push_str("-other") }
            let error = check_request(32, &base, c, &request(&changed, c)).unwrap_err();
            assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
        }
        for fault in 0..6 {
            let mut r = request(&identities, c);
            match fault { 0 => r.chunk_count = 0, 1 => r.chunk_count -= 1, 2 => r.chunk_count = 257,
                3 => r.kv_reservation_bytes -= 1, 4 => r.kv_reservation_bytes += 1,
                _ => r.identities = &identities[..3] }
            assert_eq!(check_request(32, &base, c, &r).unwrap_err().fault.code, BatchCode::Admission);
        }
    }
}

#[test]
fn every_native_axis_and_full_vocabulary_work_is_admitted_independently() {
    let identities = [identity(), identity()]; let base = identity(); let c = capacity(true);
    for axis in 0..5 {
        for zero in [false, true] {
            let mut r = request(&identities, c);
            let maximum = c.document.mapping.max_model_work;
            let (slot, ceiling) = match axis {
                0 => (&mut r.model_work.forward_positions, maximum.forward_positions),
                1 => (&mut r.model_work.projected_logits, maximum.projected_logits),
                2 => (&mut r.model_work.attention_pairs, maximum.attention_pairs),
                3 => (&mut r.model_work.projections.dot_products, maximum.projections.dot_products),
                _ => (&mut r.model_work.projections.multiply_accumulates, maximum.projections.multiply_accumulates),
            };
            *slot = if zero { 0 } else { ceiling + 1 };
            let error = check_request(32, &base, c, &r).unwrap_err();
            assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
        }
    }
    let mut fractional = request(&identities, c); fractional.model_work.projected_logits += 1;
    assert!(check_request(32, &base, c, &fractional).is_err());
    let mut impossible = request(&identities, c);
    impossible.model_work.forward_positions = impossible.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64 - 1;
    assert!(check_request(32, &base, c, &impossible).is_err());
}

#[test]
fn masks_cover_every_chunk_and_cannot_shrink_or_change_traversal_limits() {
    let identities = [identity(), identity(), identity()]; let base = identity(); let c = capacity(true);
    for fault in 0..4 {
        let mut r = request(&identities, c);
        match fault { 0 => r.mask_node_visits -= 1, 1 => r.mask_node_visits += 1,
            2 => r.mask_limits.max_trie_node_visits += 1, _ => r.mask_limits.checkpoint_interval_nodes += 1 }
        assert_eq!(check_request(32, &base, c, &r).unwrap_err().fault.code, BatchCode::InvalidExecution);
    }
    let mut insufficient = c;
    insufficient.document.mapping.max_mask_visits = c.document.mapping.mask_visits_per_chunk * 3 - 1;
    assert!(check_request(32, &base, insufficient, &request(&identities, c)).is_err());
    let mut overflowing = c; overflowing.document.mapping.mask_visits_per_chunk = u64::MAX;
    let mut r = request(&identities, c); r.mask_node_visits = u64::MAX;
    assert!(check_request(32, &base, overflowing, &r).is_err());
}

#[test]
fn output_admission_requires_the_whole_semantic_document_and_all_token_owners() {
    let identities = [identity(), identity(), identity(), identity()]; let base = identity();
    for cohorts in [false, true] {
        let c = capacity(cohorts);
        for fault in 0..5 {
            let mut r = request(&identities, c);
            match fault { 0 => r.max_result_bytes = 0, 1 => r.max_result_bytes -= 1,
                2 => r.max_result_bytes += 4096, 3 => r.max_output_tokens -= 1,
                _ => r.max_output_tokens += 1 }
            let error = check_request(32, &base, c, &r).unwrap_err();
            assert!(error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
        }
    }
}

#[test]
fn deadlines_transports_and_complete_output_are_refused_by_the_pure_host_quote() {
    for fault in 0..7 {
        let (budget, document, mut limits, mut reserve) = fixture();
        match fault { 0 => limits.native.run.max_checkpoints = 1,
            1 => limits.native.run.max_elapsed = std::time::Duration::ZERO,
            2 => limits.native.run.cleanup_reserve_bytes = 0,
            3 => limits.transport.max_output_line_bytes = document.mapping.reduction.max_result_bytes - 1,
            4 => limits.preparation_reserve_bytes = 0, 5 => limits.io_reserve_bytes = 0, _ => reserve = 0 }
        assert!(int4_extraction_document_corpus_requirements(32, budget, document, limits, reserve, None).is_err(),
            "fault {fault}");
    }
    let (budget, document, mut limits, reserve) = fixture();
    limits.transport.max_output_line_bytes = document.mapping.reduction.max_result_bytes;
    // Cohorts publish the same semantic map, so they do not require an
    // unpriced physical-statistics wrapper in the completed document frame.
    for chunks in [None, Some(3)] {
        int4_extraction_document_corpus_requirements(32, budget, document, limits, reserve, chunks).unwrap();
    }
}

#[test]
fn supplied_io_and_all_document_owners_can_cross_the_physical_invocation() {
    fn send<T: Send + 'static>() {}
    send::<Charged<StreamInput<std::io::Cursor<Vec<u8>>, Vec<u8>>>>();
    send::<DeferredExtractionStream<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    let _host = NlpEngine::batch_int4_extract_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int4_extract_documents_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _public_quote = crate::hosted::int4_extraction_document_corpus_requirements;
}

#[test]
fn numerical_document_teams_refuse_both_explicit_and_hidden_chunk_cohorts() {
    let (_, mut document, _, _) = fixture();
    for chunks in [1, 3] {
        document.mapping.reduction.map_batch_chunks = chunks;
        check_team_schedule(None, false, document).unwrap();
        check_team_schedule(None, true, document).unwrap();
        for threads in [2, 4, 64] {
            assert!(matches!(check_team_schedule(Some(threads), true, document), Err(HostedError::Limits(_))));
            assert_eq!(check_team_schedule(Some(threads), false, document).is_ok(), chunks == 1);
        }
    }
}

mod team {
    use super::*;
    use std::{io::{self, Cursor, Read}, sync::Mutex, time::Duration};
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        batch::extract::grouped_int4::long::stream::Int4ExtractionDocumentBatchConfig,
        grammar::runtime::SourceRuntimeLimits, hosted::{RunLimits, RunStop},
        native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture}};

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
                let kv = snapshot.by_class.get(&MemoryClass::KvPages).map_or(0, |v| v.committed_bytes);
                if self.threads > 1 {
                    assert_eq!(kv, 0, "all native state must drop before post-join completion");
                    assert_eq!(snapshot.by_class[&MemoryClass::ActivationScratch].committed_bytes,
                        int4_task_team_workspace_bytes(self.threads, 32).unwrap());
                } else { assert!(kv > 0, "legacy serial terminal scheduling must remain unchanged"); }
                self.observed.lock().unwrap().terminals += 1;
            }
            self.observed.lock().unwrap().writes += 1;
            if self.terminal && matches!(self.delivery, Delivery::PanicAtTerminal) {
                panic!("injected extraction terminal writer panic");
            }
            let mut observed = self.observed.lock().unwrap();
            if self.terminal && matches!(self.delivery, Delivery::PartialTerminal) {
                observed.bytes.extend_from_slice(&bytes[..bytes.len().min(7)]);
                return Err(io::Error::other("injected extraction terminal partial write"));
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
                .map_or(0, |v| v.committed_bytes + v.reserved_bytes);
            observed.drops += 1;
        }
    }
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused extraction read input"); }
    }
    impl BufRead for NoRead {
        fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("refused extraction filled input"); }
        fn consume(&mut self, _: usize) { panic!("refused extraction consumed input"); }
    }
    fn planner(model: &ResidentInt4) -> (Int4ExtractionDocumentBatchPlanner, CorpusLimits, u64) {
        let (budget, mut document, mut limits, reserve) = fixture();
        document.mapping.reduction.map_batch_chunks = 1;
        limits.preparation_reserve_bytes = PREPARATION;
        limits.native.run.max_elapsed = Duration::from_secs(60);
        limits.native.run.max_checkpoints = 100_000;
        let request = super::super::super::tests::config().request;
        let planner = Int4ExtractionDocumentBatchPlanner::for_model(model.model_identity(), budget,
            CompileLimits::default(), SourceRuntimeLimits::default(),
            Int4ExtractionDocumentBatchConfig { request, limits: document,
                max_model_work: document.mapping.max_model_work, max_mask_visits: document.mapping.max_mask_visits,
                max_verification: document.verification }).unwrap();
        (planner, limits, reserve)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute<R: BufRead + Send + 'static>(host: &Arc<NlpEngine>, model: &ResidentInt4,
        vocabulary: &Arc<ExtractionVocabulary>, threads: usize, selected: bool, delivery: Delivery,
        input: R, stop: CancellationToken) -> (Result<BatchSummary, HostedError>, Arc<Mutex<Observation>>) {
        let (planner, limits, reserve) = planner(model);
        let planner = if selected {
            planner.with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: 7 }).unwrap()
        } else { planner };
        let observed = Arc::new(Mutex::new(Observation::default()));
        let writer = Writer { host: Arc::clone(host), observed: Arc::clone(&observed), run: limits.native.run,
            threads, delivery, stop: stop.clone(), terminal: false, eof: false };
        let result = if threads == 1 {
            host.batch_int4_extract_documents(model, planner, Arc::clone(vocabulary), limits, reserve,
                PackedInt4Kernel::scalar(), input, writer, stop)
        } else {
            host.batch_int4_extract_documents_with_threads(model, planner, Arc::clone(vocabulary),
                limits, reserve, PackedInt4Kernel::scalar(), threads, input, writer, stop)
        };
        (result, observed)
    }
    fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
        assert_eq!(&host.resources().memory_snapshot(), baseline);
        let closures = host.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0);
        assert_eq!(closures.wrapper_cancelled_closures, 0);
    }
    #[test]
    #[ignore = "controller-only fresh runtime: tiny weights prove transport/ownership, never native extraction parity"]
    fn document_stream_terminal_delivery_joins_and_retains_input_until_success_cancel_or_failure() {
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
        let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
        let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls()).unwrap());
        let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(),
            || Ok(projection_fixture(32).0)).unwrap();
        let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
        let baseline = host.resources().memory_snapshot();
        let (p, limits, reserve) = planner(&model);
        let charge = int4_extraction_document_corpus_requirements(32, p.document_budget(),
            p.document_limits(), limits, reserve, None).unwrap().input_bytes;
        drop(p);
        for selected in [false, true] {
            let mut expected = None;
            for threads in [1, 2, 3] {
                let (result, observed) = execute(&host, &model, &vocabulary, threads, selected,
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
            let (result, observed) = execute(&host, &model, &vocabulary, 3, true, delivery,
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
        for threads in [0, 4, 65, usize::MAX] {
            let (result, observed) = execute(&host, &model, &vocabulary, threads, false,
                Delivery::Complete, NoRead, CancellationToken::default());
            assert!(matches!(result, Err(HostedError::Limits(_))));
            let observed = observed.lock().unwrap();
            assert_eq!(observed.writes, 0); assert_eq!(observed.drops, 1);
            drained(&host, &baseline);
        }
        drop(model); drop(vocabulary); drop(preparation);
        drained(&host, &idle);
    }
}


#[test]
fn selected_document_work_uses_the_factory_ceiling_in_serial_and_cohort_streams() {
    let base = identity(); let identities = [identity(), identity(), identity()];
    for cohorts in [false, true] {
        let dense = capacity(cohorts);
        let dense_request = request(&identities, dense);
        check_request(32, &base, dense, &dense_request).unwrap();
        for cap in [1, 7, 32, 65, NANBEIGE_VOCAB_SIZE] {
            let mut selected = dense; selected.selected_rows = Some(Int4JsonSparseLimits { max_rows_per_step: cap });
            let claim = request(&identities, selected);
            assert_eq!(claim.model_work.projected_logits, claim.max_output_tokens * cap as u64);
            assert_eq!(claim.max_output_tokens, dense_request.max_output_tokens);
            assert_eq!(claim.max_result_bytes, dense_request.max_result_bytes);
            assert_eq!(claim.kv_reservation_bytes, dense_request.kv_reservation_bytes);
            if cap == 32 { assert_ne!(claim.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0); }
            check_request(32, &base, selected, &claim).unwrap();
            if cap != NANBEIGE_VOCAB_SIZE {
                assert_eq!(check_request(32, &base, selected, &dense_request).unwrap_err().fault.code,
                    BatchCode::InvalidExecution);
                assert_eq!(check_request(32, &base, dense, &claim).unwrap_err().fault.code,
                    BatchCode::InvalidExecution);
            } else {
                assert_eq!(claim.model_work, dense_request.model_work);
            }
            let mut another_policy = selected;
            another_policy.selected_rows = Some(Int4JsonSparseLimits { max_rows_per_step: if cap == 1 { 2 } else { cap - 1 } });
            assert!(check_request(32, &base, another_policy, &claim).is_err());
        }
    }
}

#[test]
fn planned_sparse_rows_cannot_shrink_to_an_actual_subtotal_or_renew_output_ownership() {
    let base = identity(); let identities = [identity(), identity()];
    for cohorts in [false, true] {
        let mut c = capacity(cohorts); c.selected_rows = Some(Int4JsonSparseLimits { max_rows_per_step: 32 });
        let valid = request(&identities, c);
        for change in [-32_i64, -1, 1, 32] {
            let mut claim = request(&identities, c);
            claim.model_work.projected_logits = valid.model_work.projected_logits.checked_add_signed(change).unwrap();
            // A smaller divisible subtotal was accepted by a modulo-only gate.
            if change == -32 { claim.max_output_tokens -= 1; }
            let error = check_request(32, &base, c, &claim).unwrap_err();
            assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
        }
        for cap in [0, NANBEIGE_VOCAB_SIZE + 1] {
            let mut invalid = c; invalid.selected_rows = Some(Int4JsonSparseLimits { max_rows_per_step: cap });
            assert_eq!(check_request(32, &base, invalid, &valid).unwrap_err().fault.code, BatchCode::Admission);
        }
        let mut unselected = capacity(cohorts);
        let mut dense = request(&identities, unselected);
        dense.model_work.projected_logits -= NANBEIGE_VOCAB_SIZE as u64;
        assert_eq!(check_request(32, &base, unselected, &dense).unwrap_err().fault.code, BatchCode::InvalidExecution);
        unselected.budget.max_output_tokens = 0;
        assert_eq!(check_request(32, &base, unselected, &dense).unwrap_err().fault.code, BatchCode::InvalidExecution);
        let mut claim = request(&identities, c); claim.max_output_tokens -= 1;
        assert_eq!(check_request(32, &base, c, &claim).unwrap_err().fault.code, BatchCode::OutputLineLimit);
        let mut claim = request(&identities, c); claim.model_work.forward_positions = claim.max_output_tokens - 1;
        assert_eq!(check_request(32, &base, c, &claim).unwrap_err().fault.code, BatchCode::InvalidExecution);
    }
}
