//! Reservation/admission fixtures and isolated real-host transport lifetimes.
//! The tiny projection fixture never executes a neural forward.
use super::*;
use crate::{
    batch::{BatchLimits, BatchWork},
    tasks::{ner::{EntityType, NerOptions},
        source_planning::grouped_int4::long::{
            question::{SourceQuestion, synthesis::{QuestionSynthesisLimits, SourceQuestionSynthesis,
                hierarchy::QuestionHierarchyLimits}},
            summary::synthesis::{SourceSummarySynthesis, SummarySynthesisLimits,
                hierarchy::SummaryHierarchyLimits},
        }},
    validation::grounded_fields::GroundingBudget,
};

const WIDTH: usize = 32;
fn fixture() -> (Int4SourceMapConfig, Int4SourceDocumentBatchLimits, CorpusLimits) {
    let (_, base) = super::super::tests::fixture(SourceMapTask::Ner(Default::default()), WIDTH);
    let run = Int4SourceDocumentBatchLimits {
        max_model_work: base.mapping.max_model_work, max_mask_visits: base.mapping.max_mask_visits,
        max_verification: GroundingBudget { max_fields: 1_000_000, max_matches: 1_000_000,
            max_scan_steps: 1_000_000_000_000 },
        max_summary_scan_steps: 1_000_000_000_000, max_keyphrase_scan_work: 1_000_000_000_000,
    };
    let host = CorpusLimits {
        native: base.native, preparation_reserve_bytes: base.preparation_reserve_bytes,
        io_reserve_bytes: 131072,
        transport: BatchLimits { max_document_bytes: base.mapping.chunks.max_input_bytes,
            max_work: BatchWork { forward_positions: run.max_model_work.forward_positions,
                projected_logits: run.max_model_work.projected_logits }, ..Default::default() },
    };
    (base, run, host)
}
fn quote(base: &Int4SourceMapConfig, task: &Int4SourceDocumentTask,
    run: Int4SourceDocumentBatchLimits, host: CorpusLimits, discovery: bool,
    hierarchy: Option<Int4HierarchyCohortSchedule>) -> Result<Int4SourceDocumentCorpusRequirement, HostedError> {
    int4_source_document_corpus_requirements(WIDTH, task, base.budget, base.planning, base.mapping,
        run, host, base.reduction_reserve_bytes, discovery, hierarchy)
}
fn question(mapping: Int4SourceMapLimits, capacity: usize) -> SourceQuestion {
    let mut text = String::with_capacity(capacity); text.push_str("Who founded the company?");
    SourceQuestion { question: text, options: Default::default(), verification: mapping.verification }
}
fn summary(mapping: Int4SourceMapLimits) -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: Default::default(), synthesis_options: Default::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 32, max_evidence_bytes: 1024,
            verification: mapping.verification } }
}
fn qa(mapping: Int4SourceMapLimits, capacity: usize) -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: question(mapping, capacity),
        limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 1024 } }
}

#[test]
fn pure_quote_keeps_source_stream_frozen_metadata_and_complete_frontier_together() {
    let (base, run, host) = fixture();
    let task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
    let need = quote(&base, &task, run, host, false, None).unwrap();
    assert_eq!(need.input_bytes, host.reservation_bytes().unwrap()
        + base.mapping.chunks.max_input_bytes as u64
        + base.mapping.chunks.max_chunks as u64 * SOURCE_DOCUMENT_CHUNK_METADATA_BYTES
        + task.retained_bytes().unwrap());
    assert_eq!(need.document_workspace_bytes, base.workspace_bytes(base.mapping.chunks.max_chunks).unwrap());
    assert_eq!(need.native.kv_bytes, base.budget.max_kv_bytes);
    assert_eq!(need.native_workspace_bytes, need.native.rope_bytes
        + need.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    assert_eq!(need.output_reservation_bytes, 4 * base.mapping.reduction.max_result_bytes as u64
        + 8 * base.mapping.chunks.max_chunks as u64 * u64::from(base.budget.max_output_tokens) + 4096);
    assert_eq!(need.total_bytes, need.native.kv_bytes + need.native_workspace_bytes
        + need.input_bytes + need.document_workspace_bytes + need.output_reservation_bytes);
    assert!(need.cohort.is_none());
}

#[test]
fn retained_ner_vector_capacity_is_priced_separately_from_its_semantic_contents() {
    let (base, run, host) = fixture();
    let small = Int4SourceDocumentTask::Map(SourceMapTask::Ner(NerOptions::default()));
    let mut options = NerOptions::default();
    let mut types = Vec::with_capacity(4096); types.extend_from_slice(&options.types); options.types = types;
    let large = Int4SourceDocumentTask::Map(SourceMapTask::Ner(options));
    let a = quote(&base, &small, run, host, false, None).unwrap();
    let b = quote(&base, &large, run, host, false, None).unwrap();
    assert_eq!(b.input_bytes - a.input_bytes, large.retained_bytes().unwrap() - small.retained_bytes().unwrap());
    assert!(b.input_bytes - a.input_bytes >= 4093 * std::mem::size_of::<EntityType>() as u64);
    assert_eq!(a.document_workspace_bytes, b.document_workspace_bytes);
    assert_eq!(a.native, b.native);
    assert_eq!(a.output_reservation_bytes, b.output_reservation_bytes);
}

#[test]
fn every_question_mode_retains_private_capacity_and_only_its_public_transcripts() {
    let (base, run, host) = fixture();
    for mode in 0..3 {
        let task = |capacity| match mode {
            0 => Int4SourceDocumentTask::Question(question(base.mapping, capacity)),
            1 => Int4SourceDocumentTask::QuestionSynthesis(qa(base.mapping, capacity)),
            _ => Int4SourceDocumentTask::QuestionHierarchy { request: qa(base.mapping, capacity),
                limits: QuestionHierarchyLimits { max_passes: 4, ..Default::default() } },
        };
        let small = task(32); let large = task(8192);
        let a = quote(&base, &small, run, host, false, None).unwrap();
        let b = quote(&base, &large, run, host, false, None).unwrap();
        assert_eq!(b.input_bytes - a.input_bytes, large.retained_bytes().unwrap() - small.retained_bytes().unwrap());
        let tokens = if mode == 2 { 4 * u64::from(base.budget.max_output_tokens) } else { 0 };
        assert_eq!(a.output_reservation_bytes, 4 * base.mapping.reduction.max_result_bytes as u64 + 8 * tokens + 4096);
        assert_eq!(a.document_workspace_bytes, b.document_workspace_bytes);
    }
}

#[test]
fn summary_hierarchy_keeps_discovery_and_every_permitted_native_transcript() {
    let (base, run, host) = fixture();
    let flat = Int4SourceDocumentTask::SummarySynthesis(summary(base.mapping));
    let hierarchy = Int4SourceDocumentTask::SummaryHierarchy { request: summary(base.mapping),
        limits: SummaryHierarchyLimits { max_passes: 4, max_levels: 3, ..Default::default() } };
    let a = quote(&base, &flat, run, host, false, None).unwrap();
    let b = quote(&base, &hierarchy, run, host, false, None).unwrap();
    assert_eq!(b.output_reservation_bytes - a.output_reservation_bytes,
        3 * u64::from(base.budget.max_output_tokens) * 8);
    assert_eq!(b.document_workspace_bytes - a.document_workspace_bytes,
        3 * (4 * base.budget.max_output_bytes + 8 * u64::from(base.budget.max_output_tokens))
            + 4 * (8192 + 256) + 3 * 128);
}

#[test]
fn independent_hierarchy_and_discovery_phases_keep_one_full_native_peak() {
    let (mut base, run, host) = fixture(); base.mapping.reduction.map_batch_chunks = 2;
    let task = Int4SourceDocumentTask::QuestionHierarchy { request: qa(base.mapping, 8192),
        limits: QuestionHierarchyLimits { max_passes: 4, ..Default::default() } };
    let serial = quote(&base, &task, run, host, false, None).unwrap();
    for discovery_cohorts in [false, true] {
        let schedule = Int4HierarchyCohortSchedule { max_groups: 3, discovery_cohorts };
        let grouped = quote(&base, &task, run, host, discovery_cohorts, Some(schedule)).unwrap();
        assert_eq!(grouped.native.kv_bytes, 3 * base.budget.max_kv_bytes);
        assert_eq!(grouped.input_bytes, serial.input_bytes + 2 * host.preparation_reserve_bytes);
        assert_eq!(grouped.document_workspace_bytes, serial.document_workspace_bytes);
        assert_eq!(grouped.output_reservation_bytes, serial.output_reservation_bytes);
        let decoder = grouped.cohort.unwrap();
        assert_eq!(grouped.total_bytes, grouped.native.kv_bytes + grouped.native_workspace_bytes
            + grouped.input_bytes + grouped.document_workspace_bytes + decoder.workspace_bytes
            + grouped.output_reservation_bytes);
        assert!(quote(&base, &task, run, host, !discovery_cohorts, Some(schedule)).is_err());
    }
    let wrong_task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
    assert!(quote(&base, &wrong_task, run, host, false,
        Some(Int4HierarchyCohortSchedule { max_groups: 3, discovery_cohorts: false })).is_err());
}

#[test]
fn partial_memory_and_lifetime_declarations_refuse_before_stream_or_model_access() {
    for axis in 0..15 {
        let (mut base, mut run, mut host) = fixture();
        let task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
        match axis {
            0 => base.budget.max_kv_bytes -= 1,
            1 => base.budget.max_kv_bytes += 1,
            2 => base.planning.max_context_tokens -= 1,
            3 => host.native.allocator_reserve_bytes = 0,
            4 => host.preparation_reserve_bytes = 0,
            5 => host.io_reserve_bytes = 0,
            6 => base.reduction_reserve_bytes = 0,
            7 => host.transport.max_output_line_bytes = base.mapping.reduction.max_result_bytes - 1,
            8 => run.max_model_work.forward_positions -= 1,
            9 => run.max_model_work.projected_logits -= 1,
            10 => run.max_model_work.attention_pairs -= 1,
            11 => run.max_model_work.projections.dot_products -= 1,
            12 => run.max_model_work.projections.multiply_accumulates -= 1,
            13 => run.max_mask_visits -= 1,
            _ => run.max_verification.max_matches = base.mapping.verification.max_matches - 1,
        }
        assert!(quote(&base, &task, run, host, false, None).is_err(), "axis {axis}");
    }
    let (base, run, mut host) = fixture();
    let task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
    host.io_reserve_bytes = u64::MAX;
    assert!(quote(&base, &task, run, host, false, None).is_err());
    assert!(output_bytes(u64::MAX, 0).is_err());
    assert!(output_bytes(1, u64::MAX).is_err());
}

#[test]
fn admission_keeps_complete_sequence_work_and_frozen_physical_peak() {
    let (base, _, _) = fixture();
    let identities = [base.identity.clone(), base.identity.clone()];
    let request = || Int4SourceDocumentAdmission { identities: &identities,
        model_work: base.mapping.max_model_work, mask_node_visits: base.mapping.max_mask_visits,
        mask_limits: base.mapping.mask_limits, kv_reservation_bytes: 3 * base.budget.max_kv_bytes,
        max_result_bytes: base.mapping.reduction.max_result_bytes as u64,
        max_output_tokens: 2 * u64::from(base.budget.max_output_tokens), chunk_count: 2 };
    let check = |claim: &Int4SourceDocumentAdmission<'_>| admission::check_document_request(WIDTH, "ner-v1",
        3 * base.budget.max_kv_bytes, base.mapping, base.mapping.reduction.max_result_bytes as u64,
        base.mapping.chunks.max_chunks as u64 * u64::from(base.budget.max_output_tokens), claim);
    check(&request()).unwrap();
    for axis in 0..8 {
        let mut claim = request();
        match axis {
            0 => claim.kv_reservation_bytes = base.budget.max_kv_bytes,
            1 => claim.model_work.attention_pairs += 1,
            2 => claim.model_work.projections.dot_products += 1,
            3 => claim.model_work.projections.multiply_accumulates += 1,
            4 => claim.mask_node_visits += 1,
            5 => claim.mask_limits.checkpoint_interval_nodes += 1,
            6 => claim.max_result_bytes -= 1,
            _ => claim.chunk_count = 1,
        }
        assert!(check(&claim).is_err(), "axis {axis}");
    }
}

#[test]
fn compact_factory_workspace_fits_once_inside_live_preparation_with_exact_boundary() {
    use crate::{native_engine::strict_int4::weights::projection_fixture,
        tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding}};
    let (base, run, mut host) = fixture();
    let (source_planner, _) = super::super::tests::fixture(SourceMapTask::Ner(Default::default()), WIDTH);
    let (weights, _) = projection_fixture(WIDTH);
    let task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
    let compiler = Int4SourceDocumentBatchPlanner::for_model(weights.identity(), Arc::new(source_planner),
        task, base.budget, base.planning, base.mapping, run).unwrap();
    assert_eq!(compiler.encoding_workspace_bytes().unwrap(), 0);
    check_encoding_preparation(0, host.preparation_reserve_bytes).unwrap();
    let byte_identity = compiler.execution_identity().clone();
    let byte_quote = quote(&base, compiler.task(), run, host, false, None).unwrap();
    let old_preparation = host.preparation_reserve_bytes;
    let call = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let complete = SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 };
    let compiler = compiler.with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), complete).unwrap();
    let bytes = compiler.encoding_workspace_bytes().unwrap();
    assert_eq!(bytes, call.requirements().unwrap().workspace_bytes + 512);
    assert_eq!(compiler.map_document_encoding_limits(), Some(complete));
    assert_ne!(compiler.execution_identity().template_digest, byte_identity.template_digest);
    host.preparation_reserve_bytes = bytes;
    check_encoding_preparation(bytes, host.preparation_reserve_bytes).unwrap();
    let exact = quote(&base, compiler.task(), run, host, false, None).unwrap();
    assert_eq!(exact.input_bytes - bytes, byte_quote.input_bytes - old_preparation);
    assert_eq!(exact.native, byte_quote.native);
    assert_eq!(exact.document_workspace_bytes, byte_quote.document_workspace_bytes);
    assert_eq!(exact.output_reservation_bytes, byte_quote.output_reservation_bytes);
    host.preparation_reserve_bytes -= 1;
    assert!(check_encoding_preparation(bytes, host.preparation_reserve_bytes).is_err());
    let short = quote(&base, compiler.task(), run, host, false, None).unwrap();
    assert_eq!(exact.input_bytes - short.input_bytes, 1);
}

#[test]
fn fixed_document_host_binds_encoding_identity_but_accepts_independent_chunk_prompts() {
    let (base, _, _) = fixture();
    let mut chunk = base.identity.clone();
    chunk.prompt_digest = Sha256Digest::of_bytes(b"independent chunk prompt");
    chunk.taskir_digest = Sha256Digest::of_bytes(b"independent chunk task");
    admission::check_document_encoding_identity(&base.identity, &chunk).unwrap();
    chunk.tokenizer_digest = Sha256Digest::of_bytes(b"different source tokenizer");
    assert!(admission::check_document_encoding_identity(&base.identity, &chunk).is_err());
    chunk.tokenizer_digest = base.identity.tokenizer_digest;
    chunk.template_digest = Sha256Digest::of_bytes(b"different compact policy or cumulative ceiling");
    assert!(admission::check_document_encoding_identity(&base.identity, &chunk).is_err());
    // A correct model/task profile alone cannot authorize another encoder.
    assert!(constrained_int4::check_profile(&chunk, WIDTH).is_ok());
}

mod team_lifecycle {
    use super::*;
    use std::{io::{self, Cursor, Read}, sync::Mutex};
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        batch::source::grouped_int4::question::{Int4QuestionDocumentBatchConfig,
            Int4QuestionDocumentBatchPlanner},
        native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture}};

    const INPUT: &[u8] = b"invalid-json\n{\"flush\":true}\n";
    const PREPARATION: u64 = 128 << 20;

    #[derive(Clone, Copy)]
    enum Workload { Source, Questions }
    #[derive(Clone, Copy)]
    enum Schedule { SerialChunks, UnselectedChunks, ChunkCohorts }
    #[derive(Clone, Copy)]
    enum Delivery { Complete, PartialTerminal, CancelAtEof, CancelAtTerminalFlush, PanicAtTerminal }

    struct Fixture {
        source: Arc<SourceTaskPlanner>, base: Int4SourceMapConfig,
        run: Int4SourceDocumentBatchLimits, limits: CorpusLimits,
    }
    impl Fixture {
        fn new() -> Self {
            let (source, mut base) = super::super::super::tests::fixture(
                SourceMapTask::Ner(Default::default()), WIDTH);
            // No valid document reaches this incomplete projection model.
            // Keep the real resident owner small while retaining all host seams.
            base.native.context_tokens = 128;
            base.planning.max_context_tokens = 128;
            base.planning.max_input_bytes = 4096;
            base.budget.max_input_tokens = 64;
            base.budget.max_output_tokens = 64;
            base.budget.max_kv_bytes = Int4MemoryRequirement::for_context(128, WIDTH).unwrap().kv_bytes;
            base.mapping.chunks.context_tokens = 128;
            base.mapping.chunks.reserved_tokens = 64;
            base.mapping.chunks.max_chunk_tokens = 64;
            base.mapping.chunks.max_input_bytes = 4096;
            base.mapping.chunks.max_chunks = 3;
            let run = Int4SourceDocumentBatchLimits {
                max_model_work: base.mapping.max_model_work,
                max_mask_visits: base.mapping.max_mask_visits,
                max_verification: base.mapping.verification,
                max_summary_scan_steps: 1 << 30, max_keyphrase_scan_work: 1 << 30,
            };
            let limits = CorpusLimits { native: base.native,
                preparation_reserve_bytes: base.preparation_reserve_bytes, io_reserve_bytes: 131_072,
                transport: BatchLimits { max_document_bytes: base.mapping.chunks.max_input_bytes,
                    max_work: BatchWork { forward_positions: run.max_model_work.forward_positions,
                        projected_logits: run.max_model_work.projected_logits }, ..Default::default() } };
            Self { source: Arc::new(source), base, run, limits }
        }
        fn question_config(&self, mapping: Int4SourceMapLimits) -> Int4QuestionDocumentBatchConfig {
            Int4QuestionDocumentBatchConfig { options: Default::default(), budget: self.base.budget,
                planning: self.base.planning, mapping, verification: mapping.verification,
                synthesis: None, hierarchy: None, max_question_bytes: 32,
                max_model_work: self.run.max_model_work, max_mask_visits: self.run.max_mask_visits,
                max_map_verification: self.run.max_verification,
                max_verification: self.run.max_verification }
        }
        fn charges(&self, workload: Workload) -> (u64, u64) {
            match workload {
                Workload::Source => {
                    let need = int4_source_document_corpus_requirements(WIDTH,
                        &Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default())),
                        self.base.budget, self.base.planning, self.base.mapping, self.run, self.limits,
                        self.base.reduction_reserve_bytes, false, None).unwrap();
                    (need.input_bytes, need.document_workspace_bytes)
                }
                Workload::Questions => {
                    let need = int4_question_document_corpus_requirements(WIDTH,
                        self.question_config(self.base.mapping), self.limits,
                        self.base.reduction_reserve_bytes, None, None).unwrap();
                    (need.input_bytes, need.document_workspace_bytes)
                }
            }
        }
        #[allow(clippy::too_many_arguments)]
        fn execute<R: BufRead + Send + 'static>(&self, host: &NlpEngine, model: &ResidentInt4,
            vocabulary: &Arc<ExtractionVocabulary>, workload: Workload, threads: Option<usize>,
            schedule: Schedule, reader: R, writer: Writer, cancellation: CancellationToken)
            -> Result<BatchSummary, HostedError> {
            let mut mapping = self.base.mapping;
            if !matches!(schedule, Schedule::SerialChunks) { mapping.reduction.map_batch_chunks = 2; }
            let planner = Int4SourceDocumentBatchPlanner::for_model(model.model_identity(),
                Arc::clone(&self.source), Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default())),
                self.base.budget, self.base.planning, mapping, self.run).unwrap();
            match workload {
                Workload::Source => {
                    let planner = if matches!(schedule, Schedule::ChunkCohorts) {
                        planner.with_chunk_cohorts().unwrap()
                    } else { planner };
                    match threads {
                        None => host.batch_int4_source_documents(model, planner, Arc::clone(vocabulary),
                            self.limits, self.base.reduction_reserve_bytes, PackedInt4Kernel::scalar(),
                            reader, writer, cancellation),
                        Some(width) => host.batch_int4_source_documents_with_threads(model, planner,
                            Arc::clone(vocabulary), self.limits, self.base.reduction_reserve_bytes,
                            PackedInt4Kernel::scalar(), width, reader, writer, cancellation),
                    }
                }
                Workload::Questions => {
                    let mut identity = planner.execution_identity().clone();
                    identity.task_spec = crate::tasks::answer::ANSWER_TASK_VERSION.to_owned();
                    drop(planner);
                    let planner = Int4QuestionDocumentBatchPlanner::for_model(model.model_identity(),
                        Arc::clone(&self.source), identity, self.question_config(mapping)).unwrap();
                    let planner = if matches!(schedule, Schedule::ChunkCohorts) {
                        planner.with_chunk_cohorts().unwrap()
                    } else { planner };
                    match threads {
                        None => host.batch_int4_question_documents(model, planner, Arc::clone(vocabulary),
                            self.limits, self.base.reduction_reserve_bytes, PackedInt4Kernel::scalar(),
                            reader, writer, cancellation),
                        Some(width) => host.batch_int4_question_documents_with_threads(model, planner,
                            Arc::clone(vocabulary), self.limits, self.base.reduction_reserve_bytes,
                            PackedInt4Kernel::scalar(), width, reader, writer, cancellation),
                    }
                }
            }
        }
    }

    #[derive(Default)]
    struct Observation { bytes: Vec<u8>, writes: usize, terminal_checks: usize,
        dropped: usize, job_bytes_at_drop: u64 }
    struct Writer {
        host: Arc<NlpEngine>, observation: Arc<Mutex<Observation>>, run: RunLimits,
        threads: Option<usize>, delivery: Delivery, cancellation: CancellationToken,
        terminal_job_bytes: u64, last_terminal: bool, last_eof: bool,
    }
    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let event: serde_json::Value = serde_json::from_slice(bytes).expect("canonical transport record");
            self.last_terminal = event["event"] == "run_complete";
            self.last_eof = event["event"] == "flush" && event["eof"] == true;
            assert_eq!(self.host.resources().outstanding_closure_snapshot().active_closures, 1);
            assert!(matches!(dispatch::preflight(&self.host, self.run), Err(HostedError::Reentrant)),
                "delivery must remain inside the original owned blocking boundary");
            if self.last_terminal {
                let snapshot = self.host.resources().memory_snapshot();
                let kv = snapshot.by_class.get(&MemoryClass::KvPages).map_or(0, |charge| charge.committed_bytes);
                if let Some(threads) = self.threads {
                    assert_eq!(kv, 0, "native state must drop before post-join completion");
                    assert_eq!(snapshot.by_class[&MemoryClass::ActivationScratch].committed_bytes,
                        int4_task_team_workspace_bytes(threads, WIDTH).unwrap(),
                        "exactly one team scratch owner must survive terminal delivery");
                } else { assert!(kv > 0, "legacy serial terminal schedule changed"); }
                let job = &snapshot.by_class[&MemoryClass::JobBuffers];
                assert!(job.committed_bytes + job.reserved_bytes >= self.terminal_job_bytes,
                    "complete source frontier and owned IO must survive terminal delivery");
                self.observation.lock().unwrap().terminal_checks += 1;
            }
            self.observation.lock().unwrap().writes += 1;
            if self.last_terminal && matches!(self.delivery, Delivery::PanicAtTerminal) {
                panic!("injected complete-document terminal panic");
            }
            let mut observed = self.observation.lock().unwrap();
            if self.last_terminal && matches!(self.delivery, Delivery::PartialTerminal) {
                observed.bytes.extend_from_slice(&bytes[..bytes.len().min(7)]);
                return Err(io::Error::other("injected complete-document partial write"));
            }
            observed.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.last_eof && matches!(self.delivery, Delivery::CancelAtEof) {
                self.cancellation.cancel(DecodeCancellationKind::Deadline);
            }
            if self.last_terminal && matches!(self.delivery, Delivery::CancelAtTerminalFlush) {
                self.cancellation.cancel(DecodeCancellationKind::Shutdown);
            }
            Ok(())
        }
    }
    impl Drop for Writer {
        fn drop(&mut self) {
            let snapshot = self.host.resources().memory_snapshot();
            let mut observed = self.observation.lock().unwrap();
            observed.job_bytes_at_drop = snapshot.by_class.get(&MemoryClass::JobBuffers)
                .map_or(0, |charge| charge.committed_bytes + charge.reserved_bytes);
            observed.dropped += 1;
        }
    }
    fn writer(host: &Arc<NlpEngine>, fixture: &Fixture, workload: Workload, threads: Option<usize>,
        delivery: Delivery, cancellation: &CancellationToken) -> (Writer, Arc<Mutex<Observation>>) {
        let observation = Arc::new(Mutex::new(Observation::default()));
        let (input, frontier) = fixture.charges(workload);
        (Writer { host: Arc::clone(host), observation: Arc::clone(&observation), run: fixture.limits.native.run,
            threads, delivery, cancellation: cancellation.clone(), terminal_job_bytes: input + frontier + PREPARATION,
            last_terminal: false, last_eof: false }, observation)
    }

    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted document input read"); }
    }
    impl BufRead for NoRead {
        fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("unadmitted document input filled"); }
        fn consume(&mut self, _: usize) { panic!("unadmitted document input consumed"); }
    }
    fn drained(host: &NlpEngine, baseline: &crate::orchestrator::MemorySnapshot) {
        assert_eq!(&host.resources().memory_snapshot(), baseline);
        let closures = host.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0);
        assert_eq!(closures.wrapper_cancelled_closures, 0);
    }

    #[test]
    #[ignore = "run alone under controller-admitted real runtime; protocol fixture performs no neural inference"]
    fn complete_document_teams_preserve_delivery_cancellation_and_memory_ownership() {
        assert!(crate::installed_process_resources().is_none(), "fixture requires a fresh process");
        let host = Arc::new(NlpEngine::builder().resource_config(ResourceHostConfig {
            runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
            max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 2,
            helper_threads: 0, thread_ceiling: 4, memory_ceiling_bytes: 512 << 20,
            leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
        }).build().unwrap());
        let idle = host.resources().memory_snapshot();
        let lease = host.resources().acquire_lease();
        let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, PREPARATION).unwrap().commit().unwrap();
        let fixture = Fixture::new();
        let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
        let vocabulary = Arc::new(ExtractionVocabulary::pinned(registry.template_controls()).unwrap());
        let (weights, _) = projection_fixture(WIDTH);
        let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(), || Ok(weights)).unwrap();
        let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
        let baseline = host.resources().memory_snapshot();
        for workload in [Workload::Source, Workload::Questions] {
            let (input_charge, _) = fixture.charges(workload);
            let mut expected = None;
            for threads in [None, Some(2), Some(3)] {
                let cancellation = CancellationToken::default();
                let (writer, observed) = writer(&host, &fixture, workload, threads, Delivery::Complete, &cancellation);
                let summary = fixture.execute(&host, &model, &vocabulary, workload, threads,
                    Schedule::SerialChunks, Cursor::new(INPUT.to_vec()), writer, cancellation).unwrap();
                assert_eq!(summary.failed, 1); assert_eq!(summary.succeeded, 0); assert_eq!(summary.flushes, 2);
                assert_eq!(summary.reserved_work, BatchWork::default());
                let observed = observed.lock().unwrap();
                assert_eq!(observed.dropped, 1); assert_eq!(observed.terminal_checks, 1);
                assert!(observed.job_bytes_at_drop >= input_charge + PREPARATION);
                if let Some(expected) = &expected { assert_eq!(&observed.bytes, expected); }
                else { expected = Some(observed.bytes.clone()); }
                drained(&host, &baseline);
            }
            for delivery in [Delivery::PartialTerminal, Delivery::CancelAtEof,
                Delivery::CancelAtTerminalFlush, Delivery::PanicAtTerminal] {
                let cancellation = CancellationToken::default();
                let (writer, observed) = writer(&host, &fixture, workload, Some(3), delivery, &cancellation);
                let error = fixture.execute(&host, &model, &vocabulary, workload, Some(3),
                    Schedule::SerialChunks, Cursor::new(INPUT.to_vec()), writer, cancellation).unwrap_err();
                match delivery {
                    Delivery::PartialTerminal => assert!(matches!(error, HostedError::Batch(error)
                        if error.fault.code == BatchCode::OutputIo)),
                    Delivery::CancelAtEof => assert!(matches!(error, HostedError::Stopped { stop, .. }
                        if stop.kind == DecodeCancellationKind::Deadline)),
                    Delivery::CancelAtTerminalFlush => assert!(matches!(error, HostedError::Stopped { stop, .. }
                        if stop.kind == DecodeCancellationKind::Shutdown)),
                    Delivery::PanicAtTerminal => assert!(matches!(error, HostedError::Panicked)),
                    Delivery::Complete => unreachable!(),
                }
                let observed = observed.lock().unwrap();
                assert_eq!(observed.dropped, 1);
                assert!(observed.job_bytes_at_drop >= input_charge + PREPARATION);
                let text = std::str::from_utf8(&observed.bytes).unwrap();
                if matches!(delivery, Delivery::CancelAtEof) {
                    assert_eq!(observed.terminal_checks, 0); assert!(!text.contains("run_complete"));
                    assert!(text.contains("run_error"));
                } else {
                    assert_eq!(observed.terminal_checks, 1); assert!(!text.contains("run_error"));
                    if matches!(delivery, Delivery::PanicAtTerminal) { assert!(!text.contains("run_complete")); }
                }
                drained(&host, &baseline);
            }
            for (threads, schedule) in [(0, Schedule::SerialChunks), (1, Schedule::SerialChunks),
                (4, Schedule::SerialChunks), (65, Schedule::SerialChunks),
                (3, Schedule::UnselectedChunks), (3, Schedule::ChunkCohorts)] {
                let cancellation = CancellationToken::default();
                let (writer, observed) = writer(&host, &fixture, workload, Some(threads), Delivery::Complete, &cancellation);
                let error = fixture.execute(&host, &model, &vocabulary, workload, Some(threads), schedule,
                    NoRead, writer, cancellation).unwrap_err();
                assert!(matches!(error, HostedError::Limits(_)));
                let observed = observed.lock().unwrap();
                assert_eq!(observed.writes, 0); assert_eq!(observed.dropped, 1);
                drained(&host, &baseline);
            }
            let cancellation = CancellationToken::default();
            cancellation.cancel(DecodeCancellationKind::Shutdown);
            let (writer, observed) = writer(&host, &fixture, workload, Some(3), Delivery::Complete, &cancellation);
            let error = fixture.execute(&host, &model, &vocabulary, workload, Some(3), Schedule::SerialChunks,
                NoRead, writer, cancellation).unwrap_err();
            assert!(matches!(error, HostedError::Stopped { stop, .. }
                if stop.kind == DecodeCancellationKind::Shutdown));
            let observed = observed.lock().unwrap();
            assert_eq!(observed.writes, 0); assert_eq!(observed.dropped, 1);
            drained(&host, &baseline);
        }
        drop(model); drop(fixture); drop(vocabulary); drop(preparation);
        drained(&host, &idle);
    }
}
