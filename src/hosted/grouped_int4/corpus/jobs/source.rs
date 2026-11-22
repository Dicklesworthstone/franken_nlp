//! Authenticated resumable NER, keyphrases, cited summaries and passage QA
//! on the actual resident grouped-INT4 engine.
use super::*;
use crate::{
    batch::source::grouped_int4::{Int4SourceAdmission, Int4SourceBatchAdmission},
    jobs::runner::source::grouped_int4::{Int4SourceJobPlanner, Int4SourceJobProcessor},
    native_engine::strict_int4::projection::Int4ProjectionExecutor,
};

struct SourceJobInput<R> {
    preparation: Option<(Int4SourceJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}

// Capture the whole owner across the blocking boundary. Queued cancellation
// drops the private reader/factory/key before their charge and the lease last.
struct SourceJobExecution<R> {
    input: Charged<SourceJobInput<R>>,
    kv: Pending,
    scratch: Pending,
    prefill: Option<Int4PrefillLimits>,
    limits: JobHostLimits,
    required: Requirements,
    lease: EngineLease,
}

/// Retain the caller-owned reader and native reservation/charges through
/// physical join on completed jobs and ordinary population/runner failures.
struct SourceJobCompletion<R> {
    result: Result<JobProgress, JobRunError>,
    input: Charged<SourceJobInput<R>>,
    native: NativeJobCharges,
    lease: EngineLease,
}

impl NlpEngine {
    /// Create or explicitly resume an owned source-task job from the complete
    /// original NDJSON population. The immutable factory fixes the task,
    /// actual model/group identity, compiler, defaults and lifetime work.
    ///
    /// The job owns its authenticated result spool. Input remains supplied by
    /// the caller on every invocation. Population ingestion, authentication,
    /// native execution, durable acknowledgement and optional materialization
    /// run inside one physical blocking invocation. No result text is emitted
    /// to stdout by this API.
    /// Explicit selected rows retain their exact cap and source finalizer
    /// through native admission, durable publication and authenticated resume.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source<R>(&self, model: &ResidentInt4, planner: Int4SourceJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_job(model, planner, vocabulary, request, limits,
            kernel, None, None, reader, cancellation)
    }

    /// Start or resume NER, keyphrases, cited summaries or passage QA with
    /// bounded layer-major prompt morsels. One extra prompt workspace is
    /// admitted before input and retained through physical completion. All
    /// finalizers, private recipes and lifetime work/attempt debits stay fixed;
    /// explicit resume may choose another physical row width. This entry
    /// uses one coordinator; the threaded variant also admits a numerical team.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source_layer_major<R>(&self, model: &ResidentInt4,
        planner: Int4SourceJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        prefill: Int4PrefillLimits, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_job(model, planner, vocabulary, request, limits,
            kernel, Some(prefill), None, reader, cancellation)
    }

    /// Run the same authenticated source job with one persistent team of
    /// 2..=64 compute threads, including the coordinator. Every pending item,
    /// constrained token, durable commit and optional materialization shares
    /// the original invocation budget and one fixed numerical team.
    ///
    /// Worker width changes physical scheduling only. An explicit resume may
    /// choose another width without changing the private recipe or refunding
    /// persisted native/mask work and attempts. Worker scratch is separately
    /// admitted before population IO. Return waits for every worker to join;
    /// cancellation may still leave authenticated durable progress.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4SourceJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_job(model, planner, vocabulary, request, limits,
            kernel, None, Some(compute_threads), reader, cancellation)
    }

    /// Reuse one bounded numerical team across layer-major prompts, every
    /// constrained token and all durable attempts. Worker buffers retain the
    /// declared prompt row capacity, separately from coordinator prefill
    /// storage, and are admitted before private IO. Either physical width may
    /// change on authenticated resume without changing task recipes, source
    /// finalizers, committed results or spent work. Return waits for full join.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_source_layer_major_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4SourceJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, prefill: Int4PrefillLimits, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_source_job(model, planner, vocabulary, request, limits,
            kernel, Some(prefill), Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_source_job<R>(&self, model: &ResidentInt4, planner: Int4SourceJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, prefill: Option<Int4PrefillLimits>, compute_threads: Option<usize>,
        reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        // The factory prices its complete planning context into task_ceiling.
        // Equality binds that context to the actual resident KV before any
        // population bytes are read or a private job directory is opened.
        let required = with_encoding_workspace(
            job_requirements(model.model_identity().group_width(), planner.task_ceiling(),
                request.limits, limits, request.mode.input_memory())?,
            planner.encoding_workspace_bytes())?;
        let required = with_job_prefill_workspace(required, prefill)?;
        let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
        let team_scratch = compute_threads.map(|width|
            dispatch::int4_prefill_team_workspace_bytes(width, model.model_identity().group_width(), max_batch_rows))
            .transpose()?.unwrap_or(0);
        // The dispatcher reserves this exact worker quote and retains it
        // through join. Check total arithmetic before reserving native inputs.
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes, team_scratch])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(SourceJobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let execution = SourceJobExecution { input, kv, scratch, prefill, limits, required, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_prefill_team(self, limits.native.run, cancellation,
                model, width, kernel, max_batch_rows, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        // Physical cancellation/drain failure takes precedence over a job
        // return value, including a durable commit with a lost acknowledgement.
        let SourceJobCompletion { result, input, native, lease } = completed;
        drop(input);
        drop(native);
        drop(lease);
        result.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> SourceJobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<SourceJobCompletion<R>, HostedError> {
        let Self { mut input, kv, scratch, prefill, limits, required, lease } = self;
        checkpoint(control)?;
        let (planner, vocabulary, request) = input.value.preparation.take()
            .ok_or(HostedError::CompletionMissing)?;
        let native_work = planner.native_limits();
        let selected_rows = planner.selected_rows();
        let population = match read_population(&mut input.value.reader, &request,
            limits.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(SourceJobCompletion {
                result: Err(JobRunError::Storage(error)), input,
                native: NativeJobCharges::Pending { _kv: kv, _scratch: scratch }, lease,
            }),
        };
        checkpoint(control)?;
        let mut engine = allocate_native(kv, scratch, || {
            let engine = StrictInt4Engine::new(weights, limits.native.context_tokens, Int4MemoryBudget {
                max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound,
            }).map_err(native_error)?.with_kernel(kernel);
            match projection {
                Some(team) => engine.with_projection_executor(team).map_err(native_error),
                None => Ok(engine),
            }
        })?;
        let admission = SourceJobAdmission { inner: CorpusAdmission {
            lease: &lease, model: weights.identity(), kv_bytes: required.native.kv_bytes,
            masks: native_work.masks, output_bytes: request.limits.max_result_bytes as u64,
            selected_rows,
        }, task_spec: &planner.execution_identity().task_spec };
        let result = (|| {
            let processor = match prefill {
                None => Int4SourceJobProcessor::new(&planner, &mut engine.value, &vocabulary, admission),
                Some(prefill) => Int4SourceJobProcessor::new_layer_major(
                    &planner, &mut engine.value, &vocabulary, admission, prefill),
            }.map_err(JobRunError::Processor)?;
            // Both schedules use the same sealed processor and runner.
            // Authentication precedes replay/tail repair; native output guards
            // stay intact through spool sync and journal acknowledgement.
            run_population(request, &population, processor, control)
        })();
        let ChargedNative { value, _kv, _scratch } = engine;
        drop(value);
        drop(population);
        drop(vocabulary);
        drop(planner);
        Ok(SourceJobCompletion {
            result, input, native: NativeJobCharges::Committed { _kv, _scratch }, lease,
        })
    }
}

// One transient encoder workspace coexists with the charged population and
// native state. The original planner Arc keeps its cumulative ledger; this
// host neither forks it per record nor refunds preparation/reconstruction.
fn with_encoding_workspace(mut required: Requirements, workspace: u64)
    -> Result<Requirements, HostedError> {
    required.input_bytes = sum(&[required.input_bytes, workspace])?;
    sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes])?;
    Ok(required)
}

struct SourceJobAdmission<'a> { inner: CorpusAdmission<'a>, task_spec: &'a str }
impl Int4SourceBatchAdmission for SourceJobAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4SourceAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        self.inner.admit_for_task(request, self.task_spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn private_population_factory_and_result_owners_cross_the_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<Int4SourceJobPlanner>();
        send::<SourceJobInput<Cursor<Vec<u8>>>>();
        send::<Charged<SourceJobInput<Cursor<Vec<u8>>>>>();
        send::<SourceJobExecution<Cursor<Vec<u8>>>>();
        send::<SourceJobCompletion<Cursor<Vec<u8>>>>();
        let _entry = NlpEngine::job_int4_source::<Cursor<Vec<u8>>>;
        let _team_entry = NlpEngine::job_int4_source_with_threads::<Cursor<Vec<u8>>>;
        let _prefill_entry = NlpEngine::job_int4_source_layer_major::<Cursor<Vec<u8>>>;
        let _threaded_prefill_entry = NlpEngine::job_int4_source_layer_major_with_threads::<Cursor<Vec<u8>>>;
    }

    #[test]
    fn source_encoder_and_prompt_workspaces_are_independent_admitted_owners() {
        let native = Int4MemoryRequirement::for_context(64, 32).unwrap();
        let base = Requirements { native, workspace_bytes: 8192, input_bytes: 4096 };
        for rows in [1, 3, 64] {
            let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let prefill = Some(Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra });
            let encoded = with_encoding_workspace(base, 16384).unwrap();
            let combined = with_job_prefill_workspace(encoded, prefill).unwrap();
            assert_eq!(combined.native, base.native);
            assert_eq!(combined.input_bytes, base.input_bytes + 16384);
            assert_eq!(combined.workspace_bytes, base.workspace_bytes + extra);
            assert_eq!(combined, with_encoding_workspace(with_job_prefill_workspace(base, prefill).unwrap(), 16384).unwrap());
            assert!(with_job_prefill_workspace(with_encoding_workspace(base,
                u64::MAX - base.input_bytes - base.native.kv_bytes - base.workspace_bytes).unwrap(), prefill).is_err());
        }
    }

    #[cfg(feature = "asupersync-runtime")]
    mod native {
        use super::*;
        use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
            batch::{extract::{ExtractionBatchArgs, ExtractionBatchGrounding}, source::SourceBatchArgs},
            grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
            hosted::{Int4LoadLimits, NativeLimits, RunLimits, corpus::jobs::JobOpenMode},
            jobs::{InputRetentionLimits, JobId, JobSecret, JobWork, TailPolicy, population::PopulationReadLimits},
            native_engine::strict_int4::{Int4ModelLoadBudget, Int4ModelLoadRequirement},
            tasks::{BuiltInTask, answer::{AnswerOptions, AnswerPassage}, keyphrases::KeyphraseOptions,
                ner::NerOptions, source_planning::{SourcePlanningLimits, SourceTaskPlanner},
                summarize::SummaryOptions},
            tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits}};
        use std::{fs, io::{self, Read}, os::unix::fs::DirBuilderExt,
            path::{Path, PathBuf}, time::{Duration, SystemTime}};

        struct NeverRead;
        impl Read for NeverRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("retained replay read original input") }
        }
        impl BufRead for NeverRead {
            fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("retained replay read original input") }
            fn consume(&mut self, _: usize) { panic!("retained replay consumed original input") }
        }
        fn request(root: &Path, limits: JobLimits, mode: JobOpenMode) -> SourceJobRequest {
            SourceJobRequest { root: root.to_owned(), key: JobSecret::from_bytes([61; 32]),
                job_id: JobId([62; 16]), limits, mode, materialize: true }
        }
        fn drained(engine: &NlpEngine, baseline: &crate::MemorySnapshot) {
            assert_eq!(&engine.resources().memory_snapshot(), baseline);
            let closures = engine.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
        }
        #[derive(Clone, Copy)]
        enum Task { Extract { selected: bool }, Source { task: BuiltInTask, selected: bool } }

        #[allow(clippy::too_many_arguments)]
        fn execute<R: BufRead + Send + 'static>(engine: &NlpEngine, model: &ResidentInt4,
            vocabulary: &Arc<ExtractionVocabulary>, task: Task, encoding: &UntrustedEncoding,
            budget: TaskBudget, work: Int4ExtractionBatchLimits, request: SourceJobRequest,
            host: JobHostLimits, rows: Option<usize>, threads: Option<usize>, reader: R) -> JobProgress {
            let prefill = rows.map(|rows| Int4PrefillLimits { max_batch_rows: rows,
                max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() });
            let total = SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
                max_pair_checks: 1 << 26, max_pair_bytes: 1 << 30 };
            match task {
                Task::Extract { selected } => {
                    let mut planner = Int4ExtractionJobPlanner::for_model(model.model_identity(), budget,
                        CompileLimits::default(), SourceRuntimeLimits::default(), ExtractionBatchArgs {
                            schema: r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#.to_owned(),
                            grounding: ExtractionBatchGrounding::SourceMembership, budget,
                        }, work).unwrap().with_document_encoding(encoding.clone(),
                            (encoding != &UntrustedEncoding::ByteFallback).then_some(total)).unwrap();
                    if selected { planner = planner.with_selected_rows(Int4JsonSparseLimits {
                        max_rows_per_step: NANBEIGE_VOCAB_SIZE }).unwrap(); }
                    match (prefill, threads) {
                        (None, None) => engine.job_int4_extract(model, planner, Arc::clone(vocabulary), request,
                            host, PackedInt4Kernel::scalar(), reader, CancellationToken::default()),
                        (Some(prefill), None) => engine.job_int4_extract_layer_major(model, planner,
                            Arc::clone(vocabulary), request, host, PackedInt4Kernel::scalar(), prefill,
                            reader, CancellationToken::default()),
                        (None, Some(threads)) => engine.job_int4_extract_with_threads(model, planner,
                            Arc::clone(vocabulary), request, host, PackedInt4Kernel::scalar(), threads,
                            reader, CancellationToken::default()),
                        (Some(prefill), Some(threads)) => engine.job_int4_extract_layer_major_with_threads(
                            model, planner, Arc::clone(vocabulary), request, host, PackedInt4Kernel::scalar(),
                            threads, prefill, reader, CancellationToken::default()),
                    }.unwrap()
                }
                Task::Source { task, selected } => {
                    let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
                    let eos = registry.template_controls().entries().iter()
                        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
                    let mut source = SourceTaskPlanner::pinned(registry.template_controls(), eos).unwrap();
                    if encoding != &UntrustedEncoding::ByteFallback {
                        source = source.with_map_document_encoding(encoding.clone(), total).unwrap();
                    }
                    let defaults = match task {
                        BuiltInTask::Ner => SourceBatchArgs::Ner { options: NerOptions {
                            max_entities: 2, max_mention_scalars: 32, ..Default::default() }, budget },
                        BuiltInTask::Keyphrases => SourceBatchArgs::Keyphrases { options: KeyphraseOptions {
                            max_phrases: 2, max_phrase_scalars: 32 }, budget },
                        BuiltInTask::Summarize => SourceBatchArgs::Summarize { options: SummaryOptions {
                            max_bullets: 1, max_bullet_scalars: 64, max_citations_per_bullet: 1,
                            max_quote_scalars: 32 }, budget },
                        BuiltInTask::Answer => SourceBatchArgs::Answer { passages: vec![AnswerPassage {
                            id: "source".to_owned(), text: "Alice lives in Paris. Bob lives in London. café 上海".to_owned() }],
                            options: AnswerOptions { max_answer_scalars: 64, max_citations: 2,
                                max_quote_scalars: 32 }, budget },
                        _ => unreachable!(),
                    };
                    let planner = Int4SourceJobPlanner::for_model_with_document_encoding(model.model_identity(),
                        Arc::new(source), task, budget, SourcePlanningLimits {
                            max_context_tokens: host.native.context_tokens, ..Default::default() },
                        Some(defaults), work).unwrap();
                    let planner = if selected { planner.with_selected_rows(Int4JsonSparseLimits {
                        max_rows_per_step: NANBEIGE_VOCAB_SIZE }).unwrap() } else { planner };
                    match (prefill, threads) {
                        (None, None) => engine.job_int4_source(model, planner, Arc::clone(vocabulary), request,
                            host, PackedInt4Kernel::scalar(), reader, CancellationToken::default()),
                        (Some(prefill), None) => engine.job_int4_source_layer_major(model, planner,
                            Arc::clone(vocabulary), request, host, PackedInt4Kernel::scalar(), prefill,
                            reader, CancellationToken::default()),
                        (None, Some(threads)) => engine.job_int4_source_with_threads(model, planner,
                            Arc::clone(vocabulary), request, host, PackedInt4Kernel::scalar(), threads,
                            reader, CancellationToken::default()),
                        (Some(prefill), Some(threads)) => engine.job_int4_source_layer_major_with_threads(
                            model, planner, Arc::clone(vocabulary), request, host, PackedInt4Kernel::scalar(),
                            threads, prefill, reader, CancellationToken::default()),
                    }.unwrap()
                }
            }
        }

        #[test]
        #[ignore = "controller-only pinned model: complete selected/dense results, durable work and retained replay across prompt/worker schedules"]
        fn complete_structured_jobs_match_serial_results_and_resume_without_reexecution() {
            assert!(crate::installed_process_resources().is_none(), "requires a fresh process");
            let source = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source"));
            let manifest = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest"));
            let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_BYTES").expect("explicit memory cap").parse().unwrap();
            let parent = PathBuf::from(std::env::var("FNLP_INT4_TEST_JOB_DIR").expect("explicit protected test parent"));
            let unique = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
            let parent = parent.join(format!("structured-job-prefill-{}-{unique}", std::process::id()));
            fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
            let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
                runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
                max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
                helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory,
                leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
            }).build().unwrap();
            let idle = engine.resources().memory_snapshot();
            let run = RunLimits { max_elapsed: Duration::from_secs(86_400), max_checkpoints: 1_000_000_000,
                cleanup_reserve_bytes: 65536 };
            let load = Int4ModelLoadRequirement::for_group_width(32).unwrap();
            let model = engine.load_source_candidate_int4(source, manifest, 32, Int4LoadLimits {
                model: Int4ModelLoadBudget { max_payload_bytes: load.peak_payload_bytes().unwrap(),
                    max_source_values: load.source_values, max_groups: load.groups,
                    max_shard_revalidation_bytes: load.shard_revalidation_bytes },
                source_preparation_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run,
            }, CancellationToken::default()).unwrap();
            let lease = engine.resources().acquire_lease();
            let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 << 20).unwrap().commit().unwrap();
            let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
            let vocabulary = Arc::new(ExtractionVocabulary::pinned(registry.template_controls()).unwrap());
            let native = Int4MemoryRequirement::for_context(4096, 32).unwrap();
            let budget = TaskBudget { max_input_tokens: 3072, max_output_tokens: 256,
                max_output_bytes: 1 << 20, max_grammar_states: 8192, max_kv_bytes: native.kv_bytes };
            let work = Int4ExtractionBatchLimits {
                max_model_work: Int4Work::for_sequence(32, 0, 8192, 512 * NANBEIGE_VOCAB_SIZE).unwrap(),
                masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 2_000_000,
                    checkpoint_interval_nodes: 256 }, max_visits_per_item: 1_000_000_000,
                    max_visits_per_run: 2_000_000_000 },
            };
            let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
                max_snapshot_bytes: 65536, max_result_bytes: 1 << 20, max_spool_bytes: 16 << 20,
                max_materialized_bytes: 16 << 20, max_journal_bytes: 16 << 20,
                max_attempts: 4, max_work: JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run) };
            let host = JobHostLimits { native: NativeLimits { context_tokens: 4096,
                allocator_reserve_bytes: 64 << 20, run }, transport: PopulationReadLimits {
                    max_stream_bytes: 1 << 20, max_lines: 100 },
                preparation_reserve_bytes: 512 << 20, io_reserve_bytes: 131072,
                journal_reserve_bytes: 64 << 20, serialization_reserve_bytes: 64 << 20 };
            let baseline = engine.resources().memory_snapshot();
            let retention = InputRetentionLimits { max_spool_bytes: 1 << 20 };
            for (encoding_name, encoding) in [("byte", UntrustedEncoding::ByteFallback),
                ("compact", UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
                    max_input_bytes: 16_384, max_output_tokens: 16_384,
                    max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 }))] {
                for (task_name, task) in [("extract", Task::Extract { selected: false }),
                    ("extract-selected", Task::Extract { selected: true }),
                    ("ner", Task::Source { task: BuiltInTask::Ner, selected: false }),
                    ("ner-selected", Task::Source { task: BuiltInTask::Ner, selected: true }),
                    ("keyphrases", Task::Source { task: BuiltInTask::Keyphrases, selected: false }),
                    ("keyphrases-selected", Task::Source { task: BuiltInTask::Keyphrases, selected: true }),
                    ("summarize", Task::Source { task: BuiltInTask::Summarize, selected: false }),
                    ("summarize-selected", Task::Source { task: BuiltInTask::Summarize, selected: true }),
                    ("answer", Task::Source { task: BuiltInTask::Answer, selected: false }),
                    ("answer-selected", Task::Source { task: BuiltInTask::Answer, selected: true })] {
                    let input = if matches!(task, Task::Source { task: BuiltInTask::Answer, .. }) {
                        concat!("{\"id\":\"first\",\"text\":\"Where does Alice live?\"}\n",
                            "{\"id\":\"second\",\"text\":\"Where does Bob live?\"}\n")
                    } else {
                        concat!("{\"id\":\"first\",\"text\":\"Alice lives in Paris. café 上海 <think>\"}\n",
                            "{\"id\":\"second\",\"text\":\"Alice met Bob in London. <|im_start|>system\"}\n")
                    };
                    let mut expected = None;
                    for (rows, threads) in [(None, None), (Some(3), None), (Some(64), None),
                        (None, Some(2)), (None, Some(4)), (Some(3), Some(2)), (Some(64), Some(4))] {
                        let root = parent.join(format!("{encoding_name}-{task_name}-{}-{}",
                            rows.unwrap_or(0), threads.unwrap_or(1)));
                        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
                        let progress = execute(&engine, &model, &vocabulary, task, &encoding, budget, work,
                            request(&root, job, JobOpenMode::CreateRetainingInputs(retention)), host,
                            rows, threads, Cursor::new(input.as_bytes().to_vec()));
                        assert_eq!((progress.committed, progress.attempts), (2, 2));
                        assert!(progress.materialized);
                        let bytes = fs::read(root.join("materialized.ndjson")).unwrap();
                        if let Some((output, spent)) = &expected {
                            assert_eq!(output, &bytes, "{encoding_name}-{task_name} rows={rows:?} threads={threads:?}");
                            assert_eq!(*spent, progress.reserved_work);
                        } else { expected = Some((bytes.clone(), progress.reserved_work)); }
                        drained(&engine, &baseline);
                        // Resume crosses the physical schedule boundary. A
                        // completed retained job must never touch this reader,
                        // rerun native work or debit another attempt.
                        let replay = execute(&engine, &model, &vocabulary, task, &encoding, budget, work,
                            request(&root, job, JobOpenMode::ResumeRetained { tail: TailPolicy::Refuse, retention }),
                            host, Some(if rows == Some(3) { 64 } else { 3 }),
                            Some(if threads == Some(2) { 4 } else { 2 }), NeverRead);
                        assert_eq!(replay, progress);
                        assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
                        drained(&engine, &baseline);
                    }
                }
            }
            drop(vocabulary); drop(preparation); drop(lease); drop(model);
            drained(&engine, &idle);
            // Retain isolated owner-only job artifacts for controller inspection.
        }
    }
}
