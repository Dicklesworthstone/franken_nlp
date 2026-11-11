//! Real dispatcher ownership with a private finalization seam, without inference.
//! The model-present serial/team comparisons remain separate controller gates.
use super::*;
use crate::{
    LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::weights::projection_fixture},
    tasks::source_planning::grouped_int4::long::cohort::{
        Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement,
    },
};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
enum Finish { Complete, Fail, Cancel(DecodeCancellationKind) }
struct ProbeTask {
    maps: SourceMapTask,
    private_question: String,
    cancellation: CancellationToken,
    entered: Arc<AtomicBool>,
    finish: Finish,
}
#[derive(Clone, Copy)]
struct ProbeMode;
struct ProbePlan<'s> {
    maps: PreparedInt4SourceMap<'s>,
    cancellation: CancellationToken,
    entered: Arc<AtomicBool>,
    finish: Finish,
}
impl SynthesisMode for ProbeMode {
    type Task = ProbeTask;
    type Output = ();
    type Prepared<'s> = ProbePlan<'s>;
    fn requirements(self, config: &Int4SourceMapConfig<Self::Task>, bytes: usize, width: usize,
        planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
        common_requirements(config, bytes, width, planner)
    }
    fn input_bytes(self, capacity: usize, config: &Int4SourceMapConfig<Self::Task>)
        -> Result<u64, HostedError> {
        sum(&[capacity as u64, config.task.private_question.capacity() as u64,
            config.preparation_reserve_bytes])
    }
    fn workspace_bytes(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize)
        -> Result<u64, HostedError> { config.workspace_bytes(chunks) }
    fn output_tokens(self, config: &Int4SourceMapConfig<Self::Task>, chunks: usize)
        -> Result<u64, HostedError> {
        (chunks as u64).checked_mul(u64::from(config.budget.max_output_tokens))
            .ok_or(HostedError::Limits("test synthesis transcript arithmetic"))
    }
    fn prepare<'s, C: DecodeStepControl>(self, source: &'s str, planner: &SourceTaskPlanner,
        config: &Int4SourceMapConfig<Self::Task>, context: &PlanContext<'_>, width: usize, control: &mut C)
        -> Result<Self::Prepared<'s>, HostedError> {
        let maps = planner.plan_int4_map_with_control(source, &config.task.maps, config.budget, context,
            config.planning, config.mapping, width, control).map_err(map_error)?;
        Ok(ProbePlan { maps, cancellation: config.task.cancellation.clone(),
            entered: Arc::clone(&config.task.entered), finish: config.task.finish })
    }
}
impl CohortSynthesisPlan for ProbePlan<'_> {
    type Output = ();
    fn group_width(&self) -> usize { self.maps.group_width() }
    fn source_chunks(&self) -> usize { self.maps.chunk_count() }
    fn native_chunks(&self) -> usize { self.maps.chunk_count() }
    fn max_kv_bytes(&self) -> u64 { self.maps.max_kv_bytes() }
    fn max_result_bytes(&self) -> u64 { self.maps.max_result_bytes() }
    fn execution_identities(&self) -> impl ExactSizeIterator<Item = &ExecutionIdentity> {
        self.maps.execution_identities()
    }
    fn cohort_requirements(&self, _: usize) -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        Err(Int4SourceMapError::Admission)
    }
    fn run<C: DecodeStepControl>(self, _: &[ExecutionIdentity], _: &SourceTaskPlanner,
        _: &Int4ModelWeights, _: &ExtractionVocabulary, _: Int4SourceMapCohortBudget,
        _: PackedInt4Kernel, _: &mut C) -> Result<(), HostedError> {
        panic!("single-engine synthesis must never execute a cohort")
    }
}
impl SynthesisPlan for ProbePlan<'_> {
    type Output = ();
    fn validate_metadata(&self) -> Result<(), HostedError> {
        assert_eq!(self.maps.preflight_metadata().group_width(), self.maps.group_width());
        Ok(())
    }
    fn run_on_engine(self, admitted: &[ExecutionIdentity], _: &SourceTaskPlanner,
        engine: &mut StrictInt4Engine<'_>, _: &ExtractionVocabulary, _: &mut RunControl)
        -> Result<(), HostedError> {
        assert!(self.maps.chunk_count() > 1);
        assert_eq!(admitted.len(), self.maps.chunk_count());
        assert!(engine.kv_cache().all_slots_have_len(0));
        self.entered.store(true, Ordering::SeqCst);
        match self.finish {
            Finish::Complete => Ok(()),
            Finish::Fail => Err(map_error(Int4SourceMapError::Admission)),
            Finish::Cancel(cause) => { self.cancellation.cancel(cause); Ok(()) },
        }
    }
}

#[test]
fn complete_private_capture_is_sendable_before_admission() {
    fn send<T: Send + 'static>() {}
    send::<SynthesisAdmission<ProbeTask>>();
    send::<HostedInt4TeamResult<()>>();
}

/// Use real admission/dispatch/engine allocation, but deliberately stop at the
/// finalization seam before a native forward. Tiny projection-only weights do
/// not claim a complete model result or numerical parity.
#[test]
#[ignore = "run alone with controller-admitted runtime; synthetic lifecycle, not model parity"]
fn one_team_drains_on_invalid_width_planning_failure_and_late_finalization() {
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
    let make = |finish, cancellation: CancellationToken, entered| {
        let task = SourceMapTask::Ner(crate::tasks::ner::NerOptions {
            types: vec![crate::tasks::ner::EntityType::Person], max_entities: 2, max_mention_scalars: 32,
        });
        let (planner, mut base) = super::super::tests::fixture(task, 32);
        base.identity = bind_candidate_model(base.identity, model.model_identity()).unwrap();
        base.budget.max_output_tokens = 64;
        base.mapping.chunks.reserved_tokens = 64;
        base.mapping.chunks.max_chunk_bytes = 16;
        base.mapping.chunks.max_chunks = 8;
        base.mapping.max_model_work = crate::native_engine::strict_int4::Int4Work::for_sequence(
            32, 0, 16_384, 512 * NANBEIGE_VOCAB_SIZE).unwrap();
        base.preparation_reserve_bytes = 64 << 20;
        base.native.run.max_elapsed = Duration::from_secs(120);
        base.native.run.max_checkpoints = 1_000_000;
        let mut private_question = String::with_capacity(8192);
        private_question.push_str("Who visited?");
        let config = Int4SourceMapConfig {
            identity: base.identity, task: ProbeTask { maps: base.task, private_question,
                cancellation, entered, finish }, budget: base.budget, planning: base.planning,
            mapping: base.mapping, native: base.native,
            preparation_reserve_bytes: base.preparation_reserve_bytes,
            reduction_reserve_bytes: base.reduction_reserve_bytes,
        };
        (Arc::new(planner), config)
    };
    let drained = || {
        assert_eq!(host.resources().memory_snapshot(), baseline);
        assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
        assert_eq!(Arc::strong_count(&vocabulary), 1);
    };
    let source = "Alice visits Paris. Bob visits Tokyo.";

    let entered = Arc::new(AtomicBool::new(false));
    let cancellation = CancellationToken::default();
    let (planner, config) = make(Finish::Complete, cancellation.clone(), Arc::clone(&entered));
    let error = execute_with_threads::<_, ()>(&host, &model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, ProbeMode, PackedInt4Kernel::scalar(), 4, cancellation)
        .err().expect("unadmitted team must fail before source capture");
    assert!(matches!(error, HostedError::Limits("INT4 team exceeds installed child envelope")));
    assert!(!entered.load(Ordering::SeqCst));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    let entered = Arc::new(AtomicBool::new(false));
    let cancellation = CancellationToken::default();
    cancellation.cancel(DecodeCancellationKind::ParentCancelled);
    let (planner, config) = make(Finish::Complete, cancellation.clone(), Arc::clone(&entered));
    let error = execute_with_threads::<_, ()>(&host, &model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, ProbeMode, PackedInt4Kernel::scalar(), 3, cancellation)
        .err().expect("pre-cancelled synthesis must fail");
    assert!(matches!(error, HostedError::Stopped { stop, .. }
        if stop.kind == DecodeCancellationKind::ParentCancelled));
    assert!(!entered.load(Ordering::SeqCst));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    let entered = Arc::new(AtomicBool::new(false));
    let cancellation = CancellationToken::default();
    let (planner, mut config) = make(Finish::Complete, cancellation.clone(), Arc::clone(&entered));
    config.mapping.chunks.max_tokenizer_calls = 0;
    let error = execute_with_threads::<_, ()>(&host, &model, source.to_owned(), Arc::clone(&planner),
        Arc::clone(&vocabulary), config, ProbeMode, PackedInt4Kernel::scalar(), 3, cancellation)
        .err().expect("invalid complete planning must fail");
    assert!(matches!(error, HostedError::Int4(Int4HostError::SourceMap(_))));
    assert!(!entered.load(Ordering::SeqCst));
    assert_eq!(Arc::strong_count(&planner), 1);
    drained();

    for finish in [Finish::Complete, Finish::Fail,
        Finish::Cancel(DecodeCancellationKind::Deadline),
        Finish::Cancel(DecodeCancellationKind::ParentCancelled),
        Finish::Cancel(DecodeCancellationKind::ResourceUnavailable)] {
        let entered = Arc::new(AtomicBool::new(false));
        let cancellation = CancellationToken::default();
        let (planner, config) = make(finish, cancellation.clone(), Arc::clone(&entered));
        let result = execute_with_threads::<_, ()>(&host, &model, source.to_owned(), Arc::clone(&planner),
            Arc::clone(&vocabulary), config, ProbeMode, PackedInt4Kernel::scalar(), 3, cancellation);
        assert!(entered.load(Ordering::SeqCst));
        match finish {
            Finish::Complete => {
                let output = result.expect("completed fixture remains guarded until caller drop");
                assert_ne!(host.resources().memory_snapshot(), baseline);
                drop(output);
            },
            Finish::Fail => assert!(matches!(result, Err(HostedError::Int4(Int4HostError::SourceMap(_))))),
            Finish::Cancel(cause) => assert!(matches!(result, Err(HostedError::Stopped { stop, .. }) if stop.kind == cause)),
        }
        assert_eq!(Arc::strong_count(&planner), 1);
        assert_eq!(Arc::strong_count(&entered), 1);
        drained();
    }
}
