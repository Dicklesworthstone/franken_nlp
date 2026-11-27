//! Recipe, retained-replay and debit fixtures; no native model is simulated as proof.
use super::*;
use std::{cell::{Cell, RefCell}, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::{OnceLock, atomic::{AtomicU64, Ordering}}};
use crate::{
    batch::sentiment::sampled::quantized::long::Int8SampledSentimentDocumentAdmission,
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField,
        InputRetentionLimits, TailPolicy, runner::{JobRunner, JobRunError}},
    native_engine::{artifact_bridge::ArtifactIdentity, constrained_int8::{planned_work, json_sampling_workspace_bytes},
        decode::DecodeCancellationKind},
    tokenizer::untrusted::SafeBpeLimits,
};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "nanbeige42-int8-v1".into(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 1024, max_output_bytes: 65_536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn options() -> SampledSentimentOptions {
    SampledSentimentOptions { effective_seed: [37; 32], samples: 2, temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 128, max_justification_chars: 64 }
}
fn planning() -> SampledSentimentLimits {
    SampledSentimentLimits { max_input_bytes: 4096, max_result_bytes: 65_536, ..Default::default() }
}
fn work(units: usize) -> Int8Work {
    let one = planned_work(4096, 128).unwrap();
    (0..64 * units).try_fold(Int8Work::default(), |sum, _| sum.checked_add(one)).unwrap()
}
fn native() -> Int8SampledSentimentBatchLimits {
    Int8SampledSentimentBatchLimits { max_item_work: work(1), max_model_work: work(3),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 100,
            checkpoint_interval_nodes: 8 }, max_visits_per_item: 819_200, max_visits_per_run: 2_457_600 },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 32768,
        max_snapshot_bytes: 131072, max_result_bytes: 262144, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 8,
        max_work: JobWork { model: native().max_model_work, mask_node_visits: native().masks.max_visits_per_run } }
}
fn mapping() -> Int8SampledSentimentMapLimits {
    Int8SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 32768,
        max_chunk_bytes: 4096, max_chunk_tokens: 4096, context_tokens: 4096,
        reserved_tokens: 127, max_chunks: 8, max_tokenizer_calls: 4096 },
        max_model_work: native().max_item_work, max_retained_result_bytes: 131072, max_result_bytes: 262144 }
}
fn recipe() -> Int8SampledSentimentDocumentJobRecipe {
    Int8SampledSentimentDocumentJobRecipe::new(task(), options(), planning(), mapping(), native(),
        UntrustedEncoding::ByteFallback, limits()).unwrap()
}
fn identity() -> ExecutionIdentity {
    static IDENTITY: OnceLock<ExecutionIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| Int8SampledSentimentPlanner::for_model(&facts(), options(), task(), planning())
        .unwrap().execution_identity().clone()).clone()
}
fn inputs() -> [JobInput<'static>; 2] {
    let a = b" {\"id\":\"alpha\", \"text\":\"caf\xc3\xa9\\r\\nShanghai complete source\"}\r";
    let b = br#"{"id":"beta","text":"the complete second source","task_args":{"axes":["valence"]}}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn freeze(recipe: &Int8SampledSentimentDocumentJobRecipe) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: &identity(), recipe, limits: recipe.job }, inputs(), &mut Continue).unwrap()
}

#[test]
fn recipe_binds_every_partition_sampler_compiler_native_and_encoding_policy() {
    let original = freeze(&recipe());
    for field in 0..25 {
        let mut changed = recipe();
        match field {
            0 => changed.mapping.chunks.max_input_bytes -= 1,
            1 => changed.mapping.chunks.max_chunk_bytes -= 1,
            2 => changed.mapping.chunks.max_chunk_tokens -= 1,
            3 => changed.mapping.chunks.context_tokens -= 1,
            4 => changed.mapping.chunks.reserved_tokens += 1,
            5 => changed.mapping.chunks.max_chunks -= 1,
            6 => changed.mapping.chunks.max_tokenizer_calls -= 1,
            7 => changed.mapping.max_retained_result_bytes -= 1,
            8 => changed.mapping.max_result_bytes -= 1,
            9 => changed.aggregation_version = "other-reducer",
            10 => changed.document_address_version = "other-document-address",
            11 => changed.chunk_address_version = "other-chunk-address",
            12 => changed.sampling.effective_seed[0] ^= 1,
            13 => changed.sampling.samples += 1,
            14 => changed.sampling.temperature_milli += 1,
            15 => changed.sampling.top_k = Some(32),
            16 => changed.sampling.top_p_ppm -= 1,
            17 => changed.sampling.max_sample_tokens -= 1,
            18 => changed.sampling.max_justification_chars -= 1,
            19 => changed.planning.compile.max_states -= 1,
            20 => changed.native.max_mask_visits_per_item -= 1,
            21 => changed.native.mask_checkpoint_interval_nodes += 1,
            22 => changed.native.max_sampler_bytes += 1,
            23 => changed.sample_address_stride += 1,
            _ => changed.input_encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
                max_input_bytes: 32768, max_output_tokens: 32768,
                max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }),
        }
        assert_eq!(original.binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "field {field}");
    }
}
#[test]
fn complete_document_limits_are_separate_from_chunk_limits_and_cannot_widen_on_resume() {
    let original = recipe(); original.check_job_limits(limits()).unwrap();
    assert!(original.mapping.max_result_bytes > original.task_budget.max_output_bytes);
    assert!(original.mapping.chunks.max_input_bytes > original.planning.max_input_bytes);
    for field in 0..9 {
        let mut changed = recipe();
        match field {
            0 => changed.mapping.chunks.max_input_bytes += 1,
            1 => changed.mapping.max_result_bytes += 1,
            2 => changed.mapping.max_model_work.forward_positions += 1,
            3 => changed.mapping.max_model_work.projected_logits += 1,
            4 => changed.mapping.max_model_work.attention_pairs += 1,
            5 => changed.mapping.max_model_work.projections.dot_products += 1,
            6 => changed.mapping.max_model_work.projections.multiply_accumulates += 1,
            7 => changed.native.max_mask_visits_per_run += 1,
            _ => changed.native.max_model_work.projections.multiply_accumulates += 1,
        }
        assert!(changed.check_job_limits(limits()).is_err(), "field {field}");
    }
    let mut changed = limits(); changed.max_attempts += 1;
    assert!(original.check_job_limits(changed).is_err());
}
#[test]
fn physical_prefill_binds_resume_without_refunding_cold_native_or_mask_debits() {
    let cold = recipe(); let original = freeze(&cold);
    let limits = Int8PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap() };
    let prefetched = recipe().with_prefill(limits).unwrap();
    assert_eq!(original.binding.compare(&freeze(&prefetched).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(prefetched.mapping.max_model_work, cold.mapping.max_model_work);
    assert_eq!(prefetched.job.max_work, cold.job.max_work);
    assert_eq!(prefetched.prefill_limits().unwrap().max_batch_rows, 4);
}
#[test]
fn actual_factory_keeps_effective_seed_native_identity_encoding_and_retained_owners() {
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
    let planner = Int8SampledSentimentPlanner::for_model(&facts(), options(), task(), planning()).unwrap();
    let expected = planner.execution_identity().clone();
    let factory = Int8SampledSentimentDocumentJobPlanner::new(planner, vocabulary,
        mapping(), native(), limits()).unwrap();
    assert_eq!(factory.base_execution_identity(), &expected);
    assert_eq!(factory.sampling_options().effective_seed, options().effective_seed);
    assert_eq!(factory.job_limits(), limits());
    assert_eq!(factory.max_result_bytes(), mapping().max_result_bytes);
    assert_eq!(factory.preparation_requirement_bytes(), factory.compiler.preparation_requirement_bytes()
        + SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64);
    let (owner, vocabulary) = factory.retained_owners();
    assert_eq!(owner.execution_identity(), &expected);
    assert!(Arc::strong_count(&vocabulary) >= 2);
    let factory = factory.with_layer_major_prefill(Int8PrefillLimits { max_batch_rows: 2,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(2).unwrap() }).unwrap();
    assert_eq!(factory.base_execution_identity(), &expected);
    factory.check_job_limits(limits()).unwrap();
}
struct RefuseAdmission;
impl Int8SampledSentimentDocumentBatchAdmission for RefuseAdmission {
    type Guard = ();
    fn admit(&mut self, _: Int8SampledSentimentDocumentAdmission<'_>) -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn complete_native_processor_implements_existing_durable_boundary() {
    fn durable<T: DurableBatchProcessor>() {} fn send<T: Send + 'static>() {}
    durable::<Int8SampledSentimentDocumentJobProcessor<'static, 'static, RefuseAdmission>>();
    send::<Int8SampledSentimentDocumentJobPlanner>();
    let _new = Int8SampledSentimentDocumentJobProcessor::<RefuseAdmission>::new;
}

/// Real protected storage and runner with synthetic outputs only. The native
/// adapter is separately type-checked above; these fixtures prove recipe/replay
/// and journal semantics, not model execution or native completion authority.
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-int8-sampled-document-job-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
#[derive(Default)]
struct Events { ids: RefCell<Vec<String>>, live: Cell<bool>, serializations: Cell<usize> }
struct Held { id: String, events: Rc<Events> }
impl Serialize for Held {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.live.get());
        self.events.serializations.set(self.events.serializations.get() + 1);
        self.id.serialize(serializer)
    }
}
impl Drop for Held { fn drop(&mut self) { assert!(self.events.live.replace(false)); } }
struct Fixture {
    identity: ExecutionIdentity, recipe: Int8SampledSentimentDocumentJobRecipe,
    events: Rc<Events>, fail: bool,
}
impl Fixture {
    fn new(events: Rc<Events>, fail: bool) -> Self {
        Self { identity: identity(), recipe: recipe(), events, fail }
    }
}
impl BatchProcessor for Fixture {
    type Args = SampledSentimentBatchArgs;
    type Prepared = String;
    type Output = Held;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<String, BatchItemFailure> { Ok(document.id) }
    fn planned_work(&self, _: &String) -> BatchWork {
        BatchWork { forward_positions: work(1).forward_positions, projected_logits: work(1).projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: String, _: &mut C) -> Result<Held, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, id: String, _: BatchRequestContext,
        _: &mut C) -> Result<Held, BatchItemFailure> {
        self.events.ids.borrow_mut().push(id.clone());
        if self.fail { return Err(BatchItemFailure::reject(BatchCode::Execution)); }
        assert!(!self.events.live.replace(true));
        Ok(Held { id, events: Rc::clone(&self.events) })
    }
}
impl DurableBatchProcessor for Fixture {
    type Recipe = Int8SampledSentimentDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, _: &String) -> Result<JobWork, BatchItemFailure> {
        Ok(JobWork { model: work(1), mask_node_visits: native().masks.max_visits_per_item })
    }
    fn max_result_bytes(&self, _: &String) -> u64 { self.recipe.mapping.max_result_bytes }
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn retention() -> InputRetentionLimits { InputRetentionLimits { max_spool_bytes: 1 << 20 } }

#[test]
fn retained_restart_keeps_failed_attempt_debits_and_commits_each_complete_item_once() {
    let root = Root::new(); let events = Rc::new(Events::default()); let population = inputs();
    let mut first = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(),
        retention(), &population, Fixture::new(Rc::clone(&events), true), &mut Continue).unwrap();
    assert!(first.step(&mut Continue).is_err()); assert_eq!(first.progress().attempts, 1);
    assert_eq!(first.progress().committed, 0); drop(first);
    let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(),
        retention(), Fixture::new(Rc::clone(&events), false), TailPolicy::Refuse, &mut Continue).unwrap();
    let progress = resumed.run(&mut Continue).unwrap();
    assert_eq!(progress.attempts, 3); assert_eq!(progress.committed, 2);
    assert_eq!(progress.reserved_work, limits().max_work);
    assert_eq!(*events.ids.borrow(), vec!["alpha", "alpha", "beta"]);
    assert_eq!(resumed.read_committed(0, &mut Continue).unwrap(), br#""alpha""#);
    assert_eq!(resumed.read_committed(1, &mut Continue).unwrap(), br#""beta""#);
    assert!(!events.live.get()); assert!(events.serializations.get() >= 2);
    drop(resumed);
    let mut complete = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(),
        retention(), Fixture::new(Rc::clone(&events), false), TailPolicy::Refuse, &mut Continue).unwrap();
    assert!(complete.step(&mut Continue).unwrap().is_none());
    assert_eq!(*events.ids.borrow(), vec!["alpha", "alpha", "beta"]);
}
#[test]
fn changed_private_seed_refuses_before_replay_or_result_tail_repair() {
    use std::io::Write;
    let root = Root::new(); let events = Rc::new(Events::default()); let population = inputs();
    drop(JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(), retention(),
        &population, Fixture::new(Rc::clone(&events), false), &mut Continue).unwrap());
    let mut spool = fs::OpenOptions::new().append(true).open(root.0.join("results.spool")).unwrap();
    spool.write_all(b"uncommitted tail").unwrap(); spool.sync_all().unwrap(); drop(spool);
    let before = fs::read(root.0.join("results.spool")).unwrap();
    let mut changed = Fixture::new(Rc::clone(&events), false); changed.recipe.sampling.effective_seed[0] ^= 1;
    let failure = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(),
        retention(), changed, TailPolicy::DiscardUncommitted, &mut Continue).err().unwrap();
    assert_eq!(failure, JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(fs::read(root.0.join("results.spool")).unwrap(), before);
    assert!(events.ids.borrow().is_empty());
}
#[test]
fn each_lifetime_work_axis_remains_spent_after_restart() {
    for axis in 0..6 {
        let root = Root::new(); let events = Rc::new(Events::default()); let population = inputs();
        let mut selected = limits();
        selected.max_work = JobWork { model: work(1), mask_node_visits: native().masks.max_visits_per_item };
        match axis { 0 => selected.max_work.model.forward_positions += work(1).forward_positions - 1,
            1 => selected.max_work.model.projected_logits += work(1).projected_logits - 1,
            2 => selected.max_work.model.attention_pairs += work(1).attention_pairs - 1,
            3 => selected.max_work.model.projections.dot_products += work(1).projections.dot_products - 1,
            4 => selected.max_work.model.projections.multiply_accumulates += work(1).projections.multiply_accumulates - 1,
            _ => selected.max_work.mask_node_visits += native().masks.max_visits_per_item - 1 }
        // Set all other axes to exactly two attempts; only this selected
        // counter is one unit short of paying for the resumed second attempt.
        let mut cap = JobWork { model: work(2), mask_node_visits: 2 * native().masks.max_visits_per_item };
        match axis { 0 => cap.model.forward_positions = selected.max_work.model.forward_positions,
            1 => cap.model.projected_logits = selected.max_work.model.projected_logits,
            2 => cap.model.attention_pairs = selected.max_work.model.attention_pairs,
            3 => cap.model.projections.dot_products = selected.max_work.model.projections.dot_products,
            4 => cap.model.projections.multiply_accumulates = selected.max_work.model.projections.multiply_accumulates,
            _ => cap.mask_node_visits = selected.max_work.mask_node_visits }
        selected.max_work = cap;
        let make = |fail| {
            let mut fixture = Fixture::new(Rc::clone(&events), fail);
            fixture.recipe.native.max_model_work = selected.max_work.model;
            fixture.recipe.native.max_mask_visits_per_run = selected.max_work.mask_node_visits;
            fixture.recipe.job = selected; fixture
        };
        let mut first = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), selected,
            retention(), &population, make(true), &mut Continue).unwrap();
        assert!(first.step(&mut Continue).is_err()); drop(first);
        let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), selected,
            retention(), make(false), TailPolicy::Refuse, &mut Continue).unwrap();
        let error = resumed.step(&mut Continue).unwrap_err();
        assert_eq!(error, JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(resumed.progress().attempts, 1);
        assert_eq!(events.ids.borrow().len(), 1, "axis {axis}");
    }
}
