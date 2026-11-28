//! Model-free factory contracts and real retained-job storage fixtures.
//! Synthetic results below never represent native inference or model quality.
use super::*;
use crate::{
    canonjson,
    execution_identity::Sha256Digest,
    jobs::{FrozenManifest, InputRetentionLimits, JobContract, JobError, JobId, JobInput, JobSecret,
        MismatchField, TailPolicy, runner::{JobRunError, JobRunner}},
    native_engine::{artifact_bridge::ArtifactIdentity, constrained::JsonDecodeError,
        decode::DecodeCancellationKind, strict_int8::StrictInt8Error},
    tokenizer::untrusted::SafeBpeLimits,
};
use serde::Serializer;
use std::{cell::Cell, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::{OnceLock, atomic::{AtomicU64, Ordering}}};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn options() -> SampledSentimentOptions {
    SampledSentimentOptions { effective_seed: [37; 32], samples: 3, temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 256, max_justification_chars: 64 }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 4096, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn planning() -> SampledSentimentLimits {
    SampledSentimentLimits { max_input_bytes: 4096, max_result_bytes: 65536, ..Default::default() }
}
fn work(units: u64) -> Int8Work {
    Int8Work { forward_positions: 40 * units, projected_logits: 50 * units, attention_pairs: 60 * units,
        projections: ProjectionWork { dot_products: 70 * units, multiply_accumulates: 80 * units } }
}
fn native() -> Int8SampledSentimentJobLimits {
    Int8SampledSentimentJobLimits { max_item_work: work(2), max_model_work: work(10),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 100,
            checkpoint_interval_nodes: 8 }, max_visits_per_item: 10, max_visits_per_run: 50 },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn limits(native: Int8SampledSentimentJobLimits) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 8,
        max_work: JobWork { model: native.max_model_work, mask_node_visits: native.masks.max_visits_per_run } }
}
fn recipe(cap: Int8SampledSentimentJobLimits) -> Int8SampledSentimentJobRecipe {
    Int8SampledSentimentJobRecipe::new(task(), options(), planning(), cap,
        UntrustedEncoding::ByteFallback, None, limits(cap)).unwrap()
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Private fixture A","task_args":{"axes":["valence"]}}"#;
    let b = br#"{"id":"beta","text":"Private fixture B","task_args":null}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn freeze(recipe: &Int8SampledSentimentJobRecipe) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]),
        execution: &crate::jobs::tests::identity(), recipe, limits: recipe.job }, inputs(), &mut Continue).unwrap()
}
fn distinct(changed: &Int8SampledSentimentJobRecipe) {
    assert_eq!(freeze(&recipe(native())).binding.compare(&freeze(changed).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
}
fn work_axis(work: &mut Int8Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}
fn failure<T>(result: Result<T, BatchItemFailure>) -> BatchItemFailure {
    match result { Err(error) => error, Ok(_) => panic!("expected refusal") }
}
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(error) => error, Ok(_) => panic!("expected job refusal") }
}

#[test]
fn every_sampling_option_and_execution_address_binds_resume() {
    for axis in 0..7 {
        let mut changed = options();
        match axis { 0 => changed.effective_seed[31] += 1, 1 => changed.samples += 1,
            2 => changed.temperature_milli += 1, 3 => changed.top_k = Some(10),
            4 => changed.top_p_ppm -= 1, 5 => changed.max_sample_tokens += 1,
            _ => changed.max_justification_chars += 1 }
        distinct(&Int8SampledSentimentJobRecipe::new(task(), changed, planning(), native(),
            UntrustedEncoding::ByteFallback, None, limits(native())).unwrap());
    }
    for axis in 0..8 {
        let mut changed = recipe(native());
        match axis {
            0 => changed.item_address_version = "other-address",
            1 => changed.aggregation_version = "other-aggregation",
            2 => changed.sampler_version = "other-sampler",
            3 => changed.native_execution = "other-native",
            4 => changed.prompt_version = "other-prompt",
            5 => changed.grammar_version = "other-grammar",
            6 => changed.sample_address_stride += 1,
            _ => changed.default_axes.reverse(),
        }
        distinct(&changed);
    }
    let encoded = serde_json::to_value(recipe(native())).unwrap();
    assert_eq!(encoded["execution"], INT8_SAMPLED_SENTIMENT_EXECUTION);
    assert_eq!(encoded["sampling"]["effective_seed"], serde_json::to_value([37_u8; 32]).unwrap());
    assert!(encoded.get("compute_threads").is_none());
}

#[test]
fn all_planning_compiler_task_native_and_mask_caps_are_authenticated() {
    for axis in 0..11 {
        let mut changed = planning();
        match axis { 0 => changed.max_input_bytes += 1, 1 => changed.max_total_prompt_tokens += 1,
            2 => changed.max_sample_json_bytes += 1, 3 => changed.max_result_bytes += 1,
            4 => changed.compile.max_schema_bytes += 1, 5 => changed.compile.max_string_bytes += 1,
            6 => changed.compile.max_array_items += 1, 7 => changed.compile.max_output_bytes += 1,
            8 => changed.compile.max_states += 1, 9 => changed.compile.max_transitions += 1,
            _ => changed.compile.max_mask_bytes += 1 }
        distinct(&Int8SampledSentimentJobRecipe::new(task(), options(), changed, native(),
            UntrustedEncoding::ByteFallback, None, limits(native())).unwrap());
    }
    for axis in 0..5 {
        let mut changed = task();
        match axis { 0 => changed.max_input_tokens += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_output_bytes += 1, 3 => changed.max_grammar_states += 1,
            _ => changed.max_kv_bytes += 1 }
        distinct(&Int8SampledSentimentJobRecipe::new(changed, options(), planning(), native(),
            UntrustedEncoding::ByteFallback, None, limits(native())).unwrap());
    }
    for axis in 0..15 {
        let mut changed = native();
        match axis { 0..=4 => *work_axis(&mut changed.max_item_work, axis) += 1,
            5..=9 => *work_axis(&mut changed.max_model_work, axis - 5) += 1,
            10 => changed.masks.per_mask.max_trie_node_visits += 1,
            11 => changed.masks.per_mask.checkpoint_interval_nodes += 1,
            12 => changed.masks.max_visits_per_item += 1,
            13 => changed.masks.max_visits_per_run += 1, _ => changed.max_sampler_bytes += 1 }
        distinct(&recipe(changed));
    }
}

fn compact() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 4 * 1024 * 1024, max_pair_bytes: 64 * 1024 * 1024 })
}
fn total() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 32768, max_output_tokens: 32768,
        max_pair_checks: 16 * 1024 * 1024, max_pair_bytes: 256 * 1024 * 1024 }
}
#[test]
fn compact_policy_and_invocation_work_and_prefill_are_private_recipe_authority() {
    let make = |policy, total| Int8SampledSentimentJobRecipe::new(task(), options(), planning(),
        native(), policy, total, limits(native()));
    assert!(make(compact(), None).is_err());
    assert!(make(UntrustedEncoding::ByteFallback, Some(total())).is_err());
    let original = freeze(&make(compact(), Some(total())).unwrap());
    for axis in 0..8 {
        let UntrustedEncoding::SafeBpeV1(mut policy) = compact() else { unreachable!() };
        let mut total = total();
        match axis { 0 => policy.max_input_bytes += 1, 1 => policy.max_output_tokens += 1,
            2 => policy.max_pair_checks += 1, 3 => policy.max_pair_bytes += 1,
            4 => total.max_input_bytes += 1, 5 => total.max_output_tokens += 1,
            6 => total.max_pair_checks += 1, _ => total.max_pair_bytes += 1 }
        let changed = make(UntrustedEncoding::SafeBpeV1(policy), Some(total)).unwrap();
        assert_eq!(original.binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    let prefill = Int8PrefillLimits { max_batch_rows: 2,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(2).unwrap() };
    distinct(&recipe(native()).with_prefill(prefill).unwrap());
    let packed = freeze(&recipe(native()).with_prefill(prefill).unwrap());
    let changed = Int8PrefillLimits { max_extra_scratch_bytes: prefill.max_extra_scratch_bytes + 1, ..prefill };
    assert_eq!(packed.binding.compare(&freeze(&recipe(native()).with_prefill(changed).unwrap()).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
}

#[test]
fn selected_projection_cap_and_execution_profiles_are_authenticated_without_changing_legacy_recipes() {
    let original = recipe(native());
    assert!(serde_json::to_value(&original).unwrap().get("max_selected_rows").is_none());
    let selected = recipe(native()).with_projection(Some(Int8JsonSparseLimits { max_rows_per_step: 32 })).unwrap();
    selected.check_job_limits(limits(native())).unwrap();
    distinct(&selected);
    let value = serde_json::to_value(&selected).unwrap();
    assert_eq!(value["max_selected_rows"], 32);
    assert_eq!(value["execution"], INT8_SAMPLED_SENTIMENT_SPARSE_EXECUTION);
    assert_eq!(value["native_execution"], INT8_JSON_SAMPLED_SPARSE_EXECUTION);
    assert_eq!(selected.native.max_sampler_bytes, original.native.max_sampler_bytes);
    assert_eq!(selected.job.max_work, original.job.max_work);
    let changed = recipe(native()).with_projection(Some(Int8JsonSparseLimits { max_rows_per_step: 33 })).unwrap();
    assert_eq!(freeze(&selected).binding.compare(&freeze(&changed).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    let mut mixed = recipe(native()).with_projection(Some(Int8JsonSparseLimits { max_rows_per_step: 32 })).unwrap();
    mixed.native_execution = INT8_JSON_SAMPLED_EXECUTION;
    assert!(mixed.check_job_limits(limits(native())).is_err());
    assert!(recipe(native()).with_projection(Some(Int8JsonSparseLimits { max_rows_per_step: 0 })).is_err());
}

#[test]
fn immutable_job_envelope_requires_exact_native_and_mask_lifetime_caps() {
    let exact = limits(native());
    for axis in 0..6 {
        for increase in [false, true] {
            let mut changed = exact;
            let value = if axis < 5 { work_axis(&mut changed.max_work.model, axis) }
                else { &mut changed.max_work.mask_node_visits };
            if increase { *value += 1; } else { *value -= 1; }
            assert!(recipe(native()).check_job_limits(changed).is_err());
            assert!(Int8SampledSentimentJobRecipe::new(task(), options(), planning(), native(),
                UntrustedEncoding::ByteFallback, None, changed).is_err());
        }
    }
    let mut changed = exact; changed.max_result_bytes -= 1;
    assert!(Int8SampledSentimentJobRecipe::new(task(), options(), planning(), native(),
        UntrustedEncoding::ByteFallback, None, changed).is_err());
    let mut changed = exact; changed.max_id_bytes = 257;
    assert!(Int8SampledSentimentJobRecipe::new(task(), options(), planning(), native(),
        UntrustedEncoding::ByteFallback, None, changed).is_err());
    let mut changed = exact; changed.max_attempts += 1;
    assert!(recipe(native()).check_job_limits(changed).is_err());
}

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn vocabulary() -> Arc<SampledSentimentVocabulary> {
    static VOCABULARY: OnceLock<Arc<SampledSentimentVocabulary>> = OnceLock::new();
    Arc::clone(VOCABULARY.get_or_init(|| Arc::new(SampledSentimentVocabulary::pinned().unwrap())))
}
fn ample() -> Int8SampledSentimentJobLimits {
    let cap = u64::MAX / 4;
    let work = Int8Work { forward_positions: cap, projected_logits: cap, attention_pairs: cap,
        projections: ProjectionWork { dot_products: cap, multiply_accumulates: cap } };
    Int8SampledSentimentJobLimits { max_item_work: work, max_model_work: work,
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
                max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 1024 },
            max_visits_per_item: 4 * 3 * 256 * 2_000_000,
            max_visits_per_run: 2 * 4 * 3 * 256 * 2_000_000 },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn factory(encoding: UntrustedEncoding, total: Option<SafeBpeWorkLimits>) -> Int8SampledSentimentJobPlanner {
    let planner = Int8SampledSentimentPlanner::for_model_with_encoding(&facts(), options(), task(), planning(), encoding).unwrap();
    Int8SampledSentimentJobPlanner::new_with_document_encoding(planner, vocabulary(), ample(),
        limits(ample()), total).unwrap()
}
fn document(id: &str) -> BatchDocument<SampledSentimentBatchArgs> {
    BatchDocument { id: id.to_owned(),
        text: "A reassuring and helpful service with repeated reassuring words.".to_owned(),
        task_args: Some(SampledSentimentBatchArgs { axes: vec![SentimentAxis::Valence], budget: None }) }
}
#[test]
fn sealed_item_addresses_match_existing_byte_and_compact_task_planning() {
    for policy in [UntrustedEncoding::ByteFallback, compact()] {
        let total = (policy != UntrustedEncoding::ByteFallback).then(total);
        let declared = match &policy { UntrustedEncoding::ByteFallback => None, UntrustedEncoding::SafeBpeV1(cap) => Some(*cap) };
        let task_planner = Int8SampledSentimentPlanner::for_model_with_encoding(
            &facts(), options(), task(), planning(), policy.clone()).unwrap();
        let doc = document("stable-id");
        let request = SentimentRequest { document: doc.text, axes: vec![SentimentAxis::Valence], budget: task() };
        let original = task_planner.plan_for_item_with_control(&request, "stable-id", &mut Continue).unwrap();
        let explicit = task_planner.plan_for_item_with_encoding_budget(&request, "stable-id", &mut Continue,
            &mut SafeBpeWork::default(), declared).unwrap();
        assert_eq!(original.execution_identity(), explicit.execution_identity());
        assert_eq!(original.planned_work(), explicit.planned_work());
        assert_eq!(original.encoding_work(), explicit.encoding_work());
        let compiler = factory(policy.clone(), total);
        let prepared = compiler.prepare_with_control(document("stable-id"), &mut Continue).unwrap();
        assert_eq!(prepared.execution_identity(), original.execution_identity());
        assert_eq!(prepared.model_work(), original.planned_work());
        assert_eq!(compiler.sampler_workspace_bytes(), json_sampling_workspace_bytes().unwrap());
        compiler.verify(&prepared).unwrap();
        let other = factory(policy.clone(), total);
        assert_eq!(failure(other.verify(&prepared)).fault.code, BatchCode::Admission);
        let changed = compiler.prepare_with_control(document("other-id"), &mut Continue).unwrap();
        assert_ne!(prepared.execution_identity().decision_policy_digest, changed.execution_identity().decision_policy_digest);
        let expected = sampled_sentiment_job_preparation_bytes(task(), planning(), &policy).unwrap()
            + if total.is_some() { 1024 } else { 0 };
        assert_eq!(compiler.preparation_requirement_bytes(), expected);
    }
}

#[test]
fn rejected_compact_planning_spends_each_invocation_axis_without_a_fresh_ledger() {
    let baseline = factory(compact(), Some(total()));
    baseline.prepare_with_control(document("baseline"), &mut Continue).unwrap();
    let spent = baseline.encoding_work().unwrap();
    let counts = [spent.input_bytes() as u64, spent.output_tokens() as u64, spent.pair_checks(), spent.pair_bytes()];
    for axis in 0..4 {
        assert!(counts[axis] > 0);
        let mut total = total();
        match axis { 0 => total.max_input_bytes = counts[axis], 1 => total.max_output_tokens = counts[axis],
            2 => total.max_pair_checks = counts[axis], _ => total.max_pair_bytes = counts[axis] }
        let compiler = factory(compact(), Some(total));
        let mut bad = document("rejected");
        bad.task_args.as_mut().unwrap().budget = Some(TaskBudget { max_input_tokens: 1, ..task() });
        assert_eq!(failure(compiler.prepare_with_control(bad, &mut Continue)).fault.code, BatchCode::WorkLimit);
        assert_eq!(compiler.encoding_work().unwrap(), spent);
        assert_eq!(failure(compiler.prepare_with_control(document("next"), &mut Continue)).fault.code, BatchCode::WorkLimit);
        assert_eq!(compiler.encoding_work().unwrap(), spent);
    }
}

#[test]
fn local_six_axis_attempt_debit_is_atomic_and_context_cannot_replay() {
    let context = BatchRequestContext { request_seq: 1, epoch: 1, input_line: 1, byte_offset: 0 };
    for axis in 0..6 {
        let mut cap = native(); cap.max_item_work = work(1); cap.max_model_work = work(2);
        cap.masks.max_visits_per_item = 5; cap.masks.max_visits_per_run = 10;
        if axis < 5 { *work_axis(&mut cap.max_model_work, axis) -= 1; }
        else { cap.masks.max_visits_per_run -= 1; }
        cap.validate().unwrap();
        let mut ledger = RunState::new(cap); ledger.state = State::Running;
        ledger.charge(work(1), 5, context).unwrap();
        let left = (ledger.remaining, ledger.remaining_masks);
        assert_eq!(failure(ledger.charge(work(1), 5, BatchRequestContext { request_seq: 2, ..context })).fault.code,
            BatchCode::WorkLimit);
        assert_eq!((ledger.remaining, ledger.remaining_masks), left);
        assert_eq!(ledger.last_sequence, 1);
        assert_eq!(failure(ledger.charge(work(1), 5, context)).fault.code, BatchCode::InvalidExecution);
        assert_eq!((ledger.remaining, ledger.remaining_masks), left);
    }
}

#[test]
fn every_cancellation_cause_survives_shared_native_and_decoder_errors() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for error in [Int8SampledSentimentError::Shared(SampledSentimentError::Cancelled(cause)),
            Int8SampledSentimentError::Native(Int8JsonError::Native(StrictInt8Error::Cancelled(cause))),
            Int8SampledSentimentError::Native(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause)))] {
            let failure = task_failure(error);
            assert_eq!(failure.fault, BatchFault::cancelled(cause)); assert!(failure.stop);
        }
    }
}
struct RefuseAdmission;
impl Int8SampledSentimentJobAdmission for RefuseAdmission {
    type Guard = ();
    fn admit(&mut self, _: &ExecutionIdentity, _: Int8Work, _: u64, _: u64, _: u64)
        -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn sealed_factory_and_guarded_processor_fit_existing_owned_runtime_boundary() {
    fn durable<T: DurableBatchProcessor>() {} fn send<T: Send + 'static>() {} fn sync<T: Sync>() {}
    durable::<Int8SampledSentimentJobProcessor<'static, 'static, RefuseAdmission>>();
    send::<Arc<Int8SampledSentimentJobPlanner>>(); sync::<Int8SampledSentimentJobPlanner>();
    send::<PreparedInt8SampledSentimentJob>();
    let _factory = Int8SampledSentimentJobPlanner::new;
    let _compact = Int8SampledSentimentJobPlanner::new_with_document_encoding;
    let _processor = Int8SampledSentimentJobProcessor::<RefuseAdmission>::new;
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-int8-sampled-job-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
#[derive(Default)]
struct Events { held: Cell<bool>, executions: Cell<usize>, serializations: Cell<usize>,
    releases: Cell<usize>, held_polls: Cell<usize> }
struct Hold(Rc<Events>);
impl Drop for Hold {
    fn drop(&mut self) { assert!(self.0.held.replace(false)); self.0.releases.set(self.0.releases.get() + 1); }
}
struct Datum { item: String, draw: Sha256Digest, events: Rc<Events> }
impl Serialize for Datum {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get()); self.events.serializations.set(self.events.serializations.get() + 1);
        ("storage-fixture-not-native-inference", &self.item, self.draw).serialize(serializer)
    }
}
struct Processor {
    identity: ExecutionIdentity,
    recipe: Int8SampledSentimentJobRecipe,
    events: Rc<Events>,
    fail: Option<BatchFault>,
}
impl Processor {
    fn new(cap: Int8SampledSentimentJobLimits, events: Rc<Events>) -> Self {
        Self { identity: crate::jobs::tests::identity(), recipe: recipe(cap), events, fail: None }
    }
}
impl BatchProcessor for Processor {
    type Args = SampledSentimentBatchArgs;
    type Prepared = BatchDocument<Self::Args>;
    type Output = GuardedOutput<Datum, Hold>;
    fn prepare(&mut self, _: Self::Prepared) -> Result<Self::Prepared, BatchItemFailure> { panic!("missing control") }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, doc: Self::Prepared, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> { checkpoint(control)?; Ok(doc) }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork {
        BatchWork { forward_positions: work(1).forward_positions, projected_logits: work(1).projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> { panic!("missing durable context") }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        self.events.executions.set(self.events.executions.get() + 1);
        if let Some(error) = self.fail { return Err(BatchItemFailure::fatal(error)); }
        assert!(!self.events.held.replace(true));
        let draw = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            self.recipe.sampling.effective_seed, &prepared.id, &prepared.text)).unwrap());
        Ok(GuardedOutput::new(Datum { item: prepared.id, draw, events: Rc::clone(&self.events) },
            Hold(Rc::clone(&self.events))))
    }
}
impl DurableBatchProcessor for Processor {
    type Recipe = Int8SampledSentimentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, _: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        Ok(JobWork { model: work(1), mask_node_visits: 5 })
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { task().max_output_bytes }
}

#[test]
fn retained_resume_skips_committed_items_and_keeps_guard_through_acknowledgement() {
    let root = Root::new(); let full = Root::new(); let input = inputs(); let cap = native();
    let retention = InputRetentionLimits { max_spool_bytes: 65536 };
    let events = Rc::new(Events::default()); let make = || Processor::new(cap, Rc::clone(&events));
    let mut run = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, &input, make(), &mut Continue).unwrap();
    assert_eq!(run.step(&mut Continue).unwrap().unwrap().committed, 1); drop(run);
    let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let done = resumed.run(&mut Continue).unwrap();
    assert_eq!((done.committed, done.attempts), (2, 2));
    assert_eq!(done.reserved_work, JobWork { model: work(2), mask_node_visits: 10 });
    assert_eq!(events.executions.get(), 2); assert_eq!(events.releases.get(), 2);
    assert!(!events.held.get()); assert!(events.serializations.get() >= 4);
    resumed.verify(&mut Continue).unwrap(); resumed.materialize_ordered(&mut Continue).unwrap();
    let mut baseline = JobRunner::create(&full.0, key(), JobId([9; 16]), limits(cap), &input,
        Processor::new(cap, Rc::new(Events::default())), &mut Continue).unwrap();
    baseline.run(&mut Continue).unwrap(); baseline.materialize_ordered(&mut Continue).unwrap();
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(),
        fs::read(full.0.join("materialized.ndjson")).unwrap());
}

#[test]
fn failures_and_cancellation_cannot_refund_any_durable_work_axis_on_resume() {
    for axis in 0..6 {
        let mut cap = native(); cap.max_item_work = work(1); cap.masks.max_visits_per_item = 5;
        if axis < 5 { *work_axis(&mut cap.max_model_work, axis) = *work_axis(&mut work(1), axis); }
        else { cap.masks.max_visits_per_run = 5; }
        let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
        let expected = if axis % 2 == 0 { BatchCode::Execution.into() }
            else { BatchFault::cancelled(DecodeCancellationKind::CostBudget) };
        let make = || Processor::new(cap, Rc::clone(&events));
        let mut processor = make(); processor.fail = Some(expected);
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(cap), &input,
            processor, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(expected));
        assert_eq!(run.progress().committed, 0);
        assert_eq!(run.progress().reserved_work, JobWork { model: work(1), mask_node_visits: 5 }); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
            make(), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1); assert_eq!(resumed.progress().attempts, 1);
        assert_eq!(resumed.progress().reserved_work, JobWork { model: work(1), mask_node_visits: 5 });
    }
}

struct CancelAfterSync(Rc<Events>);
impl DecodeStepControl for CancelAfterSync {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if !self.0.held.get() { return None; }
        let n = self.0.held_polls.get() + 1; self.0.held_polls.set(n);
        (n == 2).then_some(DecodeCancellationKind::PollQuota)
    }
}
#[test]
fn changed_seed_refuses_retained_resume_before_tail_repair() {
    let root = Root::new(); let input = inputs(); let cap = native();
    let retention = InputRetentionLimits { max_spool_bytes: 65536 };
    let events = Rc::new(Events::default()); let make = || Processor::new(cap, Rc::clone(&events));
    let mut control = CancelAfterSync(Rc::clone(&events));
    let mut run = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, &input, make(), &mut control).unwrap();
    assert_eq!(error(run.step(&mut control)),
        JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    assert_eq!(run.progress().committed, 0); assert_eq!(events.releases.get(), 1); drop(run);
    let mut changed = make(); changed.recipe.sampling.effective_seed[0] += 1;
    assert_eq!(error(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, changed, TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(error(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    let done = resumed.run(&mut Continue).unwrap();
    assert_eq!((done.committed, done.attempts), (2, 3));
    assert_eq!(done.reserved_work, JobWork { model: work(3), mask_node_visits: 15 });
}
