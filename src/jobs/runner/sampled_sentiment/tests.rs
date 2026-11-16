//! Real owned-job authentication, debit and replay with private synthetic
//! execution. These fixtures exercise storage, never claim native quality.

use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::sentiment::sampled::SampledSentimentAdmission,
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, InputRetentionLimits, JobContract, JobError, JobId, JobInput,
        JobSecret, MismatchField, TailPolicy, runner::{JobRunError, JobRunner}},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        constrained_int4::json_sampling_workspace_bytes, portable_int8::{ProjectionWork,
            int4::convert::GROUPED_INT4_RECIPE}, strict_int4::STRICT_INT4_EXECUTION},
    tokenizer::untrusted::SafeBpeLimits,
};
use serde::{Serialize, Serializer};
use std::{cell::Cell, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::atomic::{AtomicU64, Ordering}};

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
    SampledSentimentLimits { max_result_bytes: 65536, ..Default::default() }
}
fn work(units: u64) -> Int4Work {
    Int4Work { forward_positions: 40 * units, projected_logits: 50 * units, attention_pairs: 60 * units,
        projections: ProjectionWork { dot_products: 70 * units, multiply_accumulates: 80 * units } }
}
fn native() -> SampledSentimentBatchLimits {
    SampledSentimentBatchLimits { max_item_work: work(2), max_model_work: work(10),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 100,
            checkpoint_interval_nodes: 8 }, max_visits_per_item: 10, max_visits_per_run: 50 },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn limits(native: SampledSentimentBatchLimits) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 8,
        max_work: JobWork::from_int4(native.max_model_work, native.masks.max_visits_per_run) }
}
fn recipe(native: SampledSentimentBatchLimits) -> SampledSentimentJobRecipe {
    SampledSentimentJobRecipe::new(32, task(), options(), planning(), native,
        UntrustedEncoding::ByteFallback, limits(native)).unwrap()
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"sampled-job-storage-fixture-not-a-model");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: digest,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, 32_usize)).unwrap()),
        tokenizer_digest: digest, template_digest: digest, task_spec: "sentiment-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: JSON_SAMPLER_VERSION.to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: digest,
        decision_policy_digest: digest, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Private source A","task_args":{"axes":["valence"]}}"#;
    let b = br#"{"id":"beta","text":"Private source B","task_args":null}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn freeze(recipe: &SampledSentimentJobRecipe) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]), execution: &identity(),
        recipe, limits: recipe.job }, inputs(), &mut Continue).unwrap()
}
fn distinct(changed: &SampledSentimentJobRecipe) {
    assert_eq!(freeze(&recipe(native())).binding.compare(&freeze(changed).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
}
fn work_axis(work: &mut Int4Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates }
}

#[test]
fn every_sample_option_and_item_address_version_changes_authenticated_recipe() {
    for axis in 0..7 {
        let mut changed = options();
        match axis { 0 => changed.effective_seed[31] += 1, 1 => changed.samples += 1,
            2 => changed.temperature_milli += 1, 3 => changed.top_k = Some(10),
            4 => changed.top_p_ppm -= 1, 5 => changed.max_sample_tokens += 1,
            _ => changed.max_justification_chars += 1 }
        distinct(&SampledSentimentJobRecipe::new(32, task(), changed, planning(), native(),
            UntrustedEncoding::ByteFallback, limits(native())).unwrap());
    }
    let mut changed = recipe(native()); changed.item_address_version = "other-address"; distinct(&changed);
    let mut changed = recipe(native()); changed.aggregation_version = "other-aggregation"; distinct(&changed);
    let mut changed = recipe(native()); changed.sampler_version = "other-sampler"; distinct(&changed);
    let mut changed = recipe(native()); changed.default_axes.reverse(); distinct(&changed);
    let r = serde_json::to_value(recipe(native())).unwrap();
    assert_eq!(r["item_address_version"], SAMPLED_SENTIMENT_ITEM_ADDRESS_VERSION);
    assert_eq!(r["sampling"]["effective_seed"], serde_json::to_value([37_u8; 32]).unwrap());
}

#[test]
fn all_planning_compiler_budget_and_native_caps_bind_resume() {
    for axis in 0..11 {
        let mut changed = planning();
        match axis { 0 => changed.max_input_bytes += 1, 1 => changed.max_total_prompt_tokens += 1,
            2 => changed.max_sample_json_bytes += 1, 3 => changed.max_result_bytes += 1,
            4 => changed.compile.max_schema_bytes += 1, 5 => changed.compile.max_string_bytes += 1,
            6 => changed.compile.max_array_items += 1, 7 => changed.compile.max_output_bytes += 1,
            8 => changed.compile.max_states += 1, 9 => changed.compile.max_transitions += 1,
            _ => changed.compile.max_mask_bytes += 1 }
        distinct(&SampledSentimentJobRecipe::new(32, task(), options(), changed, native(),
            UntrustedEncoding::ByteFallback, limits(native())).unwrap());
    }
    for axis in 0..5 {
        let mut changed = task();
        match axis { 0 => changed.max_input_tokens += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_output_bytes += 1, 3 => changed.max_grammar_states += 1,
            _ => changed.max_kv_bytes += 1 }
        distinct(&SampledSentimentJobRecipe::new(32, changed, options(), planning(), native(),
            UntrustedEncoding::ByteFallback, limits(native())).unwrap());
    }
    for axis in 0..15 {
        let mut changed = native();
        match axis { 0..=4 => *work_axis(&mut changed.max_item_work, axis) += 1,
            5..=9 => *work_axis(&mut changed.max_model_work, axis - 5) += 1,
            10 => changed.masks.per_mask.max_trie_node_visits += 1,
            11 => changed.masks.per_mask.checkpoint_interval_nodes += 1,
            12 => changed.masks.max_visits_per_item += 1,
            13 => changed.masks.max_visits_per_run += 1, _ => changed.max_sampler_bytes += 1 }
        // Recipe comparison is independent of the separately bound limits.
        distinct(&recipe(changed));
    }
    distinct(&SampledSentimentJobRecipe::new(128, task(), options(), planning(), native(),
        UntrustedEncoding::ByteFallback, limits(native())).unwrap());
}

fn compact() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 4 * 1024 * 1024, max_pair_bytes: 64 * 1024 * 1024 })
}
#[test]
fn compact_policy_is_exact_private_authority_including_each_work_cap() {
    let make = |encoding| SampledSentimentJobRecipe::new(32, task(), options(), planning(), native(),
        encoding, limits(native())).unwrap();
    let original = freeze(&make(compact()));
    for axis in 0..4 {
        let UntrustedEncoding::SafeBpeV1(mut changed) = compact() else { unreachable!() };
        match axis { 0 => changed.max_input_bytes += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_pair_checks += 1, _ => changed.max_pair_bytes += 1 }
        assert_eq!(original.binding.compare(&freeze(&make(UntrustedEncoding::SafeBpeV1(changed))).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    assert_eq!(original.binding.compare(&freeze(&make(UntrustedEncoding::ByteFallback)).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
}

#[test]
fn exact_lifetime_limits_refuse_all_six_axis_mismatches_and_oversized_result() {
    let r = recipe(native()); let exact = limits(native()); r.check_job_limits(exact).unwrap();
    for axis in 0..6 {
        for increase in [false, true] {
            let mut changed = exact;
            if axis < 5 {
                let mut work = changed.max_work.int4_model_work(); let n = work_axis(&mut work, axis);
                if increase { *n += 1; } else { *n -= 1; }
                changed.max_work = JobWork::from_int4(work, changed.max_work.mask_node_visits);
            } else if increase { changed.max_work.mask_node_visits += 1; }
            else { changed.max_work.mask_node_visits -= 1; }
            assert!(r.check_job_limits(changed).is_err());
            assert!(SampledSentimentJobRecipe::new(32, task(), options(), planning(), native(),
                UntrustedEncoding::ByteFallback, changed).is_err());
        }
    }
    let mut smaller = exact; smaller.max_result_bytes -= 1;
    assert!(SampledSentimentJobRecipe::new(32, task(), options(), planning(), native(),
        UntrustedEncoding::ByteFallback, smaller).is_err());
    let mut narrow = planning(); narrow.max_result_bytes = smaller.max_result_bytes as u64;
    SampledSentimentJobRecipe::new(32, task(), options(), narrow, native(),
        UntrustedEncoding::ByteFallback, smaller).unwrap();
    let mut changed = exact; changed.max_attempts += 1; assert!(r.check_job_limits(changed).is_err());
}

struct RefuseAdmission;
impl SampledSentimentBatchAdmission for RefuseAdmission {
    type Guard = ();
    fn admit(&mut self, _: SampledSentimentAdmission<'_>) -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn public_adapter_is_owned_send_and_implements_the_existing_durable_boundary() {
    fn durable<T: DurableBatchProcessor>() {} fn send<T: Send + 'static>() {}
    durable::<SampledSentimentJobProcessor<'static, 'static, RefuseAdmission>>();
    send::<SampledSentimentJobPlanner>();
    let _constructor = SampledSentimentJobPlanner::new;
    let _serial = SampledSentimentJobProcessor::<RefuseAdmission>::new;
    let _layer_major = SampledSentimentJobProcessor::<RefuseAdmission>::new_layer_major;
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-sampled-job-{}-{}",
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
struct Processor { identity: ExecutionIdentity, recipe: SampledSentimentJobRecipe,
    events: Rc<Events>, fail: Option<BatchFault> }
impl Processor {
    fn new(cap: SampledSentimentBatchLimits, events: Rc<Events>) -> Self {
        Self { identity: identity(), recipe: recipe(cap), events, fail: None }
    }
}
impl BatchProcessor for Processor {
    type Args = SampledSentimentBatchArgs;
    type Prepared = BatchDocument<Self::Args>;
    type Output = GuardedOutput<Datum, Hold>;
    fn prepare(&mut self, _: Self::Prepared) -> Result<Self::Prepared, BatchItemFailure> { panic!("missing control") }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, doc: Self::Prepared, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        if let Some(cause) = control.prefill_checkpoint(0) { return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))); }
        Ok(doc)
    }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork {
        BatchWork { forward_positions: work(1).forward_positions, projected_logits: work(1).projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("missing durable context")
    }
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
    type Recipe = SampledSentimentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, _: &Self::Prepared) -> Result<JobWork, BatchItemFailure> { Ok(JobWork::from_int4(work(1), 5)) }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { task().max_output_bytes }
}
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(error) => error, Ok(_) => panic!("expected refusal") }
}

#[test]
fn retained_resume_skips_committed_items_preserves_draws_and_holds_results_through_sync() {
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
    assert_eq!(done.reserved_work, JobWork::from_int4(work(2), 10));
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
fn failed_and_cancelled_generations_retain_all_six_lifetime_debits_on_resume() {
    for axis in 0..6 {
        let mut cap = native(); cap.max_item_work = work(1); cap.masks.max_visits_per_item = 5;
        if axis < 5 { *work_axis(&mut cap.max_model_work, axis) = *work_axis(&mut work(1), axis); }
        else { cap.masks.max_visits_per_run = 5; }
        let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
        let expected = if axis % 2 == 0 { BatchCode::Execution.into() }
            else { BatchFault::cancelled(DecodeCancellationKind::CostBudget) };
        let mut p = Processor::new(cap, Rc::clone(&events)); p.fail = Some(expected);
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(cap), &input, p, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(expected));
        assert_eq!(run.progress().committed, 0);
        assert_eq!(run.progress().reserved_work, JobWork::from_int4(work(1), 5)); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
            Processor::new(cap, Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1); assert_eq!(resumed.progress().attempts, 1);
        assert_eq!(resumed.progress().reserved_work, JobWork::from_int4(work(1), 5));
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
fn changed_seed_refuses_retained_resume_before_repair_and_original_seed_recovers() {
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
    assert_eq!(done.reserved_work, JobWork::from_int4(work(3), 15));
}
