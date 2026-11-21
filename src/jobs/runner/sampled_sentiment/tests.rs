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

mod selected;

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
    let _layer_major = SampledSentimentJobPlanner::with_layer_major_prefill;
    let _reused = SampledSentimentJobPlanner::with_reused_prefill;
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
    fn with_reused_prefill(mut self) -> Self {
        self.recipe = self.recipe.with_reused_prefill().unwrap();
        self
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
    for reuse in [false, true] {
    for axis in 0..6 {
        let mut cap = native(); cap.max_item_work = work(1); cap.masks.max_visits_per_item = 5;
        if axis < 5 { *work_axis(&mut cap.max_model_work, axis) = *work_axis(&mut work(1), axis); }
        else { cap.masks.max_visits_per_run = 5; }
        let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
        let expected = if axis % 2 == 0 { BatchCode::Execution.into() }
            else { BatchFault::cancelled(DecodeCancellationKind::CostBudget) };
        let make = || {
            let processor = Processor::new(cap, Rc::clone(&events));
            if reuse { processor.with_reused_prefill() } else { processor }
        };
        let mut p = make(); p.fail = Some(expected);
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(cap), &input, p, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(expected));
        assert_eq!(run.progress().committed, 0);
        assert_eq!(run.progress().reserved_work, JobWork::from_int4(work(1), 5)); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
            make(), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1); assert_eq!(resumed.progress().attempts, 1);
        assert_eq!(resumed.progress().reserved_work, JobWork::from_int4(work(1), 5));
    }
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

#[test]
fn changing_reuse_policy_refuses_before_uncommitted_tail_repair_or_execution() {
    for reuse in [false, true] {
        let root = Root::new(); let input = inputs(); let cap = native();
        let retention = InputRetentionLimits { max_spool_bytes: 65536 };
        let events = Rc::new(Events::default());
        let make = |selected| {
            let processor = Processor::new(cap, Rc::clone(&events));
            if selected { processor.with_reused_prefill() } else { processor }
        };
        let mut control = CancelAfterSync(Rc::clone(&events));
        let mut run = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
            retention, &input, make(reuse), &mut control).unwrap();
        assert_eq!(error(run.step(&mut control)),
            JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
        assert_eq!(run.progress().reserved_work, JobWork::from_int4(work(1), 5));
        assert_eq!(run.progress().committed, 0); drop(run);
        assert_eq!(error(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
            retention, make(!reuse), TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
        assert_eq!(events.executions.get(), 1);
        // A rejected policy change cannot repair the original job's spool.
        assert_eq!(error(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
            retention, make(reuse), TailPolicy::Refuse, &mut Continue)),
            JobRunError::Storage(JobError::UncommittedTail));
        let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
            retention, make(reuse), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
        let done = resumed.run(&mut Continue).unwrap();
        assert_eq!((done.committed, done.attempts), (2, 3));
        assert_eq!(done.reserved_work, JobWork::from_int4(work(3), 15));
        assert_eq!(events.executions.get(), 3); assert_eq!(events.releases.get(), 3);
        resumed.verify(&mut Continue).unwrap();
    }
}

#[test]
fn completed_reused_job_resume_performs_no_new_draws_or_work_debits() {
    let root = Root::new(); let full = Root::new(); let input = inputs(); let cap = native();
    let retention = InputRetentionLimits { max_spool_bytes: 65536 };
    let events = Rc::new(Events::default());
    let make = || Processor::new(cap, Rc::clone(&events)).with_reused_prefill();
    let mut run = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, &input, make(), &mut Continue).unwrap();
    assert_eq!(run.step(&mut Continue).unwrap().unwrap().committed, 1); drop(run);
    let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let complete = resumed.run(&mut Continue).unwrap();
    assert_eq!((complete.committed, complete.attempts), (2, 2));
    assert_eq!(complete.reserved_work, JobWork::from_int4(work(2), 10));
    resumed.materialize_ordered(&mut Continue).unwrap(); drop(resumed);
    let mut completed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(cap),
        retention, make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let unchanged = completed.run(&mut Continue).unwrap();
    assert_eq!((unchanged.committed, unchanged.attempts), (2, 2));
    assert_eq!(unchanged.reserved_work, complete.reserved_work);
    assert_eq!(events.executions.get(), 2); assert_eq!(events.releases.get(), 2);
    completed.verify(&mut Continue).unwrap(); drop(completed);
    let mut baseline = JobRunner::create(&full.0, key(), JobId([9; 16]), limits(cap), &input,
        Processor::new(cap, Rc::new(Events::default())).with_reused_prefill(), &mut Continue).unwrap();
    baseline.run(&mut Continue).unwrap(); baseline.materialize_ordered(&mut Continue).unwrap();
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(),
        fs::read(full.0.join("materialized.ndjson")).unwrap());
}

#[test]
fn reuse_versions_are_authenticated_without_changing_cold_recipe_shape_or_seed() {
    let cold = recipe(native());
    let reused = recipe(native()).with_reused_prefill().unwrap();
    assert!(!cold.reuses_prefill()); assert!(reused.reuses_prefill());
    distinct(&reused);
    let mut expected = serde_json::to_value(&cold).unwrap();
    expected["execution"] = serde_json::json!(SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION);
    expected["native_execution"] = serde_json::json!(INT4_JSON_SAMPLED_REUSED_PREFILL_EXECUTION);
    assert_eq!(serde_json::to_value(&reused).unwrap(), expected);
    assert_eq!(canonjson::canonical_bytes(&reused).unwrap(),
        canonjson::canonical_bytes(&recipe(native()).with_reused_prefill().unwrap().with_reused_prefill().unwrap()).unwrap());
    for rows in [1, 4, 64] {
        let prefill = Int4PrefillLimits { max_batch_rows: rows,
            max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
        let first = recipe(native()).with_prefill(prefill).unwrap().with_reused_prefill().unwrap();
        let second = recipe(native()).with_reused_prefill().unwrap().with_prefill(prefill).unwrap();
        assert_eq!(canonjson::canonical_bytes(&first).unwrap(), canonjson::canonical_bytes(&second).unwrap());
        assert_eq!(freeze(&first).binding.compare(&freeze(&recipe(native()).with_prefill(prefill).unwrap()).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
        let changed = Int4PrefillLimits { max_extra_scratch_bytes: prefill.max_extra_scratch_bytes + 1, ..prefill };
        let changed = recipe(native()).with_reused_prefill().unwrap().with_prefill(changed).unwrap();
        assert_eq!(freeze(&first).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}

#[test]
fn prompt_schedule_rows_workspace_and_backend_version_are_authenticated() {
    let original = recipe(native());
    for rows in [1, 4, 64] {
        let prefill = Int4PrefillLimits { max_batch_rows: rows,
            max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
        let selected = recipe(native()).with_prefill(prefill).unwrap();
        distinct(&selected);
        assert_eq!(selected.prefill_limits().unwrap().max_batch_rows, rows);
        assert_eq!(selected.prefill_limits().unwrap().max_extra_scratch_bytes, prefill.max_extra_scratch_bytes);
        let mut changed = prefill; changed.max_extra_scratch_bytes += 1;
        let changed = recipe(native()).with_prefill(changed).unwrap();
        assert_eq!(freeze(&selected).binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    let mut changed = recipe(native()); changed.native_execution = "other-native-execution"; distinct(&changed);
    let mut changed = recipe(native()); changed.sample_address_stride += 1; distinct(&changed);
    assert!(original.prefill_limits().is_none());
    for rows in [0, 65] {
        assert!(recipe(native()).with_prefill(Int4PrefillLimits {
            max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX }).is_err());
    }
    assert!(recipe(native()).with_prefill(Int4PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() - 1 }).is_err());
}

#[test]
fn paired_execution_versions_reject_mixed_cold_and_reused_schedules() {
    let cold = recipe(native());
    let reused = recipe(native()).with_reused_prefill().unwrap();
    cold.check_job_limits(cold.job).unwrap(); reused.check_job_limits(reused.job).unwrap();
    assert_eq!(cold.version, 2); assert_eq!(reused.version, cold.version);
    for (execution, native_execution) in [
        (SAMPLED_SENTIMENT_EXECUTION, INT4_JSON_SAMPLED_REUSED_PREFILL_EXECUTION),
        (SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION, INT4_JSON_SAMPLED_EXECUTION),
    ] {
        let mut mixed = recipe(native());
        mixed.execution = execution; mixed.native_execution = native_execution;
        assert!(mixed.check_job_limits(mixed.job).is_err(), "mixed schedules cannot authorize a processor");
    }
}

#[test]
fn storage_limits_and_record_bounds_are_part_of_the_immutable_recipe() {
    for axis in 0..10 {
        let mut changed = limits(native());
        match axis { 0 => changed.max_items += 1, 1 => changed.max_id_bytes += 1,
            2 => changed.max_input_bytes_per_item += 1, 3 => changed.max_snapshot_bytes += 1,
            4 => changed.max_result_bytes += 1, 5 => changed.max_spool_bytes += 1,
            6 => changed.max_materialized_bytes += 1, 7 => changed.max_journal_bytes += 4096,
            8 => changed.max_attempts += 1, _ => changed.max_id_bytes = 256 }
        distinct(&SampledSentimentJobRecipe::new(32, task(), options(), planning(), native(),
            UntrustedEncoding::ByteFallback, changed).unwrap());
    }
    let mut oversized = limits(native()); oversized.max_id_bytes = 257;
    assert!(SampledSentimentJobRecipe::new(32, task(), options(), planning(), native(),
        UntrustedEncoding::ByteFallback, oversized).is_err());
    let mut narrow = limits(native()); narrow.max_input_bytes_per_item = planning().max_input_bytes - 1;
    assert!(SampledSentimentJobRecipe::new(32, task(), options(), planning(), native(),
        UntrustedEncoding::ByteFallback, narrow).is_err());
}

#[test]
fn actual_model_bound_factory_quotes_recipe_and_uses_the_same_native_compiler() {
    use crate::native_engine::strict_int4::weights::projection_fixture;
    use crate::batch::sentiment::sampled::sampled_sentiment_batch_preparation_bytes;
    let (model, _) = projection_fixture(32);
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
    let make = |encoding: UntrustedEncoding| {
        let compiler = SampledSentimentPlanner::for_model_with_encoding(model.identity(),
            options(), task(), planning(), encoding).unwrap();
        SampledSentimentJobPlanner::new(compiler, Arc::clone(&vocabulary), native(), limits(native())).unwrap()
    };
    let byte = make(UntrustedEncoding::ByteFallback);
    assert_eq!(byte.preparation_requirement_bytes(),
        sampled_sentiment_batch_preparation_bytes(task(), planning(), &UntrustedEncoding::ByteFallback).unwrap()
            + SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64);
    assert_eq!(byte.max_result_bytes(), task().max_output_bytes);
    assert_eq!(byte.base_execution_identity().task_spec, "sentiment-v1");
    assert_eq!(byte.base_execution_identity().sampler_version, JSON_SAMPLER_VERSION);
    assert_eq!(byte.group_width(), 32); assert_eq!(byte.task_budget(), task());
    assert_eq!(byte.job_limits(), limits(native())); byte.check_job_limits(limits(native())).unwrap();
    assert!(byte.prefill_limits().is_none());
    assert!(!byte.reuses_prefill());
    let compact = make(compact());
    assert_eq!(compact.document_encoding(), &self::compact());
    assert_eq!(compact.preparation_requirement_bytes(),
        sampled_sentiment_job_preparation_bytes(task(), planning(), &self::compact()).unwrap());
    assert!(compact.preparation_requirement_bytes() > byte.preparation_requirement_bytes());
    let identity = compact.base_execution_identity().clone();
    let before = canonjson::canonical_bytes(compact.job_recipe()).unwrap();
    let selected = compact.with_layer_major_prefill(Int4PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() }).unwrap();
    assert_eq!(selected.base_execution_identity(), &identity);
    assert_ne!(canonjson::canonical_bytes(selected.job_recipe()).unwrap(), before);
    assert_eq!(selected.prefill_limits().unwrap().max_batch_rows, 4);
    assert!(!selected.reuses_prefill());
    let original_task = selected.task_budget(); let original_work = selected.native_limits();
    let original_preparation = selected.preparation_requirement_bytes();
    let reused = selected.with_reused_prefill().unwrap();
    assert!(reused.reuses_prefill()); assert_eq!(reused.prefill_limits().unwrap().max_batch_rows, 4);
    assert_eq!(reused.base_execution_identity(), &identity);
    assert_eq!(reused.task_budget(), original_task);
    assert_eq!(canonjson::canonical_bytes(&NativeRecipe::from(reused.native_limits())).unwrap(),
        canonjson::canonical_bytes(&NativeRecipe::from(original_work)).unwrap());
    assert_eq!(reused.preparation_requirement_bytes(), original_preparation);
    assert_eq!(reused.job_limits(), limits(native())); reused.check_job_limits(limits(native())).unwrap();
}

#[test]
fn exact_per_record_axes_and_tighter_budget_are_authenticated_before_any_resume() {
    let root = Root::new(); let input = inputs(); let cap = native();
    let events = Rc::new(Events::default()); let make = || Processor::new(cap, Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(cap), &input, make(), &mut Continue).unwrap();
    run.step(&mut Continue).unwrap(); drop(run);
    for axes in [vec!["valence"], vec!["arousal", "dominance"]] {
        let mut smaller = task(); smaller.max_output_tokens -= 1; smaller.max_input_tokens -= 1;
        let changed = serde_json::to_vec(&serde_json::json!({
            "id": "beta", "text": "Private source B",
            "task_args": { "axes": axes, "budget": smaller }
        })).unwrap();
        let replaced = [JobInput { id: input[0].id, original: input[0].original, normalized: input[0].normalized },
            JobInput { id: "beta", original: &changed, normalized: &changed }];
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap),
            &replaced, make(), TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Population)));
        assert_eq!(events.executions.get(), 1);
    }
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    assert_eq!(resumed.run(&mut Continue).unwrap().committed, 2);
    assert_eq!(events.executions.get(), 2);
}

#[test]
fn invalid_task_data_is_rejected_for_the_complete_population_before_storage() {
    for invalid in [r#"{"id":"beta","text":"private","task_args":{"seed":"replacement"}}"#,
        r#"{"id":"beta","text":"private","task_args":{"samples":1}}"#,
        r#"{"id":"beta","text":"private","task_args":{"axes":["unknown"]}}"#,
        r#"{"id":"beta","text":"private","task_args":{"budget":{}}}"#] {
        let root = Root::new(); let input = inputs();
        let replaced = [JobInput { id: input[0].id, original: input[0].original, normalized: input[0].normalized },
            JobInput { id: "beta", original: invalid.as_bytes(), normalized: invalid.as_bytes() }];
        let events = Rc::new(Events::default());
        assert_eq!(error(JobRunner::create(&root.0, key(), JobId([9; 16]), limits(native()),
            &replaced, Processor::new(native(), Rc::clone(&events)), &mut Continue)), JobRunError::InvalidEnvelope);
        assert_eq!(events.executions.get(), 0);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }
}
