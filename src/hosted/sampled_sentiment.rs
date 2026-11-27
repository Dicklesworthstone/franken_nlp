//! Complete independent INT8 score/justification bundles on one charged host.
//! Every sample has fresh logical KV and its own addressed draws. Native,
//! sampler, preparation and output ownership survive physical completion.

mod long;
pub use long::Int8SampledSentimentMapHostConfig;

use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{
        constrained::JsonWorkBudget,
        constrained_int8::{Int8JsonBudget, json_sampling_workspace_bytes},
        simd::Int8Kernel,
        strict_int8::{Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits},
    },
    tasks::{ir::TaskBudget, sentiment::sampled::{
        SampledSentimentVocabulary,
        quantized::{PreparedInt8SampledSentiment, Int8SampledSentimentRun,
            Int8SampledSentimentError, INT8_SAMPLED_SENTIMENT_EXECUTION},
    }},
};

/// Nonrenewable limits for the complete axis-by-sample bundle. Preparation
/// covers transferred plans and vocabulary in addition to any caller-owned
/// preparation reservation; it is modeled admission, not measured RSS.
#[derive(Clone, Copy, Debug)]
pub struct Int8SampledSentimentHostLimits {
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int8Work,
    pub max_sampler_bytes: u64,
    pub mask_limits: MaskWorkLimits,
    pub max_mask_node_visits: u64,
}

impl NlpEngine {
    pub fn execute_int8_sampled_sentiment(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SampledSentimentRun>, HostedError> {
        self.execute_int8_sampled_sentiment_scheduled(model, prepared, vocabulary,
            limits, None, cancellation)
    }

    /// Only prompt scheduling changes: each sample still begins with empty KV,
    /// scores the complete vocabulary and consumes its independently addressed
    /// random sequence. The declared prompt workspace is charged before dispatch.
    pub fn execute_int8_sampled_sentiment_layer_major(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, prefill: Int8PrefillLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8SampledSentimentRun>, HostedError> {
        self.execute_int8_sampled_sentiment_scheduled(model, prepared, vocabulary,
            limits, Some(prefill), cancellation)
    }

    /// Reuse one admitted row team across the entire independent sample
    /// bundle. Width is physical metadata and never enters the addressed RNG.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_sampled_sentiment_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, kernel: Int8Kernel, compute_threads: usize,
        cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SampledSentimentRun>, HostedError> {
        self.execute_int8_sampled_sentiment_team(model, prepared, vocabulary,
            limits, kernel, compute_threads, None, cancellation)
    }

    /// The same fixed workers execute prompt morsels and subsequent sampled
    /// tokens. Coordinator, child, sampler and output storage are all charged.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_sampled_sentiment_layer_major_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SampledSentimentRun>, HostedError> {
        self.execute_int8_sampled_sentiment_team(model, prepared, vocabulary,
            limits, kernel, compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int8_sampled_sentiment_scheduled(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, prefill: Option<Int8PrefillLimits>,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8SampledSentimentRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let required = requirements(limits.native)?;
        let work = prepared.planned_work();
        let scratch = validate(limits, prepared.task_budget(), prepared.required_context(),
            required, work, prepared.preparation_requirement_bytes())?;
        let scratch = super::scored::scoring_scratch(scratch, prefill)?;
        let budget = execution_budget(work, required.kv_bytes, limits);
        prepared.complete_mask_charge(&vocabulary, budget, limits.max_sampler_bytes)
            .map_err(HostedError::SampledSentiment)?;
        let lease = self.resources().acquire_lease();
        let output = output_claim(&lease, prepared.max_result_bytes(),
            u64::from(prepared.task_budget().max_output_tokens))?;
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers,
            limits.preparation_reserve_bytes)?, || Ok((prepared, vocabulary)))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let input = input;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required))
                .map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let (prepared, vocabulary) = &input.value;
                execute_prepared(prepared, vocabulary, &mut engine.value, budget,
                    limits.max_sampler_bytes, prefill, control)
            })?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int8_sampled_sentiment_team(&self, model: &ResidentInt8,
        prepared: PreparedInt8SampledSentiment, vocabulary: Arc<SampledSentimentVocabulary>,
        limits: Int8SampledSentimentHostLimits, kernel: Int8Kernel, width: usize,
        prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SampledSentimentRun>, HostedError> {
        dispatch::preflight_int8_team(self, limits.native.run, width)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let native = requirements(limits.native)?;
        let scratch = validate(limits, prepared.task_budget(), prepared.required_context(),
            native, prepared.planned_work(), prepared.preparation_requirement_bytes())?;
        let (scratch, max_batch_rows) = team_scratch(scratch, width, prefill)?;
        let budget = execution_budget(prepared.planned_work(), native.kv_bytes, limits);
        prepared.complete_mask_charge(&vocabulary, budget, limits.max_sampler_bytes)
            .map_err(HostedError::SampledSentiment)?;
        let lease = self.resources().acquire_lease();
        let output = output_claim(&lease, prepared.max_result_bytes(),
            u64::from(prepared.task_budget().max_output_tokens))?;
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers,
            limits.preparation_reserve_bytes)?, || Ok((prepared, vocabulary)))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch)?;
        let admission = TeamAdmission { input, kv: Some(kv), scratch: Some(scratch), output: Some(output),
            _kv: None, _scratch: None, _lease: lease };
        dispatch::run_int8_prefill_team(self, limits.native.run, cancellation, model.clone(),
            width, kernel, max_batch_rows, admission, move |model, admission, control, team| {
                let mut engine = model.engine(limits.native.context_tokens, memory_budget(native))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission.commit_native()?;
                let (prepared, vocabulary) = &admission.input.value;
                let result = execute_prepared(prepared, vocabulary, &mut engine, budget,
                    limits.max_sampler_bytes, prefill, control)?;
                drop(engine);
                let compute = team.stats().into();
                admission.finish(result, compute)
            })
    }
}

/// The dispatcher owns this aggregate outside its scoped callback. Every
/// reservation, plan and vocabulary outlives physical child drain on errors.
struct TeamAdmission {
    input: Charged<(PreparedInt8SampledSentiment, Arc<SampledSentimentVocabulary>)>,
    kv: Option<Pending>, scratch: Option<Pending>, output: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>, _lease: EngineLease,
}
impl TeamAdmission {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::Limits("sampled INT8 KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::Limits("sampled INT8 scratch admission consumed"))?.commit()?);
        Ok(())
    }
    fn finish(&mut self, result: Int8SampledSentimentRun, compute: Int8ComputeReceipt)
        -> Result<HostedInt8TeamResult<Int8SampledSentimentRun>, HostedError> {
        let output = self.output.take().ok_or(HostedError::Limits("sampled INT8 output admission consumed"))?;
        match output.commit() {
            Ok(memory) => Ok(HostedInt8TeamResult::from_guarded(GuardedOutput::new(result, memory), compute)),
            Err(error) => { drop(result); Err(error) }
        }
    }
}

fn team_scratch(ordinary: u64, width: usize, prefill: Option<Int8PrefillLimits>)
    -> Result<(u64, usize), HostedError> {
    let scratch = super::scored::scoring_scratch(ordinary, prefill)?;
    let rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    // The dispatcher separately retains the child charge; it never replaces
    // the coordinator's native buffers, prompt morsels or sampler workspace.
    sum(&[scratch, dispatch::int8_prefill_team_workspace_bytes(width, rows)?])?;
    Ok((scratch, rows))
}

#[allow(clippy::too_many_arguments)]
fn execute_prepared(prepared: &PreparedInt8SampledSentiment, vocabulary: &SampledSentimentVocabulary,
    engine: &mut StrictInt8Engine<'_>, budget: Int8JsonBudget, max_sampler_bytes: u64,
    prefill: Option<Int8PrefillLimits>, control: &mut RunControl)
    -> Result<Int8SampledSentimentRun, HostedError> {
    let run = match prefill {
        None => prepared.execute_with_control(engine, prepared.execution_identity(), vocabulary,
            budget, max_sampler_bytes, control),
        Some(prefill) => prepared.execute_layer_major_with_control(engine, prepared.execution_identity(),
            vocabulary, budget, max_sampler_bytes, prefill, control),
    }.map_err(HostedError::SampledSentiment)?;
    if run.execution != INT8_SAMPLED_SENTIMENT_EXECUTION || !fits(run.model_work, prepared.planned_work())
        || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
        return Err(HostedError::Limits("sampled sentiment completion accounting"));
    }
    if let Some(cause) = crate::native_engine::decode::DecodeStepControl::checkpoint(control, 0) {
        return Err(HostedError::SampledSentiment(Int8SampledSentimentError::from(
            crate::tasks::sentiment::sampled::SampledSentimentError::Cancelled(cause))));
    }
    Ok(run)
}

fn execution_budget(work: Int8Work, kv_bytes: u64, limits: Int8SampledSentimentHostLimits)
    -> Int8JsonBudget {
    Int8JsonBudget { native: Int8RunBudget::exact(work),
        json: JsonWorkBudget { max_forward_positions: work.forward_positions,
            max_projected_logits: work.projected_logits, max_kv_bytes: kv_bytes,
            max_total_mask_node_visits: limits.max_mask_node_visits,
            mask_limits: limits.mask_limits } }
}
fn fits(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn validate(limits: Int8SampledSentimentHostLimits, task: TaskBudget, context: usize,
    required: Int8MemoryRequirement, work: Int8Work, preparation_bytes: u64)
    -> Result<u64, HostedError> {
    limits.native.run.validate()?;
    task.validate().map_err(|_| HostedError::Limits("sampled sentiment task budget"))?;
    let sampler = json_sampling_workspace_bytes()
        .map_err(Int8SampledSentimentError::from).map_err(HostedError::SampledSentiment)?;
    if preparation_bytes == 0 || preparation_bytes > limits.preparation_reserve_bytes
        || context == 0 || context > limits.native.context_tokens
        || required.kv_bytes > task.max_kv_bytes || limits.native.allocator_reserve_bytes == 0
        || work.forward_positions == 0 || work.projected_logits == 0
        || !fits(work, limits.max_model_work) || sampler > limits.max_sampler_bytes
        || limits.mask_limits.max_trie_node_visits == 0
        || limits.mask_limits.checkpoint_interval_nodes == 0 || limits.max_mask_node_visits == 0 {
        return Err(HostedError::Limits("sampled sentiment whole-bundle admission"));
    }
    sum(&[required.rope_bytes, required.scratch_payload_bound,
        limits.native.allocator_reserve_bytes, sampler])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Int8SampledSentimentHostLimits, TaskBudget, Int8MemoryRequirement, Int8Work) {
        let required = Int8MemoryRequirement::for_context(128).unwrap();
        let one = crate::native_engine::constrained_int8::planned_work(32, 64).unwrap();
        let work = one.checked_add(one).unwrap();
        let limits = Int8SampledSentimentHostLimits {
            native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 1 << 20,
                run: RunLimits { max_elapsed: Duration::from_secs(30),
                    max_checkpoints: 1000, cleanup_reserve_bytes: 65_536 } },
            preparation_reserve_bytes: 1 << 20, max_model_work: work,
            max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
            mask_limits: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 256 },
            max_mask_node_visits: 128_000,
        };
        let task = TaskBudget { max_input_tokens: 64, max_output_tokens: 128,
            max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: required.kv_bytes };
        (limits, task, required, work)
    }
    #[test]
    fn every_whole_bundle_resource_is_admitted_before_dispatch() {
        let (limits, task, required, work) = fixture();
        assert_eq!(validate(limits, task, 95, required, work, 1 << 20).unwrap(),
            required.rope_bytes + required.scratch_payload_bound + (1 << 20)
                + json_sampling_workspace_bytes().unwrap());
        for axis in 0..10 {
            let mut limits = limits;
            match axis {
                0 => limits.preparation_reserve_bytes -= 1,
                1 => limits.max_model_work.forward_positions -= 1,
                2 => limits.max_model_work.projected_logits -= 1,
                3 => limits.max_model_work.attention_pairs -= 1,
                4 => limits.max_model_work.projections.dot_products -= 1,
                5 => limits.max_model_work.projections.multiply_accumulates -= 1,
                6 => limits.max_sampler_bytes -= 1,
                7 => limits.mask_limits.max_trie_node_visits = 0,
                8 => limits.mask_limits.checkpoint_interval_nodes = 0,
                _ => limits.max_mask_node_visits = 0,
            }
            assert!(validate(limits, task, 95, required, work, 1 << 20).is_err());
        }
    }
    #[test]
    fn independent_samples_share_capacity_but_keep_aggregate_work() {
        let (limits, mut task, required, work) = fixture();
        assert!(work.forward_positions > limits.native.context_tokens as u64);
        validate(limits, task, 95, required, work, 1 << 20).unwrap();
        assert!(validate(limits, task, 129, required, work, 1 << 20).is_err());
        task.max_kv_bytes -= 1;
        assert!(validate(limits, task, 95, required, work, 1 << 20).is_err());
    }
    #[test]
    fn sampled_task_errors_preserve_cancellation_without_private_diagnostics() {
        let cause = crate::native_engine::decode::DecodeCancellationKind::Deadline;
        let error = HostedError::SampledSentiment(Int8SampledSentimentError::from(
            crate::tasks::sentiment::sampled::SampledSentimentError::Cancelled(cause)));
        assert_eq!(error.to_string(), "hosted native sampled sentiment failed");
        assert_eq!(format!("{error:?}"), error.to_string());
        assert!(error.source().is_some());
        let HostedError::SampledSentiment(inner) = error else { unreachable!() };
        assert_eq!(inner.cancellation(), Some(cause));
    }
    #[test]
    fn transferred_input_and_guarded_output_are_send() {
        fn send<T: Send + 'static>() {}
        send::<PreparedInt8SampledSentiment>();
        send::<Int8SampledSentimentRun>();
        send::<SampledSentimentVocabulary>();
        send::<TeamAdmission>();
        send::<HostedInt8TeamResult<Int8SampledSentimentRun>>();
        send::<Arc<crate::tasks::sentiment::sampled::quantized::Int8SampledSentimentPlanner>>();
    }
    #[test]
    fn team_and_prompt_workspace_are_additive_and_invalid_capacity_refuses() {
        let ordinary = 1 << 20;
        for width in [2, 3, 64] {
            let (scratch, rows) = team_scratch(ordinary, width, None).unwrap();
            assert_eq!((scratch, rows), (ordinary, 1));
            for rows in [1, 4, 64] {
                let extra = Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
                let prefill = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
                let (scratch, actual_rows) = team_scratch(ordinary, width, Some(prefill)).unwrap();
                assert_eq!(actual_rows, rows);
                assert_eq!(scratch, ordinary + extra);
                assert!(dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap() > 0);
                assert!(team_scratch(ordinary, width,
                    Some(Int8PrefillLimits { max_extra_scratch_bytes: extra - 1, ..prefill })).is_err());
            }
        }
        for width in [0, 1, 65, usize::MAX] { assert!(team_scratch(ordinary, width, None).is_err()); }
        assert!(team_scratch(u64::MAX, 2, None).is_err());
        for rows in [0, 65, usize::MAX] {
            assert!(team_scratch(ordinary, 2,
                Some(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX })).is_err());
        }
    }

    /// This gate executes real model weights and remains ignored until a
    /// controller supplies independent identity pins and admitted memory.
    mod model {
        use super::*;
        use crate::{canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
            batch::{BatchLimits, BatchSummary, BatchWork},
            hosted::corpus::{CorpusLimits, SampledSentimentCorpusConfig},
            native_engine::{decode::DecodeCancellationKind,
                simd::{Int8KernelKind, config::configure_process_kernel}},
            tasks::sentiment::{SentimentAxis, SentimentRequest,
                sampled::{SampledSentimentLimits, SampledSentimentOptions,
                    quantized::Int8SampledSentimentPlanner}}};
        use std::{io::{self, Cursor, Write}, sync::Mutex};

        fn native() -> NativeLimits {
            NativeLimits { context_tokens: 2048, allocator_reserve_bytes: 64 << 20,
                run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
                    cleanup_reserve_bytes: 65_536 } }
        }
        fn task() -> TaskBudget {
            TaskBudget { max_input_tokens: 1792, max_output_tokens: 1024, max_output_bytes: 1 << 20,
                max_grammar_states: 4096,
                max_kv_bytes: Int8MemoryRequirement::for_context(native().context_tokens).unwrap().kv_bytes }
        }
        fn planner(model: &ResidentInt8) -> Int8SampledSentimentPlanner {
            Int8SampledSentimentPlanner::for_model(model.artifact_identity(), SampledSentimentOptions {
                effective_seed: [37; 32], samples: 2, temperature_milli: 1000, top_k: Some(32),
                top_p_ppm: 950_000, max_sample_tokens: 256, max_justification_chars: 32,
            }, task(), SampledSentimentLimits::default()).unwrap()
        }
        const DOCUMENT: &str = "The service was reassuring and helpful. café 上海 <think> literal data.";
        fn prepared(model: &ResidentInt8) -> PreparedInt8SampledSentiment {
            planner(model).plan(&SentimentRequest { document: DOCUMENT.to_owned(),
                axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget: task() }).unwrap()
        }
        fn limits(work: Int8Work) -> Int8SampledSentimentHostLimits {
            Int8SampledSentimentHostLimits { native: native(), preparation_reserve_bytes: 1 << 30,
                max_model_work: work, max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
                mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
                max_mask_node_visits: 32_000_000_000 }
        }
        fn prefill() -> Int8PrefillLimits {
            Int8PrefillLimits { max_batch_rows: 4,
                max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap() }
        }
        fn drained(host: &NlpEngine, baseline: u64) {
            let memory = host.resources().memory_snapshot();
            assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
            assert_eq!(memory.recorded_leaks, 0);
            assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
        }
        #[derive(Clone)]
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        fn corpus(host: &NlpEngine, model: &ResidentInt8, vocabulary: Arc<SampledSentimentVocabulary>,
            width: usize, prefill: Option<Int8PrefillLimits>) -> (BatchSummary, Vec<u8>) {
            let work = prepared(model).planned_work();
            let work = work.checked_add(work).unwrap();
            let host_limits = limits(work);
            let config = SampledSentimentCorpusConfig { max_model_work: work,
                mask_limits: host_limits.mask_limits, max_mask_node_visits: host_limits.max_mask_node_visits };
            let limits = CorpusLimits { native: native(), preparation_reserve_bytes: 1 << 30,
                io_reserve_bytes: 32 << 20, transport: BatchLimits {
                    max_requests: 8, max_input_bytes: 1 << 20, max_output_bytes: 16 << 20,
                    max_output_line_bytes: 2 << 20,
                    max_work: BatchWork { forward_positions: work.forward_positions,
                        projected_logits: work.projected_logits }, ..BatchLimits::default() } };
            let record = serde_json::json!({"id":"stable-id","text":DOCUMENT,
                "task_args":{"axes":["valence","arousal"]}}).to_string();
            let input = Cursor::new(format!("{record}\n{{\"id\":\"malformed\",\"text\":17}}\n{{\"flush\":true}}\n{record}").into_bytes());
            let capture = Capture(Arc::new(Mutex::new(Vec::new())));
            let result = if width == 1 {
                host.batch_int8_sampled_sentiment(model, planner(model), vocabulary, config, limits,
                    prefill, input, capture.clone(), CancellationToken::default())
            } else {
                host.batch_int8_sampled_sentiment_with_threads(model, planner(model), vocabulary, config, limits,
                    prefill, Int8Kernel::scalar(), width, input, capture.clone(), CancellationToken::default())
            }.unwrap();
            assert_eq!((result.succeeded, result.failed, result.flushes), (2, 1, 2));
            let output = capture.0.lock().unwrap().clone();
            (result, output)
        }

        #[test]
        #[ignore = "controller-only pinned INT8 model, admitted memory and sampled team parity/drain gate"]
        fn sampled_bundles_and_resident_streams_preserve_serial_bytes_and_cancel_then_drain() {
            const CHILD: &str = "FNLP_INT8_SAMPLED_TEAM_CHILD";
            const COMPLETE: &str = "FNLP_INT8_SAMPLED_TEAM_COMPLETE";
            let full = concat!(module_path!(),
                "::sampled_bundles_and_resident_streams_preserve_serial_bytes_and_cancel_then_drain");
            let name = full.split_once("::").unwrap().1;
            if std::env::var(CHILD).ok().as_deref() != Some(name) {
                let result = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
                    .output().expect("isolated sampled model gate must start");
                let stdout = String::from_utf8_lossy(&result.stdout);
                assert!(result.status.success(), "sampled gate failed:\n{stdout}\n{}",
                    String::from_utf8_lossy(&result.stderr));
                assert!(stdout.contains(COMPLETE), "exact filter must complete the model gate");
                return;
            }
            assert!(crate::installed_process_resources().is_none());
            configure_process_kernel(Int8KernelKind::Scalar).unwrap();
            let path = PathBuf::from(std::env::var_os("FNLP_INT8_SAMPLED_TEAM_MODEL").expect("explicit local model required"));
            let logical = std::env::var("FNLP_INT8_SAMPLED_TEAM_LOGICAL_SHA256").expect("independent logical pin required");
            let source = std::env::var("FNLP_INT8_SAMPLED_TEAM_SOURCE_ROOT_SHA256").expect("independent source pin required");
            Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source).unwrap();
            let memory: u64 = std::env::var("FNLP_INT8_SAMPLED_TEAM_MEMORY_MIB")
                .expect("controller-admitted memory required").parse().unwrap();
            let host = NlpEngine::builder().resource_config(ResourceHostConfig {
                runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
                max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
                helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory.checked_mul(1 << 20).unwrap(),
                leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
            }).build().unwrap();
            let lease = host.resources().acquire_lease();
            let caller_preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 1 << 30).unwrap().commit().unwrap();
            let model = host.load_current_candidate_int8(path, LoadLimits {
                artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
                tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run: native().run,
            }, CancellationToken::default()).unwrap();
            assert_eq!(model.artifact_identity().logical_model_sha256, logical);
            assert_eq!(model.artifact_identity().source_root_sha256, source);
            let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
            let baseline = host.resources().memory_snapshot().committed_bytes;
            let plan = prepared(&model);
            let output_charge = plan.max_result_bytes() * 4 + plan.max_output_tokens() * 8 + 4096;
            let host_limits = limits(plan.planned_work());
            let serial = host.execute_int8_sampled_sentiment(&model, plan, Arc::clone(&vocabulary),
                host_limits, CancellationToken::default()).unwrap();
            let expected = canonjson::canonical_bytes(serial.result()).unwrap();
            drop(serial); drained(&host, baseline);
            for width in [2, 4] { for schedule in [None, Some(prefill())] {
                let result = match schedule {
                    None => host.execute_int8_sampled_sentiment_with_threads(&model, prepared(&model),
                        Arc::clone(&vocabulary), host_limits, Int8Kernel::scalar(), width, CancellationToken::default()),
                    Some(prefill) => host.execute_int8_sampled_sentiment_layer_major_with_threads(&model,
                        prepared(&model), Arc::clone(&vocabulary), host_limits, Int8Kernel::scalar(), width,
                        prefill, CancellationToken::default()),
                }.unwrap();
                assert_eq!(canonjson::canonical_bytes(result.result()).unwrap(), expected);
                assert_eq!(result.compute().compute_threads, width);
                assert_eq!(result.compute().completed_rows, result.result().model_work.projections.dot_products);
                assert!(result.compute().completed_worker_rows > 0);
                drained(&host, baseline + output_charge);
                drop(result); drained(&host, baseline);
            } }
            let expected = corpus(&host, &model, Arc::clone(&vocabulary), 1, None);
            drained(&host, baseline);
            for (width, schedule) in [(2, None), (4, Some(prefill()))] {
                assert_eq!(corpus(&host, &model, Arc::clone(&vocabulary), width, schedule), expected);
                drained(&host, baseline);
            }
            use DecodeCancellationKind::*;
            for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
                ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
                let token = CancellationToken::default(); token.cancel(cause);
                let error = host.execute_int8_sampled_sentiment_layer_major_with_threads(&model,
                    prepared(&model), Arc::clone(&vocabulary), host_limits, Int8Kernel::scalar(), 4, prefill(), token)
                    .err().expect("cancelled sampling must not publish");
                assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
                drained(&host, baseline);
            }
            drop(vocabulary); drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
            println!("{COMPLETE}");
        }
    }
}
