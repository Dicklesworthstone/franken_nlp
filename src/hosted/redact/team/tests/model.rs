//! Controller-only real-artifact schedule and physical-drain gate.
//! Run alone in a fresh resource-host process with independent artifact pins.
//! A residual finding is a typed failure, never a failed recall assertion.
use std::{path::PathBuf, sync::Arc, time::Duration};
use crate::{
    canonjson, LeakResponsePolicy, MemoryClass, NlpEngine, ResourceHostConfig, RuntimePreset,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    hosted::{CancellationToken, HostedError, HostedInt8TeamResult, HostedOutput,
        Int8ComputeReceipt, LoadLimits, NativeLimits, Pending, RedactConfig, ResidentInt8, RunLimits, RunStop},
    native_engine::{artifact_bridge::{ArtifactIdentity, ArtifactLoadBudget},
        decode::DecodeCancellationKind, simd::config::process_kernel,
        strict_int8::{Int8MemoryRequirement, Int8Work, STRICT_INT8_EXECUTION, prefill::Int8PrefillLimits}},
    tasks::{extract::ExtractionVocabulary, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        ner::{EntityType, NerOptions, NER_TASK_VERSION},
        redact::{RedactError, RedactionRequest, actions::{RedactionAction, VerificationStatus},
            long::{LongRedactionConfig, LongRedactionRun},
            quantized::{Int8RedactionConfig, Int8RedactionError, Int8RedactionRun}},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, quantized::long::Int8SourceMapLimits}},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding}},
};

const SOURCE: &str = "Alice visits Paris. Email a@example.org. Bob visits Rome. é 上海 <think>.";
const RESULT_BYTES: u64 = 4 << 20;

fn native() -> NativeLimits {
    NativeLimits { context_tokens: 2048, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 } }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 1536, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(2048).unwrap().kv_bytes }
}
fn work() -> Int8Work {
    Int8Work::for_sequence(0, 65_536, 8192 * crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE).unwrap()
}
fn assets(facts: &ArtifactIdentity, compact: bool, verify: bool) -> (Arc<SourceTaskPlanner>, RedactConfig) {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.surface == crate::template::IM_END).unwrap().id;
    let mut planner = SourceTaskPlanner::pinned(controls, eos).unwrap();
    if compact {
        planner = planner.with_map_document_encoding(UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 1536, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        }), SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
            max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }).unwrap();
    }
    let absent = Sha256Digest::of_bytes(b"null");
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    let ner_identity = ExecutionIdentity { schema_version: 1, source_revision: facts.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap(),
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: planner.tokenizer_digest(),
        template_digest: *planner.template_digest(), task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: absent, prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let mut request = RedactionRequest { verify, ..RedactionRequest::default() };
    request.actions.default_action = RedactionAction::Mask; request.actions.include_map = true;
    request.rule_budget.max_input_bytes = 4096; request.edit_budget.max_output_bytes = RESULT_BYTES as usize;
    let detector = Int8RedactionConfig {
        ner: NerOptions { types: vec![EntityType::Person], max_entities: 4, max_mention_scalars: 64 },
        per_pass: budget(), planning: SourcePlanningLimits { max_input_bytes: 4096,
            max_context_tokens: 2048, ..SourcePlanningLimits::default() },
        max_model_work: work(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        mask_visits_per_pass: 1_024_000_000, max_mask_visits: 2_048_000_000, max_result_bytes: RESULT_BYTES,
    };
    (Arc::new(planner), RedactConfig { ner_identity, request, detector, native: native(),
        preparation_reserve_bytes: 512 << 20, edit_reserve_bytes: 64 << 20 })
}
fn document(config: RedactConfig) -> RedactConfig<LongRedactionConfig> {
    let detector = LongRedactionConfig {
        ner: config.detector.ner, per_chunk: config.detector.per_pass, planning: config.detector.planning,
        mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 32, max_chunk_tokens: 256,
                context_tokens: 2048, reserved_tokens: 512, max_chunks: 16, max_tokenizer_calls: 4096 },
            reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2,
                max_value_bytes: 4 << 20, max_result_bytes: 16 << 20,
                max_live_value_bytes: 32 << 20, max_total_value_bytes: 64 << 20, ..ExecutionLimits::default() },
            max_model_work: work(), mask_limits: config.detector.mask_limits,
            mask_visits_per_chunk: config.detector.mask_visits_per_pass, max_mask_visits: 32_768_000_000,
        },
        max_result_bytes: RESULT_BYTES,
    };
    RedactConfig { ner_identity: config.ner_identity, request: config.request, detector, native: config.native,
        preparation_reserve_bytes: config.preparation_reserve_bytes, edit_reserve_bytes: config.edit_reserve_bytes }
}
fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline);
    assert_eq!(memory.reserved_bytes, 0); assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

enum Output<T> { Serial(HostedOutput<T>), Team(HostedInt8TeamResult<T>) }
impl<T> Output<T> {
    fn result(&self) -> &T { match self { Self::Serial(out) => out.result(), Self::Team(out) => out.result() } }
    fn compute(&self) -> Option<&Int8ComputeReceipt> {
        match self { Self::Serial(_) => None, Self::Team(out) => Some(out.compute()) }
    }
}
trait Receipt: serde::Serialize {
    fn work(&self) -> Int8Work;
    fn check(&self, verify: bool);
}
impl Receipt for Int8RedactionRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn check(&self, verify: bool) {
        assert_eq!(self.ner_passes, 1 + usize::from(verify));
        assert_eq!(self.result.verification(), if verify { VerificationStatus::CleanDeclaredUnion }
            else { VerificationStatus::NotRequested });
        assert!(!self.result.text().contains("a@example.org"));
    }
}
impl Receipt for LongRedactionRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn check(&self, verify: bool) {
        assert!(self.original.preflight.chunks > 1);
        assert_eq!(self.verification.is_some(), verify);
        assert_eq!(self.result.verification(), if verify { VerificationStatus::CleanDeclaredUnion }
            else { VerificationStatus::NotRequested });
        assert!(!self.result.text().contains("a@example.org"));
    }
}
#[derive(Debug, Eq, PartialEq)]
enum Outcome { Complete(Vec<u8>), Residual(usize) }
fn residual(error: HostedError) -> usize {
    match error {
        HostedError::Redaction(Int8RedactionError::Redaction(RedactError::VerificationResidual { count })) => count,
        HostedError::Join { physical_error: Some(error), .. } | HostedError::Scope { physical_error: Some(error), .. } => residual(*error),
        other => panic!("unexpected redaction failure: {other:?}"),
    }
}
fn capture<T: Receipt>(host: &NlpEngine, baseline: u64, result: Result<Output<T>, HostedError>,
    width: usize, verify: bool) -> Outcome {
    match result {
        Ok(output) => {
            output.result().check(verify);
            let bytes = canonjson::canonical_bytes(output.result()).unwrap();
            if let Some(compute) = output.compute() {
                assert_eq!(compute.compute_threads, width);
                assert_eq!(compute.completed_rows, output.result().work().projections.dot_products);
                assert!(compute.completed_projections > 0); assert!(compute.completed_worker_rows > 0);
            } else { assert_eq!(width, 1); }
            // All source, native, secret and worker storage has drained. Only
            // the completed result and its serialization allowance stay owned.
            drained(host, baseline + RESULT_BYTES * 4 + 4096);
            drop(output); drained(host, baseline);
            Outcome::Complete(bytes)
        }
        Err(error) => {
            assert!(verify, "nonverification execution must produce a complete receipt");
            let count = residual(error); assert!(count > 0);
            drained(host, baseline);
            Outcome::Residual(count)
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn execute(host: &NlpEngine, model: &ResidentInt8, vocabulary: &Arc<ExtractionVocabulary>,
    baseline: u64, long: bool, compact: bool, verify: bool, width: usize,
    schedule: Option<Int8PrefillLimits>) -> Outcome {
    let (planner, config) = assets(model.artifact_identity(), compact, verify);
    let source = SOURCE.to_owned(); let vocabulary = Arc::clone(vocabulary);
    let cancellation = CancellationToken::default(); let kernel = process_kernel();
    if long {
        let config = document(config);
        let result = if width == 1 { match schedule {
            None => host.redact_int8_document(model, source, planner, vocabulary, config, None, cancellation),
            Some(prefill) => host.redact_int8_document_layer_major(model, source, planner, vocabulary, config, None, prefill, cancellation),
        }.map(Output::Serial) } else { match schedule {
            None => host.redact_int8_document_with_threads(model, source, planner, vocabulary, config, None, kernel, width, cancellation),
            Some(prefill) => host.redact_int8_document_layer_major_with_threads(model, source, planner, vocabulary, config, None, kernel, width, prefill, cancellation),
        }.map(Output::Team) };
        return capture(host, baseline, result, width, verify);
    }
    let result = if width == 1 { match schedule {
        None => host.redact_int8(model, source, planner, vocabulary, config, None, cancellation),
        Some(prefill) => host.redact_int8_layer_major(model, source, planner, vocabulary, config, None, prefill, cancellation),
    }.map(Output::Serial) } else { match schedule {
        None => host.redact_int8_with_threads(model, source, planner, vocabulary, config, None, kernel, width, cancellation),
        Some(prefill) => host.redact_int8_layer_major_with_threads(model, source, planner, vocabulary, config, None, kernel, width, prefill, cancellation),
    }.map(Output::Team) };
    capture(host, baseline, result, width, verify)
}

#[test]
#[ignore = "controller-only real INT8 artifact, independent pins and fresh host; no parity claim until executed"]
fn short_and_complete_redaction_preserve_both_passes_with_prefill_and_persistent_workers() {
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_REDACTION_TEAM_MODEL").expect("admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_REDACTION_TEAM_LOGICAL_SHA256").expect("independent logical pin required");
    let source_root = std::env::var("FNLP_INT8_REDACTION_TEAM_SOURCE_ROOT_SHA256").expect("independent source-root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16_u64 << 30,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 << 20).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run: native().run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(pinned_controls::pinned().unwrap().template_controls()).unwrap());
    let baseline = host.resources().memory_snapshot().committed_bytes;
    for long in [false, true] { for compact in [false, true] { for verify in [false, true] {
        let expected = execute(&host, &model, &vocabulary, baseline, long, compact, verify, 1, None);
        for width in [1, 2, 4] { for schedule in [Some(prefill(1)), Some(prefill(7))] {
            assert_eq!(execute(&host, &model, &vocabulary, baseline, long, compact, verify, width, schedule), expected);
            assert_eq!(Arc::strong_count(&vocabulary), 1);
        } }
        for width in [2, 4] {
            assert_eq!(execute(&host, &model, &vocabulary, baseline, long, compact, verify, width, None), expected);
        }
    } } }

    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let (planner, config) = assets(model.artifact_identity(), true, true);
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = host.redact_int8_document_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
            Arc::clone(&vocabulary), document(config), None, process_kernel(), 4, prefill(7), cancellation)
            .err().expect("cancelled verification pipeline cannot publish");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
    }
    for width in [0, 1, 5, 65, usize::MAX] {
        let (planner, config) = assets(model.artifact_identity(), false, false);
        assert!(matches!(host.redact_int8_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
            Arc::clone(&vocabulary), config, None, process_kernel(), width, prefill(7), CancellationToken::default()),
            Err(HostedError::Limits(_))));
        drained(&host, baseline);
    }
    for schedule in [Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_extra_scratch_bytes: prefill(7).max_extra_scratch_bytes - 1, ..prefill(7) }] {
        let (planner, config) = assets(model.artifact_identity(), true, true);
        assert!(matches!(host.redact_int8_document_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
            Arc::clone(&vocabulary), document(config), None, process_kernel(), 2, schedule, CancellationToken::default()),
            Err(HostedError::Native(_))));
        drained(&host, baseline);
    }
    let (planner, mut config) = assets(model.artifact_identity(), false, true);
    config.request.actions.default_action = RedactionAction::Pseudonymize;
    let error = host.redact_int8_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
        Arc::clone(&vocabulary), config, None, process_kernel(), 2, prefill(7), CancellationToken::default())
        .err().expect("missing key must refuse before either neural pass");
    assert!(matches!(error, HostedError::Redaction(Int8RedactionError::Redaction(RedactError::MissingKey))));
    drained(&host, baseline);
    drop(vocabulary); drop(model); drop(preparation); drop(lease); drained(&host, 0);
}
