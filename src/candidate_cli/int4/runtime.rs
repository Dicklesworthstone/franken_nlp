//! End-to-end source verification, charged loading, planning and INT4 execution.
use super::*;
use std::{fs::File, sync::Arc, time::{Duration, Instant}};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    execution_identity::ExecutionIdentity,
    batch::{BatchDocument, extract::{ExtractionBatchArgs, grouped_int4::Int4ExtractionBatchPlanner}},
    grammar::mask::MaskWorkLimits,
    hosted::{CancellationToken, HostedOutput, Int4LoadLimits, NativeLimits, ResidentInt4, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl}, int4::PackedInt4Kernel,
        strict_int4::{Int4ModelLoadBudget, weights::snapshot::INT4_SNAPSHOT_VERSION}},
    tasks::{chat::{ChatLimits, ChatRequest, GenerateRequest,
            grouped_int4::{Int4ChatPlanner, Int4ChatResult, PreparedInt4Chat}},
        classify::{ClassificationRequest,
            grouped_int4::{Int4ClassificationPlanner, Int4ClassificationRun, PreparedInt4Classification}},
        extract::{ExtractionVocabulary, grouped_int4::{Int4ExtractPlan, Int4ExtractRun}},
        ir::{PlanContext, TaskBudget}, source_planning::{SourceTaskRequest,
            grouped_int4::{PreparedInt4SourceTask, Int4SourceTaskRun}}}};

// Typed task alternatives share one loader, preparation guard and publication
// envelope. Only real hosted execution can construct the guarded completion.
enum Input {
    Text(TextInput),
    Classify(ClassificationRequest, classify::ScoringArgs),
    Extract { document: String, request: ExtractionBatchArgs, options: extract::ExtractionArgs },
    Source { request: SourceTaskRequest, options: source::SourceArgs },
}
enum Prepared {
    Text(PreparedInt4Chat),
    Classify(PreparedInt4Classification),
    Extract { plan: Int4ExtractPlan, vocabulary: Arc<ExtractionVocabulary>,
        masks: MaskWorkLimits, max_mask_node_visits: u64 },
    Source { plan: PreparedInt4SourceTask, vocabulary: Arc<ExtractionVocabulary>,
        masks: MaskWorkLimits, max_mask_node_visits: u64 },
}
enum Completed {
    Text(HostedOutput<Int4ChatResult>),
    Classify(HostedOutput<Int4ClassificationRun>),
    Extract(HostedOutput<Int4ExtractRun>),
    Source(HostedOutput<Int4SourceTaskRun>),
}
impl Serialize for Completed {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(result) => result.result().serialize(serializer),
            Self::Classify(result) => result.result().serialize(serializer),
            Self::Extract(result) => result.result().serialize(serializer),
            Self::Source(result) => result.result().serialize(serializer),
        }
    }
}
impl Prepared {
    fn identity(&self) -> &ExecutionIdentity {
        match self {
            Self::Text(plan) => plan.execution_identity(),
            Self::Classify(plan) => plan.execution_identity(),
            Self::Extract { plan, .. } => plan.execution_identity(),
            Self::Source { plan, .. } => plan.execution_identity(),
        }
    }
    fn execute(self, engine: &NlpEngine, model: &ResidentInt4, native: NativeLimits,
        kernel: PackedInt4Kernel, prefill: Option<Int4PrefillLimits>, cancellation: CancellationToken)
        -> Result<Completed, CandidateError> {
        match (self, prefill) {
            (Self::Text(plan), None) => engine.execute_int4_chat(model, plan, 1, native, SAMPLER_BYTES,
                kernel, cancellation).map(Completed::Text),
            (Self::Text(plan), Some(limits)) => engine.execute_int4_chat_layer_major(model, plan, 1, native,
                SAMPLER_BYTES, kernel, limits, cancellation).map(Completed::Text),
            (Self::Classify(plan), None) => engine.execute_int4_classify(model, plan, native,
                kernel, cancellation).map(Completed::Classify),
            (Self::Extract { plan, vocabulary, masks, max_mask_node_visits }, None) =>
                engine.execute_int4_extract(model, plan, vocabulary, native, masks,
                    max_mask_node_visits, kernel, cancellation).map(Completed::Extract),
            (Self::Source { plan, vocabulary, masks, max_mask_node_visits }, None) =>
                engine.execute_int4_source_task(model, plan, vocabulary, native, masks,
                    max_mask_node_visits, kernel, cancellation).map(Completed::Source),
            (Self::Classify(_) | Self::Extract { .. } | Self::Source { .. }, Some(_)) => return Err(CandidateError::Arguments),
        }.map_err(|_| CandidateError::Execution)
    }
}
struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}

pub(super) fn execute_task(task: Int4Task, args: Int4Args, limits: Int4Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    if matches!(&task, Int4Task::RedactCorpus(_) | Int4Task::RedactDocument(_) | Int4Task::Redact(_) | Int4Task::Judge(_) | Int4Task::Semantic(_) | Int4Task::Resolve(_) | Int4Task::Sentiment(_) | Int4Task::ExtractBatch(_) | Int4Task::SourceBatch(_) | Int4Task::ScoredBatch(_) | Int4Task::ExtractMap(_) | Int4Task::SourceMap(_) | Int4Task::TextJob | Int4Task::ExtractJob | Int4Task::ExtractDocumentJob | Int4Task::ScoredJob | Int4Task::SourceJob | Int4Task::DocumentJob | Int4Task::RedactJob)
        || limits.minimum_memory_bytes > limits.memory_bytes || limits.kernel.kind() != args.kernel
        || !prefill::matches(&args, limits)?
        || (!matches!(&task, Int4Task::Text(_)) && limits.prefill.is_some()) {
        return Err(CandidateError::Arguments);
    }
    let started = Instant::now();
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: limits.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Declared before every input/planner allocation and kept through output
    // serialization, external write AND flush, on success or any failure.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" {
        read_input(input, args.max_input_bytes)?
    } else {
        let mut file = File::open(&args.input).map_err(|_| CandidateError::Input)?;
        read_input(&mut file, args.max_input_bytes)?
    };
    let request = match task {
        Int4Task::Text(task) => Input::Text(parse_input(task, text, args.max_input_bytes)?),
        Int4Task::Classify(scoring) => Input::Classify(
            classify::parse_request(&text, scoring.budget(limits), args.max_input_bytes)?, scoring),
        Int4Task::Extract(options) => {
            remaining()?;
            let mut file = File::open(options.schema_path()).map_err(|_| CandidateError::Input)?;
            let schema = read_input(&mut file, crate::candidate_cli::extract::SCHEMA_BYTES)?;
            let request = options.check_schema(schema, &text, options.budget(&args, limits))?;
            Input::Extract { document: text, request, options }
        }
        Int4Task::Source(kind, options) => {
            remaining()?;
            let option_json = options.options_path().map(|path| {
                let mut file = File::open(path).map_err(|_| CandidateError::Input)?;
                read_input(&mut file, crate::candidate_cli::source::OPTIONS_BYTES)
            }).transpose()?;
            let request = crate::candidate_cli::source::request(kind, text, option_json.as_deref(),
                options.budget(&args, limits), args.max_input_bytes)?;
            Input::Source { request, options }
        }
        Int4Task::RedactCorpus(_) | Int4Task::RedactDocument(_) | Int4Task::Redact(_) | Int4Task::Judge(_) | Int4Task::Semantic(_) | Int4Task::Resolve(_) | Int4Task::Sentiment(_) | Int4Task::ExtractBatch(_) | Int4Task::SourceBatch(_) | Int4Task::ScoredBatch(_) | Int4Task::ExtractMap(_) | Int4Task::SourceMap(_)
            | Int4Task::TextJob | Int4Task::ExtractJob | Int4Task::ExtractDocumentJob | Int4Task::ScoredJob | Int4Task::SourceJob | Int4Task::DocumentJob | Int4Task::RedactJob => return Err(CandidateError::Arguments),
    };
    remaining()?;
    let cancellation = CancellationToken::default();
    // Snapshot failure is terminal. Never catch it to convert a source, read a
    // header as its own trust anchor, look up a catalog, or change the group width.
    let model = load_model(&engine, &args, limits, remaining()?, cancellation.clone())?;
    remaining()?;
    // The actual model owns its recomputed fingerprint. No fabricated pre-load
    // identity is used for token planning. Wire shape/labels were checked above.
    let prepared = prepare(request, &model, &args, limits, &mut DeadlineControl { started, deadline })?;
    remaining()?;
    let identity = prepared.identity().clone();
    let mut snapshot = limits.snapshot_pin.map(|_| SnapshotRecord::new("loaded", limits.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Save only after complete task planning and all work-axis checks.
        // Preserve a complete file if execution fails, and partial files on IO failure.
        let pin = engine.save_int4_snapshot(&model, path.clone(), limits.snapshot_budget(false)?,
            snapshot_run(&args, remaining()?), cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", limits.snapshot_need));
    }
    let result = prepared.execute(&engine, &model, NativeLimits {
        context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES },
    }, limits.kernel, limits.prefill, cancellation)?;
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let response = SourceResponse {
        schema_version: 1, scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        output: &result, snapshot, integer_kernel: kernel_record(limits.kernel),
    };
    // Completed retains the exact hosted output guard while Serialize borrows
    // its result. No unguarded JSON value or partially completed head escapes.
    publish(&response, limits.result_bytes + 4096, output)
}

fn prepare(request: Input, model: &ResidentInt4, args: &Int4Args, limits: Int4Limits,
    control: &mut DeadlineControl) -> Result<Prepared, CandidateError> {
    match request {
        Input::Text(request) => {
            let budget = TaskBudget {
                max_input_tokens: u32::try_from(limits.max_prompt_tokens).map_err(|_| CandidateError::Arguments)?,
                max_output_tokens: u32::try_from(args.max_new_tokens).map_err(|_| CandidateError::Arguments)?,
                max_output_bytes: limits.result_bytes as u64, max_grammar_states: 1,
                max_kv_bytes: limits.native.kv_bytes,
            };
            let planner = Int4ChatPlanner::for_model(model.model_identity(), budget, ChatLimits {
                max_messages: 128, max_message_bytes: args.max_input_bytes,
                max_total_message_bytes: args.max_input_bytes, generation: args.generation_limits(limits),
            }).map_err(|_| CandidateError::Planning)?;
            let options = args.options(planner.eos_token_id())?;
            let plan = match request {
                TextInput::Generate(prompt) => planner.plan_generate(&GenerateRequest {
                    item_id: "cli".to_owned(), sample_index: 0, prompt, generation: options, budget }),
                TextInput::Chat(messages) => planner.plan_chat(&ChatRequest {
                    item_id: "cli".to_owned(), sample_index: 0, messages, generation: options, budget }),
            }.map_err(|_| CandidateError::Planning)?;
            Ok(Prepared::Text(plan))
        }
        Input::Classify(request, scoring) => {
            let planner = Int4ClassificationPlanner::for_model(model.model_identity(), scoring.budget(limits),
                scoring.classification_limits(args)).map_err(|_| CandidateError::Planning)?;
            let plan = planner.plan_with_control(&request, control).map_err(|error| {
                if error.cancellation() == Some(DecodeCancellationKind::Deadline) { CandidateError::Timeout }
                else { CandidateError::Planning }
            })?;
            scoring.admit_plan(plan.required_context(), args.context_tokens, plan.planned_work())?;
            Ok(Prepared::Classify(plan))
        }
        Input::Extract { document, request, options } => {
            let planner = Int4ExtractionBatchPlanner::for_model(model.model_identity(), options.budget(args, limits),
                options.compiler(), options.source_limits(), None).map_err(|_| CandidateError::Planning)?;
            let prepared = planner.prepare_with_control(BatchDocument {
                id: "cli".to_owned(), text: document, task_args: Some(request),
            }, control).map_err(|error| match error.fault.cancellation {
                Some(DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) => CandidateError::Timeout,
                Some(_) => CandidateError::Execution,
                None => CandidateError::Planning,
            })?;
            if prepared.planned_work().forward_positions > args.context_tokens as u64 {
                return Err(CandidateError::Planning);
            }
            prepared.verify_identity(prepared.execution_identity()).map_err(|_| CandidateError::Identity)?;
            drop(planner);
            let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
            let vocabulary = ExtractionVocabulary::pinned(controls.template_controls())
                .map(Arc::new).map_err(|_| CandidateError::Planning)?;
            if let Some(cause) = control.checkpoint(0) {
                return Err(if matches!(cause, DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) {
                    CandidateError::Timeout
                } else { CandidateError::Execution });
            }
            Ok(Prepared::Extract { plan: prepared.into_extraction_plan(), vocabulary,
                masks: options.masks(), max_mask_node_visits: options.mask_visits() })
        }
        Input::Source { request, options } => {
            if let Some(cause) = control.checkpoint(0) {
                return Err(if matches!(cause, DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) {
                    CandidateError::Timeout
                } else { CandidateError::Execution });
            }
            let (planner, vocabulary) = source::planner()?;
            let identity = source::identity(model.model_identity(), &planner, request.task())?;
            let context = PlanContext::new(&identity, options.budget(args, limits))
                .map_err(|_| CandidateError::Planning)?;
            let prepared = planner.plan_int4_with_control(&request, &context, options.planning(args),
                model.model_identity().group_width(), control).map_err(|error| {
                    match error.cancellation() {
                        Some(DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout) => CandidateError::Timeout,
                        Some(_) => CandidateError::Execution,
                        None => CandidateError::Planning,
                    }
                })?;
            if prepared.planned_work().forward_positions > args.context_tokens as u64 {
                return Err(CandidateError::Planning);
            }
            prepared.verify_identity(prepared.execution_identity()).map_err(|_| CandidateError::Identity)?;
            Ok(Prepared::Source { plan: prepared, vocabulary: Arc::new(vocabulary),
                masks: options.masks(), max_mask_node_visits: options.mask_visits() })
        }
    }
}

/// Source and snapshot loading share the same checked limits for single and
/// corpus extraction. A failed snapshot never falls back to another source.
pub(super) fn load_model(engine: &NlpEngine, args: &Int4Args, limits: Int4Limits,
    remaining: Duration, cancellation: CancellationToken) -> Result<ResidentInt4, CandidateError> {
    match (&args.snapshot, limits.snapshot_pin) {
        (Some(path), Some(pin)) => engine.load_int4_snapshot(path.clone(), pin, limits.snapshot_budget(true)?,
            ALLOCATOR_BYTES, snapshot_run(args, remaining), cancellation),
        (None, None) => engine.load_source_candidate_int4(
            args.source_dir.clone().ok_or(CandidateError::Arguments)?,
            args.source_manifest.clone().ok_or(CandidateError::Arguments)?,
            args.group_width, load_limits(args, limits, remaining), cancellation),
        _ => return Err(CandidateError::Arguments),
    }.map_err(|_| CandidateError::Model)
}

fn load_limits(args: &Int4Args, limits: Int4Limits, remaining: Duration) -> Int4LoadLimits {
    Int4LoadLimits {
        model: Int4ModelLoadBudget { max_payload_bytes: limits.max_load_bytes,
            max_source_values: limits.load.source_values, max_groups: limits.load.groups,
            max_shard_revalidation_bytes: limits.load.shard_revalidation_bytes },
        source_preparation_bytes: limits.source_preparation_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: RunLimits { max_elapsed: remaining, max_checkpoints: args.load_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES },
    }
}
fn snapshot_run(args: &Int4Args, remaining: Duration) -> RunLimits {
    RunLimits { max_elapsed: remaining, max_checkpoints: args.snapshot_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES }
}
#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct SourceResponse<'a, T: Serialize> {
    schema_version: u32, scope: &'static str, evidence: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, output: &'a T,
    // No path, arbitrary file-supplied facts or duplicate trust pin. The outer
    // model_content_sha256 is the independently recomputed model identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    // Physical integer leaf, NOT model identity, packing, quality or timing.
    // Omit the scalar default to preserve the existing response contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    integer_kernel: Option<&'static str>,
}
fn kernel_record(kernel: PackedInt4Kernel) -> Option<&'static str> {
    (kernel.kind() != Int8KernelKind::Scalar).then_some(kernel.kind().id())
}

// Existing model-present text fixtures use the same extended runtime path.
#[cfg(test)]
pub(super) fn execute(task: Task, args: Int4Args, limits: Int4Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    execute_task(Int4Task::Text(task), args, limits, input, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cli_resident_staging_and_work_budgets_match_the_host_for_group_partitions() {
        for width in [1, 3, 128, 10752] {
            let mut args = super::super::tests::args("generate", &[]);
            args.group_width = width; args.max_load_mib = 65536; args.memory_mib = 65536;
            let limits = args.validate().unwrap();
            let load = load_limits(&args, limits, Duration::from_secs(1));
            let host = load.requirements(width).unwrap();
            assert_eq!(host.resident_bytes, limits.load.weight_payload_bytes + ALLOCATOR_BYTES);
            assert_eq!(host.staging_bytes, limits.load.scratch_payload_bytes + limits.source_preparation_bytes);
            assert_eq!(load.model.max_source_values, limits.load.source_values);
            assert_eq!(load.model.max_groups, limits.load.groups);
            assert_eq!(load.model.max_shard_revalidation_bytes, limits.load.shard_revalidation_bytes);
            assert!(limits.minimum_memory_bytes >= limits.preparation_bytes + host.resident_bytes + host.staging_bytes);
        }
    }
    #[test]
    fn source_response_uses_a_separate_namespace_and_preserves_full_native_result() {
        let output = serde_json::json!({"result":{"content":"A"},"model_work":{"attention_pairs":7}});
        let response = SourceResponse { schema_version: 1, scope: "real-source-current-candidate-int4",
            evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
            group_width: 128, output: &output, snapshot: None, integer_kernel: None };
        let mut bytes = Vec::new();
        publish(&response, 4096, &mut bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"], output);
        assert_eq!(value["evidence"], "non_authoritative");
        assert_eq!(value["group_width"], 128);
        for absent in ["logical_model_sha256", "prompt_digest", "source_dir", "source_manifest", "snapshot", "integer_kernel"] {
            assert!(value.get(absent).is_none());
        }
    }
    #[test]
    fn snapshot_response_binds_operation_and_length_without_changing_native_output() {
        let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
        let output = serde_json::json!({"result":{"content":"A"},"model_work":{"attention_pairs":7}});
        for operation in ["saved", "loaded"] {
            let response = SourceResponse { schema_version: 1, scope: "real-source-current-candidate-int4",
                evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
                source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
                candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
                group_width: 128, output: &output, snapshot: Some(SnapshotRecord::new(operation, need)), integer_kernel: None };
            let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["output"], output);
            assert_eq!(value["snapshot"]["operation"], operation);
            assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
            assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
            assert_eq!(value["snapshot"].as_object().unwrap().len(), 3);
            let mut refused = Vec::new(); assert!(publish(&response, 1, &mut refused).is_err());
            assert!(refused.is_empty());
        }
    }
    #[test]
    fn physical_kernel_metadata_is_checked_separate_and_default_omitted() {
        let output = serde_json::json!({"head_count":2,"model_work":{"attention_pairs":7}});
        for kind in Int8KernelKind::ALL {
            let Ok(kernel) = PackedInt4Kernel::detect(kind) else { continue; };
            let response = SourceResponse { schema_version: 1, scope: "real-source-current-candidate-int4",
                evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
                source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
                candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
                group_width: 128, output: &output, snapshot: None, integer_kernel: kernel_record(kernel) };
            let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["output"], output);
            assert_eq!(value["model_content_sha256"], "fixture-content");
            assert_eq!(value["quant_recipe"], "fixture:group=128");
            if kind == Int8KernelKind::Scalar { assert!(value.get("integer_kernel").is_none()); }
            else { assert_eq!(value["integer_kernel"], kind.id()); }
            assert!(value.get("qualified").is_none());
            assert!(value.get("speedup").is_none());
        }
    }
    #[test]
    fn a_changed_cli_kernel_cannot_reuse_another_admitted_handle_before_io() {
        struct NoRead;
        impl Read for NoRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("mismatched kernel reached input"); }
        }
        let mut args = super::super::tests::args("generate", &[]);
        let limits = args.validate().unwrap();
        args.kernel = Int8KernelKind::Avx2WidenedI16;
        let mut output = Vec::new();
        assert_eq!(execute_task(Int4Task::Text(Task::Generate), args, limits, &mut NoRead, &mut output),
            Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    #[test]
    fn complete_extraction_serialization_retains_exact_json_and_source_offsets() {
        // Publication must retain the native JSON text as a string. A
        // serde_json::Value round trip over this inner value would round the
        // numeric constant whose spelling is part of the extraction contract.
        let exact = "{\"value\":12345678901234567890123456789012345678}";
        let output = serde_json::json!({
            "schema_version":1,
            "group_width":128,
            "result":{"output":{"json":exact},"source_fields":[{"byte_start":4,"byte_end":10}]},
            "model_work":{"forward_positions":19}
        });
        let response = SourceResponse { schema_version: 1, scope: "real-source-current-candidate-int4",
            evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
            group_width: 128, output: &output, snapshot: None, integer_kernel: None };
        let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"]["result"]["output"]["json"].as_str(), Some(exact));
        assert_eq!(value["output"]["result"]["source_fields"][0]["byte_start"], 4);
        assert_eq!(value["output"]["model_work"]["forward_positions"], 19);
        let mut refused = Vec::new();
        assert!(publish(&response, 1, &mut refused).is_err());
        assert!(refused.is_empty());
    }
    #[test]
    #[ignore = "requires controller-admitted pinned source and an explicit local extraction schema fixture"]
    fn model_present_source_cli_extraction_is_independently_validated() {
        let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory required");
        let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
        let schema = std::env::var("FNLP_INT4_TEST_EXTRACTION_SCHEMA").expect("explicit local schema required");
        let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", "extract", "--source-dir", &source, "--source-manifest", &manifest,
            "--memory-mib", &memory, "--schema", &schema, "--context-tokens", "1024", "--max-new-tokens", "128",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong extraction route"); };
        let Int4Task::Extract(options) = &command.task else { panic!("wrong task"); };
        let program = crate::grammar::runtime::JsonProgram::compile(
            &std::fs::read_to_string(&schema).unwrap(), options.compiler()).unwrap();
        let mut output = Vec::new();
        command.execute(&mut "Alice works in Shanghai.".as_bytes(), &mut output).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["evidence"], "non_authoritative");
        assert_eq!(value["output"]["execution"], crate::tasks::extract::grouped_int4::INT4_EXTRACT_VERSION);
        assert_eq!(value["output"]["result"]["output"]["numerics_profile"],
            crate::native_engine::strict_int4::STRICT_INT4_PROFILE);
        program.validate_json(value["output"]["result"]["output"]["json"].as_str().unwrap()).unwrap();
    }

    #[test]
    #[ignore = "requires controller-admitted pinned source and full-model CLI execution"]
    fn model_present_source_cli_generate_and_chat_publish_matching_greedy_output() {
        let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory required");
        let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
        let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
            .parse().expect("memory ceiling must be MiB");
        let prompt = "Reply with a short greeting.";
        let mut completed = Vec::new();
        for task in [Task::Generate, Task::Chat] {
            let mut args = super::super::tests::args("generate", &["--context-tokens", "256",
                "--max-new-tokens", "4", "--max-output-bytes", "256"]);
            args.source_dir = Some(PathBuf::from(&source)); args.source_manifest = Some(PathBuf::from(&manifest));
            args.memory_mib = memory;
            let text = match task {
                Task::Generate => prompt.to_owned(),
                Task::Chat => serde_json::json!([{"role":"user","content":prompt}]).to_string(),
            };
            let limits = args.validate().unwrap();
            let mut output = Vec::new();
            execute(task, args, limits, &mut text.as_bytes(), &mut output).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(value["scope"], "real-source-current-candidate-int4");
            assert_eq!(value["evidence"], "non_authoritative");
            assert_eq!(value["candidate_format"], "fnlp-int4-source-candidate-v1");
            assert_eq!(value["model_content_sha256"].as_str().unwrap().len(), 64);
            assert_eq!(value["output"]["result"]["numerics_profile"], crate::native_engine::strict_int4::STRICT_INT4_PROFILE);
            completed.push(value);
        }
        for path in ["token_ids", "content", "native_work"] {
            assert_eq!(completed[0]["output"]["result"][path], completed[1]["output"]["result"][path]);
        }
        assert_eq!(completed[0]["output"]["model_work"], completed[1]["output"]["model_work"]);
        assert_eq!(completed[0]["model_content_sha256"], completed[1]["model_content_sha256"]);
    }

}
