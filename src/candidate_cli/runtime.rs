//! One-shot candidate CLI on the existing process resource host.
pub(super) mod source_tasks;
pub(super) mod batch_tasks;
pub(super) mod scored_tasks;
pub(super) mod extraction;
#[cfg(all(feature = "metadata-store", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(super) mod owned_jobs;
use std::{fs::File, time::{Duration, Instant}};
use super::*;
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    hosted::{CancellationToken, LoadLimits, NativeLimits, RunLimits},
    native_engine::{artifact_bridge::{ArtifactIdentity, ArtifactLoadBudget,
        CheckedArtifactSource, CurrentCandidateArtifactSource},
        strict_int8::STRICT_INT8_EXECUTION},
    tasks::{chat::{ChatLimits, ChatRequest, GenerateRequest, quantized::Int8ChatPlanner}, ir::TaskBudget},
    tokenizer::pinned_controls};

pub(super) fn execute(task: Task, args: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let started = Instant::now();
    let elapsed_limit = Duration::from_secs(args.timeout_seconds);
    let remaining = || elapsed_limit.checked_sub(started.elapsed())
        .filter(|duration| !duration.is_zero()).ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: limits.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Declared before every preparation allocation: drops AFTER input,
    // metadata, planners and completed-result staging on every return path.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let text = if args.input.as_os_str() == "-" {
        read_input(input, args.max_input_bytes)?
    } else {
        let mut file = File::open(&args.input).map_err(|_| CandidateError::Input)?;
        read_input(&mut file, args.max_input_bytes)?
    };
    let messages = match task {
        Task::Chat => Some(parse_messages(&text, args.max_input_bytes)?),
        Task::Generate => None,
    };
    remaining()?;
    // Read the checked current-candidate metadata without materializing any
    // model weights. The source reader performs its own bounded verification.
    // It is dropped before loading, so no duplicate envelope is retained.
    let facts = {
        let source = CurrentCandidateArtifactSource::open(&args.model).map_err(|_| CandidateError::Model)?;
        source.identity().clone()
    };
    let identity = candidate_identity(&facts)?;
    let controls = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    let budget = TaskBudget {
        max_input_tokens: u32::try_from(limits.max_prompt_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_tokens: u32::try_from(args.max_new_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_bytes: limits.result_bytes as u64, max_grammar_states: 1,
        max_kv_bytes: limits.kv_bytes,
    };
    let planner = Int8ChatPlanner::pinned(controls.template_controls(), eos, identity, budget,
        ChatLimits { max_messages: 128, max_message_bytes: args.max_input_bytes,
            max_total_message_bytes: args.max_input_bytes, generation: args.generation_limits(limits) })
        .map_err(|_| CandidateError::Planning)?;
    let options = args.options(eos)?;
    let prepared = match messages {
        Some(messages) => planner.plan_chat(&ChatRequest { item_id: "cli".to_owned(), sample_index: 0,
            messages, generation: options, budget }),
        None => planner.plan_generate(&GenerateRequest { item_id: "cli".to_owned(), sample_index: 0,
            prompt: text, generation: options, budget }),
    }.map_err(|_| CandidateError::Planning)?;
    if prepared.planned_work().forward_positions > args.context_tokens as u64 {
        return Err(CandidateError::Planning);
    }
    let cancellation = CancellationToken::default();
    let model = engine.load_current_candidate_int8(args.model.clone(), LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(limits.weight_bytes, 64 * 1024),
        tokenizer_and_metadata_bytes: 128 * MIB, allocator_reserve_bytes: 64 * MIB,
        // Loader checkpoints are a separately declared finite stage budget;
        // they do not replenish the native execution checkpoint allowance.
        run: RunLimits { max_elapsed: remaining()?, max_checkpoints: 64,
            cleanup_reserve_bytes: 65_536 },
    }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
    // A pathname can change between metadata inspection and loading. Compare
    // ALL retained facts (including source-root identity), not just a filename.
    // The host independently checks the prepared plan against resident weights.
    if model.artifact_identity() != &facts { return Err(CandidateError::Identity); }
    let result = engine.execute_int8_chat(&model, prepared, 1, NativeLimits {
        context_tokens: args.context_tokens, allocator_reserve_bytes: 64 * MIB,
        run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
            cleanup_reserve_bytes: 65_536 },
    }, SAMPLER_BYTES, cancellation).map_err(|_| CandidateError::Execution)?;
    remaining()?;
    let response = CandidateResponse { schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id,
        output: &result };
    // Keep both the hosted output charge and preparation/staging charge alive
    // until serialization, external delivery and flush have actually finished.
    publish(&response, limits.result_bytes + 4096, output)
}

#[derive(Serialize)]
struct CandidateResponse<'a, T: Serialize> {
    schema_version: u32,
    scope: &'static str,
    evidence: &'static str,
    model_id: &'a str,
    source_revision: &'a str,
    source_root_sha256: &'a str,
    logical_model_sha256: &'a str,
    quant_recipe: &'a str,
    output: &'a T,
}

/// Model facts come from the checked file, never from a fabricated fixture or
/// an arbitrary user identity. The planner replaces all uncompiled task,
/// prompt, schema, template and policy fields before the identity is admitted.
/// No candidate identity is offered as a release receipt or cache certificate.
fn candidate_identity(facts: &ArtifactIdentity) -> Result<ExecutionIdentity, CandidateError> {
    if facts.model_id != "Nanbeige4.2-3B"
        || facts.revision != "f56ec5a9650268aa098496734743c25ea778bd2d"
        || facts.recipe_id.is_empty() || facts.recipe_id.len() > 256
    { return Err(CandidateError::Identity); }
    let logical_model_digest = Sha256Digest::from_hex(&facts.logical_model_sha256)
        .map_err(|_| CandidateError::Identity)?;
    Sha256Digest::from_hex(&facts.source_root_sha256).map_err(|_| CandidateError::Identity)?;
    let absent = Sha256Digest::of_bytes(b"null");
    // This route has no native sidecars. Bind the actual portable layout and
    // empty sidecar set, NOT a made-up artifact or native-cache checksum.
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new()))
        .map_err(|_| CandidateError::Identity)?;
    ExecutionIdentity::new(ExecutionIdentity {
        schema_version: 1, source_revision: facts.revision.clone(), logical_model_digest,
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: absent,
        template_digest: absent, task_spec: "uncompiled-candidate-chat".to_owned(),
        taskir_digest: absent, prompt_digest: absent, grammar_compiler_version: "none".to_owned(),
        schema_digest: absent, numerics_profile: NumericsProfile::StrictQuantized { version: 1 },
        kv_dtype: "bf16".to_owned(), sampler_version: "uncompiled".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }).map_err(|_| CandidateError::Identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn facts() -> ArtifactIdentity {
        ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
            revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
            recipe_id: "metadata-only-unit-fixture".to_owned(),
            source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
    }
    #[test]
    fn candidate_identity_keeps_actual_model_facts_and_explicit_nonrelease_format() {
        let facts = facts(); let identity = candidate_identity(&facts).unwrap();
        assert_eq!(identity.source_revision, facts.revision);
        assert_eq!(identity.quant_recipe, facts.recipe_id);
        assert_eq!(identity.logical_model_digest.to_hex(), facts.logical_model_sha256);
        assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
        assert_eq!(identity.backend_semantic_version, STRICT_INT8_EXECUTION);
        assert!(identity.artifact_format.contains("current-candidate"));
        assert_eq!(identity.thinking_mode, ThinkingMode::Disabled);
        assert_eq!(identity.tool_mode, ToolMode::None);
    }
    #[test]
    fn malformed_or_different_model_facts_never_get_an_identity() {
        for axis in 0..5 {
            let mut changed = facts();
            match axis { 0 => changed.model_id = "other".to_owned(),
                1 => changed.revision = "other".to_owned(), 2 => changed.recipe_id.clear(),
                3 => changed.logical_model_sha256 = "invalid".to_owned(),
                _ => changed.source_root_sha256 = "invalid".to_owned() }
            assert!(candidate_identity(&changed).is_err());
        }
    }
    #[test]
    fn real_pinned_planner_replaces_uncompiled_fields_before_admission() {
        let args = super::super::tests::args(&[]);
        let limits = args.validate().unwrap();
        let base = candidate_identity(&facts()).unwrap();
        let controls = pinned_controls::pinned().unwrap();
        let eos = 166_101;
        let budget = TaskBudget { max_input_tokens: limits.max_prompt_tokens as u32,
            max_output_tokens: args.max_new_tokens as u32, max_output_bytes: limits.result_bytes as u64,
            max_grammar_states: 1, max_kv_bytes: limits.kv_bytes };
        let planner = Int8ChatPlanner::pinned(controls.template_controls(), eos, base.clone(), budget,
            ChatLimits { generation: args.generation_limits(limits), ..ChatLimits::default() }).unwrap();
        let plan = planner.plan_generate(&GenerateRequest { item_id: "cli".to_owned(), sample_index: 0,
            prompt: "é <tool_call> <|im_start|> 上海".to_owned(), generation: args.options(eos).unwrap(), budget }).unwrap();
        let identity = plan.execution_identity();
        assert_ne!(identity.task_spec, base.task_spec);
        assert_ne!(identity.sampler_version, base.sampler_version);
        for (compiled, uncompiled) in [(identity.tokenizer_digest, base.tokenizer_digest),
            (identity.template_digest, base.template_digest), (identity.prompt_digest, base.prompt_digest),
            (identity.taskir_digest, base.taskir_digest), (identity.decision_policy_digest, base.decision_policy_digest)] {
            assert_ne!(compiled, uncompiled);
        }
        assert!(plan.planned_work().forward_positions <= args.context_tokens as u64);
    }
}