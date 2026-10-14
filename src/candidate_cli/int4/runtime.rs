//! End-to-end source verification, charged loading, planning and INT4 execution.
use super::*;
use std::{fs::File, time::{Duration, Instant}};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    hosted::{CancellationToken, Int4LoadLimits, NativeLimits, RunLimits},
    native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelLoadBudget},
    tasks::{chat::{ChatLimits, ChatRequest, GenerateRequest, grouped_int4::Int4ChatPlanner}, ir::TaskBudget}};

pub(super) fn execute(task: Task, args: Int4Args, limits: Int4Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    if limits.minimum_memory_bytes > limits.memory_bytes { return Err(CandidateError::Arguments); }
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
    let request = parse_input(task, text, args.max_input_bytes)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let load = load_limits(&args, limits, remaining()?);
    // No candidate fnlpq reader, catalog lookup, caller identity, weight clone
    // or uncharged alternate constructor can reach this dispatch branch.
    let model = engine.load_source_candidate_int4(args.source_dir.clone(), args.source_manifest.clone(),
        args.group_width, load, cancellation.clone()).map_err(|_| CandidateError::Model)?;
    remaining()?;
    let budget = TaskBudget {
        max_input_tokens: u32::try_from(limits.max_prompt_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_tokens: u32::try_from(args.max_new_tokens).map_err(|_| CandidateError::Arguments)?,
        max_output_bytes: limits.result_bytes as u64, max_grammar_states: 1,
        max_kv_bytes: limits.native.kv_bytes,
    };
    // The real quantized content fingerprint exists only after conversion.
    // Never fabricate a pre-load identity to make a planning claim. Input
    // bytes/JSON/roles/options were checked earlier; exact token planning is here.
    let planner = Int4ChatPlanner::for_model(model.model_identity(), budget, ChatLimits {
        max_messages: 128, max_message_bytes: args.max_input_bytes,
        max_total_message_bytes: args.max_input_bytes, generation: args.generation_limits(limits),
    }).map_err(|_| CandidateError::Planning)?;
    let options = args.options(planner.eos_token_id())?;
    let prepared = match request {
        TextInput::Generate(prompt) => planner.plan_generate(&GenerateRequest {
            item_id: "cli".to_owned(), sample_index: 0, prompt, generation: options, budget }),
        TextInput::Chat(messages) => planner.plan_chat(&ChatRequest {
            item_id: "cli".to_owned(), sample_index: 0, messages, generation: options, budget }),
    }.map_err(|_| CandidateError::Planning)?;
    let identity = prepared.execution_identity().clone();
    let result = engine.execute_int4_chat(&model, prepared, 1, NativeLimits {
        context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES },
    }, SAMPLER_BYTES, PackedInt4Kernel::scalar(), cancellation).map_err(|_| CandidateError::Execution)?;
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let response = SourceResponse {
        schema_version: 1, scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        output: result.result(),
    };
    // Both charges remain live here. The wrapper is explicitly source-candidate
    // evidence, not a canonical fnlpq digest, release receipt or activation claim.
    publish(&response, limits.result_bytes + 4096, output)
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
#[derive(Serialize)]
struct SourceResponse<'a, T: Serialize> {
    schema_version: u32, scope: &'static str, evidence: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, output: &'a T,
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
            group_width: 128, output: &output };
        let mut bytes = Vec::new();
        publish(&response, 4096, &mut bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"], output);
        assert_eq!(value["evidence"], "non_authoritative");
        assert_eq!(value["group_width"], 128);
        for absent in ["logical_model_sha256", "prompt_digest", "source_dir", "source_manifest"] {
            assert!(value.get(absent).is_none());
        }
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
            args.source_dir = PathBuf::from(&source); args.source_manifest = PathBuf::from(&manifest);
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
