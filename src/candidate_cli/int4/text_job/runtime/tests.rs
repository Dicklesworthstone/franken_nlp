//! UNRUN runtime-boundary and memory/privacy checks without native model IO.
use super::*;
use super::super::tests::{command, lifetime};
use crate::jobs::{JobId, JobProgress};

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected private input access"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected report publication"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unexpected report flush"); }
}

#[test]
fn owned_boundary_rechecks_options_before_runtime_or_private_file_access() {
    let (mut c, args) = command("start", "generate", &[]);
    let admitted = c.args.validate(&args).unwrap();
    c.args.storage.max_input_lines = 0;
    assert_eq!(execute(c.mode, args, c.args, admitted, NoRead, &mut NoWrite).unwrap_err().code,
        "explicit_text_job_storage");
}

#[test]
fn fixed_configuration_uses_exact_context_kv_and_effective_sampling_policy() {
    let seed = "cd".repeat(32);
    for task in ["generate", "chat"] {
        let (c, args) = command("resume", task, &["--context-tokens", "512", "--max-new-tokens", "32",
            "--seed", &seed, "--temperature-milli", "700", "--top-k", "10",
            "--stop", " END\n", "--logprobs", "--logit-bias", "42=-1500"]);
        let common = c.args.validate(&args).unwrap();
        let job = lifetime(&args, common);
        let limits = c.args.lifetime_limits(&args, common, job).unwrap();
        let config = configuration(&c.args, &args, limits, 166_101).unwrap();
        assert_eq!(config.task, c.args.task);
        assert_eq!(config.context_tokens, 512);
        assert_eq!(config.budget.max_input_tokens, 481);
        assert_eq!(config.budget.max_output_tokens, 32);
        assert_eq!(config.budget.max_output_bytes, common.result_bytes as u64);
        assert_eq!(config.budget.max_kv_bytes, Int4MemoryRequirement::for_context(512, args.group_width).unwrap().kv_bytes);
        assert_eq!(config.native.max_kv_bytes, config.budget.max_kv_bytes);
        assert_eq!(config.native.max_sampler_bytes, SAMPLER_BYTES);
        assert_eq!(config.native.native.max_forward_positions, job.max_work.model.forward_positions);
        assert_eq!(config.native.native.max_attention_pairs, job.max_work.model.attention_pairs);
        assert_eq!(config.native.native.max_projection_work, job.max_work.model.projections);
        assert_eq!(config.planning.max_messages, 128);
        assert_eq!(config.planning.max_message_bytes, args.max_input_bytes);
        assert_eq!(config.planning.max_total_message_bytes, args.max_input_bytes);
        assert_eq!(config.planning.generation.max_prompt_tokens, 481);
        assert_eq!(config.planning.generation.max_sampler_bytes, SAMPLER_BYTES);
        assert!(config.generation == args.options(166_101).unwrap());
        assert!(matches!(config.generation.sampling, GenerationSampling::Seeded {
            effective_seed, temperature_milli: 700, top_k: Some(10), top_p_ppm: 1_000_000,
        } if effective_seed == [0xcd; 32]));
    }
}

#[test]
fn pure_host_quote_matches_the_runtime_owner_and_rejects_missing_staging() {
    let (c, args) = command("start", "generate", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, common);
    let limits = c.args.lifetime_limits(&args, common, job).unwrap();
    let host = host_limits(&args, limits, Duration::from_secs(30));
    check_host_quote(host, job, limits.buffer_bytes).unwrap();
    assert_eq!(check_host_quote(host, job, limits.buffer_bytes - 1).unwrap_err().code, "job_host_memory_quote");
    let mut missing = host; missing.journal_reserve_bytes = 0;
    assert_eq!(check_host_quote(missing, job, limits.buffer_bytes).unwrap_err().code, "job_host_admission");
    assert_eq!(host.native.context_tokens, args.context_tokens);
    assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
    assert_eq!(host.native.run.max_elapsed, Duration::from_secs(30));
    assert_eq!(host.native.allocator_reserve_bytes, ALLOCATOR_BYTES);
}

#[test]
fn native_quote_includes_full_kv_sampler_and_output_vectors_in_addition_to_population() {
    let (c, args) = command("start", "chat", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, common);
    let limits = c.args.lifetime_limits(&args, common, job).unwrap();
    let config = configuration(&c.args, &args, limits, 166_101).unwrap();
    let host = host_limits(&args, limits, Duration::from_secs(30));
    let total = simultaneous_memory(args.group_width, config.native, config.budget, host, job, common).unwrap();
    assert!(total <= limits.common.minimum_memory_bytes);
    assert!(total > limits.buffer_bytes + common.preparation_bytes + common.load.weight_payload_bytes + common.native.kv_bytes);
    let mut sampler = config.native; sampler.max_sampler_bytes += MIB;
    assert_eq!(simultaneous_memory(args.group_width, sampler, config.budget, host, job, common).unwrap(), total + MIB);
    let mut output = config.budget; output.max_output_tokens += 1;
    assert_eq!(simultaneous_memory(args.group_width, config.native, output, host, job, common).unwrap(), total + 8);
    let mut output = config.budget; output.max_output_bytes += 1024;
    assert_eq!(simultaneous_memory(args.group_width, config.native, output, host, job, common).unwrap(), total + 4096);
    let mut context = host; context.native.context_tokens -= 1;
    assert_eq!(simultaneous_memory(args.group_width, config.native, config.budget, context, job, common).unwrap_err().code,
        "text_job_native_quote");
    let mut kv = config.native; kv.max_kv_bytes -= 1;
    assert_eq!(simultaneous_memory(args.group_width, kv, config.budget, host, job, common).unwrap_err().code,
        "text_job_native_quote");
}

#[test]
fn complete_envelope_publishes_only_public_model_identity_and_authenticated_progress() {
    let (c, args) = command("resume", "chat", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, common);
    let id = JobId([0x12; 16]);
    // Serialization fixture only: this is not a durable or native success proof.
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: job.max_work, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, job, true,
        RunMode::Resume { discard_uncommitted: true }, "chat").unwrap();
    let content = "ab".repeat(32); let source = "cd".repeat(32);
    let response = Response {
        protocol: "fnlp-candidate-int4-text-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
        source_root_sha256: &source, model_content_sha256: &content,
        candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
        group_width: args.group_width, integer_kernel: common.kernel.kind().id(),
        snapshot: Some(SnapshotRecord::new("loaded", common.snapshot_need)), output: &report,
    };
    let mut bytes = Vec::new(); publish(&response, job_cli::REPORT_BYTES, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-text-job-v1");
    assert_eq!(value["evidence"], "non_authoritative");
    assert_eq!(value["model_content_sha256"], content);
    assert_eq!(value["group_width"], args.group_width);
    assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(value["output"]["operation"], "resume");
    assert_eq!(value["output"]["task"], "chat");
    assert_eq!(value["output"]["attempts"], 3);
    assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(job.max_work).unwrap());
    for private in ["text", "history", "prompt", "seed", "effective_seed", "options", "token_ids", "logprobs",
        "key", "job_dir", "results", "decision_policy_digest"] {
        assert!(value.get(private).is_none());
        assert!(value["output"].get(private).is_none());
    }
    assert!(bytes.ends_with(b"\n"));
    assert_eq!(publish(&response, 1, &mut NoWrite), Err(CandidateError::Output));
}
