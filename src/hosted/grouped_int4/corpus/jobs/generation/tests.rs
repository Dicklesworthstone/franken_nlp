//! UNRUN admission/ownership fixtures; no model inference or crash proof.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    hosted::RunLimits,
    jobs::population::PopulationReadLimits,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
};
use std::{io::Cursor, time::Duration};

const WIDTH: usize = 32;
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-generation-host-admission");
    ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH,
        )).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fnlp-sampler-v1".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }
}
fn fixture() -> (TaskBudget, GenerationEnvelope, JobLimits, JobHostLimits) {
    let memory = Int4MemoryRequirement::for_context(64, WIDTH).unwrap();
    let work = Int4Work::for_sequence(WIDTH, 0, 64, 16 * NANBEIGE_VOCAB_SIZE).unwrap();
    let task = TaskBudget { max_input_tokens: 48, max_output_tokens: 16,
        max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: memory.kv_bytes };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 131072, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(work, 0) };
    let envelope = GenerationEnvelope { context_tokens: 64, kv_bytes: memory.kv_bytes,
        native: Int4GenerationBudget { native: Int4RunBudget::exact(work),
            max_kv_bytes: memory.kv_bytes, max_sampler_bytes: 32 << 20 },
        lifetime: job.max_work, result_bytes: task.max_output_bytes, output_tokens: 16 };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 10000,
                cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 131072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (task, envelope, job, host)
}
fn request(identity: &ExecutionIdentity, envelope: GenerationEnvelope) -> Int4GenerationAdmission<'_> {
    Int4GenerationAdmission { identity,
        model_work: Int4Work::for_sequence(WIDTH, 0, 48, 8 * NANBEIGE_VOCAB_SIZE).unwrap(),
        sampler_bytes: 16 << 20, kv_reservation_bytes: envelope.kv_bytes,
        max_result_bytes: envelope.result_bytes, max_output_tokens: envelope.output_tokens }
}

#[test]
fn full_population_context_and_sampler_coexist_in_the_same_admission() {
    let (task, envelope, job, host) = fixture();
    let need = requirements(WIDTH, task, envelope, job, host).unwrap();
    assert_eq!(need.input_bytes, host.required_buffer_bytes(job).unwrap());
    assert_eq!(need.native.kv_bytes, task.max_kv_bytes);
    assert_eq!(need.workspace_bytes, need.native.rope_bytes + need.native.scratch_payload_bound
        + host.native.allocator_reserve_bytes + envelope.native.max_sampler_bytes);
    // The fixed task result and larger immutable storage cap have separate
    // accounting roles; the per-item claim must not silently widen to storage.
    assert!(envelope.result_bytes < job.max_result_bytes as u64);
}

#[test]
fn looser_or_tighter_context_and_kv_cannot_change_the_frozen_recipe() {
    for axis in 0..6 {
        let (mut task, mut envelope, job, mut host) = fixture();
        match axis { 0 => task.max_kv_bytes -= 1, 1 => task.max_kv_bytes += 1,
            2 => envelope.native.max_kv_bytes -= 1, 3 => envelope.kv_bytes += 1,
            4 => envelope.context_tokens -= 1, _ => host.native.context_tokens += 1 }
        assert!(requirements(WIDTH, task, envelope, job, host).is_err(), "axis {axis}");
    }
}

#[test]
fn missing_reserves_and_workspace_overflow_refuse_before_input() {
    for axis in 0..7 {
        let (task, mut envelope, job, mut host) = fixture();
        match axis { 0 => host.preparation_reserve_bytes = 0, 1 => host.io_reserve_bytes = 0,
            2 => host.journal_reserve_bytes = 0, 3 => host.serialization_reserve_bytes = 0,
            4 => host.native.allocator_reserve_bytes = 0, 5 => envelope.native.max_sampler_bytes = 0,
            _ => envelope.native.max_sampler_bytes = u64::MAX }
        assert!(requirements(WIDTH, task, envelope, job, host).is_err(), "axis {axis}");
    }
}

#[test]
fn every_actual_work_axis_remains_within_the_persisted_lifetime_envelope() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    let req = request(&base, envelope);
    for axis in 0..5 {
        let mut tighter = envelope;
        tighter.lifetime = JobWork::from_int4(req.model_work, 0);
        match axis { 0 => tighter.lifetime.model.forward_positions -= 1,
            1 => tighter.lifetime.model.projected_logits -= 1,
            2 => tighter.lifetime.model.attention_pairs -= 1,
            3 => tighter.lifetime.model.projections.dot_products -= 1,
            _ => tighter.lifetime.model.projections.multiply_accumulates -= 1 }
        assert!(check_request(WIDTH, &base, tighter, &req).is_err(), "axis {axis}");
    }
}

#[test]
fn a_different_group_partition_cannot_relabel_cheaper_work_as_the_actual_model() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    let mut req = request(&base, envelope);
    assert!(check_request(WIDTH, &base, envelope, &req).is_ok());
    req.model_work = Int4Work::for_sequence(128, 0, 48, 8 * NANBEIGE_VOCAB_SIZE).unwrap();
    assert!(check_request(WIDTH, &base, envelope, &req).is_err());
    req.model_work = request(&base, envelope).model_work;
    req.model_work.attention_pairs -= 1;
    assert!(check_request(WIDTH, &base, envelope, &req).is_err());
}

#[test]
fn one_empty_cache_sequence_must_fit_the_resident_context_and_projection_schedule() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    for (positions, rows) in [(0, 0), (65, NANBEIGE_VOCAB_SIZE), (1, 2 * NANBEIGE_VOCAB_SIZE), (48, 1)] {
        let mut req = request(&base, envelope);
        req.model_work = Int4Work::for_sequence(WIDTH, 0, positions, rows).unwrap();
        assert!(check_request(WIDTH, &base, envelope, &req).is_err());
    }
}

#[test]
fn per_item_prompt_and_sampling_key_change_but_fixed_task_assets_and_modes_cannot() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    let mut item = base.clone();
    item.prompt_digest = Sha256Digest::of_bytes(b"different exact prompt");
    item.taskir_digest = Sha256Digest::of_bytes(b"different item task graph");
    item.decision_policy_digest = Sha256Digest::of_bytes(b"different item/sample addressed key");
    assert!(check_request(WIDTH, &base, envelope, &request(&item, envelope)).is_ok());
    for axis in 0..8 {
        let mut wrong = item.clone();
        match axis { 0 => wrong.task_spec = "chat-v1".to_owned(),
            1 => wrong.tokenizer_digest = Sha256Digest::of_bytes(b"wrong tokenizer"),
            2 => wrong.template_digest = Sha256Digest::of_bytes(b"wrong template"),
            3 => wrong.sampler_version = "different-sampler".to_owned(),
            4 => wrong.thinking_mode = ThinkingMode::Enabled, 5 => wrong.tool_mode = ToolMode::Json,
            6 => wrong.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            _ => wrong.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=128") }
        assert!(check_request(WIDTH, &base, envelope, &request(&wrong, envelope)).is_err(), "axis {axis}");
    }
}

#[test]
fn output_and_sampler_claims_cannot_shrink_or_widen_the_fixed_admission() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    for axis in 0..8 {
        let mut req = request(&base, envelope);
        match axis { 0 => req.max_result_bytes -= 1, 1 => req.max_result_bytes += 1,
            2 => req.max_output_tokens -= 1, 3 => req.max_output_tokens += 1,
            4 => req.sampler_bytes = 0, 5 => req.sampler_bytes = envelope.native.max_sampler_bytes + 1,
            6 => req.kv_reservation_bytes -= 1, _ => req.kv_reservation_bytes += 1 }
        assert!(check_request(WIDTH, &base, envelope, &req).is_err(), "axis {axis}");
    }
}

#[test]
fn owned_population_factory_and_native_result_fit_the_existing_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4GenerationJobPlanner>();
    send::<GenerationInput<Cursor<Vec<u8>>>>();
    send::<crate::tasks::chat::grouped_int4::Int4ChatResult>();
    let _entry = NlpEngine::job_int4_generation::<Cursor<Vec<u8>>>;
}
