//! Owned redaction admission fixtures. They exercise real budget/ownership
//! boundaries without native model execution or durable-execution claims.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    hosted::{NativeLimits, RunLimits},
    jobs::{JobSecret, JobWork, population::PopulationReadLimits},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, grouped_int4::long::Int4SourceMapLimits},
        redact::{actions::RedactionAction, grouped_int4::long::Int4LongRedactionConfig,
            pseudonym::PseudonymKey}},
    validation::grounded_fields::GroundingBudget,
};
use std::{io::Cursor, time::Duration};

fn config() -> RedactionCorpusConfig<Int4RedactionJobConfig> {
    let width = 127;
    let d = Sha256Digest::of_bytes(b"int4-owned-redaction-host-fixture");
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
        )).unwrap()), tokenizer_digest: d, template_digest: d, task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    let per = constrained_int4::planned_work(width, 64, 64).unwrap();
    let mut item = per;
    for _ in 1..6 { item = item.checked_add(per).unwrap(); }
    let mut total = item;
    for _ in 1..4 { total = total.checked_add(item).unwrap(); }
    let mut request = RedactionRequest::default();
    request.actions.default_action = RedactionAction::Mask;
    request.edit_budget.max_output_bytes = 65_536;
    request.rule_budget.max_input_bytes = 65_536;
    RedactionCorpusConfig {
        batch: Int4RedactionJobConfig {
            batch: Int4DocumentRedactionBatchConfig {
                ner_identity: identity,
                detector: Int4LongRedactionConfig {
                    ner: Default::default(),
                    per_chunk: TaskBudget { max_input_tokens: 64, max_output_tokens: 64,
                        max_output_bytes: 65_536, max_grammar_states: 4096,
                        max_kv_bytes: Int4MemoryRequirement::for_context(128, width).unwrap().kv_bytes },
                    planning: SourcePlanningLimits { max_input_bytes: 128, max_context_tokens: 128, ..Default::default() },
                    mapping: Int4SourceMapLimits {
                        chunks: ChunkLimits { max_input_bytes: 65_536, max_chunk_bytes: 128,
                            max_chunk_tokens: 64, context_tokens: 128, reserved_tokens: 64,
                            max_chunks: 3, ..Default::default() },
                        reduction: ExecutionLimits { max_value_bytes: 65_536, max_live_value_bytes: 131_072,
                            max_result_bytes: 65_536, ..Default::default() },
                        max_model_work: item, mask_limits: Default::default(), mask_visits_per_chunk: 1000,
                        max_mask_visits: 6000,
                        verification: GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 1_000_000 },
                    },
                    max_result_bytes: 65_536,
                },
                request, max_model_work: total, max_mask_visits: 24_000,
            },
            max_source_bytes: 4096,
        },
        edit_reserve_bytes: 1 << 20,
    }
}
fn limits(c: &RedactionCorpusConfig<Int4RedactionJobConfig>) -> (JobLimits, JobHostLimits) {
    let job = JobLimits {
        max_items: 4, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 131072, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 8,
        max_work: JobWork::from_int4(c.batch.batch.max_model_work, c.batch.batch.max_mask_visits),
    };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
                cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 1 << 20,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (job, host)
}
fn admission_request(fixed: &FixedItem) -> Int4RedactionAdmission<'_> {
    Int4RedactionAdmission {
        identity: &fixed.identity, model_work: fixed.model_work, mask_node_visits: fixed.mask_visits,
        mask_limits: fixed.mask_limits, kv_reservation_bytes: fixed.kv_bytes, max_result_bytes: fixed.result_bytes,
    }
}
fn secret() -> RedactionPseudonyms {
    let mut namespace = String::with_capacity(4096);
    namespace.push_str("fixed-owned-scope");
    RedactionPseudonyms { key: Arc::new(PseudonymKey::from_bytes(&[0x93; 32], "opaque-key-id").unwrap()), namespace }
}

#[test]
fn the_job_quote_keeps_complete_population_map_edit_recipe_and_native_storage_together() {
    let c = config(); let (job, host) = limits(&c);
    let required = job_requirements(&c, job, host, 127, None).unwrap();
    assert_eq!(required.input_bytes, host.required_buffer_bytes(job).unwrap()
        + long::document_workspace_bytes(&c.batch.batch.detector, c.edit_reserve_bytes).unwrap()
        + 4 * INT4_REDACTION_JOB_RECIPE_BYTES as u64);
    assert_eq!(required.native.kv_bytes, c.batch.batch.detector.per_chunk.max_kv_bytes);
    assert_eq!(required.workspace_bytes, required.native.rope_bytes
        + required.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    assert_eq!(INT4_REDACTION_JOB_RECIPE_BYTES, 1024 * 1024 - 1024);
}

#[test]
fn a_larger_stored_frame_ceiling_does_not_expand_native_output_authority() {
    let c = config(); let (job, host) = limits(&c);
    let required = job_requirements(&c, job, host, 127, None).unwrap();
    assert!(job.max_result_bytes as u64 > c.batch.batch.detector.max_result_bytes);
    let fixed = FixedItem::new(&c.batch.batch, 127, required.native.kv_bytes, job.max_result_bytes as u64);
    assert_eq!(fixed.result_bytes, c.batch.batch.detector.max_result_bytes);
    check_admission(&fixed, &admission_request(&fixed)).unwrap();
    let mut enlarged = admission_request(&fixed); enlarged.max_result_bytes = job.max_result_bytes as u64;
    let error = check_admission(&fixed, &enlarged).unwrap_err();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    let mut too_small = job; too_small.max_result_bytes = c.batch.batch.detector.max_result_bytes as usize - 1;
    assert!(job_requirements(&c, too_small, host, 127, None).is_err());
}

#[test]
fn actual_allocated_kv_and_each_planning_context_must_fit_the_host() {
    let c = config(); let (job, host) = limits(&c);
    for position in [0, 64, 127] {
        let mut small = host; small.native.context_tokens = position;
        assert!(job_requirements(&c, job, small, 127, None).is_err());
    }
    let mut short = config(); short.batch.batch.detector.per_chunk.max_kv_bytes -= 1;
    assert!(job_requirements(&short, job, host, 127, None).is_err());
    let mut too_long = config(); too_long.batch.batch.detector.planning.max_context_tokens = 129;
    assert!(job_requirements(&too_long, job, host, 127, None).is_err());
    let mut too_long = config(); too_long.batch.batch.detector.mapping.chunks.context_tokens = 129;
    assert!(job_requirements(&too_long, job, host, 127, None).is_err());
    assert!(job_requirements(&c, job, host, 0, None).is_err());
}

#[test]
fn all_persisted_work_axes_and_mask_authority_must_equal_the_frozen_corpus_ceiling() {
    let c = config(); let (job, host) = limits(&c);
    for axis in 0..6 {
        for increase in [false, true] {
            let mut changed = job;
            let counter = match axis {
                0 => &mut changed.max_work.model.forward_positions,
                1 => &mut changed.max_work.model.projected_logits,
                2 => &mut changed.max_work.model.attention_pairs,
                3 => &mut changed.max_work.model.projections.dot_products,
                4 => &mut changed.max_work.model.projections.multiply_accumulates,
                _ => &mut changed.max_work.mask_node_visits,
            };
            if increase { *counter += 1; } else { *counter -= 1; }
            assert!(job_requirements(&c, changed, host, 127, None).is_err(), "{axis} {increase}");
        }
    }
}

#[test]
fn original_decoded_text_encoded_envelope_and_transformed_source_have_distinct_bounds() {
    let c = config(); let (job, host) = limits(&c);
    assert!(c.batch.max_source_bytes > c.batch.batch.detector.planning.max_input_bytes);
    assert!(job.max_input_bytes_per_item > c.batch.max_source_bytes);
    assert!(c.batch.batch.detector.mapping.chunks.max_input_bytes > job.max_input_bytes_per_item);
    job_requirements(&c, job, host, 127, None).unwrap();
    for cap in [0, 65_537] {
        let mut wrong = config(); wrong.batch.max_source_bytes = cap;
        assert!(job_requirements(&wrong, job, host, 127, None).is_err());
    }
}

#[test]
fn missing_host_reserves_and_overflow_refuse_before_any_storage_access() {
    let c = config(); let (job, host) = limits(&c);
    for axis in 0..6 {
        let mut changed = config(); let mut limits = host;
        match axis {
            0 => changed.edit_reserve_bytes = 0,
            1 => limits.native.allocator_reserve_bytes = 0,
            2 => limits.preparation_reserve_bytes = 0,
            3 => limits.io_reserve_bytes = 0,
            4 => limits.journal_reserve_bytes = 0,
            _ => limits.serialization_reserve_bytes = 0,
        }
        assert!(job_requirements(&changed, job, limits, 127, None).is_err(), "{axis}");
    }
    let mut overflowing = host; overflowing.serialization_reserve_bytes = u64::MAX;
    assert!(job_requirements(&c, job, overflowing, 127, None).is_err());
    let mut oversized_map = config();
    oversized_map.batch.batch.detector.mapping.verification.max_matches = usize::MAX;
    assert!(job_requirements(&oversized_map, job, host, 127, None).is_err());
}

#[test]
fn namespace_capacity_and_key_owner_are_in_addition_to_all_job_buffers() {
    let c = config(); let (job, host) = limits(&c); let secret = secret();
    let without = job_requirements(&c, job, host, 127, None).unwrap();
    let with = job_requirements(&c, job, host, 127, Some(&secret)).unwrap();
    assert_eq!(with.input_bytes - without.input_bytes, secret.namespace.capacity() as u64 + 2048);
    for namespace in [String::new(), "x".repeat(257)] {
        let invalid = RedactionPseudonyms { key: secret.key.clone(), namespace };
        assert!(job_requirements(&c, job, host, 127, Some(&invalid)).is_err());
    }
}

#[test]
fn pseudonym_scope_checks_the_actual_key_independently_of_retention_authentication() {
    let mut c = config(); let secret = secret();
    c.batch.batch.request.actions.default_action = RedactionAction::Pseudonymize;
    c.batch.batch.request.actions.expected_key_commitment = Some(secret.key.commitment());
    let scope = pseudonym_context(Some(&secret), &c.batch.batch.request).unwrap().unwrap();
    assert_eq!(scope.identity().key_commitment, secret.key.commitment());
    assert_eq!(scope.identity().encoding, crate::tasks::redact::pseudonym::PseudonymEncoding::Full256);
    assert!(pseudonym_context(None, &c.batch.batch.request).is_err());
    let wrong = RedactionPseudonyms {
        key: Arc::new(PseudonymKey::from_bytes(&[0x94; 32], "opaque-key-id").unwrap()),
        namespace: secret.namespace.clone(),
    };
    assert!(pseudonym_context(Some(&wrong), &c.batch.batch.request).is_err());
    // The retained job secret has its own non-Debug, non-Clone owner and is
    // supplied separately in SourceJobRequest, never inferred from this key.
    fn send<T: Send + 'static>() {}
    send::<JobSecret>();
}

#[test]
fn complete_document_work_remains_independent_contexts_inside_the_durable_guard() {
    let c = config(); let (job, host) = limits(&c);
    let required = job_requirements(&c, job, host, 127, None).unwrap();
    let fixed = FixedItem::new(&c.batch.batch, 127, required.native.kv_bytes, job.max_result_bytes as u64);
    assert!(fixed.model_work.forward_positions > host.native.context_tokens as u64);
    let concatenated = Int4Work::for_sequence(127, 0, fixed.model_work.forward_positions as usize,
        fixed.model_work.projected_logits as usize).unwrap();
    assert_ne!(fixed.model_work.attention_pairs, concatenated.attention_pairs);
    check_admission(&fixed, &admission_request(&fixed)).unwrap();
    let mut wrong = admission_request(&fixed); wrong.model_work = concatenated;
    assert!(check_admission(&fixed, &wrong).unwrap_err().stop);
}

#[test]
fn owned_population_preparation_secrets_and_reader_cross_one_physical_boundary() {
    fn send<T: Send + 'static>() {}
    send::<JobPreparation>();
    send::<JobInput<Cursor<Vec<u8>>>>();
    send::<Charged<JobInput<Cursor<Vec<u8>>>>>();
    send::<SourceJobRequest>();
    send::<RedactionCorpusConfig<Int4RedactionJobConfig>>();
    let _host = NlpEngine::job_int4_redact_document::<Cursor<Vec<u8>>>;
}
