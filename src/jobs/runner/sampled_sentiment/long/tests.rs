use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::sentiment::sampled::long::SampledSentimentDocumentAdmission,
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{constrained_int4::json_sampling_workspace_bytes,
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<crate::native_engine::decode::DecodeCancellationKind> { None }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 4096, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
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
fn limits() -> JobLimits {
    let native = native();
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 32768,
        max_snapshot_bytes: 131072, max_result_bytes: 262144, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 8,
        max_work: JobWork::from_int4(native.max_model_work, native.masks.max_visits_per_run) }
}
fn mapping() -> SampledSentimentMapLimits {
    SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 32768,
        max_chunk_bytes: 2048, max_chunk_tokens: 2048, context_tokens: 4096,
        reserved_tokens: 0, max_chunks: 32, max_tokenizer_calls: 4096 },
        max_model_work: native().max_item_work, max_retained_result_bytes: 131072,
        max_result_bytes: 262144 }
}
fn recipe() -> SampledSentimentDocumentJobRecipe {
    let options = SampledSentimentOptions { effective_seed: [37; 32], samples: 3,
        temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
        max_sample_tokens: 256, max_justification_chars: 64 };
    let planning = SampledSentimentLimits { max_input_bytes: 2048, max_result_bytes: 65536,
        ..Default::default() };
    let chunks = SampledSentimentJobRecipe::new(32, task(), options, planning, native(),
        UntrustedEncoding::ByteFallback, limits()).unwrap();
    SampledSentimentDocumentJobRecipe::new(chunks, mapping()).unwrap()
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"sampled-document-job-recipe-fixture-not-native-evidence");
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
fn freeze(recipe: &SampledSentimentDocumentJobRecipe) -> FrozenManifest {
    let line = br#"{"id":"complete","text":"original complete document"}"#;
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]), JobContract { job_id: JobId([9; 16]),
        execution: &identity(), recipe, limits: recipe.chunk_sampling.job },
        [JobInput { id: "complete", original: line, normalized: line }], &mut Continue).unwrap()
}

#[test]
fn whole_document_result_is_not_capped_by_a_single_chunk_envelope() {
    let recipe = recipe();
    assert!(recipe.mapping.max_result_bytes > recipe.chunk_sampling.task_budget.max_output_bytes);
    recipe.check_job_limits(limits()).unwrap();
    assert_eq!(recipe.mapping.chunks.max_input_bytes, limits().max_input_bytes_per_item);
    assert!(recipe.mapping.chunks.max_input_bytes > recipe.chunk_sampling.planning.max_input_bytes);
}

#[test]
fn every_partition_and_reducer_setting_changes_authenticated_resume() {
    let original = freeze(&recipe());
    for axis in 0..11 {
        let mut changed = recipe();
        match axis {
            0 => changed.mapping.chunks.max_input_bytes -= 1,
            1 => changed.mapping.chunks.max_chunk_bytes -= 1,
            2 => changed.mapping.chunks.max_chunk_tokens -= 1,
            3 => changed.mapping.chunks.context_tokens -= 1,
            4 => changed.mapping.chunks.reserved_tokens += 1,
            5 => changed.mapping.chunks.max_chunks -= 1,
            6 => changed.mapping.chunks.max_tokenizer_calls -= 1,
            7 => changed.mapping.max_retained_result_bytes -= 1,
            8 => changed.mapping.max_result_bytes -= 1,
            9 => changed.aggregation_version = "other-reducer",
            _ => changed.item_address_version = "other-document-address",
        }
        assert_eq!(original.binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}

#[test]
fn source_result_and_all_native_axes_must_fit_the_fixed_job_contract() {
    let mut changed = recipe(); changed.mapping.chunks.max_input_bytes += 1;
    assert!(changed.check_job_limits(limits()).is_err());
    let mut changed = recipe(); changed.mapping.max_result_bytes += 1;
    assert!(changed.check_job_limits(limits()).is_err());
    for axis in 0..5 {
        let mut changed = recipe();
        match axis { 0 => changed.mapping.max_model_work.forward_positions += 1,
            1 => changed.mapping.max_model_work.projected_logits += 1,
            2 => changed.mapping.max_model_work.attention_pairs += 1,
            3 => changed.mapping.max_model_work.projections.dot_products += 1,
            _ => changed.mapping.max_model_work.projections.multiply_accumulates += 1 }
        assert!(changed.check_job_limits(limits()).is_err());
    }
    let mut changed = limits(); changed.max_attempts += 1;
    assert!(recipe().check_job_limits(changed).is_err());
}

#[test]
fn reuse_and_prefill_bind_resume_without_reducing_the_cold_attempt_allowance() {
    let cold = recipe(); let original = freeze(&cold);
    let mut reused = recipe();
    reused.chunk_sampling = reused.chunk_sampling.with_reused_prefill().unwrap();
    reused.execution = SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION;
    reused.check_job_limits(limits()).unwrap();
    assert_eq!(original.binding.compare(&freeze(&reused).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(reused.mapping.max_model_work, cold.mapping.max_model_work);
    assert_eq!(reused.chunk_sampling.job.max_work, cold.chunk_sampling.job.max_work);
    let mut prefetched = recipe();
    prefetched.chunk_sampling = prefetched.chunk_sampling.with_prefill(Int4PrefillLimits {
        max_batch_rows: 2, max_extra_scratch_bytes: 64 << 20 }).unwrap();
    assert_eq!(original.binding.compare(&freeze(&prefetched).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    let mut bad = recipe(); bad.execution = SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION;
    assert!(bad.check_job_limits(limits()).is_err());
}

#[test]
fn complete_document_recipe_binds_selected_child_policy_and_unchanged_cold_work() {
    let dense = recipe();
    assert!(serde_json::to_value(&dense).unwrap()["chunk_sampling"].get("selected_rows").is_none());
    for reuse in [false, true] {
        let make = |rows| {
            let mut child = recipe().chunk_sampling.with_selected_rows(
                Some(Int4JsonSparseLimits { max_rows_per_step: rows })).unwrap();
            if reuse { child = child.with_reused_prefill().unwrap(); }
            SampledSentimentDocumentJobRecipe::new(child, mapping()).unwrap()
        };
        let sparse = make(4096);
        sparse.check_job_limits(limits()).unwrap();
        assert_eq!(sparse.execution, if reuse { SAMPLED_SENTIMENT_MAP_SPARSE_REUSED_PREFILL_EXECUTION }
            else { SAMPLED_SENTIMENT_MAP_SPARSE_EXECUTION });
        assert_eq!(sparse.chunk_sampling.checked_selected_rows().unwrap(),
            Some(Int4JsonSparseLimits { max_rows_per_step: 4096 }));
        assert_eq!(sparse.mapping.max_model_work, dense.mapping.max_model_work);
        assert_eq!(sparse.chunk_sampling.job.max_work, dense.chunk_sampling.job.max_work);
        assert_eq!(sparse.chunk_sampling.sampling.effective_seed, dense.chunk_sampling.sampling.effective_seed);
        assert_eq!(freeze(&dense).binding.compare(&freeze(&sparse).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
        assert_eq!(freeze(&sparse).binding.compare(&freeze(&make(4095)).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
        let mut expected = serde_json::to_value(&sparse).unwrap();
        expected["chunk_sampling"].as_object_mut().unwrap().remove("selected_rows");
        expected["execution"] = serde_json::json!(SAMPLED_SENTIMENT_MAP_EXECUTION);
        expected["chunk_sampling"]["execution"] = serde_json::json!(SAMPLED_SENTIMENT_EXECUTION);
        expected["chunk_sampling"]["native_execution"] = serde_json::json!(INT4_JSON_SAMPLED_EXECUTION);
        assert_eq!(expected, serde_json::to_value(&dense).unwrap());
        for axis in 0..5 {
            let mut changed = make(4096);
            match axis {
                0 => changed.execution = SAMPLED_SENTIMENT_MAP_EXECUTION,
                1 => changed.chunk_sampling.execution = SAMPLED_SENTIMENT_EXECUTION,
                2 => changed.chunk_sampling.native_execution = INT4_JSON_SAMPLED_EXECUTION,
                3 => changed.chunk_sampling.selected_rows.as_mut().unwrap().max_rows_per_step = 0,
                _ => changed.chunk_sampling.selected_rows = None,
            }
            assert!(changed.check_job_limits(limits()).is_err(), "document projection binding {axis}");
        }
    }
}

#[test]
fn actual_document_factory_keeps_selected_cap_encoding_and_full_preparation_across_schedules() {
    use crate::native_engine::strict_int4::weights::projection_fixture;
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let (model, _) = projection_fixture(32);
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
    let compact = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 32768,
        max_output_tokens: 32768, max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 });
    for encoding in [UntrustedEncoding::ByteFallback, compact] {
        let make = |projection: Option<Int4JsonSparseLimits>| {
            let recipe = recipe();
            let planner = SampledSentimentPlanner::for_model_with_encoding(model.identity(),
                recipe.chunk_sampling.sampling, task(), SampledSentimentLimits {
                    max_input_bytes: 2048, max_result_bytes: 65536, ..Default::default() }, encoding.clone()).unwrap();
            let planner = match projection { Some(cap) => planner.with_selected_rows(cap).unwrap(), None => planner };
            SampledSentimentDocumentJobPlanner::new(planner, Arc::clone(&vocabulary), mapping(), native(), limits()).unwrap()
        };
        let dense = make(None);
        let cap = Int4JsonSparseLimits { max_rows_per_step: 4096 };
        let sparse = make(Some(cap));
        assert_eq!(sparse.selected_rows(), Some(cap));
        assert_eq!(sparse.preparation_requirement_bytes(), dense.preparation_requirement_bytes());
        assert_eq!(sparse.document_encoding(), dense.document_encoding());
        assert_eq!(sparse.native_limits().max_model_work, dense.native_limits().max_model_work);
        assert_ne!(sparse.base_execution_identity(), dense.base_execution_identity());
        let identity = sparse.base_execution_identity().clone();
        let sparse = sparse.with_layer_major_prefill(Int4PrefillLimits { max_batch_rows: 4,
            max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() }).unwrap()
            .with_reused_prefill().unwrap();
        sparse.check_job_limits(limits()).unwrap();
        assert_eq!(sparse.base_execution_identity(), &identity);
        assert_eq!(sparse.selected_rows(), Some(cap));
        assert_eq!(sparse.job_recipe().execution, SAMPLED_SENTIMENT_MAP_SPARSE_REUSED_PREFILL_EXECUTION);
    }
}

struct RefuseAdmission;
impl SampledSentimentDocumentBatchAdmission for RefuseAdmission {
    type Guard = ();
    fn admit(&mut self, _: SampledSentimentDocumentAdmission<'_>)
        -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn document_processor_types_use_the_existing_owned_job_boundary() {
    fn durable<T: DurableBatchProcessor>() {} fn send<T: Send + 'static>() {}
    durable::<SampledSentimentDocumentJobProcessor<'static, 'static, RefuseAdmission>>();
    send::<SampledSentimentDocumentJobPlanner>();
    let _constructor = SampledSentimentDocumentJobPlanner::new;
    let _reuse = SampledSentimentDocumentJobPlanner::with_reused_prefill;
    let _prefill = SampledSentimentDocumentJobPlanner::with_layer_major_prefill;
}
