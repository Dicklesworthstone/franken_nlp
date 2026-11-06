//! Pure admission and unchanged pipeline semantics; no synthetic native proof.
use super::*;
use crate::{
    native_engine::strict_int4::{Int4MemoryBudget, Int4MemoryRequirement},
    tasks::source_planning::grouped_int4::long::cohort::{Int4SourceMapCohortBudget,
        Int4SourceMapCohortRequirement, Int4SourceMapCohortStats},
};

fn budget(need: Int4SourceMapCohortRequirement) -> Int4SourceMapCohortBudget {
    Int4SourceMapCohortBudget { context_limit: 8192,
        native_memory: Int4MemoryBudget { max_kv_bytes: need.native.kv_bytes,
            max_rope_bytes: need.native.rope_bytes, max_scratch_payload_bytes: need.native.scratch_payload_bound },
        max_workspace_bytes: need.workspace_bytes, max_result_bytes: need.max_result_bytes }
}

#[test]
fn declaration_prices_every_live_chunk_without_widening_task_authority() {
    for width in [3, 127, 128, 10752] {
        let mut config = config(); config.mapping.reduction.map_batch_chunks = 3;
        let task = config.per_chunk;
        let quote = config.cohort_requirements(8192, width).unwrap();
        assert_eq!(quote.native, Int4MemoryRequirement::for_cohort(&[8192; 3], width).unwrap());
        assert_eq!(config.per_chunk, task);
        assert!(quote.native.kv_bytes > task.max_kv_bytes, "the window has separate authority from each row");
        assert!(quote.workspace_bytes > quote.max_result_bytes);
        assert_eq!(quote.max_result_bytes, 3 * task.max_output_bytes + 4096 + 3);
    }
    for selected in [0, 65, 257, usize::MAX] {
        let mut config = config(); config.mapping.reduction.map_batch_chunks = selected;
        assert!(config.cohort_requirements(8192, 127).is_err(), "no clamping for {selected}");
    }
    let mut config = config(); config.mapping.reduction.map_batch_chunks = 3;
    config.mapping.chunks.max_chunks = 2;
    assert!(config.cohort_requirements(8192, 127).is_err());
    config.mapping.chunks.max_chunks = 256;
    config.per_chunk.max_kv_bytes = Int4MemoryRequirement::for_context(8192, 127).unwrap().kv_bytes - 1;
    assert!(config.cohort_requirements(8192, 127).is_err());
    config.per_chunk.max_kv_bytes += 1;
    config.mapping.reduction.map_batch_chunks = 64;
    assert!(config.cohort_requirements(8192, 127).is_err(), "64 one-MiB rows exceed the native envelope cap");
    config.per_chunk.max_output_bytes = 524_288;
    assert!(config.cohort_requirements(8192, 127).is_ok());
}

#[test]
fn every_actual_window_is_admitted_before_native_work_and_all_peak_axes_bind() {
    let p = planner(); let mut config = config(); config.mapping.reduction.map_batch_chunks = 3;
    let recipe = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), config, 127).unwrap();
    let source = "Alice Alice Alice Alice Alice";
    let prepared = recipe.prepare(source, recipe.config.mapping, &mut Continue).unwrap();
    assert!(prepared.chunk_count() > 3 && prepared.chunk_count() % 3 != 0);
    let quote = prepared.cohort_requirements(8192).unwrap();
    let exact = budget(quote);
    assert_eq!(recipe.preflight_cohorts(source, exact, &mut Continue).unwrap(),
        recipe.preflight(source, &mut Continue).unwrap());
    for axis in 0..5 {
        let mut short = exact;
        match axis { 0 => short.native_memory.max_kv_bytes -= 1,
            1 => short.native_memory.max_rope_bytes -= 1,
            2 => short.native_memory.max_scratch_payload_bytes -= 1,
            3 => short.max_workspace_bytes -= 1,
            _ => short.max_result_bytes -= 1 }
        assert!(matches!(recipe.preflight_cohorts(source, short, &mut Continue),
            Err(Int4LongRedactionError::Map(Int4SourceMapError::Admission))), "axis {axis}");
    }
    struct Cancel;
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::User) }
    }
    assert_eq!(recipe.preflight_cohorts(source, exact, &mut Cancel).unwrap_err().cancellation(),
        Some(DecodeCancellationKind::User));
}

#[test]
fn cohort_preflight_never_renews_any_whole_document_work_axis() {
    let p = planner(); let source = "Alice Alice Alice Alice Alice";
    let mut config = config(); config.mapping.reduction.map_batch_chunks = 3;
    let recipe = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), config.clone(), 127).unwrap();
    let expected = recipe.preflight(source, &mut Continue).unwrap();
    let quote = budget(config.cohort_requirements(8192, 127).unwrap());
    for axis in 0..5 {
        let mut limited = config.clone(); limited.mapping.max_model_work = expected.planned_model_work;
        let work = &mut limited.mapping.max_model_work;
        match axis { 0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1,
            2 => work.attention_pairs -= 1, 3 => work.projections.dot_products -= 1,
            _ => work.projections.multiply_accumulates -= 1 }
        let recipe = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), limited, 127).unwrap();
        assert!(recipe.preflight_cohorts(source, quote, &mut Continue).is_err(), "axis {axis}");
    }
}

fn stats(stage: &Int4LongRedactionStage, width: usize) -> Int4SourceMapCohortStats {
    // Deliberately synthetic receipt counters for the shape validator only.
    Int4SourceMapCohortStats { chunks: stage.preflight.chunks as u64,
        cohorts: stage.preflight.chunks.div_ceil(width) as u64,
        group_steps: stage.model_work.forward_positions,
        projection_groups: stage.model_work.projected_logits / crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64,
        planned_work: stage.preflight.planned_model_work, model_work: stage.model_work }
}

#[test]
fn physical_schedule_never_changes_edit_key_scope_or_fresh_verification_semantics() {
    let p = planner();
    let key = PseudonymKey::from_bytes(&[9; 32], "fixture-key").unwrap();
    let pseudonyms = Pseudonyms::full256(&key, "private-scope", None).unwrap();
    for action in [RedactionAction::Mask, RedactionAction::Placeholder, RedactionAction::Pseudonymize] {
        for verify in [false, true] {
            let mut request = RedactionRequest::default();
            request.rules.enabled.clear(); request.verify = verify; request.actions.include_map = true;
            request.actions.default_action = action;
            if action == RedactionAction::Pseudonymize { request.actions.expected_key_commitment = Some(key.commitment()); }
            let mut config = config(); config.mapping.reduction.map_batch_chunks = 1;
            let serial = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), config.clone(), 127).unwrap()
                .run_pipeline("Alice Alice", &request, Some(&pseudonyms), &mut FixtureDetection::new()).unwrap();
            config.mapping.reduction.map_batch_chunks = 3;
            let output = Int4DocumentRedactor::from_bound_parts(&p, identity(&p, 127), config, 127).unwrap()
                .run_pipeline("Alice Alice", &request, Some(&pseudonyms), &mut FixtureDetection::new()).unwrap();
            assert_eq!(canonjson::canonical_bytes(&output).unwrap(), canonjson::canonical_bytes(&serial).unwrap());
            let mut wrapped = Int4LongRedactionCohortRun { schema_version: 1,
                execution: INT4_LONG_REDACTION_COHORT_EXECUTION, cohort_chunks: 3,
                original_cohorts: stats(&output.original, 3),
                verification_cohorts: output.verification.as_ref().map(|stage| stats(stage, 3)), output };
            wrapped.validate_schedule().unwrap();
            wrapped.original_cohorts.chunks += 1;
            assert!(wrapped.validate_schedule().is_err());
            wrapped.original_cohorts.chunks -= 1;
            wrapped.original_cohorts.group_steps = 0;
            assert!(wrapped.validate_schedule().is_err());
        }
    }
}
