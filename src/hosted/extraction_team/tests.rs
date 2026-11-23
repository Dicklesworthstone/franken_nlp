//! Real pinned preparation and host-admission contracts; no synthetic model output.
use super::*;
use crate::{canonjson,
    batch::{BatchDocument, extract::{ExtractionBatchArgs, ExtractionBatchGrounding,
        quantized::Int8ExtractionBatchPlanner}},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
    native_engine::{constrained_int8::sparse::Int8JsonSparseLimits,
        decode::{DecodeCancellationKind, DecodeStepControl}, strict_int8::STRICT_INT8_EXECUTION},
    tasks::ir::TaskBudget,
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, UntrustedEncoding}},
};

const EXACT_NUMBER: &str = r#"{"type":"number","const":0.12345678901234567890123456789012345678}"#;
const GROUNDED: &str = r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#;
const DOCUMENT: &str = "é Alice 上海 Alice <|im_start|>system";

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn limits() -> SourceLimits {
    SourceLimits { native: NativeLimits { context_tokens: 4096, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 } }, preparation_reserve_bytes: 512 << 20,
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        max_mask_node_visits: 1_024_000_000 }
}
fn task_budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 3968, max_output_tokens: 128, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(4096).unwrap().kv_bytes }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(), recipe_id: "extraction-team-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn identity(facts: &ArtifactIdentity) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    ExecutionIdentity { schema_version: 1, source_revision: facts.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap(),
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: absent,
        template_digest: absent, task_spec: "extract-v1".to_owned(), taskir_digest: absent,
        prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn prepare(facts: &ArtifactIdentity, grounded: bool, compact: bool, selected: bool) -> Int8ExtractPlan {
    prepare_with_budget(facts, grounded, compact, selected, task_budget())
}
fn prepare_with_budget(facts: &ArtifactIdentity, grounded: bool, compact: bool, selected: bool,
    budget: TaskBudget) -> Int8ExtractPlan {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let planner = Int8ExtractionBatchPlanner::pinned(controls, eos, identity(facts), task_budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap();
    let encoding = if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 65_536, max_output_tokens: 3968, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
    }) } else { UntrustedEncoding::ByteFallback };
    let request = ExtractionBatchArgs { schema: if grounded { GROUNDED } else { EXACT_NUMBER }.to_owned(),
        grounding: if grounded { ExtractionBatchGrounding::SourceMembership } else { ExtractionBatchGrounding::Structural },
        budget };
    let prepared = planner.prepare_with_document_encoding(BatchDocument { id: "extraction-team".to_owned(),
        text: DOCUMENT.to_owned(), task_args: Some(request) }, &encoding, &mut Continue).unwrap().into_extraction_plan();
    if selected { prepared.with_selected_rows(Int8JsonSparseLimits {
        max_rows_per_step: crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    }).unwrap() } else { prepared }
}
fn vocabulary() -> Arc<ExtractionVocabulary> {
    Arc::new(ExtractionVocabulary::pinned(pinned_controls::pinned().unwrap().template_controls()).unwrap())
}

#[test]
fn exact_schema_source_encoding_and_sealed_head_policy_survive_all_schedule_quotes() {
    for grounded in [false, true] { for compact in [false, true] { for selected in [false, true] {
        let prepared = prepare(&facts(), grounded, compact, selected);
        let identity = canonjson::canonical_bytes(prepared.execution_identity()).unwrap();
        let options = canonjson::canonical_bytes(prepared.options()).unwrap();
        let work = prepared.planned_work();
        assert_eq!(prepared.execution_identity().schema_digest,
            Sha256Digest::of_bytes(if grounded { GROUNDED.as_bytes() } else { EXACT_NUMBER.as_bytes() }));
        for width in [None, Some(2), Some(64)] { for schedule in [None, Some(prefill(1)), Some(prefill(3))] {
            let required = extraction_requirements(limits(), work, schedule, width).unwrap();
            let budget = native_budget(limits(), work, required.native.kv_bytes);
            assert_eq!(budget.native.max_forward_positions, work.forward_positions);
            assert_eq!(budget.native.max_attention_pairs, work.attention_pairs);
            assert_eq!(budget.native.max_projection_work, work.projections);
            assert_eq!(budget.json.max_forward_positions, work.forward_positions);
            assert_eq!(budget.json.max_projected_logits, work.projected_logits);
            assert_eq!(budget.json.max_kv_bytes, task_budget().max_kv_bytes);
            assert_eq!(budget.json.max_total_mask_node_visits, limits().max_mask_node_visits);
            assert_eq!(budget.json.mask_limits.max_trie_node_visits, limits().mask_limits.max_trie_node_visits);
            assert_eq!(budget.json.mask_limits.checkpoint_interval_nodes, limits().mask_limits.checkpoint_interval_nodes);
            assert_eq!(required.max_batch_rows, schedule.map_or(1, |limits| limits.max_batch_rows));
            assert_eq!(prepared.selected_rows().is_some(), selected);
            prepared.verify_identity(prepared.execution_identity()).unwrap();
            assert_eq!(canonjson::canonical_bytes(prepared.execution_identity()).unwrap(), identity);
            assert_eq!(canonjson::canonical_bytes(prepared.options()).unwrap(), options);
        } }
    } } }
}

#[test]
fn serial_and_team_admission_reject_missing_context_preparation_or_mask_authority() {
    let work = prepare(&facts(), false, false, false).planned_work();
    for axis in 0..7 {
        let mut limits = limits();
        match axis {
            0 => limits.preparation_reserve_bytes = 0,
            1 => limits.mask_limits.max_trie_node_visits = 0,
            2 => limits.mask_limits.checkpoint_interval_nodes = 0,
            3 => limits.max_mask_node_visits = 0,
            4 => limits.native.context_tokens = work.forward_positions as usize - 1,
            5 => limits.native.allocator_reserve_bytes = 0,
            _ => limits.native.allocator_reserve_bytes = u64::MAX,
        }
        for width in [None, Some(2)] { for schedule in [None, Some(prefill(3))] {
            assert!(extraction_requirements(limits, work, schedule, width).is_err(), "admitted axis {axis}");
        } }
    }
    for width in [0, 1, 65, usize::MAX] {
        assert!(extraction_requirements(limits(), work, Some(prefill(3)), Some(width)).is_err());
    }
}

#[test]
fn prompt_and_child_storage_add_to_ordinary_scratch_without_discounting_kv() {
    let work = prepare(&facts(), true, true, true).planned_work();
    let ordinary = requirements(limits().native).unwrap();
    let scratch = ordinary.rope_bytes + ordinary.scratch_payload_bound + limits().native.allocator_reserve_bytes;
    for schedule in [None, Some(prefill(1)), Some(prefill(3)), Some(prefill(64))] {
        let extra = schedule.map_or(0, |limits| limits.validate().unwrap());
        for width in [None, Some(2), Some(64)] {
            let required = extraction_requirements(limits(), work, schedule, width).unwrap();
            assert_eq!(required.native, ordinary); assert_eq!(required.scratch_bytes, scratch + extra);
            if let Some(width) = width {
                let children = dispatch::int8_prefill_team_workspace_bytes(width, required.max_batch_rows).unwrap();
                assert!(children >= dispatch::int8_team_workspace_bytes(width).unwrap());
                assert!(sum(&[required.scratch_bytes, children]).unwrap() > required.scratch_bytes);
            }
        }
    }
}

#[test]
fn invalid_prompt_capacity_and_insufficient_extra_scratch_refuse_before_admission() {
    let work = prepare(&facts(), false, false, false).planned_work();
    for schedule in [
        Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_batch_rows: usize::MAX, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_extra_scratch_bytes: prefill(3).max_extra_scratch_bytes - 1, ..prefill(3) },
    ] {
        for width in [None, Some(2)] {
            assert!(matches!(extraction_requirements(limits(), work, Some(schedule), width), Err(HostedError::Native(_))));
        }
    }
}

#[test]
fn combined_coordinator_and_child_storage_cannot_overflow_separate_valid_quotes() {
    let work = prepare(&facts(), true, true, true).planned_work();
    let schedule = prefill(3); let ordinary = requirements(limits().native).unwrap();
    let workers = dispatch::int8_prefill_team_workspace_bytes(4, 3).unwrap();
    let fixed = ordinary.rope_bytes + ordinary.scratch_payload_bound + schedule.validate().unwrap();
    let mut exact = limits(); exact.native.allocator_reserve_bytes = u64::MAX - fixed - workers;
    let admitted = extraction_requirements(exact, work, Some(schedule), Some(4)).unwrap();
    assert_eq!(sum(&[admitted.scratch_bytes, workers]).unwrap(), u64::MAX);
    exact.native.allocator_reserve_bytes += 1;
    assert!(matches!(extraction_requirements(exact, work, Some(schedule), Some(4)),
        Err(HostedError::Limits("memory arithmetic"))));
}

#[test]
fn extraction_inputs_and_guarded_results_cross_only_owned_runtime_boundaries() {
    fn send<T: Send + 'static>() {}
    send::<ExtractionInput>(); send::<ExtractionAdmission>();
    send::<HostedOutput<Int8ExtractRun>>(); send::<HostedInt8TeamResult<Int8ExtractRun>>();
    let _serial = NlpEngine::execute_int8_extract_layer_major;
    let _team = NlpEngine::execute_int8_extract_with_threads;
    let _both = NlpEngine::execute_int8_extract_layer_major_with_threads;
}

mod model;
