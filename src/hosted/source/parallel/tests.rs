//! Admission and ownership definitions; synthetic plans are not model output.
use super::*;
use crate::{canonjson,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{constrained_int8::sparse::Int8JsonSparseLimits,
        decode::{DecodeCancellationKind, DecodeStepControl}, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{answer::{AnswerOptions, AnswerPassage}, ir::PlanContext,
        keyphrases::KeyphraseOptions, ner::NerOptions, summarize::SummaryOptions,
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SourceTaskRequest}},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, UntrustedEncoding}}};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn limits() -> SourceLimits {
    SourceLimits { native: NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 } }, preparation_reserve_bytes: 512 << 20,
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        max_mask_node_visits: 1_024_000_000 }
}
fn task_budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 7680, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(8192).unwrap().kv_bytes }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(), recipe_id: "source-team-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn request(kind: usize) -> SourceTaskRequest {
    let document = "Alice visits Paris. Alice likes maps. é 上海".to_owned(); let budget = task_budget();
    match kind {
        0 => SourceTaskRequest::Ner { document, options: NerOptions::default(), budget },
        1 => SourceTaskRequest::Keyphrases { document, options: KeyphraseOptions::default(), budget },
        2 => SourceTaskRequest::Summarize { document, options: SummaryOptions::default(), budget },
        3 => SourceTaskRequest::Answer { question: "Who visits Paris?".to_owned(),
            passages: vec![AnswerPassage { id: "original".to_owned(), text: document }],
            options: AnswerOptions::default(), budget },
        _ => unreachable!(),
    }
}
fn prepare(facts: &ArtifactIdentity, kind: usize, compact: bool, selected: bool) -> PreparedInt8SourceTask {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.surface == crate::template::IM_END).unwrap().id;
    let encoding = if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 65_536, max_output_tokens: 7680, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
    }) } else { UntrustedEncoding::ByteFallback };
    let planner = SourceTaskPlanner::pinned(controls, eos).unwrap().with_document_encoding(encoding).unwrap();
    let request = request(kind); let absent = Sha256Digest::of_bytes(b"null");
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    let identity = ExecutionIdentity { schema_version: 1, source_revision: facts.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap(),
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: planner.tokenizer_digest(),
        template_digest: *planner.template_digest(), task_spec: request.task().spec().identity(),
        taskir_digest: absent, prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let prepared = planner.plan_int8_with_control(&request, &PlanContext::new(&identity, task_budget()).unwrap(),
        SourcePlanningLimits::default(), &mut Continue).unwrap();
    if selected { prepared.with_selected_rows(Int8JsonSparseLimits {
        max_rows_per_step: crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    }).unwrap() } else { prepared }
}
fn vocabulary() -> Arc<ExtractionVocabulary> {
    Arc::new(ExtractionVocabulary::pinned(pinned_controls::pinned().unwrap().template_controls()).unwrap())
}

#[test]
fn all_source_modes_keep_their_sealed_identity_head_policy_and_complete_native_allowance() {
    for kind in 0..4 { for compact in [false, true] { for selected in [false, true] {
        let prepared = prepare(&facts(), kind, compact, selected);
        let identity = canonjson::canonical_bytes(prepared.execution_identity()).unwrap();
        let work = prepared.planned_work();
        for width in [2, 3, 64] { for schedule in [None, Some(prefill(1)), Some(prefill(3))] {
            let required = team_requirements(limits(), prepared.task_budget(), work, width, schedule).unwrap();
            let budget = native_budget(limits(), work, required.native.kv_bytes);
            assert_eq!(budget.native.max_projection_work, work.projections);
            assert_eq!(budget.native.max_forward_positions, work.forward_positions);
            assert_eq!(budget.native.max_attention_pairs, work.attention_pairs);
            assert_eq!(budget.json.max_forward_positions, work.forward_positions);
            assert_eq!(budget.json.max_projected_logits, work.projected_logits);
            assert_eq!(budget.json.max_kv_bytes, prepared.task_budget().max_kv_bytes);
            assert_eq!(budget.json.max_total_mask_node_visits, limits().max_mask_node_visits);
            assert_eq!(budget.json.mask_limits.max_trie_node_visits, limits().mask_limits.max_trie_node_visits);
            assert_eq!(required.max_batch_rows, schedule.map_or(1, |limits| limits.max_batch_rows));
            assert_eq!(prepared.selected_rows().is_some(), selected);
            prepared.verify_identity(prepared.execution_identity()).unwrap();
            assert_eq!(canonjson::canonical_bytes(prepared.execution_identity()).unwrap(), identity);
        } }
    } } }
}

#[test]
fn team_admission_preserves_every_context_preparation_and_mask_refusal() {
    let prepared = prepare(&facts(), 0, false, false); let work = prepared.planned_work();
    for axis in 0..8 {
        let mut limits = limits(); let mut task = prepared.task_budget();
        match axis {
            0 => limits.preparation_reserve_bytes = 0,
            1 => limits.mask_limits.max_trie_node_visits = 0,
            2 => limits.mask_limits.checkpoint_interval_nodes = 0,
            3 => limits.max_mask_node_visits = 0,
            4 => limits.native.context_tokens = work.forward_positions as usize - 1,
            5 => task.max_kv_bytes -= 1,
            6 => limits.native.allocator_reserve_bytes = 0,
            _ => limits.native.allocator_reserve_bytes = u64::MAX,
        }
        for schedule in [None, Some(prefill(3))] {
            assert!(team_requirements(limits, task, work, 2, schedule).is_err(), "admitted axis {axis}");
        }
    }
    for width in [0, 1, 65, usize::MAX] { for schedule in [None, Some(prefill(3))] {
        assert!(team_requirements(limits(), prepared.task_budget(), work, width, schedule).is_err());
    } }
}

#[test]
fn worker_storage_never_replaces_coordinator_scratch_or_whole_context_kv() {
    let prepared = prepare(&facts(), 3, true, true);
    let ordinary = requirements(limits().native).unwrap();
    let scratch = ordinary.rope_bytes + ordinary.scratch_payload_bound + limits().native.allocator_reserve_bytes;
    for width in [2, 7, 64] {
        let admitted = team_requirements(limits(), prepared.task_budget(), prepared.planned_work(), width, None).unwrap();
        assert_eq!(admitted.native, ordinary); assert_eq!(admitted.scratch_bytes, scratch);
        assert_eq!(admitted.max_batch_rows, 1);
        assert_eq!(dispatch::int8_prefill_team_workspace_bytes(width, admitted.max_batch_rows).unwrap(),
            dispatch::int8_team_workspace_bytes(width).unwrap());
    }
}

#[test]
fn packed_source_prompts_admit_coordinator_and_persistent_child_storage_separately() {
    let prepared = prepare(&facts(), 3, true, true);
    let ordinary = requirements(limits().native).unwrap();
    let scratch = ordinary.rope_bytes + ordinary.scratch_payload_bound + limits().native.allocator_reserve_bytes;
    for rows in [1, 3, 64] { for width in [2, 7, 64] {
        let schedule = prefill(rows);
        let admitted = team_requirements(limits(), prepared.task_budget(), prepared.planned_work(),
            width, Some(schedule)).unwrap();
        assert_eq!(admitted.native, ordinary);
        assert_eq!(admitted.max_batch_rows, rows);
        assert_eq!(admitted.scratch_bytes, scratch + schedule.validate().unwrap());
        let children = dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap();
        assert!(children >= dispatch::int8_team_workspace_bytes(width).unwrap());
        assert!(children > 0);
        assert!(sum(&[admitted.scratch_bytes, children]).unwrap() > admitted.scratch_bytes);
    } }
}

#[test]
fn source_team_prefill_refuses_underpriced_or_unbounded_prompt_rows_before_dispatch() {
    let prepared = prepare(&facts(), 0, false, false);
    for rows in [1, 3, 64] {
        let mut schedule = prefill(rows);
        schedule.max_extra_scratch_bytes -= 1;
        assert!(matches!(team_requirements(limits(), prepared.task_budget(), prepared.planned_work(),
            2, Some(schedule)), Err(HostedError::Native(_))));
    }
    for rows in [0, 65, usize::MAX] {
        let schedule = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX };
        assert!(team_requirements(limits(), prepared.task_budget(), prepared.planned_work(), 2,
            Some(schedule)).is_err());
    }
}

#[test]
fn combined_source_team_scratch_overflow_cannot_hide_in_independent_quotes() {
    let prepared = prepare(&facts(), 0, false, false);
    let schedule = prefill(3);
    let native = requirements(limits().native).unwrap();
    let workers = dispatch::int8_prefill_team_workspace_bytes(4, 3).unwrap();
    let fixed = native.rope_bytes + native.scratch_payload_bound + schedule.validate().unwrap();
    let mut exact = limits();
    exact.native.allocator_reserve_bytes = u64::MAX - fixed - workers;
    let admitted = team_requirements(exact, prepared.task_budget(), prepared.planned_work(),
        4, Some(schedule)).unwrap();
    assert_eq!(sum(&[admitted.scratch_bytes, workers]).unwrap(), u64::MAX);
    exact.native.allocator_reserve_bytes += 1;
    assert!(matches!(team_requirements(exact, prepared.task_budget(), prepared.planned_work(),
        4, Some(schedule)), Err(HostedError::Limits("memory arithmetic"))));
}

#[test]
fn transferred_source_and_guarded_result_cross_only_owned_runtime_boundaries() {
    fn send<T: Send + 'static>() {}
    send::<SourceTeamAdmission>(); send::<SourceInput>();
    send::<HostedInt8TeamResult<Int8SourceTaskRun>>();
    let _typed_entry = NlpEngine::execute_int8_source_with_threads;
    let _typed_prefill_entry = NlpEngine::execute_int8_source_layer_major_with_threads;
}

mod model;
