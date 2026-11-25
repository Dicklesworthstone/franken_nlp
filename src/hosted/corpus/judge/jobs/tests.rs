//! Real keyed manifests and pinned settings, not neural or filesystem evidence.
use super::*;
use super::super::planning::tests::{compact_fixture, fixture, Continue};
use crate::{canonjson, jobs::{FrozenManifest, JobContract, JobId, JobInput, JobLimits, JobSecret,
    JobError, MismatchField}};

const INPUT: &[u8] = br#"{"id":"item","text":"Alice"}"#;
fn freeze(config: &JudgeCorpusConfig, recipe: &JudgeJobRecipe, input: &[u8], secret: u8) -> FrozenManifest {
    let limits = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 1 << 20, max_spool_bytes: 4 << 20,
        max_materialized_bytes: 4 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: config.max_model_work, mask_node_visits: 0 } };
    FrozenManifest::freeze(&JobSecret::from_bytes([secret; 32]), JobContract { job_id: JobId([9; 16]),
        execution: &config.identity, recipe, limits },
        [JobInput { id: "item", original: input, normalized: input }], &mut Continue).unwrap()
}
fn limits(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
#[test]
fn changed_criterion_comparison_or_policy_cannot_resume_identical_input() {
    let (_, mut config) = fixture();
    let original = JudgeJobRecipe::new(&config, None).unwrap();
    let frozen = freeze(&config, &original, INPUT, 7);
    frozen.binding.compare(&freeze(&config, &original, INPUT, 7).binding).unwrap();
    for axis in 0..4 {
        let previous = config.defaults.clone();
        if let Some(JudgeBatchArgs::Pairwise { criterion, b, policy, .. }) = &mut config.defaults {
            match axis { 0 => criterion.push('!'), 1 => b.push('!'), 2 => policy.minimum_margin_milli += 1,
                _ => policy.maximum_order_disagreement_milli += 1 }
        }
        let changed = JudgeJobRecipe::new(&config, None).unwrap();
        assert_eq!(frozen.binding.compare(&freeze(&config, &changed, INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
        config.defaults = previous;
    }
    assert_eq!(frozen.binding.compare(&freeze(&config, &original, br#"{"id":"item","text":"Bob"}"#, 7).binding),
        Err(JobError::Mismatch(MismatchField::Population)));
    assert!(frozen.binding.compare(&freeze(&config, &original, INPUT, 8).binding).is_err());
}
#[test]
fn every_planning_budget_and_native_counter_is_frozen() {
    let (_, config) = fixture(); let before = canonjson::canonical_bytes(&JudgeJobRecipe::new(&config, None).unwrap()).unwrap();
    for axis in 0..20 {
        let (_, mut changed) = fixture();
        match axis {
            0 => changed.planning.per_head.max_candidates += 1,
            1 => changed.planning.per_head.max_total_tokens += 1,
            2 => changed.planning.per_head.max_nodes += 1,
            3 => changed.planning.per_head.max_depth += 1,
            4 => changed.planning.per_head.max_candidate_id_bytes += 1,
            5 => changed.planning.per_head.max_projected_logits += 1,
            6 => changed.planning.max_total_prompt_tokens += 1,
            7 => changed.planning.max_total_candidate_tokens += 1,
            8 => changed.planning.max_total_projected_logits += 1,
            9 => changed.planning.max_output_bytes += 1,
            10 => changed.task_ceiling.max_input_tokens += 1,
            11 => changed.task_ceiling.max_output_tokens += 1,
            12 => changed.task_ceiling.max_output_bytes += 1,
            13 => changed.task_ceiling.max_grammar_states += 1,
            14 => changed.task_ceiling.max_kv_bytes += 1,
            15 => changed.max_model_work.forward_positions += 1,
            16 => changed.max_model_work.projected_logits += 1,
            17 => changed.max_model_work.attention_pairs += 1,
            18 => changed.max_model_work.projections.dot_products += 1,
            _ => changed.max_model_work.projections.multiply_accumulates += 1,
        }
        assert_ne!(before, canonjson::canonical_bytes(&JudgeJobRecipe::new(&changed, None).unwrap()).unwrap(), "axis {axis}");
    }
}
#[test]
fn physical_schedule_is_frozen_without_trusting_a_caller_workspace_price() {
    let (_, config) = fixture(); let serial = JudgeJobRecipe::new(&config, None).unwrap();
    let baseline = freeze(&config, &serial, INPUT, 7);
    for rows in [1, 4, 64] {
        let grouped = JudgeJobRecipe::new(&config, Some(limits(rows))).unwrap();
        assert_eq!(baseline.binding.compare(&freeze(&config, &grouped, INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
        let mut over = limits(rows); over.max_extra_scratch_bytes = u64::MAX;
        assert_eq!(canonjson::canonical_bytes(&grouped).unwrap(),
            canonjson::canonical_bytes(&JudgeJobRecipe::new(&config, Some(over)).unwrap()).unwrap());
    }
    let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    assert!(JudgeJobRecipe::new(&config, Some(short)).is_err());
}
#[test]
fn complete_resident_kv_and_output_envelope_cannot_be_underfunded() {
    let (_, config) = fixture(); let task = config.task_ceiling;
    check_capacity(task, task.max_kv_bytes, task.max_output_bytes as usize).unwrap();
    assert!(check_capacity(task, 0, task.max_output_bytes as usize).is_err());
    assert!(check_capacity(task, task.max_kv_bytes + 1, task.max_output_bytes as usize).is_err());
    assert!(check_capacity(task, task.max_kv_bytes, task.max_output_bytes as usize - 1).is_err());
}
#[test]
fn owned_judge_inputs_cross_the_existing_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<StreamInput<(Arc<JudgePlanner>, JudgeCorpusConfig, SourceJobRequest), std::io::Cursor<Vec<u8>>, ()>>();
    send::<Int8JudgeRun>();
}

#[test]
fn byte_default_recipe_remains_exactly_the_original_private_projection() {
    let (planner, config) = fixture();
    let encoding = EncodingWorkScope::new(planner.document_encoding(), None).unwrap();
    for prefill in [None, Some(limits(4))] {
        let original = JudgeJobRecipe::new(&config, prefill).unwrap();
        let explicit = JudgeJobRecipe::with_encoding(&config, prefill, &encoding).unwrap();
        assert_eq!(canonjson::canonical_bytes(&original).unwrap(), canonjson::canonical_bytes(&explicit).unwrap());
        let value = serde_json::to_value(&explicit).unwrap();
        assert!(value.get("input_encoding").is_none());
        assert!(value.get("encoding_work_limits").is_none());
    }
}

#[test]
fn all_compact_policy_and_invocation_caps_authenticate_resume() {
    let (planner, config, total) = compact_fixture();
    let UntrustedEncoding::SafeBpeV1(call) = planner.document_encoding().clone() else { panic!("compact fixture") };
    let encoding = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
    let original = JudgeJobRecipe::with_encoding(&config, None, &encoding).unwrap();
    let baseline = freeze(&config, &original, INPUT, 7);
    let value = serde_json::to_value(&original).unwrap();
    assert_eq!(value["input_encoding"], serde_json::to_value(planner.document_encoding()).unwrap());
    assert_eq!(value["encoding_work_limits"], serde_json::to_value(total).unwrap());
    for axis in 0..8 {
        let mut changed_call = call; let mut changed_total = total;
        match axis {
            0 => changed_call.max_input_bytes += 1,
            1 => changed_call.max_output_tokens += 1,
            2 => changed_call.max_pair_checks += 1,
            3 => changed_call.max_pair_bytes += 1,
            4 => changed_total.max_input_bytes += 1,
            5 => changed_total.max_output_tokens += 1,
            6 => changed_total.max_pair_checks += 1,
            _ => changed_total.max_pair_bytes += 1,
        }
        let changed = EncodingWorkScope::new(&UntrustedEncoding::SafeBpeV1(changed_call), Some(changed_total)).unwrap();
        let recipe = JudgeJobRecipe::with_encoding(&config, None, &changed).unwrap();
        assert_eq!(baseline.binding.compare(&freeze(&config, &recipe, INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
    }
    let grouped = JudgeJobRecipe::with_encoding(&config, Some(limits(4)), &encoding).unwrap();
    assert_eq!(baseline.binding.compare(&freeze(&config, &grouped, INPUT, 7).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
}

#[test]
fn consumed_tokenizer_work_does_not_rewrite_the_immutable_recipe() {
    let (planner, config, total) = compact_fixture();
    let encoding = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
    let before = canonjson::canonical_bytes(&JudgeJobRecipe::with_encoding(&config, None, &encoding).unwrap()).unwrap();
    let item = BatchDocument { id: "item".into(), text: "Alice writes accurate answers.".into(), task_args: None };
    let prepared = planning::prepare_with_encoding_scope(&planner, &config, &encoding, item, &mut Continue).unwrap();
    assert!(encoding.work().unwrap().input_bytes() > 0);
    assert_eq!(prepared.input_encoding_work(), Some(encoding.work().unwrap()));
    assert_eq!(before, canonjson::canonical_bytes(
        &JudgeJobRecipe::with_encoding(&config, None, &encoding).unwrap()).unwrap());
}

#[test]
fn judge_job_retains_actual_compiler_and_ledger_through_physical_join() {
    let (planner, config, total) = compact_fixture();
    let expected = config.encoding_requirements(&planner, Some(total)).unwrap().2;
    let planner = Arc::new(planner);
    let encoding = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
    let compiler_lifetime = Arc::downgrade(&planner);
    let ledger_lifetime = Arc::downgrade(&encoding);
    let task = JudgeTask { planner, config, encoding };
    assert_eq!(task.preparation_workspace_bytes().unwrap(), expected);
    let retained = task.retain_for_join().unwrap();
    drop(task);
    assert!(compiler_lifetime.upgrade().is_some());
    assert!(ledger_lifetime.upgrade().is_some());
    drop(retained);
    assert!(compiler_lifetime.upgrade().is_none());
    assert!(ledger_lifetime.upgrade().is_none());
}
