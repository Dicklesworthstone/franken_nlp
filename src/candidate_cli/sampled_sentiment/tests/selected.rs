//! Explicit projection policy, pre-IO admission and complete cold task work.
use super::*;

#[test]
fn selected_rows_keep_sampling_and_the_full_simultaneous_memory_quote() {
    for reuse in [false, true] {
        for width in ["1", "4"] {
            for rows in [None, Some("4")] {
                let mut flags = vec!["--samples", "2", "--max-new-tokens", "64", "--compute-threads", width];
                if reuse { flags.push("--reuse-prefill"); }
                if let Some(rows) = rows { flags.extend(["--prefill-rows", rows]); }
                let dense = command(&flags);
                let (_, dense_limits) = dense.common().unwrap();
                for cap in ["1", "4096", "166144"] {
                    let mut selected_flags = flags.clone(); selected_flags.extend(["--max-selected-rows", cap]);
                    let selected = command(&selected_flags);
                    let (_, selected_limits) = selected.common().unwrap();
                    assert_eq!(selected.selected_rows().unwrap(), Some(Int8JsonSparseLimits {
                        max_rows_per_step: cap.parse().unwrap() }));
                    assert_eq!(dense.selected_rows().unwrap(), None);
                    assert_eq!(dense.options().unwrap(), selected.options().unwrap());
                    assert_eq!(dense.work_ceiling(), selected.work_ceiling());
                    assert_eq!(dense.sampler_bytes().unwrap(), selected.sampler_bytes().unwrap());
                    assert_eq!(dense_limits.kv_bytes, selected_limits.kv_bytes);
                    assert_eq!(dense_limits.preparation_bytes, selected_limits.preparation_bytes);
                    assert_eq!(dense_limits.compute, selected_limits.compute);
                    let native = Int8MemoryRequirement::for_context(dense.context_tokens).unwrap();
                    let peak = |command: &SampledSentimentCommand, limits: Limits|
                        compute::request_peak_bytes(limits, native, command.max_result_bytes as u64,
                            command.total_tokens(4).unwrap() as u64, command.sampler_bytes().unwrap(),
                            limits.preparation_bytes).unwrap();
                    assert_eq!(peak(&dense, dense_limits), peak(&selected, selected_limits));
                    assert_eq!(selected.prefill_reuse_execution().is_some(), reuse);
                    if reuse { assert_ne!(dense.prefill_reuse_execution(), selected.prefill_reuse_execution()); }
                }
            }
        }
    }
}

#[test]
fn selected_planning_admits_the_complete_capped_head_but_never_discounts_prompt_work() {
    let mut selected = command(&["--samples", "2", "--max-new-tokens", "64",
        "--max-selected-rows", "4096", "--reuse-prefill"]);
    let (_, limits) = selected.common().unwrap();
    let budget = selected.budget(limits, 4).unwrap();
    let facts = crate::native_engine::artifact_bridge::ArtifactIdentity {
        model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-selected-cli-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32),
    };
    let planner = || Int8SampledSentimentPlanner::for_model(&facts, selected.options().unwrap(),
        budget, selected.task_limits()).unwrap();
    let request = request(r#"{"document":"The service was helpful."}"#, budget, selected.max_input_bytes).unwrap();
    let dense = planner().plan(&request).unwrap();
    let projected = selected.configure_planner(planner()).unwrap().plan(&request).unwrap();
    assert_eq!(projected.selected_rows(), Some(Int8JsonSparseLimits { max_rows_per_step: 4096 }));
    assert_eq!(projected.sample_count(), dense.sample_count());
    assert_eq!(projected.required_context(), dense.required_context());
    assert_eq!(projected.preparation_requirement_bytes(), dense.preparation_requirement_bytes());
    assert_eq!(projected.max_result_bytes(), dense.max_result_bytes());
    let work = projected.planned_work(); let cold = dense.planned_work();
    assert_eq!(work.forward_positions, cold.forward_positions);
    assert_eq!(work.attention_pairs, cold.attention_pairs);
    assert_eq!(work.projected_logits, projected.sample_count() as u64 * 64 * 4096);
    assert!(work.projected_logits < cold.projected_logits);
    assert!(work.projections.dot_products < cold.projections.dot_products);
    assert!(work.projections.multiply_accumulates < cold.projections.multiply_accumulates);
    assert_ne!(projected.execution_version(false), dense.execution_version(false));
    assert_ne!(projected.execution_version(true), dense.execution_version(true));
    selected.max_projected_logits = work.projected_logits;
    selected.admit_plan(projected.required_context(), work).unwrap();
    assert_eq!(selected.admit_plan(dense.required_context(), cold), Err(CandidateError::Planning));
    selected.max_projected_logits -= 1;
    assert_eq!(selected.admit_plan(projected.required_context(), work), Err(CandidateError::Planning));
}

#[test]
fn invalid_selected_row_caps_refuse_before_any_private_io() {
    for cap in ["0", "166145", "18446744073709551615"] {
        if let Ok(CandidateCommand::SampledSentiment(selected)) = parsed(&["--max-selected-rows", cap]) {
            assert_eq!(selected.execute(&mut NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
    }
}
