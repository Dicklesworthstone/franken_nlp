//! Native-free integration fixtures for complete INT4 summary composition.
//! Exact occurrence checks and pinned plans are real; outputs are scripted.
use super::*;
use super::super::tests::{planner, prepare_task, limits as mapping_limits, admitted, script, Script};
use std::rc::Rc;
use crate::tasks::summarize::SummaryResult;

fn prepare<'s>(p: &SourceTaskPlanner, source: &'s str, fan_in: usize) -> PreparedInt4SourceMap<'s> {
    let mut mapping = mapping_limits(); mapping.chunks.max_chunk_bytes = 6;
    mapping.chunks.max_chunk_tokens = 6; mapping.reduction.reduce_fan_in = fan_in;
    prepare_task(p, source, &SourceMapTask::Summarize(SummaryOptions::default()), mapping, 32)
}
fn limits() -> Int4SummaryLimits { Int4SummaryLimits { max_bullets: 1, ..Int4SummaryLimits::default() } }
fn summary(result: &mut SourceTaskResult) -> &mut SummaryResult {
    let SourceTaskResult::Summarize(result) = result else { panic!("summary fixture") }; result
}
fn run(prepared: PreparedInt4SourceMap<'_>, limits: Int4SummaryLimits, driver: Script)
    -> Result<Int4CorpusSummaryRun, Int4CorpusSummaryError> {
    let ids = admitted(&prepared); prepared.execute_summary_with_driver(&ids, limits, driver)
}
#[test]
fn final_top_k_preserves_late_support_and_every_original_unicode_occurrence() {
    let p = planner(); let source = "éAéAéAéA"; let prepared = prepare(&p, source, 2);
    assert_eq!(prepared.chunk_count(), 2); let expected = prepared.preflight_metadata();
    let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    let result = run(prepared, limits(), driver).unwrap(); expected.verify_summary_completed(&result).unwrap();
    assert_eq!(calls.get(), 2); assert_eq!(result.summary.bullets.len(), 1);
    let bullet = &result.summary.bullets[0];
    assert_eq!(bullet.text, "common"); assert_eq!(bullet.rank_sum, 4); assert_eq!(bullet.evidence.len(), 2);
    assert_eq!(result.summary.omitted_bullets, 2); assert_eq!(result.summary.mapped_chunks, 2);
    assert_eq!(result.source_span.byte_end, source.len()); assert_eq!(result.source_span.scalar_end, source.chars().count());
    let second = &bullet.evidence[1].citations[0];
    assert_eq!(second.spans[0].byte_start, 6); assert_eq!(second.spans[0].scalar_start, 4);
    for evidence in &bullet.evidence { for citation in &evidence.citations { for span in &citation.spans {
        assert_eq!(&source[span.byte_start..span.byte_end], citation.quote);
        assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
        assert_eq!(source[..span.byte_end].chars().count(), span.scalar_end);
    } } }
    assert_eq!(result.mask_node_visit_charge, 40); assert_eq!(result.reserved_mask_node_visits, 2000);
    assert_eq!(result.verification_used.fields, 4); assert_eq!(result.verification_used.matches, 8);
    assert!(fits(result.model_work, result.planned_model_work));
    assert_eq!(result.model_work.projected_logits, 4 * crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64);
}
#[test]
fn complete_union_is_deterministic_across_fanins_and_duplicate_local_proposals() {
    let p = planner(); let source = "éAéAéAéAéAéAéAéA";
    let mut baseline = None;
    for fan_in in [2, 3, 8] {
        let prepared = prepare(&p, source, fan_in); let mut driver = script(&prepared);
        for raw in &mut driver.outputs { let raw = summary(raw); raw.bullets.push(raw.bullets[1].clone()); }
        let result = run(prepared, limits(), driver).unwrap();
        assert_eq!(result.summary.bullets[0].evidence.len(), 4);
        assert_eq!(result.summary.bullets[0].rank_sum, 8);
        let json = canonjson::canonical_string(&result.summary).unwrap();
        if let Some(expected) = &baseline { assert_eq!(&json, expected); } else { baseline = Some(json); }
        if fan_in == 2 { assert!(result.reduction_levels > 1); }
    }
}
#[test]
fn final_chunk_admission_and_complete_kv_precede_all_summary_native_calls() {
    let p = planner();
    for axis in 0..3 {
        let mut prepared = prepare(&p, "éAéAéAéA", 2); let mut ids = admitted(&prepared);
        let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        match axis {
            0 => ids.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"foreign last summary"),
            1 => prepared.plans.last_mut().unwrap().budget.max_kv_bytes -= 1,
            _ => driver.fail_model_at = Some(prepared.chunk_count()),
        }
        assert!(prepared.execute_summary_with_driver(&ids, limits(), driver).is_err()); assert_eq!(calls.get(), 0);
    }
}
#[test]
fn missing_malformed_and_wrong_profile_final_proposals_abort_complete_summary() {
    let p = planner();
    for axis in 0..7 {
        let prepared = prepare(&p, "éAéAéAéA", 2); let mut driver = script(&prepared);
        let calls = Rc::clone(&driver.calls); let last = summary(driver.outputs.back_mut().unwrap());
        match axis {
            0 => { last.bullets[1].citations[0].spans.pop(); },
            1 => last.bullets[1].citations[0].spans[0].scalar_end += 1,
            2 => last.bullets[1].citations[0].occurrence = SourceOccurrence::Anchored,
            3 => last.bullets[1].citations[0].quote = "absent".to_owned(),
            4 => last.numerics_profile = crate::native_engine::strict_int8::STRICT_INT8_PROFILE.to_owned(),
            5 => last.numerics_profile = crate::native_engine::hf_bf16_eager::HF_BF16_EAGER_PROFILE.to_owned(),
            _ => { let mut bad = last.bullets[1].clone(); bad.citations[0].spans.pop(); last.bullets.push(bad); },
        }
        assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 2);
    }
}
#[test]
fn independent_map_and_union_budgets_do_not_reset_or_hide_behind_final_top_k() {
    let p = planner(); let source = "éAéAéAéA"; let prepared = prepare(&p, source, 2);
    let driver = script(&prepared); let result = run(prepared, limits(), driver).unwrap();
    for axis in 0..6 {
        let mut prepared = prepare(&p, source, 2); let driver = script(&prepared); let mut config = limits();
        match axis {
            0 => prepared.limits.verification.max_fields = result.verification_used.fields - 1,
            1 => prepared.limits.verification.max_matches = result.verification_used.matches - 1,
            2 => prepared.limits.verification.max_scan_steps = result.verification_used.scan_steps - 1,
            3 => config.aggregation.max_scan_steps = result.verification_scan_steps - 1,
            4 => config.aggregation.max_unique_bullets = 2,
            _ => config.aggregation.max_citations = 3,
        }
        assert!(run(prepared, config, driver).is_err());
    }
}
#[test]
fn all_five_native_work_axes_and_complete_summary_metadata_remain_bound() {
    let p = planner();
    for axis in 0..5 {
        let prepared = prepare(&p, "éAéA", 2); let mut driver = script(&prepared);
        driver.corrupt_axis = Some(axis); assert!(run(prepared, limits(), driver).is_err());
    }
    for axis in 0..8 {
        let prepared = prepare(&p, "éAéAéAéA", 2); let expected = prepared.preflight_metadata();
        let driver = script(&prepared); let mut result = run(prepared, limits(), driver).unwrap();
        match axis {
            0 => result.group_width = 16,
            1 => result.numerics_profile = crate::native_engine::strict_int8::STRICT_INT8_PROFILE,
            2 => result.source_span.byte_end -= 1,
            3 => result.summary.mapped_chunks -= 1,
            4 => result.planned_model_work.attention_pairs -= 1,
            5 => result.model_work.projections.dot_products = result.planned_model_work.projections.dot_products + 1,
            6 => result.mask_node_visit_charge += 1,
            _ => result.summary.forward_positions += 1,
        }
        assert!(expected.verify_summary_completed(&result).is_err());
    }
}
#[test]
fn non_summary_task_mixed_options_and_invalid_limits_refuse_before_native_calls() {
    let p = planner();
    let prepared = prepare_task(&p, "éAéA", &SourceMapTask::Ner(NerOptions::default()), mapping_limits(), 32);
    let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 0);
    for axis in 0..5 {
        let mut prepared = prepare(&p, "éAéAéAéA", 2); let mut config = limits();
        let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        match axis {
            0 => { let Finalizer::Summarize(options) = &mut prepared.plans.last_mut().unwrap().finalizer else { unreachable!() };
                options.max_bullets += 1; },
            1 => config.max_bullets = 0,
            2 => config.max_result_bytes = 0,
            3 => config.aggregation.max_value_bytes = 8 << 20,
            _ => config.max_result_bytes = 8 << 20,
        }
        assert!(run(prepared, config, driver).is_err()); assert_eq!(calls.get(), 0);
    }
}
#[test]
fn singleton_empty_bullets_and_final_cancellation_preserve_complete_coverage() {
    let p = planner(); let prepared = prepare(&p, "éAéA", 2); let driver = script(&prepared);
    let result = run(prepared, limits(), driver).unwrap(); assert_eq!(result.reduce_calls, 0); assert_eq!(result.reduction_levels, 0);
    let prepared = prepare(&p, "éAéAéAéA", 2); let mut driver = script(&prepared);
    for raw in &mut driver.outputs { summary(raw).bullets.clear(); }
    let result = run(prepared, limits(), driver).unwrap();
    assert!(result.summary.bullets.is_empty()); assert_eq!(result.summary.mapped_chunks, 2);
    assert_eq!(result.source_span.byte_end, 12); assert!(result.model_work.forward_positions > 0);
    let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared);
    let checkpoints = Rc::clone(&driver.checkpoints); run(prepared, limits(), driver).unwrap();
    let prepared = prepare(&p, "éAéAéAéA", 2); let mut driver = script(&prepared);
    driver.cancel_at = Some(checkpoints.get()); let calls = Rc::clone(&driver.calls);
    let error = run(prepared, limits(), driver).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls.get(), 2);
}
#[test]
fn outer_envelope_budget_includes_all_metadata_and_does_not_publish_private_prompts() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared);
    let result = run(prepared, limits(), driver).unwrap(); let bytes = canonjson::canonical_bytes(&result).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    for field in ["generated_token_ids", "prompt_digest", "logical_model_digest"] { assert!(!text.contains(field)); }
    let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared);
    assert!(run(prepared, Int4SummaryLimits { max_result_bytes: bytes.len() - 1, ..limits() }, driver).is_err());
}
