//! UNRUN real pinned-plan and private scripted-result fixtures. These establish
//! contracts only when executed; they do not establish neural relevance/recall.
use super::*;
use super::super::tests::{planner, prepare_task, limits as mapping_limits, admitted,
    script as native_script, Script};
use std::rc::Rc;
use crate::tasks::keyphrases::{RankedKeyphrase, KEYPHRASES_RANKING};

const SOURCE: &str = "AéAéBéBéCéCéDéDé";
fn prepare<'s>(p: &SourceTaskPlanner, source: &'s str, fan_in: usize, batch: usize) -> PreparedInt4SourceMap<'s> {
    let mut mapping = mapping_limits(); mapping.chunks.max_chunk_bytes = 6;
    mapping.chunks.max_chunk_tokens = 6; mapping.reduction.reduce_fan_in = fan_in;
    mapping.reduction.map_batch_chunks = batch;
    prepare_task(p, source, &SourceMapTask::Keyphrases(KeyphraseOptions::default()), mapping, 32)
}
fn limits() -> Int4KeyphraseLimits { Int4KeyphraseLimits { max_phrases: 1, ..Int4KeyphraseLimits::default() } }
fn keyphrases(raw: &mut SourceTaskResult) -> &mut KeyphraseResult {
    let SourceTaskResult::Keyphrases(result) = raw else { panic!("keyphrase fixture") }; result
}
fn proposals(source: &str, texts: &[&str]) -> Vec<RankedKeyphrase> {
    texts.iter().enumerate().map(|(index, text)| {
        let spans = scan_occurrences(source, text, &mut GroundingBudget::default()).unwrap();
        RankedKeyphrase { rank: index + 1, text: (*text).to_owned(),
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
    }).collect()
}
fn script(prepared: &PreparedInt4SourceMap<'_>) -> Script {
    let mut driver = native_script(prepared);
    for (raw, chunk) in driver.outputs.iter_mut().zip(prepared.chunks.chunks()) {
        let local = chunk.text().chars().next().unwrap().to_string();
        keyphrases(raw).phrases = proposals(chunk.text(), &[&local, "é"]);
    }
    driver
}
fn run(prepared: PreparedInt4SourceMap<'_>, limits: Int4KeyphraseLimits, driver: Script)
    -> Result<Int4CorpusKeyphraseRun, Int4CorpusKeyphraseError> {
    let identities = admitted(&prepared); prepared.execute_keyphrases_with_driver(&identities, limits, driver)
}

#[test]
fn complete_root_ranking_preserves_late_support_and_every_unicode_occurrence() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1);
    assert_eq!(prepared.chunk_count(), 4); let expected = prepared.preflight_metadata();
    let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    let result = run(prepared, limits(), driver).unwrap(); expected.verify_keyphrases_completed(&result).unwrap();
    assert_eq!(calls.get(), 4); assert_eq!(result.requested_max_phrases, 1); assert_eq!(result.keyphrases.phrases.len(), 1);
    let phrase = &result.keyphrases.phrases[0];
    assert_eq!(phrase.text, "é"); assert_eq!(phrase.rank_sum, 8); assert_eq!(phrase.evidence.len(), 4);
    assert_eq!(result.keyphrases.omitted_candidates, 4); assert_eq!(result.keyphrases.mapped_chunks, 4);
    for (index, evidence) in phrase.evidence.iter().enumerate() {
        assert_eq!(evidence.chunk_id, index); assert_eq!(evidence.local_rank, 2); assert_eq!(evidence.spans.len(), 2);
        for span in &evidence.spans {
            assert_eq!(&SOURCE[span.byte_start..span.byte_end], phrase.text);
            assert_eq!(SOURCE[..span.byte_start].chars().count(), span.scalar_start);
            assert_eq!(SOURCE[..span.byte_end].chars().count(), span.scalar_end);
        }
    }
    assert_eq!(result.source_span.byte_end, SOURCE.len()); assert_eq!(result.source_span.scalar_end, SOURCE.chars().count());
    assert_eq!(result.verification_used.fields, 8); assert_eq!(result.verification_used.matches, 16);
    assert!(result.verification_used.scan_steps > 0); assert!(result.verification_scan_work > 0);
    assert_eq!(result.mask_node_visit_charge, 80); assert_eq!(result.reserved_mask_node_visits, 4000);
    assert!(fits(result.model_work, result.planned_model_work));
    assert_eq!(result.keyphrases.projected_logits, result.model_work.projected_logits);
}

#[test]
fn exact_ranking_and_evidence_are_independent_of_batch_and_multilevel_tree_shape() {
    let p = planner(); let mut baseline = None;
    for fan_in in [2, 3, 8] { for batch in [1, 2, 4] {
        let prepared = prepare(&p, SOURCE, fan_in, batch); let driver = script(&prepared);
        let result = run(prepared, limits(), driver).unwrap();
        let bytes = canonjson::canonical_bytes(&result.keyphrases).unwrap();
        if let Some(saved) = &baseline { assert_eq!(&bytes, saved); } else { baseline = Some(bytes); }
        if fan_in == 2 { assert!(result.reduction_levels > 1); }
        assert_eq!(result.verification_used.matches, 16);
    } }
}

#[test]
fn equal_support_uses_rank_sum_then_original_source_order_without_case_folding() {
    let p = planner(); let source = "βaβaβaβa";
    for reverse_second in [false, true] {
        let prepared = prepare(&p, source, 2, 1); let mut driver = native_script(&prepared);
        for (index, (raw, chunk)) in driver.outputs.iter_mut().zip(prepared.chunks.chunks()).enumerate() {
            let order = if reverse_second && index == 1 { ["a", "β"] } else { ["β", "a"] };
            keyphrases(raw).phrases = proposals(chunk.text(), &order);
        }
        let result = run(prepared, Int4KeyphraseLimits { max_phrases: 2, ..limits() }, driver).unwrap();
        assert_eq!(result.keyphrases.phrases.iter().map(|p| p.text.as_str()).collect::<Vec<_>>(), ["β", "a"]);
        let expected = if reverse_second { [3, 3] } else { [2, 4] };
        assert_eq!(result.keyphrases.phrases.iter().map(|p| p.rank_sum).collect::<Vec<_>>(), expected);
    }
    let prepared = prepare(&p, "AaAaAa", 2, 1); let mut driver = native_script(&prepared);
    keyphrases(&mut driver.outputs[0]).phrases = proposals("AaAaAa", &["A", "a"]);
    let result = run(prepared, Int4KeyphraseLimits { max_phrases: 2, ..limits() }, driver).unwrap();
    assert_eq!(result.keyphrases.phrases.len(), 2);
    assert_eq!(result.keyphrases.phrases[0].text, "A"); assert_eq!(result.keyphrases.phrases[1].text, "a");
    assert_eq!(result.keyphrases.phrases[0].evidence.len(), 1);
    assert_eq!(result.keyphrases.phrases[0].evidence[0].spans.len(), 3);
    // Identical support, rank sums AND first source offsets use exact UTF-8
    // lexical order. Different strings sharing an initial occurrence stay distinct.
    let prepared = prepare(&p, "abababababab", 2, 1); let mut driver = native_script(&prepared);
    for (index, (raw, chunk)) in driver.outputs.iter_mut().zip(prepared.chunks.chunks()).enumerate() {
        let order = if index == 0 { ["ab", "a"] } else { ["a", "ab"] };
        keyphrases(raw).phrases = proposals(chunk.text(), &order);
    }
    let result = run(prepared, Int4KeyphraseLimits { max_phrases: 2, ..limits() }, driver).unwrap();
    assert_eq!(result.keyphrases.phrases.iter().map(|p| p.text.as_str()).collect::<Vec<_>>(), ["a", "ab"]);
    assert!(result.keyphrases.phrases.iter().all(|p| p.rank_sum == 3 && p.evidence.len() == 2));
}

#[test]
fn overlapping_occurrences_are_complete_but_do_not_multiply_chunk_support() {
    let p = planner(); let prepared = prepare(&p, "aaaaaa", 2, 1); let mut driver = native_script(&prepared);
    keyphrases(&mut driver.outputs[0]).phrases = proposals("aaaaaa", &["aa"]);
    let result = run(prepared, limits(), driver).unwrap(); let phrase = &result.keyphrases.phrases[0];
    assert_eq!(phrase.evidence.len(), 1); assert_eq!(phrase.rank_sum, 1);
    assert_eq!(phrase.evidence[0].spans.len(), 5); assert_eq!(result.verification_used.matches, 5);
    assert_eq!(phrase.evidence[0].spans.iter().map(|span| span.byte_start).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
}

#[test]
fn every_last_identity_resident_model_group_and_complete_kv_is_checked_before_forward() {
    let p = planner();
    for field in 0..7 {
        let mut prepared = prepare(&p, SOURCE, 2, 1); let mut identities = admitted(&prepared);
        let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        match field {
            0 => identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final keyphrase chunk"),
            1 => { identities.pop(); },
            2 => driver.fail_model_at = Some(prepared.chunk_count()),
            3 => prepared.plans.last_mut().unwrap().budget.max_kv_bytes -= 1,
            4 => driver.width = 16, 5 => driver.capacity += 1, _ => driver.dirty_after = Some(0),
        }
        assert!(prepared.execute_keyphrases_with_driver(&identities, limits(), driver).is_err());
        assert_eq!(calls.get(), 0, "field {field}");
    }
}

#[test]
fn unselected_late_candidate_corruption_refuses_the_complete_result() {
    let p = planner();
    for field in 0..7 {
        let prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared);
        let calls = Rc::clone(&driver.calls); let last = keyphrases(driver.outputs.back_mut().unwrap());
        match field {
            0 => { last.phrases[0].spans.pop(); }, 1 => last.phrases[0].spans[0].scalar_end += 1,
            2 => last.phrases[0].occurrence = SourceOccurrence::Anchored, 3 => last.phrases[0].rank = 2,
            4 => last.phrases[0].text = "absent".to_owned(),
            5 => { let mut duplicate = last.phrases[0].clone(); duplicate.rank = 3; last.phrases.push(duplicate); },
            _ => last.phrases[0].spans[0].byte_start += 1,
        }
        assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 4, "field {field}");
    }
}

#[test]
fn native_work_profiles_masks_tokens_and_dirty_completion_cannot_be_substituted() {
    let p = planner();
    for field in 0..11 {
        let prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared);
        let calls = Rc::clone(&driver.calls);
        if field < 5 { driver.corrupt_axis = Some(field); }
        else if field == 10 { driver.dirty_after = Some(1); }
        else {
            let raw = keyphrases(&mut driver.outputs[0]);
            match field {
                5 => raw.numerics_profile = crate::native_engine::strict_int8::STRICT_INT8_PROFILE.to_owned(),
                6 => raw.numerics_profile = crate::native_engine::hf_bf16_eager::HF_BF16_EAGER_PROFILE.to_owned(),
                7 => raw.mask_node_visit_charge = 1001,
                8 => { raw.generated_token_ids.pop(); },
                _ => raw.generated_token_ids[0] = *raw.generated_token_ids.last().unwrap(),
            }
        }
        assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 1, "field {field}");
    }
}

#[test]
fn map_verification_and_corpus_scan_limits_are_independent_cumulative_authority() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    let baseline = run(prepared, limits(), driver).unwrap();
    for field in 0..4 {
        let mut prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared); let mut config = limits();
        match field {
            0 => prepared.limits.verification.max_fields = baseline.verification_used.fields - 1,
            1 => prepared.limits.verification.max_matches = baseline.verification_used.matches - 1,
            2 => prepared.limits.verification.max_scan_steps = baseline.verification_used.scan_steps - 1,
            _ => config.aggregation.max_scan_work = baseline.verification_scan_work - 1,
        }
        assert!(run(prepared, config, driver).is_err(), "field {field}");
    }
    let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    let mut exact = limits(); exact.aggregation.max_scan_work = baseline.verification_scan_work;
    assert_eq!(run(prepared, exact, driver).unwrap().verification_scan_work, baseline.verification_scan_work);
}

#[test]
fn final_top_k_cannot_hide_complete_candidate_span_or_value_overflow() {
    let p = planner();
    for field in 0..4 {
        let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared); let mut config = limits();
        match field {
            0 => config.aggregation.max_unique_phrases = 4,
            1 => config.aggregation.max_evidence_spans = 15,
            2 => config.aggregation.max_value_bytes = 1,
            _ => config.max_result_bytes = 1,
        }
        assert!(run(prepared, config, driver).is_err());
    }
}

#[test]
fn every_shared_checkpoint_and_native_failure_preserves_typed_cancellation() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    let checkpoints = Rc::clone(&driver.checkpoints); run(prepared, limits(), driver).unwrap();
    for point in 1..=checkpoints.get() {
        let prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared); driver.cancel_at = Some(point);
        assert_eq!(run(prepared, limits(), driver).err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline),
            "checkpoint {point}");
    }
    let prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared); driver.fail_run_at = Some(4);
    let calls = Rc::clone(&driver.calls); let error = run(prepared, limits(), driver).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls.get(), 4);
}

#[test]
fn singleton_and_empty_selections_still_cover_every_original_chunk() {
    let p = planner(); let prepared = prepare(&p, "AéAé", 2, 1); let driver = script(&prepared);
    let result = run(prepared, limits(), driver).unwrap(); assert_eq!(result.reduce_calls, 0); assert_eq!(result.reduction_levels, 0);
    let prepared = prepare(&p, SOURCE, 2, 1); let expected = prepared.preflight_metadata();
    let mut driver = script(&prepared);
    for raw in &mut driver.outputs { keyphrases(raw).phrases.clear(); }
    let result = run(prepared, limits(), driver).unwrap(); expected.verify_keyphrases_completed(&result).unwrap();
    assert!(result.keyphrases.phrases.is_empty()); assert_eq!(result.keyphrases.mapped_chunks, 4);
    assert_eq!(result.keyphrases.omitted_candidates, 0); assert_eq!(result.source_span.byte_end, SOURCE.len());
    assert_eq!(result.verification_used.fields, 0); assert_eq!(result.verification_used.matches, 0);
    assert_eq!(result.verification_scan_work, 0); assert_eq!(result.mask_node_visit_charge, 80);
}

#[test]
fn non_keyphrase_mixed_options_and_invalid_limits_refuse_before_native_calls() {
    let p = planner();
    let prepared = prepare_task(&p, "AéAé", &SourceMapTask::Ner(NerOptions::default()), mapping_limits(), 32);
    let driver = native_script(&prepared); let calls = Rc::clone(&driver.calls);
    assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 0);
    for field in 0..7 {
        let mut prepared = prepare(&p, SOURCE, 2, 1); let mut config = limits();
        let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        match field {
            0 => { let Finalizer::Keyphrases(options) = &mut prepared.plans.last_mut().unwrap().finalizer else { unreachable!() };
                options.max_phrases += 1; },
            1 => config.max_phrases = 0, 2 => config.max_phrases = 4097, 3 => config.max_result_bytes = 0,
            4 => config.aggregation.max_value_bytes = 8 << 20, 5 => config.max_result_bytes = 8 << 20,
            _ => config.aggregation.max_scan_work = 0,
        }
        assert!(run(prepared, config, driver).is_err()); assert_eq!(calls.get(), 0, "field {field}");
    }
}

#[test]
fn completed_receipt_binds_native_map_verification_ranking_and_grounding_metadata() {
    let p = planner();
    for field in 0..22 {
        let prepared = prepare(&p, SOURCE, 2, 1); let expected = prepared.preflight_metadata();
        let driver = script(&prepared); let mut result = run(prepared, limits(), driver).unwrap();
        match field {
            0 => result.schema_version += 1, 1 => result.execution = "wrong-execution",
            2 => result.numerics_profile = crate::native_engine::strict_int8::STRICT_INT8_PROFILE,
            3 => result.group_width = 16, 4 => result.chunk_profile = "wrong-chunk-profile",
            5 => result.semantics = "wrong-semantics", 6 => result.source_span.byte_end -= 1,
            7 => result.map_batches = 0, 8 => result.planned_model_work.attention_pairs -= 1,
            9 => result.model_work.projections.dot_products = result.planned_model_work.projections.dot_products + 1,
            10 => result.reserved_mask_node_visits -= 1, 11 => result.mask_node_visit_charge += 1,
            12 => result.verification_used.fields = mapping_limits().verification.max_fields + 1,
            13 => result.keyphrases.mapped_chunks -= 1, 14 => result.keyphrases.ranking_policy = "wrong-ranking".to_owned(),
            15 => result.keyphrases.grounding = ExtractionGrounding::NotRequested,
            16 => result.keyphrases.score_space = ScoreSpace::FullVocabSequenceLogprob,
            17 => result.keyphrases.warnings.swap(0, 1), 18 => result.requested_max_phrases = 2,
            19 => result.keyphrases.omitted_candidates = result.verification_used.fields,
            20 => result.keyphrases.task_spec_version = "wrong-task".to_owned(),
            _ => result.keyphrases.forward_positions += 1,
        }
        assert!(expected.verify_keyphrases_completed(&result).is_err(), "field {field}");
    }
}

#[test]
fn complete_envelope_cap_covers_metadata_and_omits_private_native_transcripts() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    let result = run(prepared, limits(), driver).unwrap(); let bytes = canonjson::canonical_bytes(&result).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    for field in ["generated_token_ids", "prompt_digest", "logical_model_digest"] { assert!(!text.contains(field)); }
    let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    assert!(run(prepared, Int4KeyphraseLimits { max_result_bytes: bytes.len() - 1, ..limits() }, driver).is_err());
    let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    assert!(run(prepared, Int4KeyphraseLimits { max_result_bytes: bytes.len(), ..limits() }, driver).is_ok());
}

#[test]
fn eager_int8_and_int4_reducers_keep_separate_profile_admission() {
    struct Fixture { profile: &'static str }
    impl KeyphrasePass for Fixture {
        fn options(&self) -> KeyphraseOptions { KeyphraseOptions::default() }
        fn run(&mut self, _: &SourceChunk<'_>) -> Result<KeyphraseResult, KeyphraseError> {
            Ok(KeyphraseResult { schema_version: 1, task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(),
                numerics_profile: self.profile.to_owned(), ranking_policy: KEYPHRASES_RANKING.to_owned(),
                score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
                phrases: Vec::new(), generated_token_ids: Vec::new(), forward_positions: 0,
                projected_logits: 0, mask_node_visit_charge: 0 })
        }
    }
    let chunks = ChunkPlan::build("AéAé", mapping_limits().chunks, |source| Ok(source.len())).unwrap();
    let profiles = [crate::native_engine::hf_bf16_eager::HF_BF16_EAGER_PROFILE,
        crate::native_engine::strict_int8::STRICT_INT8_PROFILE, STRICT_INT4_PROFILE];
    for constructor in 0..3 { for (returned, profile) in profiles.iter().enumerate() {
        let fixture = Fixture { profile: *profile };
        let mut task = match constructor {
            0 => CorpusKeyphraseTask::new(fixture, CorpusKeyphraseLimits::default()).unwrap(),
            1 => CorpusKeyphraseTask::new_int8(fixture, CorpusKeyphraseLimits::default()).unwrap(),
            _ => CorpusKeyphraseTask::new_int4(fixture, CorpusKeyphraseLimits::default()).unwrap(),
        };
        assert_eq!(task.map_batch(chunks.chunks()).is_ok(), constructor == returned);
    } }
}
