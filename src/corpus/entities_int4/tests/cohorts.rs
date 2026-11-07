//! Private synthetic numerical boundaries exercise orchestration contracts.
//! No test in this module claims native model execution or measured speedup.
use super::*;
use super::super::cohort::{
    Int4EntitySchedule, Int4EntityCohortBudget, Int4EntityCohortRequirement,
    check_budget, preflight_discovery,
};
use crate::native_engine::strict_int4::{Int4MemoryBudget, Int4MemoryRequirement};

const CONTEXT: usize = 8192;
const PREPARATION: u64 = 16 << 20;

fn schedule(ner: Option<usize>, pairs: Option<usize>) -> Int4EntitySchedule {
    Int4EntitySchedule { ner_cohort_chunks: ner, pair_cohort_pairs: pairs }
}
fn budget(required: Int4EntityCohortRequirement) -> Int4EntityCohortBudget {
    Int4EntityCohortBudget {
        context_tokens: CONTEXT,
        native_memory: Int4MemoryBudget {
            max_kv_bytes: required.native.kv_bytes,
            max_rope_bytes: required.native.rope_bytes,
            max_scratch_payload_bytes: required.native.scratch_payload_bound,
        },
        preparation_reserve_bytes: PREPARATION,
        max_preparation_bytes: required.additional_preparation_bytes,
        max_workspace_bytes: required.workspace_bytes,
        max_result_bytes: required.max_result_bytes,
    }
}
fn grouped_maps(compiled: CompiledInputs, c: &Int4EntityConfig, width: usize, window: usize) -> Extracted {
    collect_windows(compiled.inputs, c, width, window, &mut Continue, |requests, _| {
        Ok(requests.iter().map(|request| synthetic(request.text, request.witness, width,
            request.text.contains("Smith").then_some("Smith"))).collect())
    }).unwrap()
}
fn grouped_completed(items: &[(&str, &str)], c: &Int4EntityConfig, width: usize, window: usize) -> Int4EntityRun {
    let p = prepared(items, c, width);
    let (reserved, masks_reserved, chunks) = (p.ner_work, p.mask_visits, p.chunks);
    let maps = grouped_maps(p, c, width, window);
    let resolved = resolution(&maps, c, width);
    finish(maps, resolved, reserved, masks_reserved, chunks, c, width, &mut Continue).unwrap()
}
fn native_preflight_config() -> Int4EntityConfig {
    let mut c = config();
    c.masks.max_visits_per_item = 1_000_000_000;
    c.masks.max_visits_per_run = 1_000_000_000_000;
    c
}
fn vocabulary() -> ExtractionVocabulary {
    let registry = pinned_controls::pinned().unwrap();
    ExtractionVocabulary::pinned(registry.template_controls()).unwrap()
}

#[test]
fn independently_selected_native_phases_quote_the_peak_instead_of_summing_engines() {
    let c = config();
    let serial = schedule(None, None).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    let ner = schedule(Some(3), None).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    let pairs = schedule(None, Some(2)).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    let both = schedule(Some(3), Some(2)).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    assert_eq!(serial.native, Int4MemoryRequirement::for_context(CONTEXT, 32).unwrap());
    assert_eq!(both.native.kv_bytes, ner.native.kv_bytes.max(pairs.native.kv_bytes));
    assert_eq!(both.native.rope_bytes, ner.native.rope_bytes.max(pairs.native.rope_bytes));
    assert_eq!(both.native.scratch_payload_bound,
        ner.native.scratch_payload_bound.max(pairs.native.scratch_payload_bound));
    assert_eq!(both.workspace_bytes, ner.workspace_bytes.max(pairs.workspace_bytes));
    assert!(both.native.kv_bytes < ner.native.kv_bytes + pairs.native.kv_bytes);
    assert_eq!(both.max_ner_result_bytes, ner.max_ner_result_bytes);
    assert_eq!(both.max_pair_result_bytes, pairs.max_pair_result_bytes);
    assert_eq!(both.max_result_bytes, c.max_result_bytes as u64 + 4096);
}

#[test]
fn only_simultaneous_source_plans_expand_preparation_residency() {
    let mut c = config();
    let serial = schedule(None, None).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    let one = schedule(Some(1), Some(1)).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    let four = schedule(Some(4), None).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    let pairs = schedule(None, Some(32)).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    assert_eq!(serial.additional_preparation_bytes, 0);
    assert_eq!(one.additional_preparation_bytes, 0);
    assert_eq!(pairs.additional_preparation_bytes, 0);
    assert_eq!(four.additional_preparation_bytes, 3 * PREPARATION);
    // The retained compact witnesses are already priced separately: the
    // physical window quote does not grow with the total snapshot chunk cap.
    c.max_snapshot_chunks = 1;
    let small = schedule(Some(4), None).requirements(&c, CONTEXT, PREPARATION, 32).unwrap();
    c.max_snapshot_chunks = 16_384;
    assert_eq!(small, schedule(Some(4), None).requirements(&c, CONTEXT, PREPARATION, 32).unwrap());
}

#[test]
fn each_physical_admission_component_is_independently_enforced() {
    let required = schedule(Some(4), Some(3)).requirements(&config(), CONTEXT, PREPARATION, 32).unwrap();
    check_budget(required, budget(required)).unwrap();
    for axis in 0..6 {
        let mut cap = budget(required);
        match axis {
            0 => cap.native_memory.max_kv_bytes -= 1,
            1 => cap.native_memory.max_rope_bytes -= 1,
            2 => cap.native_memory.max_scratch_payload_bytes -= 1,
            3 => cap.max_preparation_bytes -= 1,
            4 => cap.max_workspace_bytes -= 1,
            _ => cap.max_result_bytes -= 1,
        }
        assert!(matches!(check_budget(required, cap), Err(Int4EntityError::WorkBudget)), "axis {axis}");
    }
}

#[test]
fn widths_context_and_checked_preparation_arithmetic_fail_before_native_allocation() {
    for value in [schedule(Some(0), None), schedule(Some(65), None),
        schedule(None, Some(0)), schedule(None, Some(33))] {
        assert!(matches!(value.validate(), Err(Int4EntityError::InvalidLimits)));
    }
    for value in [schedule(None, None), schedule(Some(64), None), schedule(None, Some(32)),
        schedule(Some(64), Some(32))] {
        value.validate().unwrap();
    }
    let c = config();
    assert!(schedule(Some(4), None).requirements(&c, CONTEXT, u64::MAX, 32).is_err());
    assert!(schedule(None, None).requirements(&c, CONTEXT, u64::MAX, 32).is_err());
    assert!(schedule(Some(1), None).requirements(&c, CONTEXT, 0, 32).is_err());
    assert!(schedule(Some(2), Some(2)).requirements(&c, CONTEXT - 1, PREPARATION, 32).is_err());
}

#[test]
fn grouping_never_widens_either_original_single_head_kv_authority() {
    let row = Int4MemoryRequirement::for_context(CONTEXT, 32).unwrap();
    for phase in 0..2 {
        let mut c = config();
        if phase == 0 { c.ner_budget.max_kv_bytes = row.kv_bytes - 1; }
        else { c.scoring.planning.per_head.max_kv_bytes = row.kv_bytes - 1; }
        for selection in [schedule(Some(4), None), schedule(None, Some(4)), schedule(Some(4), Some(4))] {
            assert!(matches!(selection.requirements(&c, CONTEXT, PREPARATION, 32),
                Err(Int4EntityError::WorkBudget)), "phase {phase}");
        }
    }
}

#[test]
fn unchunked_documents_share_canonical_native_windows_and_a_ragged_tail() {
    let c = config();
    let p = prepared(&[("c", "Third"), ("a", "First"), ("b", "Second")], &c, 32);
    let mut windows = Vec::new();
    let output = collect_windows(p.inputs, &c, 32, 2, &mut Continue, |requests, _| {
        windows.push(requests.iter().map(|row| row.text.to_owned()).collect::<Vec<_>>());
        Ok(requests.iter().map(|row| synthetic(row.text, row.witness, 32, None)).collect())
    }).unwrap();
    assert_eq!(windows, vec![vec!["First".to_owned(), "Second".to_owned()], vec!["Third".to_owned()]]);
    assert_eq!(output.documents.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(), vec!["a", "b", "c"]);
    assert_eq!(output.chunks, 3);
    assert!(output.receipts.iter().all(|receipt| receipt.chunks.len() == 1 && receipt.chunks[0].chunk_id == 0));
}

#[test]
fn complete_semantics_are_identical_across_chunk_windows_and_group_geometries() {
    let c = chunked();
    let items = [("z", "Smith Smith Smith Smith"), ("a", "é\r\nSmith 上海😀 Smith"), ("m", "Smith")];
    for width in [16, 32] {
        let serial = canonjson::canonical_bytes(&completed(&items, &c, width)).unwrap();
        for window in [1, 2, 4, 64] {
            let grouped = grouped_completed(&items, &c, width, window);
            assert_eq!(serial, canonjson::canonical_bytes(&grouped).unwrap(), "width {width}, window {window}");
            assert_eq!(grouped.documents.iter().map(|d| d.document_id.as_str()).collect::<Vec<_>>(), vec!["a", "m", "z"]);
        }
    }
}

#[test]
fn borrowed_windows_preserve_all_original_unicode_and_crlf_coordinates() {
    let mut c = chunked(); c.chunks.as_mut().unwrap().max_chunk_bytes = 10;
    let items = [("z", "上海 😀 上海"), ("a", "é\r\n上海 😀 上海")];
    let p = prepared(&items, &c, 32);
    let expected_chunks = p.chunks;
    let mut calls = 0;
    let maps = collect_windows(p.inputs, &c, 32, 3, &mut Continue, |requests, _| {
        calls += 1;
        Ok(requests.iter().map(|request| {
            let quote = request.text.chars().find(|ch| !ch.is_whitespace()).unwrap().to_string();
            synthetic(request.text, request.witness, 32, Some(&quote))
        }).collect())
    }).unwrap();
    assert_eq!(calls, expected_chunks.div_ceil(3));
    assert_eq!(maps.chunks, expected_chunks);
    for (doc, receipt) in maps.documents.iter().zip(&maps.receipts) {
        let original = items.iter().find(|(id, _)| *id == doc.id.as_str()).unwrap().1;
        assert_eq!(doc.text, original);
        assert!(!doc.mentions.is_empty());
        for mention in &doc.mentions {
            let s = mention.span;
            validate_source_span(original, &mention.surface,
                SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
        }
        let mut byte = 0; let mut scalar = 0;
        for (index, chunk) in receipt.chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, index);
            assert_eq!(chunk.source_span.byte_start, byte);
            assert_eq!(chunk.source_span.scalar_start, scalar);
            byte = chunk.source_span.byte_end; scalar = chunk.source_span.scalar_end;
        }
        assert_eq!(byte, original.len()); assert_eq!(scalar, original.chars().count());
    }
}

#[test]
fn missing_extra_or_late_corrupt_rows_cannot_publish_a_partial_window() {
    let c = config();
    let items = [("a", "Smith"), ("b", "Smith"), ("c", "Smith")];
    for axis in 0..7 {
        let p = prepared(&items, &c, 32); let mut calls = 0;
        let output = collect_windows(p.inputs, &c, 32, 2, &mut Continue, |requests, _| {
            calls += 1;
            let mut rows: Vec<_> = requests.iter().map(|row| synthetic(row.text, row.witness, 32, Some("Smith"))).collect();
            match axis {
                0 => { rows.pop(); },
                1 => rows.push(synthetic(requests[0].text, requests[0].witness, 32, None)),
                2 => rows[1].model_work.forward_positions += 1,
                3 => rows[1].model_work.projected_logits += 1,
                4 => rows[1].model_work.attention_pairs += 1,
                5 => rows[1].model_work.projections.dot_products += 1,
                _ => rows[1].model_work.projections.multiply_accumulates += 1,
            }
            Ok(rows)
        });
        assert!(output.is_err(), "axis {axis}");
        assert_eq!(calls, 1);
    }
}

#[test]
fn grounding_allowances_are_not_renewed_at_document_or_window_boundaries() {
    let items = [("a", "Smith"), ("b", "Smith"), ("c", "Smith")];
    for axis in 0..3 {
        let mut c = config();
        let used = grouped_maps(prepared(&items[..2], &c, 32), &c, 32, 2).verification_used;
        let p = prepared(&items, &c, 32);
        match axis {
            0 => c.verification.max_fields = used.fields,
            1 => c.verification.max_matches = used.matches,
            _ => c.verification.max_scan_steps = used.scan_steps,
        }
        let mut calls = 0;
        let output = collect_windows(p.inputs, &c, 32, 2, &mut Continue, |requests, _| {
            calls += 1;
            Ok(requests.iter().map(|row| synthetic(row.text, row.witness, 32, Some("Smith"))).collect())
        });
        assert!(output.is_err(), "axis {axis}"); assert_eq!(calls, 2);
    }
}

#[test]
fn grouped_early_eos_leaves_all_five_ner_reservations_unavailable_to_pairs() {
    let items = [("a", "Smith"), ("b", "Smith"), ("c", "Smith")];
    let mut c = config(); let p = prepared(&items, &c, 32); let reserved = p.ner_work;
    let maps = grouped_maps(p, &c, 32, 2);
    assert!(maps.work.forward_positions < reserved.forward_positions);
    assert!(maps.work.projected_logits < reserved.projected_logits);
    assert!(maps.work.attention_pairs < reserved.attention_pairs);
    assert!(maps.work.projections.dot_products < reserved.projections.dot_products);
    assert!(maps.work.projections.multiply_accumulates < reserved.projections.multiply_accumulates);
    c.max_model_work = reserved;
    assert_eq!(remaining_scoring(&c, reserved).unwrap().max_model_work, Int4Work::default());
    assert_ne!(remaining_scoring(&c, maps.work).unwrap().max_model_work, Int4Work::default());
}

#[test]
fn cancellation_after_one_native_window_prevents_all_later_chunks_and_pairs() {
    struct Control(bool);
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0.then_some(DecodeCancellationKind::Deadline)
        }
    }
    let c = config();
    let p = prepared(&[("a", "Smith"), ("b", "Smith"), ("c", "Smith")], &c, 32);
    let mut calls = 0;
    let output = collect_windows(p.inputs, &c, 32, 2, &mut Control(false), |requests, control| {
        calls += 1; control.0 = true;
        Ok(requests.iter().map(|row| synthetic(row.text, row.witness, 32, None)).collect())
    });
    assert_eq!(calls, 1);
    assert_eq!(output.err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn empty_snapshots_and_invalid_windows_never_invoke_a_native_callback() {
    let c = config();
    for window in [1, 2, 64] {
        let p = prepared(&[], &c, 32);
        let maps = collect_windows(p.inputs, &c, 32, window, &mut Continue,
            |_, _| -> Result<Vec<Int4SourceTaskRun>, Int4EntityError> { panic!("empty snapshot invoked native callback"); }).unwrap();
        assert_eq!(maps.chunks, 0); assert_eq!(maps.work, Int4Work::default());
        assert_eq!(maps.mask_visits, 0); assert!(maps.documents.is_empty()); assert!(maps.receipts.is_empty());
    }
    for window in [0, 65] {
        let p = prepared(&[("a", "Smith")], &c, 32);
        assert!(matches!(collect_windows(p.inputs, &c, 32, window, &mut Continue,
            |_, _| -> Result<Vec<Int4SourceTaskRun>, Int4EntityError> { panic!("invalid width invoked native callback"); }),
            Err(Int4EntityError::InvalidLimits)));
    }
}

#[test]
fn discovery_preflight_checks_the_final_private_witness_and_complete_snapshot_totals() {
    let c = native_preflight_config(); let vocabulary = vocabulary();
    let items = [("a", "Smith"), ("b", "Another Smith"), ("c", "Last Smith")];
    let original = prepared(&items, &c, 32);
    preflight_discovery(&original, source(), &identity(32), &c, 32, &vocabulary, CONTEXT, &mut Continue).unwrap();
    for axis in 0..7 {
        let mut p = prepared(&items, &c, 32);
        match axis {
            0 => p.inputs[2].chunks[0].identity = Sha256Digest::of_bytes(b"different final source witness"),
            1 => p.inputs[2].chunks[0].work.projected_logits += 1,
            2 => p.inputs[2].chunks[0].span.scalar_end += 1,
            3 => p.maximum_positions += 1,
            4 => p.chunks += 1,
            5 => p.ner_work.attention_pairs += 1,
            _ => p.mask_visits += 1,
        }
        assert!(preflight_discovery(&p, source(), &identity(32), &c, 32, &vocabulary, CONTEXT, &mut Continue).is_err(),
            "axis {axis}");
    }
}

#[test]
fn discovery_preflight_rejects_mask_context_and_cancellation_before_any_engine() {
    let c = native_preflight_config(); let vocabulary = vocabulary();
    let p = prepared(&[("a", "Smith"), ("b", "Smith")], &c, 32);
    let mut masks = c.clone(); masks.masks.max_visits_per_item = 1;
    assert!(preflight_discovery(&p, source(), &identity(32), &masks, 32, &vocabulary, CONTEXT, &mut Continue).is_err());
    assert!(preflight_discovery(&p, source(), &identity(32), &c, 32, &vocabulary,
        p.maximum_positions - 1, &mut Continue).is_err());
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::PollQuota) }
    }
    let error = preflight_discovery(&p, source(), &identity(32), &c, 32, &vocabulary, CONTEXT, &mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota));
}
