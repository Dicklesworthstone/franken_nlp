//! Real pinned population planning. No weights or generated answers are used.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding};

// This short multilingual document leaves all four synthesis passes within
// the fixture's existing eight-mask admission ceiling.
const TEXT: &str = "Alice met Bob in 上海.";
fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn allowance() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }
}
fn exact(work: SafeBpeWork) -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: work.input_bytes() as u64, max_output_tokens: work.output_tokens() as u64,
        max_pair_checks: work.pair_checks(), max_pair_bytes: work.pair_bytes() }
}
fn rebuild(factory: &Int4SourceDocumentBatchPlanner, prepared: &PreparedInt4SourceDocumentBatch)
    -> Result<(), BatchItemFailure> {
    factory.owns(prepared)?;
    let mut config = factory.configuration(); config.planner = &prepared.operation;
    let plan = config.plan(&prepared.source, &mut Continue)?;
    plan.compare_receipt(prepared.expected, prepared.work, &prepared.identities, &mut Continue)
}
fn used(prepared: &PreparedInt4SourceDocumentBatch) -> SafeBpeWork {
    prepared.map_document_encoding_work().unwrap().unwrap()
}

#[test]
fn all_complete_source_modes_share_one_operation_per_original_document() {
    for index in 0..10 {
        let byte = factory(index);
        let byte_prepared = byte.prepare_with_control(document(TEXT), &mut Continue).unwrap();
        assert_eq!(byte.encoding_workspace_bytes().unwrap(), 0);
        assert_eq!(byte_prepared.map_document_encoding_work().unwrap(), None);
        let compact = byte.with_map_document_encoding(policy(), allowance()).unwrap();
        assert!(compact.owns(&byte_prepared).is_err());
        let first = compact.prepare_with_control(document(TEXT), &mut Continue).unwrap();
        let second = compact.prepare_with_control(document(TEXT), &mut Continue).unwrap();
        let before = used(&first);
        assert!(before.input_bytes() > TEXT.len());
        assert!(before.pair_checks() > 0 && before.pair_bytes() > 0);
        assert!(first.planned_work().forward_positions < byte_prepared.planned_work().forward_positions, "mode {index}");
        assert_eq!(before, used(&second));
        assert!(!Arc::ptr_eq(&first.operation, &second.operation));
        assert_eq!(first.execution_identities(), second.execution_identities());
        rebuild(&compact, &first).unwrap();
        assert_eq!(used(&first), before.checked_add(before).unwrap());
        assert_eq!(used(&second), before);
        assert_eq!(compact.planner.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
    }
}

#[test]
fn preparation_and_execution_reconstruction_pay_every_axis_without_refund() {
    // Discovery, flat synthesis and hierarchy retain the same original operation.
    for index in [0, 4, 5, 6, 7, 8] {
        let probe = factory(index).with_map_document_encoding(policy(), allowance()).unwrap();
        let prepared = probe.prepare_with_control(document(TEXT), &mut Continue).unwrap();
        let once = used(&prepared); let twice = once.checked_add(once).unwrap();
        let all = exact(twice);
        let compact = factory(index).with_map_document_encoding(policy(), all).unwrap();
        let prepared = compact.prepare_with_control(document(TEXT), &mut Continue).unwrap();
        rebuild(&compact, &prepared).unwrap();
        assert_eq!(used(&prepared), twice);
        assert!(rebuild(&compact, &prepared).is_err(), "a consumed operation cannot renew");
        assert_eq!(used(&prepared), twice);
        // Another original item receives its own explicitly admitted attempt.
        let other = compact.prepare_with_control(document(TEXT), &mut Continue).unwrap();
        assert_eq!(used(&other), once);
        for axis in 0..4 {
            let mut short = all;
            match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
                2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
            let limited = factory(index).with_map_document_encoding(policy(), short).unwrap();
            let prepared = limited.prepare_with_control(document(TEXT), &mut Continue).unwrap();
            assert_eq!(used(&prepared), once);
            assert!(rebuild(&limited, &prepared).is_err(), "mode {index}, axis {axis}");
            let spent = used(&prepared);
            assert!(spent.input_bytes() >= once.input_bytes() && spent.output_tokens() >= once.output_tokens()
                && spent.pair_checks() >= once.pair_checks() && spent.pair_bytes() >= once.pair_bytes());
            assert!(rebuild(&limited, &prepared).is_err());
        }
    }
}

#[test]
fn compact_policy_requires_explicit_authority_and_preserves_prepared_ownership() {
    assert!(factory(0).with_map_document_encoding(UntrustedEncoding::ByteFallback, allowance()).is_err());
    let compact = factory(0).with_map_document_encoding(policy(), allowance()).unwrap();
    assert!(crate::batch::source::grouped_int4::check_configuration(&compact.planner, &compact.identity,
        compact.budget, compact.planning, None, compact.group_width).is_err());
    let prepared = compact.prepare_with_control(document(TEXT), &mut Continue).unwrap();
    let mut changed = allowance(); changed.max_pair_bytes -= 1;
    let updated = compact.with_map_document_encoding(policy(), changed).unwrap();
    assert!(updated.owns(&prepared).is_err());
    let prepared = updated.prepare_with_control(document(TEXT), &mut Continue).unwrap();
    let cohorts = updated.with_chunk_cohorts().unwrap();
    assert!(cohorts.owns(&prepared).is_err());
    let current = cohorts.prepare_with_control(document(TEXT), &mut Continue).unwrap();
    rebuild(&cohorts, &current).unwrap();
}
