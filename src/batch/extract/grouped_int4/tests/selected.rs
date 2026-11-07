//! Pinned schema/source compilation and stream policy binding; no inference.
use super::*;
use crate::tasks::extract::grouped_int4::{INT4_EXTRACT_VERSION, INT4_SPARSE_EXTRACT_VERSION};

fn rows(cap: usize) -> Int4JsonSparseLimits { Int4JsonSparseLimits { max_rows_per_step: cap } }

#[test]
fn fixed_selected_policy_binds_each_actual_plan_without_changing_source_or_task_authority() {
    for (schema, grounding) in [(r#"{"type":"boolean"}"#, ExtractionBatchGrounding::Structural),
        (r#"{"type":"number","const":12345678901234567890123456789012345678}"#, ExtractionBatchGrounding::Structural),
        (r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#, ExtractionBatchGrounding::SourceMembership)] {
        let source = "é 上海 上海 <|im_start|>";
        let dense = planner(127, None); let selected = planner(127, None).with_selected_rows(rows(7)).unwrap();
        let a = dense.prepare(document(schema, source, grounding)).unwrap();
        let b = selected.prepare(document(schema, source, grounding)).unwrap();
        assert_eq!(a.extraction_plan().execution_version(), INT4_EXTRACT_VERSION);
        assert_eq!(b.extraction_plan().execution_version(), INT4_SPARSE_EXTRACT_VERSION);
        assert_eq!(selected.selected_rows(), Some(rows(7)));
        assert_eq!(b.extraction_plan().selected_rows(), selected.selected_rows());
        assert_eq!(a.source().text(), b.source().text());
        assert_eq!(a.request_digest(), b.request_digest());
        assert_eq!(a.task_plan().ir().budget(), b.task_plan().ir().budget());
        assert_eq!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
        assert_eq!(a.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
        assert_eq!(a.execution_identity().schema_digest, b.execution_identity().schema_digest);
        assert_eq!(a.planned_work().forward_positions, b.planned_work().forward_positions);
        assert_eq!(a.planned_work().attention_pairs, b.planned_work().attention_pairs);
        assert_eq!(b.planned_work().projected_logits, 7 * u64::from(budget().max_output_tokens));
        assert!(b.planned_work().projections.multiply_accumulates < a.planned_work().projections.multiply_accumulates);
        assert_ne!(a.comparison_policy_digest(), b.comparison_policy_digest());
        assert!(b.verify_identity(a.execution_identity()).is_err());
        b.verify_identity(b.execution_identity()).unwrap();
        selected.check_projection(&b).unwrap(); dense.check_projection(&a).unwrap();
        assert!(dense.check_projection(&b).is_err()); assert!(selected.check_projection(&a).is_err());
    }
}

#[test]
fn repeated_invalid_and_full_vocabulary_caps_keep_an_explicit_distinct_policy() {
    for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        assert!(planner(32, None).with_selected_rows(rows(cap)).is_err());
    }
    assert!(planner(32, None).with_selected_rows(rows(7)).unwrap().with_selected_rows(rows(7)).is_err());
    let dense = planner(32, None); let full = planner(32, None).with_selected_rows(rows(NANBEIGE_VOCAB_SIZE)).unwrap();
    let a = dense.prepare(boolean()).unwrap(); let b = full.prepare(boolean()).unwrap();
    assert_eq!(a.planned_work(), b.planned_work());
    assert_ne!(a.execution_identity(), b.execution_identity());
    dense.require_full_projection().unwrap(); assert!(full.require_full_projection().is_err());
    assert!(dense.check_projection(&b).is_err()); assert!(full.check_projection(&a).is_err());
    let seven = planner(32, None).with_selected_rows(rows(7)).unwrap();
    let eight = planner(32, None).with_selected_rows(rows(8)).unwrap();
    assert!(seven.check_projection(&eight.prepare(boolean()).unwrap()).is_err());
}

#[test]
fn shared_schema_and_per_item_caps_cannot_replace_the_fixed_head_policy() {
    let compiler = planner(32, Some(args(r#"{"type":"boolean"}"#, ExtractionBatchGrounding::Structural)))
        .require_shared_defaults().unwrap().with_selected_rows(rows(7)).unwrap();
    let mut first = boolean(); first.task_args = None;
    let a = compiler.prepare(first).unwrap();
    let mut other = boolean(); other.task_args = None; other.text.push_str(" changed");
    let b = compiler.prepare(other).unwrap();
    assert_ne!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    assert_eq!(a.extraction_plan().selected_rows(), b.extraction_plan().selected_rows());
    assert_eq!(compiler.prepare(boolean()).err().unwrap().fault.code, BatchCode::InvalidEnvelope);
    let changed = planner(32, None).with_selected_rows(rows(7)).unwrap();
    let mut item = boolean(); item.task_args.as_mut().unwrap().budget.max_output_tokens = 8;
    let smaller = changed.prepare(item).unwrap();
    assert_eq!(smaller.planned_work().projected_logits, 7 * 8);
    assert_eq!(smaller.extraction_plan().selected_rows(), Some(rows(7)));
}
