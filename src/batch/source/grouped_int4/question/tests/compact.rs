//! Compact source/question population ownership and exact composed budgets.
use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding};

const SOURCE: &str = "Alice met Bob in 上海. Carol saw Alice. <think>";
const QUESTION: &str = "Who met Alice? <|im_start|>";
fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn allowance() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }
}
fn used(prepared: &PreparedInt4QuestionDocument) -> SafeBpeWork {
    prepared.map_document_encoding_work().unwrap().unwrap()
}
fn rebuild(factory: &Int4QuestionDocumentBatchPlanner, prepared: &PreparedInt4QuestionDocument)
    -> Result<(), BatchItemFailure> {
    factory.owns(prepared)?;
    let plan = factory.plan_with_operation(&prepared.source, &prepared.question, &prepared.operation, &mut Continue, true)?;
    factory.compare(&plan, prepared, &mut Continue)
}

#[test]
fn question_manifest_probes_and_reconstruction_use_the_same_private_scope() {
    let byte = factory(false);
    let byte_prepared = prepared(&byte, SOURCE, QUESTION);
    let compact = byte.with_map_document_encoding(policy(), allowance()).unwrap();
    assert!(compact.owns(&byte_prepared).is_err());
    let first = prepared(&compact, SOURCE, QUESTION);
    let second = prepared(&compact, SOURCE, QUESTION);
    let before = used(&first);
    assert!(before.input_bytes() > SOURCE.len() + QUESTION.len());
    assert!(before.output_tokens() < before.input_bytes());
    assert!(first.planned_work().forward_positions < byte_prepared.planned_work().forward_positions);
    assert_eq!(used(&second), before);
    assert!(!Arc::ptr_eq(&first.operation, &second.operation));
    rebuild(&compact, &first).unwrap();
    assert_eq!(used(&first), before.checked_add(before).unwrap());
    assert_eq!(used(&second), before);
    assert_eq!(compact.planner.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
    let wrong = compact.plan_with_operation(SOURCE, "Who met Carol? <|im_start|>",
        &second.operation, &mut Continue, true).unwrap();
    assert_eq!(compact.compare(&wrong, &second, &mut Continue).unwrap_err().fault.code, BatchCode::Admission);
}

#[test]
fn all_four_exact_operation_caps_cover_both_compilations_and_one_short_refuses() {
    let probe = factory(false).with_map_document_encoding(policy(), allowance()).unwrap();
    let original = prepared(&probe, SOURCE, QUESTION);
    let once = used(&original); let twice = once.checked_add(once).unwrap();
    let total = SafeBpeWorkLimits { max_input_bytes: twice.input_bytes() as u64,
        max_output_tokens: twice.output_tokens() as u64, max_pair_checks: twice.pair_checks(),
        max_pair_bytes: twice.pair_bytes() };
    let exact = factory(false).with_map_document_encoding(policy(), total).unwrap();
    let item = prepared(&exact, SOURCE, QUESTION);
    rebuild(&exact, &item).unwrap();
    assert_eq!(used(&item), twice);
    assert!(rebuild(&exact, &item).is_err());
    for axis in 0..4 {
        let mut short = total;
        match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
            2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
        let limited = factory(false).with_map_document_encoding(policy(), short).unwrap();
        let item = prepared(&limited, SOURCE, QUESTION);
        assert_eq!(used(&item), once);
        assert!(rebuild(&limited, &item).is_err(), "axis {axis}");
        assert!(rebuild(&limited, &item).is_err(), "failure must not reset axis {axis}");
        assert_eq!(used(&prepared(&limited, SOURCE, QUESTION)), once);
    }
}

#[test]
fn compact_questions_keep_complete_policy_ownership_and_typed_cancellation() {
    assert!(factory(true).with_map_document_encoding(policy(), allowance()).is_err());
    assert!(hierarchy_factory(4).with_map_document_encoding(policy(), allowance()).is_err());
    let compact = factory(false).with_map_document_encoding(policy(), allowance()).unwrap();
    let mut item = prepared(&compact, SOURCE, QUESTION);
    let mut changed = allowance(); changed.max_output_tokens -= 1;
    let foreign = factory(false).with_map_document_encoding(policy(), changed).unwrap();
    item.operation = Arc::clone(&foreign.planner);
    assert_eq!(compact.owns(&item).unwrap_err().fault.code, BatchCode::Admission);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let failure = compact.prepare_with_control(SOURCE.to_owned(),
            QuestionDocumentArgs { question: QUESTION.to_owned() }, &mut Stop(cause)).err().unwrap();
        assert_eq!(failure.fault, BatchFault::cancelled(cause));
    }
    let old = prepared(&compact, SOURCE, QUESTION);
    let cohorts = compact.with_chunk_cohorts().unwrap();
    assert!(cohorts.owns(&old).is_err());
    let current = prepared(&cohorts, SOURCE, QUESTION);
    rebuild(&cohorts, &current).unwrap();
}
