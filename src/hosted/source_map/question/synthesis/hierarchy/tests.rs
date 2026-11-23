//! Shared host/CLI quote and complete transcript ownership, without inference.
use super::*;
use super::super::tests::config;

#[test]
fn hosted_hierarchy_uses_the_shared_complete_workspace_contract() {
    let c = config(); let h = QuestionHierarchyLimits::default();
    assert_eq!(hierarchy_bytes(&c, 3, h).unwrap(), workspace_bytes(c.budget, c.mapping, 3,
        c.reduction_reserve_bytes, c.task.limits, c.task.question.verification, h).unwrap());
    let more = QuestionHierarchyLimits { max_passes: h.max_passes + 1, ..h };
    assert_eq!(hierarchy_bytes(&c, 3, more).unwrap() - hierarchy_bytes(&c, 3, h).unwrap(),
        4 * c.budget.max_output_bytes + 8 * u64::from(c.budget.max_output_tokens) + 8192 + 256);
    assert!(hierarchy_bytes(&c, 0, h).is_err());
    assert!(hierarchy_bytes(&c, c.mapping.chunks.max_chunks + 1, h).is_err());
}
#[test]
fn all_discovery_and_allowed_hierarchy_tokens_remain_owned_through_delivery() {
    let h = QuestionHierarchyLimits { max_passes: 4, ..Default::default() };
    assert_eq!(transcript_tokens(16, 3, h).unwrap(), 7 * 16);
    assert!(transcript_tokens(0, 3, h).is_err());
    assert!(transcript_tokens(16, usize::MAX, h).is_err());
    assert!(transcript_tokens(16, 3, QuestionHierarchyLimits { max_passes: 0, ..h }).is_err());
}
