//! Pure admission/lifetime-coordinate fixtures. These do not execute a model.
use super::*;
use crate::native_engine::kv::KV_BYTES_PER_TOKEN;

fn limits() -> Int4ScoredCohortBatchLimits {
    let memory = Int4MemoryRequirement::for_cohort(&[16; 4], 32).unwrap();
    Int4ScoredCohortBatchLimits { max_documents: 3, max_heads: 4, context_limit: 16,
        native_memory: Int4MemoryBudget { max_kv_bytes: memory.kv_bytes,
            max_rope_bytes: memory.rope_bytes, max_scratch_payload_bytes: memory.scratch_payload_bound },
        max_workspace_bytes: 1 << 24, max_item_output_bytes: 4096, max_output_bytes: 16384 }
}
fn work() -> Int4Work {
    Int4Work { forward_positions: 24, projected_logits: 30, attention_pairs: 50,
        projections: ProjectionWork { dot_products: 60, multiply_accumulates: 70 } }
}
fn facts(identity: &ExecutionIdentity) -> Int4ScoredPlanFacts<'_> {
    Int4ScoredPlanFacts { identity, model_work: work(), group_width: 32, head_count: 3,
        required_context: 7, task_budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 8,
            max_output_bytes: 4096, max_grammar_states: 16, max_kv_bytes: 11 * KV_BYTES_PER_TOKEN as u64 },
        max_result_bytes: 4096 }
}
#[test]
fn ragged_documents_pay_the_sum_of_all_causal_contexts() {
    let identity = super::super::tests::identity(); let mut facts = facts(&identity);
    assert_eq!(price_plan(&facts, &[3, 7, 1], limits()).unwrap(), 11 * KV_BYTES_PER_TOKEN as u64);
    facts.task_budget.max_kv_bytes = 7 * KV_BYTES_PER_TOKEN as u64;
    let error = price_plan(&facts, &[3, 7, 1], limits()).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::Admission); assert!(!error.stop);
}
#[test]
fn one_overwide_serial_document_is_rejected_before_native_context_construction() {
    let identity = super::super::tests::identity(); let mut facts = facts(&identity);
    for heads in [0, 5, 64, 65, 128] {
        facts.head_count = heads;
        let error = check_plan_bounds(&facts, limits()).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::Admission); assert!(!error.stop);
    }
}
#[test]
fn output_and_geometry_refusals_remain_per_document() {
    let identity = super::super::tests::identity(); let mut facts = facts(&identity);
    facts.max_result_bytes = 4097;
    let error = price_plan(&facts, &[3, 7, 1], limits()).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::OutputLineLimit); assert!(!error.stop);
    facts.max_result_bytes = 4096;
    for contexts in [&[3, 7][..], &[3, 6, 1][..], &[0, 7, 1][..], &[3, 17, 1][..]] {
        let error = price_plan(&facts, contexts, limits()).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::Admission); assert!(!error.stop);
    }
}
#[test]
fn all_physical_memory_axes_are_required_before_allocation() {
    limits().validate(32).unwrap();
    for axis in 0..3 {
        let mut limits = limits();
        match axis { 0 => limits.native_memory.max_kv_bytes -= 1,
            1 => limits.native_memory.max_rope_bytes -= 1,
            _ => limits.native_memory.max_scratch_payload_bytes -= 1 }
        assert_eq!(limits.validate(32).unwrap_err().code, BatchCode::InvalidLimits);
    }
}
fn request(sequence: u64, epoch: u64) -> GroupedRequest<()> {
    GroupedRequest { prepared: (), context: BatchRequestContext { request_seq: sequence, epoch,
        input_line: sequence, byte_offset: (sequence - 1) * 20 } }
}
#[test]
fn physical_slots_cannot_replay_a_document_or_cross_an_undrained_epoch() {
    validate_contexts(&[request(1, 1), request(2, 1)], None, 3).unwrap();
    let previous = Some(request(2, 1).context);
    validate_contexts(&[request(4, 2), request(5, 2)], previous, 3).unwrap();
    for invalid in [vec![request(2, 1)], vec![request(4, 1), request(5, 2)],
        vec![request(5, 2), request(4, 2)]] {
        assert_eq!(validate_contexts(&invalid, previous, 3).unwrap_err().code, BatchCode::InvalidExecution);
    }
    let mut wrong_offset = request(4, 2); wrong_offset.context.byte_offset = previous.unwrap().byte_offset;
    assert!(validate_contexts(&[wrong_offset], previous, 3).is_err());
    assert!(validate_contexts::<()>(&[], previous, 3).is_err());
}
#[test]
fn aggregate_counters_distinguish_shared_steps_from_semantic_model_work() {
    let first = add_statistics(Int4ScoredCohortBatchStats::default(), 2, 4, 8, 3, work()).unwrap();
    let total = add_statistics(first, 1, 1, 5, 2, work()).unwrap();
    assert_eq!((total.cohorts, total.documents, total.heads), (2, 3, 5));
    assert_eq!((total.group_steps, total.projection_groups), (13, 5));
    assert_eq!(total.model_work.forward_positions, 48);
    assert_eq!(total.model_work, work().checked_add(work()).unwrap());
    let full = Int4ScoredCohortBatchStats { cohorts: u64::MAX, ..first };
    assert_eq!(add_statistics(full, 1, 1, 1, 1, work()).unwrap_err().code, BatchCode::SequenceOverflow);
}
#[test]
fn native_cancellation_keeps_its_original_cause() {
    use crate::native_engine::decode::DecodeCancellationKind;
    let fault = native_error(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline));
    assert_eq!(fault, BatchFault::cancelled(DecodeCancellationKind::Deadline));
}
