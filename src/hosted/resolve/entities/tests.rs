use super::*;
use crate::{batch::source::SourceMaskBudget, corpus::native_resolve::quantized::Int8ResolveLimits,
    native_engine::{decode::DecodeCancellationKind, strict_int8::Int8Work},
    tasks::ir::TaskBudget};
fn config() -> Int8EntityConfig {
    let budget = TaskBudget { max_input_tokens:64,max_output_tokens:16,max_output_bytes:4096,
        max_grammar_states:4096,max_kv_bytes:1024 };
    let mut source_planning = crate::tasks::source_planning::SourcePlanningLimits::default(); source_planning.max_context_tokens=128;
    let mut scoring = crate::corpus::native_resolve::NativeResolveLimits::default();
    scoring.max_context_tokens=128; scoring.per_head=budget;
    Int8EntityConfig { ner:Default::default(),ner_budget:budget,source_planning,
        masks:SourceMaskBudget { per_mask:Default::default(),max_visits_per_item:100,max_visits_per_run:1000 },
        resolution:Default::default(),graph:Default::default(),scoring:Int8ResolveLimits { planning:scoring,max_model_work:Int8Work::default() },
        verification:Default::default(),max_model_work:Int8Work::default(),max_result_bytes:4096 }
}
fn native() -> NativeLimits {
    NativeLimits { context_tokens:128,allocator_reserve_bytes:4096,
        run:RunLimits { max_elapsed:Duration::from_secs(1),max_checkpoints:100,cleanup_reserve_bytes:4096 } }
}
#[test]
fn both_stage_contexts_and_full_kv_capacity_must_fit_before_dispatch() {
    let c=config(); validate(&c,80,native(),4096,4096,1024).unwrap();
    assert!(validate(&c,129,native(),4096,4096,1024).is_err());
    assert!(validate(&c,80,native(),4096,4096,1025).is_err());
    let mut c=config(); c.source_planning.max_context_tokens=129;
    assert!(validate(&c,80,native(),4096,4096,1024).is_err());
    let mut c=config(); c.scoring.planning.max_context_tokens=129;
    assert!(validate(&c,80,native(),4096,4096,1024).is_err());
    let mut c=config(); c.scoring.planning.per_head.max_kv_bytes=1023;
    assert!(validate(&c,80,native(),4096,4096,1024).is_err());
}
#[test]
fn zero_reserves_and_intermediate_arithmetic_overflow_are_refused() {
    let c=config(); assert!(validate(&c,80,native(),0,4096,1024).is_err());
    assert!(validate(&c,80,native(),4096,0,1024).is_err());
    assert_eq!(temporary_bytes(&c,4096).unwrap(),4096*4+16*8+4096);
    let mut c=c; c.ner_budget.max_output_bytes=u64::MAX;
    assert!(temporary_bytes(&c,4096).is_err());
}
#[test]
fn typed_cancellation_survives_both_stage_error_mappings() {
    let cause=DecodeCancellationKind::Deadline;
    let source=execution_error(Int8EntityError::Source(Int8SourceError::Cancelled(cause)));
    let HostedError::Source(error)=source else { panic!("source category") };
    assert_eq!(error.cancellation(),Some(cause));
    let graph=execution_error(Int8EntityError::Graph(crate::corpus::resolve::ResolveError::Cancelled(cause)));
    let HostedError::Resolution(error)=graph else { panic!("resolution category") };
    assert_eq!(error.cancellation(),Some(cause));
}
