//! Pinned planning and budget contracts only; no native/model-quality evidence.
use super::*;
use crate::{native_engine::strict_int8::STRICT_INT8_EXECUTION, tokenizer::pinned_controls};

struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop).then_some(DecodeCancellationKind::PollQuota)
    }
}
fn continuing() -> Control { Control { calls: 0, stop: usize::MAX } }
fn planner() -> SourceTaskPlanner {
    let registry = pinned_controls::pinned().unwrap(); let c = registry.template_controls();
    let eos = c.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    SourceTaskPlanner::pinned(c, eos).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"summary-synthesis-pinned-planning-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: SUMMARIZE_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn mapping() -> Int8SourceMapLimits {
    Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work::for_sequence(0, 32_768, 32_768 * 166_144).unwrap(),
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64_000,
    }
}
fn request() -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 64, max_evidence_bytes: 4096,
            verification: GroundingBudget::default() } }
}
fn prepare<'s>(p: &SourceTaskPlanner, text: &'s str, request: SourceSummarySynthesis,
    mapping: Int8SourceMapLimits, control: &mut Control) -> Result<PreparedInt8SummarySynthesis<'s>, Int8SourceMapError> {
    let id = identity(p);
    p.plan_int8_summary_synthesis_with_control(text, request, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), mapping, control)
}
#[test]
fn all_actual_unicode_chunks_and_synthesis_are_reserved_before_execution() {
    let p = planner(); let text = "éAéAéAéA";
    let prepared = prepare(&p, text, request(), mapping(), &mut continuing()).unwrap();
    let expected = prepared.preflight_metadata();
    assert_eq!(expected.chunk_count(), 2);
    assert_eq!(expected.source_span().byte_end, text.len());
    assert_eq!(expected.source_span().scalar_end, text.chars().count());
    assert_eq!(expected.reserved_model_work().unwrap(), add_work(prepared.map.work, expected.synthesis_reserve).unwrap());
    assert_eq!(expected.reserved_mask_visits().unwrap(), 3000);
    assert_eq!(prepared.execution_identities().len(), 2);
    for id in prepared.execution_identities() { assert_eq!(id.task_spec, SUMMARIZE_TASK_VERSION); }
}
#[test]
fn every_whole_invocation_axis_includes_final_synthesis_even_at_exact_boundary() {
    let p = planner(); let source = "éAéAéAéA";
    let expected = prepare(&p, source, request(), mapping(), &mut continuing()).unwrap().preflight_metadata();
    let mut exact = mapping(); exact.max_model_work = expected.reserved_model_work().unwrap();
    exact.max_mask_visits = expected.reserved_mask_visits().unwrap();
    prepare(&p, source, request(), exact, &mut continuing()).unwrap();
    for axis in 0..6 {
        let mut limited = exact;
        match axis { 0 => limited.max_model_work.forward_positions -= 1,
            1 => limited.max_model_work.projected_logits -= 1, 2 => limited.max_model_work.attention_pairs -= 1,
            3 => limited.max_model_work.projections.dot_products -= 1,
            4 => limited.max_model_work.projections.multiply_accumulates -= 1, _ => limited.max_mask_visits -= 1 }
        assert!(prepare(&p, source, request(), limited, &mut continuing()).is_err(), "{axis}");
    }
}
#[test]
fn different_final_options_cannot_change_map_prompts_or_evade_synthesis_bounds() {
    let p = planner(); let a = prepare(&p, "source", request(), mapping(), &mut continuing()).unwrap();
    let mut req = request(); req.synthesis_options.max_bullets = 1;
    let b = prepare(&p, "source", req, mapping(), &mut continuing()).unwrap();
    assert!(a.execution_identities().eq(b.execution_identities()));
    assert_ne!(a.expected.options, b.expected.options);
    assert_eq!(a.expected.reserved_model_work().unwrap(), b.expected.reserved_model_work().unwrap());
    req.synthesis_options.max_citations_per_bullet = 0;
    assert!(prepare(&p, "source", req, mapping(), &mut continuing()).is_err());
}
#[test]
fn empty_source_invalid_verification_and_foreign_profile_are_refused() {
    let p = planner();
    assert!(prepare(&p, "", request(), mapping(), &mut continuing()).is_err());
    for axis in 0..5 {
        let mut r = request();
        match axis { 0 => r.limits.max_evidence_segments = 0, 1 => r.limits.max_evidence_bytes = 0,
            2 => r.limits.verification.max_fields = 0, 3 => r.limits.verification.max_matches = 0,
            _ => r.limits.verification.max_scan_steps = 0 }
        assert!(prepare(&p, "source", r, mapping(), &mut continuing()).is_err());
    }
    let mut id = identity(&p); id.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(p.plan_int8_summary_synthesis_with_control("source", request(), budget(),
        &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(), mapping(), &mut continuing()).is_err());
}
#[test]
fn cancellation_survives_every_pinned_preparation_checkpoint_including_the_last() {
    let p = planner(); let mut control = continuing();
    prepare(&p, "éAéA", request(), mapping(), &mut control).unwrap();
    let last = control.calls; assert!(last > 2);
    for stop in [1, last / 2, last] {
        let mut control = Control { calls: 0, stop };
        let error = prepare(&p, "éAéA", request(), mapping(), &mut control).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota));
        assert_eq!(control.calls, stop);
    }
}
#[test]
fn maximum_context_reserve_is_subtracted_without_saturating_any_axis() {
    let mut m = mapping(); m.max_model_work = Int8Work::default();
    assert!(reserve(budget(), SourcePlanningLimits::default(), m).is_err());
    let mut m = mapping(); m.max_mask_visits = m.mask_visits_per_chunk - 1;
    assert!(reserve(budget(), SourcePlanningLimits::default(), m).is_err());
}
