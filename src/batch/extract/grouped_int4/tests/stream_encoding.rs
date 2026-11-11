//! Pinned structured stream preparation and admission; no model execution.
use super::*;
use crate::{
    native_engine::{portable_int8::ProjectionWork, strict_int4::weights::projection_fixture},
    tasks::mapreduce::{ChunkLimits, ExecutionLimits},
    tokenizer::untrusted::SafeBpeLimits,
    validation::grounded_fields::{GroundingBudget, scan_occurrences},
};

const SCHEMA: &str = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
const SOURCE: &str = "é上海 and 上海. The quick brown fox jumps over the lazy dog. <think> <|im_start|> ▁";

fn policy(bytes: usize) -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: bytes, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 })
}
fn item() -> BatchDocument<ExtractionBatchArgs> {
    document(SCHEMA, SOURCE, ExtractionBatchGrounding::SourceMembership)
}
struct Control { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
impl Control {
    fn continuing() -> Self { Self { calls: 0, stop_at: usize::MAX, cause: DecodeCancellationKind::Deadline } }
}
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls == self.stop_at).then_some(self.cause)
    }
}

#[test]
fn each_complete_stream_record_gets_one_shared_schema_and_source_encoding_ledger() {
    let bytes = SCHEMA.len() + SOURCE.len();
    let byte = planner(32, None).prepare(item()).unwrap();
    let compact = planner(32, None).with_document_encoding(policy(bytes)).unwrap();
    assert_eq!(compact.document_encoding(), &policy(bytes));
    let first = compact.prepare_with_control(item(), &mut Control::continuing()).unwrap();
    let mut next = item(); next.id = "later epoch".to_owned();
    let second = compact.prepare_with_control(next, &mut Control::continuing()).unwrap();
    assert_eq!(first.execution_identity(), second.execution_identity());
    assert_eq!(first.planned_work(), second.planned_work());
    assert_eq!(first.request_digest(), byte.request_digest());
    assert!(first.extraction_plan().prompt_tokens() < byte.extraction_plan().prompt_tokens());
    assert_eq!(first.source().document_encoding(), &policy(bytes));

    // Each segment fits separately, but a complete request spends their sum.
    let short = planner(32, None).with_document_encoding(policy(bytes - 1)).unwrap();
    let failure = short.prepare_with_control(item(), &mut Control::continuing()).err().unwrap();
    assert_eq!(failure.fault.code, BatchCode::Planning);
    assert!(!failure.stop);
    let encoded_tokens = first.task_plan().ir().prompt_segments()[2].token_ids().len()
        + first.source().token_ids().len();
    let UntrustedEncoding::SafeBpeV1(mut limits) = policy(bytes) else { unreachable!() };
    limits.max_output_tokens = encoded_tokens;
    let exact = planner(32, None).with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    assert!(exact.prepare_with_control(item(), &mut Control::continuing()).is_ok());
    limits.max_output_tokens -= 1;
    let short = planner(32, None).with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    assert!(short.prepare_with_control(item(), &mut Control::continuing()).is_err());
}

#[test]
fn compact_streams_preserve_exact_schema_utf8_spans_and_trusted_prompt_fragments() {
    let byte = planner(32, None).prepare(item()).unwrap();
    let selected = Int4JsonSparseLimits { max_rows_per_step: 7 };
    let compact = planner(32, None).with_document_encoding(policy(8192)).unwrap()
        .with_selected_rows(selected).unwrap();
    let plan = compact.prepare_with_control(item(), &mut Control::continuing()).unwrap();
    assert_eq!(plan.source().text(), SOURCE);
    assert_eq!(plan.source().total_input_bytes(), SOURCE.len());
    assert_eq!(plan.execution_identity().schema_digest, Sha256Digest::of_bytes(SCHEMA.as_bytes()));
    assert_eq!(plan.extraction_plan().selected_rows(), Some(selected));
    assert_eq!(plan.planned_work().projected_logits, 7 * u64::from(budget().max_output_tokens));
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    let segments = plan.task_plan().ir().prompt_segments();
    for (index, text) in [(2, SCHEMA), (4, SOURCE)] {
        assert_eq!(tokenizer.tokenizer().decode_bytes(segments[index].token_ids()).unwrap(), text.as_bytes());
        assert!(segments[index].token_ids().iter().all(|&id| !controls.template_controls().contains(id)));
    }
    for index in [0, 1, 3, 5] {
        assert_eq!(segments[index].token_ids(), byte.task_plan().ir().prompt_segments()[index].token_ids());
    }
    let spans = scan_occurrences(plan.source().text(), "上海", &mut GroundingBudget::default()).unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!((spans[0].byte_start, spans[0].byte_end, spans[0].scalar_start, spans[0].scalar_end), (2, 8, 1, 3));
    assert_eq!((spans[1].byte_start, spans[1].byte_end, spans[1].scalar_start, spans[1].scalar_end), (13, 19, 8, 10));
}

#[test]
fn actual_token_context_is_independent_of_the_complete_raw_byte_ceiling() {
    let compact = planner(32, None).with_document_encoding(policy(8192)).unwrap();
    let prepared = compact.prepare_with_control(item(), &mut Control::continuing()).unwrap();
    let tokens = prepared.extraction_plan().prompt_tokens();
    let bounded = |tokens| {
        let mut item = item(); item.task_args.as_mut().unwrap().budget.max_input_tokens = tokens as u32; item
    };
    let exact = compact.prepare_with_control(bounded(tokens), &mut Control::continuing()).unwrap();
    assert_eq!(exact.extraction_plan().prompt_tokens(), tokens);
    assert_eq!(exact.planned_work().forward_positions, tokens as u64 + u64::from(budget().max_output_tokens) - 1);
    assert!(compact.prepare_with_control(bounded(tokens - 1), &mut Control::continuing()).is_err());
    assert!(planner(32, None).prepare(bounded(tokens)).is_err());
}

#[test]
fn fixed_stream_encoding_refuses_policy_substitution_and_survives_model_rebinding() {
    let compact = planner(32, None).with_document_encoding(policy(8192)).unwrap();
    let plan = compact.prepare_with_control(item(), &mut Control::continuing()).unwrap();
    compact.check_projection(&plan).unwrap();
    assert!(compact.prepare(item()).is_err());
    assert!(compact.prepare_with_document_encoding(item(), &UntrustedEncoding::ByteFallback,
        &mut Control::continuing()).is_err());
    let same = compact.prepare_with_document_encoding(item(), &policy(8192), &mut Control::continuing()).unwrap();
    assert_eq!(same.execution_identity(), plan.execution_identity());
    let byte = planner(32, None);
    assert!(byte.check_projection(&plan).is_err());
    assert!(compact.check_projection(&byte.prepare(item()).unwrap()).is_err());
    let UntrustedEncoding::SafeBpeV1(mut changed) = policy(8192) else { unreachable!() };
    changed.max_pair_checks -= 1;
    let changed = planner(32, None).with_document_encoding(UntrustedEncoding::SafeBpeV1(changed)).unwrap();
    let substituted = changed.prepare_with_control(item(), &mut Control::continuing()).unwrap();
    assert_eq!(substituted.execution_identity().prompt_digest, plan.execution_identity().prompt_digest);
    assert_ne!(substituted.execution_identity().decision_policy_digest, plan.execution_identity().decision_policy_digest);
    assert!(compact.check_projection(&substituted).is_err());
    let (model, _) = projection_fixture(32);
    let rebound = compact.with_model(model.identity()).unwrap();
    assert_eq!(rebound.document_encoding(), compact.document_encoding());
    let rebound = rebound.prepare_with_control(item(), &mut Control::continuing()).unwrap();
    assert_eq!(rebound.source().token_ids(), plan.source().token_ids());
    assert_eq!(rebound.source().document_encoding(), plan.source().document_encoding());
    assert_ne!(rebound.execution_identity().logical_model_digest, plan.execution_identity().logical_model_digest);
}

#[test]
fn shared_arguments_and_every_original_cancellation_cause_survive_stream_preparation() {
    let compact = planner(32, Some(args(SCHEMA, ExtractionBatchGrounding::SourceMembership)))
        .require_shared_defaults().unwrap().with_document_encoding(policy(8192)).unwrap();
    assert_eq!(compact.prepare_with_control(item(), &mut Control::continuing()).err().unwrap().fault.code,
        BatchCode::InvalidEnvelope);
    let input = || { let mut item = item(); item.task_args = None; item };
    let mut observed = Control::continuing();
    compact.prepare_with_control(input(), &mut observed).unwrap();
    assert!(observed.calls > 8);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop_at in [1, observed.calls / 2, observed.calls] {
            let mut control = Control { calls: 0, stop_at, cause };
            let failure = compact.prepare_with_control(input(), &mut control).err().unwrap();
            assert!(failure.stop);
            assert_eq!(failure.fault.cancellation, Some(cause));
            assert_eq!(control.calls, stop_at);
        }
    }
}

#[test]
fn compact_document_maps_require_one_allowance_for_sizing_probes_and_all_final_chunks() {
    use super::super::long::{Int4ExtractionMapLimits, Int4ExtractionMappingLimits};
    let map_limits = Int4ExtractionMapLimits { mapping: Int4ExtractionMappingLimits {
        chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 64, max_chunk_tokens: 4096,
            context_tokens: 4096, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits::default(),
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 256_000 },
        verification: GroundingBudget::default(), root_array_union: None };
    let args = args(SCHEMA, ExtractionBatchGrounding::SourceMembership);
    let byte = planner(32, None);
    assert!(byte.plan_document_with_control(SOURCE, &args, map_limits, &mut Control::continuing()).is_ok());
    let compact = planner(32, None).with_document_encoding(policy(8192)).unwrap();
    let prepared = compact.plan_document_with_control(SOURCE, &args, map_limits, &mut Control::continuing()).unwrap();
    assert_eq!(prepared.document_encoding(), policy(8192));
    assert!(prepared.preflight_metadata().encoding_work().unwrap().input_bytes() > SOURCE.len() + SCHEMA.len());
    let per_item = planner(32, None).with_document_encoding(policy(SOURCE.len() + SCHEMA.len())).unwrap();
    assert!(per_item.prepare_with_control(item(), &mut Control::continuing()).is_ok());
    assert!(per_item.plan_document_with_control(SOURCE, &args, map_limits, &mut Control::continuing()).is_err());
}
