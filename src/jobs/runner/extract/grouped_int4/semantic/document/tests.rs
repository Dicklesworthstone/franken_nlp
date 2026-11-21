//! Recipe authentication and durable lifetime admission, without native output.
use super::*;
use crate::{
    batch::extract::ExtractionBatchGrounding,
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork,
        strict_int4::weights::projection_fixture},
    tasks::{ir::TaskBudget, extract::semantic::{ClaimPathStep, ClaimRule, ClaimValueKind,
        SemanticLimits, SemanticVerificationSpec}, judge::FaithfulnessPolicy},
    tokenizer::untrusted::SafeBpeLimits,
};
use std::sync::OnceLock;

fn work(n: u64) -> Int4Work {
    Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }
}
fn model() -> &'static Int4ModelIdentity {
    static MODEL: OnceLock<Int4ModelIdentity> = OnceLock::new();
    MODEL.get_or_init(|| projection_fixture(32).0.identity().clone())
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64,
        max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn config() -> Int4SemanticObjectDocumentJobConfig {
    let chunks = ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 16,
        max_chunk_tokens: 8192, context_tokens: 8192, reserved_tokens: 64,
        max_chunks: 64, max_tokenizer_calls: 4096 };
    let per = GroundingBudget::default();
    Int4SemanticObjectDocumentJobConfig {
        extraction: ExtractionBatchArgs {
            schema: r#"{"type":"object","additionalProperties":false,"properties":{"amount":{"type":"number"}},"required":["amount"]}"#.to_owned(),
            grounding: ExtractionBatchGrounding::Structural, budget: budget() },
        compiler_limits: CompileLimits::default(), source_limits: SourceRuntimeLimits::default(),
        options: Int4SemanticOptions {
            specification: SemanticVerificationSpec { schema_version: 1, experimental_opt_in: true,
                revision: "retained-object-review-v1".to_owned(),
                claims: vec![ClaimRule { id: "amount".to_owned(),
                    path: vec![ClaimPathStep::Property("amount".to_owned())],
                    value_kind: ClaimValueKind::Number, prefix: "The reported amount is ".to_owned(),
                    suffix: ".".to_owned() }],
                judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 600_000,
                    minimum_margin_milli: 100, evidence_window_bytes: 16,
                    max_evidence_windows: 8, max_evidence_spans: 8 }, judge_budget: budget() },
            limits: SemanticLimits { max_output_bytes: 16 << 20, ..SemanticLimits::default() },
            max_model_work: work(1 << 56) },
        judge_planning: JudgeLimits::default(), encoding: UntrustedEncoding::ByteFallback,
        extraction_limits: Int4ObjectExtractionMapLimits {
            mapping: Int4ExtractionMapLimits {
                mapping: Int4ExtractionMappingLimits { chunks,
                    reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 2,
                        max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
                    max_model_work: work(1 << 56), mask_limits: MaskWorkLimits::default(),
                    mask_visits_per_chunk: 10_000, max_mask_visits: 640_000 },
                verification: per, root_array_union: None },
            reconciliation: RootObjectLimits { max_properties: 32, max_origins: 4096,
                max_input_bytes: 4 << 20, max_json_bytes: 1 << 20,
                max_nodes: 65536, max_result_bytes: 2 << 20 } },
        review_limits: Int4FaithfulnessMapLimits { chunks: ChunkLimits {
            max_input_bytes: chunks.max_input_bytes + SemanticLimits::default().max_claim_bytes, ..chunks },
            max_model_work: work(1 << 56), max_retained_result_bytes: 4 << 20,
            max_result_bytes: 8 << 20, aggregation: FaithfulnessAggregationPolicy::default() },
        work: Int4SemanticObjectDocumentBatchLimits { max_model_work: work(1 << 58),
            max_mask_node_visits: 2_560_000,
            max_verification: GroundingBudget { max_fields: per.max_fields * 4,
                max_matches: per.max_matches * 4, max_scan_steps: per.max_scan_steps * 4 } },
    }
}
fn limits(c: &Int4SemanticObjectDocumentJobConfig) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: c.options.limits.max_output_bytes as usize,
        max_spool_bytes: 64 << 20, max_materialized_bytes: 64 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(c.work.max_model_work, c.work.max_mask_node_visits) }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn freeze(p: &Int4SemanticObjectDocumentJobPlanner) -> FrozenManifest {
    let original = r#"{"id":"invoice","text":"The amount is 7. Café 上海."}"#.as_bytes();
    FrozenManifest::freeze(&JobSecret::from_bytes([7; 32]),
        JobContract { job_id: JobId([9; 16]), execution: p.execution_identity(),
            recipe: &p.recipe, limits: limits(p.config()) },
        [JobInput { id: "invoice", original, normalized: original }], &mut Continue).unwrap()
}

#[test]
fn actual_factory_retains_both_compilers_and_original_whole_document_policy() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Int4SemanticObjectDocumentJobPlanner>();
    let p = Int4SemanticObjectDocumentJobPlanner::for_model(model(), config()).unwrap();
    assert_eq!(p.execution_identity().task_spec, "extract-v1");
    assert_eq!(p.judge_identity().task_spec, "judge-v1");
    p.check_job_limits(limits(p.config())).unwrap();
    let retained = p.retain_compiler();
    assert!(Arc::ptr_eq(&retained, &p.compiler));
    let bytes = bounded_json(&p.recipe, SEMANTIC_OBJECT_JOB_RECIPE_BYTES).unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["execution"], INT4_SEMANTIC_OBJECT_DOCUMENT_EXECUTION);
    assert_eq!(wire["schema_derivation"], ROOT_OBJECT_DERIVATION);
    assert_eq!(wire["reconciliation"], ROOT_OBJECT_RECONCILIATION);
    assert_eq!(wire["semantic_scope"], SEMANTIC_OBJECT_DOCUMENT_SCOPE);
    assert_eq!(wire["settings"]["extraction"]["schema"], p.config().extraction.schema);
    drop(p);
    assert_eq!(Arc::strong_count(&retained), 1);
}

#[test]
fn all_native_axes_and_mask_ceiling_are_exact_immutable_job_limits() {
    let c = config();
    let exact = limits(&c);
    check_job_limits(&c, exact).unwrap();
    for axis in 0..6 { for increase in [false, true] {
        let mut changed = exact;
        let field = match axis {
            0 => &mut changed.max_work.model.forward_positions,
            1 => &mut changed.max_work.model.projected_logits,
            2 => &mut changed.max_work.model.attention_pairs,
            3 => &mut changed.max_work.model.projections.dot_products,
            4 => &mut changed.max_work.model.projections.multiply_accumulates,
            _ => &mut changed.max_work.mask_node_visits,
        };
        if increase { *field += 1; } else { *field -= 1; }
        assert_eq!(check_job_limits(&c, changed), Err(BatchCode::InvalidLimits.into()));
    } }
    let mut incomplete = exact;
    incomplete.max_result_bytes = budget().max_output_bytes as usize;
    assert_eq!(check_job_limits(&c, incomplete), Err(BatchCode::InvalidLimits.into()));
}

#[test]
fn every_permitted_retry_is_funded_in_all_lifetime_grounding_axes() {
    let original = config();
    let exact = limits(&original);
    check_job_limits(&original, exact).unwrap();
    for axis in 0..3 {
        let mut c = original.clone();
        match axis {
            0 => c.work.max_verification.max_fields -= 1,
            1 => c.work.max_verification.max_matches -= 1,
            _ => c.work.max_verification.max_scan_steps -= 1,
        }
        assert_eq!(check_job_limits(&c, exact), Err(BatchCode::InvalidLimits.into()));
    }
    let mut extra_retry = exact;
    extra_retry.max_attempts += 1;
    assert_eq!(check_job_limits(&original, extra_retry), Err(BatchCode::InvalidLimits.into()));
    let mut overflow = original;
    overflow.extraction_limits.mapping.verification.max_scan_steps = u64::MAX;
    overflow.work.max_verification.max_scan_steps = u64::MAX;
    assert_eq!(check_job_limits(&overflow, exact), Err(BatchCode::InvalidLimits.into()));
}

#[test]
fn resume_authenticates_source_geometry_reconciliation_reviews_and_work() {
    let mut p = Int4SemanticObjectDocumentJobPlanner::for_model(model(), config()).unwrap();
    let base = freeze(&p);
    let original = p.recipe.config.clone();
    // Mutate only private recipe data to isolate the commitment. These values
    // are never given executable compiler or result authority.
    for axis in 0..32 {
        p.recipe.config = original.clone();
        let c = &mut p.recipe.config;
        match axis {
            0 => c.extraction.schema.push(' '),
            1 => c.compiler_limits.max_transitions -= 1,
            2 => c.source_limits.index.max_build_steps -= 1,
            3 => c.options.specification.claims[0].prefix.push('!'),
            4 => c.options.specification.claims[0].suffix.push('!'),
            5 => c.options.specification.judge_policy.minimum_candidate_weight_ppm += 1,
            6 => c.options.specification.judge_budget.max_output_bytes -= 1,
            7 => c.options.limits.max_output_bytes -= 1,
            8 => c.judge_planning.per_head.max_projected_logits -= 1,
            9 => c.extraction_limits.mapping.mapping.chunks.max_chunk_bytes += 1,
            10 => c.extraction_limits.mapping.mapping.chunks.max_tokenizer_calls -= 1,
            11 => c.extraction_limits.mapping.mapping.reduction.max_task_calls -= 1,
            12 => c.extraction_limits.mapping.mapping.reduction.max_total_value_bytes -= 1,
            13 => c.extraction_limits.mapping.mapping.mask_limits.max_trie_node_visits -= 1,
            14 => c.extraction_limits.mapping.mapping.mask_visits_per_chunk -= 1,
            15 => c.extraction_limits.mapping.mapping.max_mask_visits -= 1,
            16 => c.extraction_limits.mapping.verification.max_scan_steps -= 1,
            17 => c.extraction_limits.reconciliation.max_properties -= 1,
            18 => c.extraction_limits.reconciliation.max_origins -= 1,
            19 => c.extraction_limits.reconciliation.max_nodes -= 1,
            20 => c.extraction_limits.reconciliation.max_json_bytes -= 1,
            21 => c.review_limits.chunks.max_chunk_bytes += 1,
            22 => c.review_limits.chunks.max_tokenizer_calls -= 1,
            23 => c.review_limits.max_model_work.projected_logits -= 1,
            24 => c.review_limits.max_retained_result_bytes -= 1,
            25 => c.review_limits.max_result_bytes -= 1,
            26 => c.review_limits.aggregation.minimum_classified_coverage_ppm -= 1,
            27 => c.work.max_verification.max_fields -= 1,
            28 => c.work.max_verification.max_matches -= 1,
            29 => c.work.max_verification.max_scan_steps -= 1,
            30 => c.options.max_model_work.projections.multiply_accumulates -= 1,
            _ => c.extraction_limits.mapping.mapping.max_model_work.attention_pairs -= 1,
        }
        assert_eq!(base.binding.compare(&freeze(&p).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "setting {axis}");
    }
}

#[test]
fn changed_model_and_compact_input_policy_cannot_reuse_authenticated_results() {
    let original = Int4SemanticObjectDocumentJobPlanner::for_model(model(), config()).unwrap();
    let base = freeze(&original);
    let (weights, _) = projection_fixture(64);
    let other = Int4SemanticObjectDocumentJobPlanner::for_model(weights.identity(), config()).unwrap();
    assert_eq!(base.binding.compare(&freeze(&other).binding),
        Err(JobError::Mismatch(MismatchField::Execution)));
    let mut c = config();
    c.encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 1 << 20,
        max_output_tokens: 1 << 20, max_pair_checks: 1 << 26, max_pair_bytes: 1 << 29 });
    let compact = Int4SemanticObjectDocumentJobPlanner::for_model(model(), c).unwrap();
    assert!(base.binding != freeze(&compact).binding);
}

#[test]
fn original_population_overrides_refuse_before_native_admission() {
    let normal = BatchDocument { id: "invoice".to_owned(), text: "amount 7".to_owned(), task_args: None };
    let accepted = shared_document(normal).unwrap();
    assert_eq!(accepted.text, "amount 7");
    let override_args = BatchDocument { id: "invoice".to_owned(), text: "amount 7".to_owned(), task_args: Some(()) };
    assert_eq!(shared_document(override_args).err().unwrap().fault.code, BatchCode::InvalidEnvelope);
}
