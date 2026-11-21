//! Planning and private composition regressions. The tiny projection fixture
//! supplies a sealed model identity only; it does not run native inference or
//! establish model quality, semantic accuracy, calibration or parity.
use super::*;
use std::sync::OnceLock;
use crate::{
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding,
        grouped_int4::{Int4ExtractionBatchPlanner, long::{Int4ExtractionMapLimits,
            Int4ExtractionMappingLimits, object::Int4ObjectExtractionPlanner}}},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
    native_engine::{portable_int8::ProjectionWork,
        strict_int4::{Int4ModelIdentity, weights::projection_fixture}},
    tasks::{extract::{object_reconcile::RootObjectLimits,
            semantic::{ClaimPathStep, ClaimRule, ClaimValueKind}},
        judge::FaithfulnessAggregationPolicy, mapreduce::{ChunkLimits, ExecutionLimits}},
    tokenizer::untrusted::SafeBpeLimits,
    validation::grounded_fields::GroundingBudget,
};
use crate::batch::extract::grouped_int4::long::object::tests::semantic_composition_fixture as object_fixture;
use crate::tasks::judge::grouped_int4::faithfulness_map::tests::semantic_composition_fixture as map_fixture;

const WIDTH: usize = 32;
const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"properties":{"amount":{"type":"number"},"name":{"type":"string","maxLength":16}},"required":["amount","name"]}"#;
const SOURCE: &str = "The amount is 7.\r\nThe name is Alice. Café 上海 😀 <think> stays literal.";

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
}

fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64,
        max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn maximum() -> Int4Work {
    Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
        attention_pairs: u64::MAX,
        projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } }
}
fn model() -> &'static Int4ModelIdentity {
    static MODEL: OnceLock<Int4ModelIdentity> = OnceLock::new();
    MODEL.get_or_init(|| projection_fixture(WIDTH).0.identity().clone())
}
fn compact_limits() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 26, max_pair_bytes: 1 << 29 }
}
fn planner(encoding: UntrustedEncoding) -> Int4ObjectExtractionPlanner {
    let inner = Int4ExtractionBatchPlanner::for_model(model(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap()
        .with_document_encoding(encoding).unwrap();
    Int4ObjectExtractionPlanner::new(inner).unwrap()
}
fn judge(encoding: UntrustedEncoding) -> Arc<Int4JudgePlanner> {
    Arc::new(Int4JudgePlanner::for_model(model(), budget(), JudgeLimits::default()).unwrap()
        .with_document_encoding(encoding).unwrap())
}
fn request(schema: &str, grounding: ExtractionBatchGrounding) -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: schema.to_owned(), grounding, budget: budget() }
}
fn extraction_limits() -> Int4ObjectExtractionMapLimits {
    Int4ObjectExtractionMapLimits {
        mapping: Int4ExtractionMapLimits {
            mapping: Int4ExtractionMappingLimits {
                chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 16,
                    max_chunk_tokens: 8192, context_tokens: 8192, reserved_tokens: 64,
                    max_chunks: 64, max_tokenizer_calls: 4096 },
                reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 2,
                    max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
                max_model_work: maximum(), mask_limits: MaskWorkLimits::default(),
                mask_visits_per_chunk: 10_000, max_mask_visits: 640_000,
            },
            verification: GroundingBudget::default(), root_array_union: None,
        },
        reconciliation: RootObjectLimits { max_properties: 32, max_origins: 4096,
            max_input_bytes: 4 << 20, max_json_bytes: 1 << 20,
            max_nodes: 65536, max_result_bytes: 2 << 20 },
    }
}
fn judge_mapping() -> Int4FaithfulnessMapLimits {
    Int4FaithfulnessMapLimits {
        chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 16,
            max_chunk_tokens: 8192, context_tokens: 8192, reserved_tokens: 1,
            max_chunks: 64, max_tokenizer_calls: 4096 },
        max_model_work: maximum(), max_retained_result_bytes: 4 << 20,
        max_result_bytes: 8 << 20, aggregation: FaithfulnessAggregationPolicy::default(),
    }
}
fn rule(name: &str, value_kind: ClaimValueKind) -> ClaimRule {
    ClaimRule { id: format!("{name}-v1"), path: vec![ClaimPathStep::Property(name.to_owned())],
        value_kind, prefix: format!("The document reports {name} as "), suffix: ".".to_owned() }
}
fn options() -> Int4SemanticOptions {
    Int4SemanticOptions { specification: SemanticVerificationSpec {
        schema_version: 1, experimental_opt_in: true, revision: "document-review-v1".to_owned(),
        claims: vec![rule("amount", ClaimValueKind::Number), rule("name", ClaimValueKind::String)],
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 600_000,
            minimum_margin_milli: 100, evidence_window_bytes: 16,
            max_evidence_windows: 8, max_evidence_spans: 8 },
        judge_budget: budget(),
    }, limits: SemanticLimits::default(), max_model_work: maximum() }
}
fn prepare<'s>(source: &'s str, schema: &str, encoding: UntrustedEncoding,
    options: Int4SemanticOptions) -> PreparedInt4SemanticObjectDocument<'s> {
    let object = planner(encoding.clone()).plan_document_with_control(source,
        &request(schema, ExtractionBatchGrounding::Structural), extraction_limits(), &mut Continue).unwrap();
    PreparedInt4SemanticObjectDocument::new(object, judge(encoding), options,
        judge_mapping(), &mut Continue).unwrap()
}

#[test]
fn genuine_preparation_keeps_complete_source_original_schema_and_both_task_contracts() {
    let p = planner(UntrustedEncoding::ByteFallback);
    let args = request(SCHEMA, ExtractionBatchGrounding::Structural);
    let object = p.plan_document_with_control(SOURCE, &args, extraction_limits(), &mut Continue).unwrap();
    let expected = object.preflight_metadata();
    assert!(expected.chunk_count() > 1);
    let admitted = object.execution_identities().cloned().collect::<Vec<_>>();
    assert_eq!(admitted.len(), expected.chunk_count());
    assert!(admitted.iter().all(|identity| identity.schema_digest != Sha256Digest::of_bytes(SCHEMA.as_bytes())));
    let review = PreparedInt4SemanticObjectDocument::new(object, judge(UntrustedEncoding::ByteFallback),
        options(), judge_mapping(), &mut Continue).unwrap();
    assert_eq!(review.extraction_preflight(), expected);
    assert_eq!(review.source_bytes(), SOURCE.len());
    assert_eq!(review.group_width(), WIDTH);
    assert_eq!(review.extraction_budget(), budget());
    assert_eq!(review.judge_budget(), budget());
    assert_eq!(review.judge_identity().task_spec, "judge-v1");
    assert_eq!(review.max_model_work(), maximum());
    assert_eq!(review.max_result_bytes(), options().limits.max_output_bytes);
    assert_eq!(review.input_encoding(), &UntrustedEncoding::ByteFallback);
    assert!(review.input_encoding_work().is_none());
    assert_eq!(review.encoding_workspace_bytes().unwrap(), 0);
}

#[test]
fn whole_source_preparation_is_not_limited_to_one_faithfulness_evidence_partition() {
    let source = "Alice reports 7.\n".repeat(24);
    let selected = options();
    assert!(crate::tasks::judge::partition_evidence(&source,
        selected.specification.judge_policy).is_err());
    let prepared = prepare(&source, SCHEMA, UntrustedEncoding::ByteFallback, selected);
    assert_eq!(prepared.source_bytes(), source.len());
    assert!(prepared.extraction_preflight().chunk_count() > 1);
    assert!(prepared.extraction_preflight().chunk_count() <= extraction_limits().mapping.mapping.chunks.max_chunks);
}

#[test]
fn original_schema_and_equal_length_source_changes_keep_distinct_object_authority() {
    let original = prepare(SOURCE, SCHEMA, UntrustedEncoding::ByteFallback, options());
    let changed_source = SOURCE.replace("Alice", "Elise");
    assert_eq!(changed_source.len(), SOURCE.len());
    let source = prepare(&changed_source, SCHEMA, UntrustedEncoding::ByteFallback, options());
    assert_ne!(source.extraction_preflight(), original.extraction_preflight());
    // Only the root required set differs. The derived partial declarations
    // can agree, but their independently validated final object contracts do not.
    let changed_schema = SCHEMA.replace(r#""required":["amount","name"]"#, r#""required":["amount"]"#);
    let schema = prepare(SOURCE, &changed_schema, UntrustedEncoding::ByteFallback, options());
    assert_ne!(schema.extraction_preflight(), original.extraction_preflight());
}

#[test]
fn claim_contract_is_checked_against_original_schema_before_any_native_execution() {
    let p = planner(UntrustedEncoding::ByteFallback);
    let judge = judge(UntrustedEncoding::ByteFallback);
    for mutation in 0..5 {
        let object = p.plan_document_with_control(SOURCE,
            &request(SCHEMA, ExtractionBatchGrounding::Structural), extraction_limits(), &mut Continue).unwrap();
        let mut options = options();
        match mutation {
            0 => options.specification.experimental_opt_in = false,
            1 => options.specification.claims[0].path = vec![ClaimPathStep::Property("absent".to_owned())],
            2 => options.specification.claims[0].value_kind = ClaimValueKind::Boolean,
            3 => options.specification.claims.push(options.specification.claims[0].clone()),
            _ => options.limits.max_fields = 1,
        }
        assert!(PreparedInt4SemanticObjectDocument::new(object, Arc::clone(&judge), options,
            judge_mapping(), &mut Continue).is_err(), "mutation {mutation}");
    }
    let schema = r#"{"type":"object","additionalProperties":false,"properties":{"quote":{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"}},"required":["quote"]}"#;
    let object = p.plan_document_with_control(SOURCE,
        &request(schema, ExtractionBatchGrounding::SourceMembership), extraction_limits(), &mut Continue).unwrap();
    let mut options = options();
    options.specification.claims = vec![rule("quote", ClaimValueKind::String)];
    assert!(PreparedInt4SemanticObjectDocument::new(object, judge, options,
        judge_mapping(), &mut Continue).is_err());
}

#[test]
fn extraction_reservation_cannot_be_borrowed_from_future_reviews_on_any_native_axis() {
    let p = planner(UntrustedEncoding::ByteFallback);
    let judge = judge(UntrustedEncoding::ByteFallback);
    for axis in 0..5 {
        let object = p.plan_document_with_control(SOURCE,
            &request(SCHEMA, ExtractionBatchGrounding::Structural), extraction_limits(), &mut Continue).unwrap();
        let mut cap = object.preflight_metadata().planned_work();
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        let mut options = options(); options.max_model_work = cap;
        assert!(PreparedInt4SemanticObjectDocument::new(object, Arc::clone(&judge), options,
            judge_mapping(), &mut Continue).is_err(), "axis {axis}");
    }
}

#[test]
fn every_compact_encoding_limit_and_policy_must_match_before_generation() {
    let compact = UntrustedEncoding::SafeBpeV1(compact_limits());
    for (source_policy, review_policy) in [
        (UntrustedEncoding::ByteFallback, compact.clone()),
        (compact.clone(), UntrustedEncoding::ByteFallback),
    ] {
        let object = planner(source_policy).plan_document_with_control(SOURCE,
            &request(SCHEMA, ExtractionBatchGrounding::Structural), extraction_limits(), &mut Continue).unwrap();
        assert!(PreparedInt4SemanticObjectDocument::new(object, judge(review_policy), options(),
            judge_mapping(), &mut Continue).is_err());
    }
    for axis in 0..4 {
        let mut limits = compact_limits();
        match axis { 0 => limits.max_input_bytes -= 1, 1 => limits.max_output_tokens -= 1,
            2 => limits.max_pair_checks -= 1, _ => limits.max_pair_bytes -= 1 }
        let object = planner(compact.clone()).plan_document_with_control(SOURCE,
            &request(SCHEMA, ExtractionBatchGrounding::Structural), extraction_limits(), &mut Continue).unwrap();
        assert!(PreparedInt4SemanticObjectDocument::new(object,
            judge(UntrustedEncoding::SafeBpeV1(limits)), options(), judge_mapping(), &mut Continue).is_err(),
            "axis {axis}");
    }
    let prepared = prepare(SOURCE, SCHEMA, compact.clone(), options());
    let initial = prepared.input_encoding_work().unwrap();
    assert_eq!(prepared.input_encoding(), &compact);
    assert_eq!(prepared.extraction_preflight().encoding_work(), Some(initial));
    assert!(initial.input_bytes() > SOURCE.len());
    assert_eq!(prepared.encoding_workspace_bytes().unwrap(), compact_limits().requirements().unwrap().workspace_bytes);
}

#[test]
fn another_model_partition_is_refused_even_when_no_field_will_be_reviewed() {
    let other = projection_fixture(127).0.identity().clone();
    let other_judge = Arc::new(Int4JudgePlanner::for_model(&other, budget(), JudgeLimits::default()).unwrap());
    let p = planner(UntrustedEncoding::ByteFallback);
    let schema = r#"{"type":"object","additionalProperties":false,"properties":{"nothing":{"type":"null"}},"required":["nothing"]}"#;
    let object = p.plan_document_with_control(SOURCE,
        &request(schema, ExtractionBatchGrounding::Structural), extraction_limits(), &mut Continue).unwrap();
    let mut options = options(); options.specification.claims.clear();
    assert!(PreparedInt4SemanticObjectDocument::new(object, other_judge, options,
        judge_mapping(), &mut Continue).is_err());
}

#[test]
fn preparation_preserves_all_typed_cancellation_causes_without_partial_authority() {
    let p = planner(UntrustedEncoding::ByteFallback);
    let judge = judge(UntrustedEncoding::ByteFallback);
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        let object = p.plan_document_with_control(SOURCE,
            &request(SCHEMA, ExtractionBatchGrounding::Structural), extraction_limits(), &mut Continue).unwrap();
        let result = PreparedInt4SemanticObjectDocument::new(object, Arc::clone(&judge), options(),
            judge_mapping(), &mut Stop(cause));
        assert_eq!(result.err().unwrap().cancellation(), Some(cause));
    }
}

struct Composition<'s> {
    source: &'s str,
    schema: RootObjectSchema,
    extraction: Int4ObjectExtractionMapRun,
    judge: Arc<Int4JudgePlanner>,
    options: Int4SemanticOptions,
    mapping: Int4FaithfulnessMapLimits,
    expected: Int4SemanticObjectDocumentPreflight,
}
fn composition<'s>(source: &'s str, schema: &str, grounding: ExtractionBatchGrounding,
    encoding: UntrustedEncoding, selected: Int4SemanticOptions,
    values: impl FnOnce(usize) -> Vec<String>) -> Composition<'s> {
    let args = request(schema, grounding);
    let object = planner(encoding.clone()).plan_document_with_control(source, &args,
        extraction_limits(), &mut Continue).unwrap();
    let count = object.preflight_metadata().chunk_count();
    let prepared = PreparedInt4SemanticObjectDocument::new(object, judge(encoding), selected,
        judge_mapping(), &mut Continue).unwrap();
    let PreparedInt4SemanticObjectDocument { extraction, source, judge, options,
        judge_mapping: mapping, expected, .. } = prepared;
    let (extraction, schema) = object_fixture(extraction, &args, values(count), &mut Continue).unwrap();
    Composition { source, schema, extraction, judge, options, mapping, expected }
}
fn standard(source: &str, encoding: UntrustedEncoding) -> Composition<'_> {
    composition(source, SCHEMA, ExtractionBatchGrounding::Structural, encoding, options(), |count| {
        assert!(count > 1);
        let mut values = vec!["{}".to_owned(); count];
        values[0] = r#"{"amount":7}"#.to_owned();
        values[count - 1] = r#"{"name":"Alice"}"#.to_owned();
        values
    })
}
fn fields(fixture: &Composition<'_>) -> Vec<RenderedField> {
    prepare_fields(&fixture.schema, &fixture.extraction, &fixture.expected,
        &fixture.options, &mut Continue).unwrap()
}
fn synthetic_results(reviews: &PreparedReviews<'_>, favorite: Option<&str>) -> Vec<Int4FaithfulnessMapRun> {
    reviews.maps.iter().map(|plan|
        map_fixture(plan, &vec![favorite; plan.chunk_count()], &mut Continue).unwrap()).collect()
}
fn complete(fixture: Composition<'_>, favorite: Option<&str>) -> Int4SemanticObjectDocumentRun {
    let fields = fields(&fixture);
    let requests = review_requests(fixture.source, &fields, &fixture.options, &mut Continue).unwrap();
    let reviews = prepare_reviews(&fixture.judge, &requests, &fixture.expected,
        &fixture.options, fixture.mapping, &mut Continue).unwrap();
    let results = synthetic_results(&reviews, favorite);
    finish_combined(fixture.extraction, fields, &reviews, results, fixture.expected,
        &fixture.options, &mut Continue).unwrap()
}

#[test]
fn complete_reconciliation_stays_unchanged_and_every_claim_uses_the_entire_original_source() {
    let fixture = standard(SOURCE, UntrustedEncoding::ByteFallback);
    let expected = fixture.expected.clone();
    let original = canonjson::canonical_bytes(&fixture.extraction).unwrap();
    assert_eq!(fixture.extraction.object.json(), r#"{"amount":7,"name":"Alice"}"#);
    let fields = fields(&fixture);
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].claim.as_deref(), Some("The document reports amount as 7."));
    assert_eq!(fields[1].claim.as_deref(), Some("The document reports name as \"Alice\"."));
    let requests = review_requests(SOURCE, &fields, &fixture.options, &mut Continue).unwrap();
    for request in &requests {
        let JudgeRequest::Faithfulness { source, .. } = request else { panic!("only faithfulness review"); };
        assert_eq!(source, SOURCE);
    }
    let run = complete(fixture, Some("unsupported"));
    expected.verify_completed_with_control(&run, &mut Continue).unwrap();
    assert_eq!(canonjson::canonical_bytes(&run.extraction).unwrap(), original);
    assert_eq!(run.semantic.attempted_fields, 2);
    assert_eq!(run.semantic.checked_fields, 2);
    assert_eq!(run.semantic.not_checked_fields, 0);
    assert!(!run.semantic.single_context_equivalence);
    assert!(run.semantic.same_model_correlated);
    assert_eq!(run.semantic.calibration, CALIBRATION);
    assert!(run.input_encoding.is_none());
    for field in &run.semantic.fields {
        assert_eq!(field.status, SemanticObjectFieldStatus::Reviewed);
        let review = field.review.as_ref().unwrap();
        assert!(review.chunks.len() > 1);
        assert_eq!(review.source_span.byte_end, SOURCE.len());
        assert_eq!(review.source_span.scalar_end, SOURCE.chars().count());
        assert_eq!(review.aggregate.relation, Some(FaithfulnessRelation::Unsupported));
        assert_eq!(review.aggregate.source_bytes, SOURCE.len());
    }
    assert_eq!(run.reserved_model_work,
        run.extraction_reserved_work.checked_add(run.semantic_model_work).unwrap());
    assert_eq!(run.model_work,
        run.extraction.partial_map.model_work.checked_add(run.semantic_model_work).unwrap());
    assert!(run.model_work.forward_positions < run.reserved_model_work.forward_positions);
}

#[test]
fn absent_null_empty_and_verbatim_fields_never_become_supported_claims() {
    let schema = r#"{"type":"object","additionalProperties":false,"properties":{"amount":{"type":"number"},"absent":{"type":"number"},"nothing":{"type":"null"},"quote":{"type":"string","maxLength":16,"x-fnlp-source":"verbatim"},"empty":{"type":"array","maxItems":1,"items":{"type":"number"}}},"required":["amount","nothing","quote","empty"]}"#;
    let mut selected = options();
    selected.specification.claims = vec![rule("amount", ClaimValueKind::Number), rule("absent", ClaimValueKind::Number)];
    let fixture = composition("Alice 7.\nAlice 7.\nAlice 7.", schema,
        ExtractionBatchGrounding::SourceMembership, UntrustedEncoding::ByteFallback, selected, |count| {
            let mut values = vec!["{}".to_owned(); count];
            values[0] = r#"{"amount":7,"nothing":null,"quote":"Alice","empty":[]}"#.to_owned();
            values
        });
    let rendered = fields(&fixture);
    assert_eq!(rendered.len(), 4);
    assert!(!rendered.iter().any(|field| field.pointer == "/absent"));
    for (pointer, reason) in [("/nothing", SemanticNotChecked::NullValue),
        ("/quote", SemanticNotChecked::VerbatimMembershipOnly), ("/empty", SemanticNotChecked::EmptyContainer)] {
        let field = rendered.iter().find(|field| field.pointer == pointer).unwrap();
        assert_eq!(field.not_checked, Some(reason));
        assert!(field.claim.is_none());
        assert!(field.rule_id.is_none());
    }
    let run = complete(fixture, None);
    assert_eq!(run.semantic.attempted_fields, 1);
    assert_eq!(run.semantic.checked_fields, 0);
    assert_eq!(run.semantic.not_checked_fields, 4);
    let amount = run.semantic.fields.iter().find(|field| field.json_pointer == "/amount").unwrap();
    assert_eq!(amount.not_checked, Some(SemanticNotChecked::JudgeAbstained));
    assert!(amount.review.as_ref().unwrap().aggregate.relation.is_none());
}

#[test]
fn all_field_maps_are_prepared_before_any_review_and_share_native_prompt_and_logit_allowances() {
    let fixture = standard(SOURCE, UntrustedEncoding::ByteFallback);
    let requests = review_requests(SOURCE, &fields(&fixture), &fixture.options, &mut Continue).unwrap();
    let reviews = prepare_reviews(&fixture.judge, &requests, &fixture.expected,
        &fixture.options, fixture.mapping, &mut Continue).unwrap();
    assert_eq!(reviews.maps.len(), 2);
    assert_eq!(reviews.prompt_tokens, reviews.maps.iter().map(|map| map.total_prompt_tokens().unwrap()).sum());
    let complete = fixture.expected.extraction.planned_work().checked_add(reviews.work).unwrap();
    let first = fixture.expected.extraction.planned_work().checked_add(reviews.maps[0].planned_work()).unwrap();
    for axis in 0..5 {
        let mut selected = fixture.options.clone(); selected.max_model_work = complete;
        match axis { 0 => selected.max_model_work.forward_positions -= 1,
            1 => selected.max_model_work.projected_logits -= 1,
            2 => selected.max_model_work.attention_pairs -= 1,
            3 => selected.max_model_work.projections.dot_products -= 1,
            _ => selected.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(fits(first, selected.max_model_work));
        assert!(matches!(prepare_reviews(&fixture.judge, &requests, &fixture.expected,
            &selected, fixture.mapping, &mut Continue), Err(Int4SemanticObjectDocumentError::Semantic(
                Int4SemanticError::WorkBudget))), "native axis {axis}");
    }
    for prompt_limit in [true, false] {
        let mut selected = fixture.options.clone();
        if prompt_limit { selected.limits.max_total_prompt_tokens = reviews.prompt_tokens - 1; }
        else { selected.limits.max_total_projected_logits = reviews.work.projected_logits - 1; }
        for map in &reviews.maps {
            assert!(map.total_prompt_tokens().unwrap() <= selected.limits.max_total_prompt_tokens);
            assert!(map.planned_work().projected_logits <= selected.limits.max_total_projected_logits);
        }
        assert!(matches!(prepare_reviews(&fixture.judge, &requests, &fixture.expected,
            &selected, fixture.mapping, &mut Continue), Err(Int4SemanticObjectDocumentError::WorkBudget)));
    }
}

#[test]
fn compact_probes_original_schema_and_all_fields_share_one_nonrenewable_four_axis_ledger() {
    let fixture = standard(SOURCE, UntrustedEncoding::SafeBpeV1(compact_limits()));
    let rendered = fields(&fixture);
    let requests = review_requests(SOURCE, &rendered, &fixture.options, &mut Continue).unwrap();
    let reviews = prepare_reviews(&fixture.judge, &requests, &fixture.expected,
        &fixture.options, fixture.mapping, &mut Continue).unwrap();
    let initial = fixture.expected.initial_encoding_work;
    let results = synthetic_results(&reviews, Some("unsupported"));
    let sum = results.iter().fold(initial, |sum, result|
        sum.checked_add(result.input_encoding.as_ref().unwrap().work).unwrap());
    assert_eq!(reviews.encoding_work, sum);
    assert!(sum.input_bytes() > initial.input_bytes());
    assert!(sum.output_tokens() > initial.output_tokens());
    assert!(sum.pair_checks() > initial.pair_checks());
    assert!(sum.pair_bytes() > initial.pair_bytes());
    for axis in 0..4 {
        let mut limits = compact_limits();
        match axis { 0 => limits.max_input_bytes = sum.input_bytes() - 1,
            1 => limits.max_output_tokens = sum.output_tokens() - 1,
            2 => limits.max_pair_checks = sum.pair_checks() - 1,
            _ => limits.max_pair_bytes = sum.pair_bytes() - 1 }
        let bounded = standard(SOURCE, UntrustedEncoding::SafeBpeV1(limits));
        let requests = review_requests(SOURCE, &fields(&bounded), &bounded.options, &mut Continue).unwrap();
        // Each field alone fits a fresh allowance. The composed path must
        // reject the second field instead of renewing the initial work.
        for request in &requests {
            bounded.judge.plan_faithfulness_document_with_control(request,
                bounded.mapping, &mut Continue).unwrap();
        }
        assert!(prepare_reviews(&bounded.judge, &requests, &bounded.expected,
            &bounded.options, bounded.mapping, &mut Continue).is_err(), "encoding axis {axis}");
    }
    let expected = fixture.expected.clone();
    let run = finish_combined(fixture.extraction, rendered, &reviews, results,
        fixture.expected, &fixture.options, &mut Continue).unwrap();
    expected.verify_completed_with_control(&run, &mut Continue).unwrap();
    assert_eq!(run.input_encoding.unwrap().work, sum);
}

#[test]
fn conflicting_complete_source_evidence_remains_abstention_with_counterevidence() {
    let fixture = standard(SOURCE, UntrustedEncoding::ByteFallback);
    let rendered = fields(&fixture);
    let requests = review_requests(SOURCE, &rendered, &fixture.options, &mut Continue).unwrap();
    let reviews = prepare_reviews(&fixture.judge, &requests, &fixture.expected,
        &fixture.options, fixture.mapping, &mut Continue).unwrap();
    let results = reviews.maps.iter().map(|plan| {
        assert!(plan.chunk_count() > 1);
        let mut favorites = vec![Some("entailed"); plan.chunk_count()];
        *favorites.last_mut().unwrap() = Some("contradicted");
        map_fixture(plan, &favorites, &mut Continue).unwrap()
    }).collect();
    let run = finish_combined(fixture.extraction, rendered, &reviews, results,
        fixture.expected, &fixture.options, &mut Continue).unwrap();
    assert_eq!(run.semantic.checked_fields, 0);
    assert_eq!(run.semantic.attempted_fields, 2);
    for field in &run.semantic.fields {
        let aggregate = &field.review.as_ref().unwrap().aggregate;
        assert_eq!(field.status, SemanticObjectFieldStatus::NotChecked);
        assert_eq!(field.not_checked, Some(SemanticNotChecked::JudgeAbstained));
        assert_eq!(aggregate.relation, None);
        assert_eq!(aggregate.abstention, Some(crate::tasks::judge::FaithfulnessAggregateAbstention::ConflictingEvidence));
        assert!(aggregate.evidence.iter().any(|e| e.relation == FaithfulnessRelation::Entailed));
        assert!(aggregate.evidence.iter().any(|e| e.relation == FaithfulnessRelation::Contradicted));
    }
}

#[test]
fn public_result_work_claims_and_whole_review_substitution_cannot_reuse_completion_authority() {
    let fixture = standard(SOURCE, UntrustedEncoding::ByteFallback);
    let expected = fixture.expected.clone();
    let mut run = complete(fixture, Some("unsupported"));
    let work = run.model_work;
    for axis in 0..5 {
        match axis { 0 => run.model_work.forward_positions += 1, 1 => run.model_work.projected_logits += 1,
            2 => run.model_work.attention_pairs += 1, 3 => run.model_work.projections.dot_products += 1,
            _ => run.model_work.projections.multiply_accumulates += 1 }
        assert!(expected.verify_completed_with_control(&run, &mut Continue).is_err());
        run.model_work = work;
    }
    run.semantic.fields[0].rendered_claim.as_mut().unwrap().push('!');
    assert!(expected.verify_completed_with_control(&run, &mut Continue).is_err());
    run.semantic.fields[0].rendered_claim.as_mut().unwrap().pop();
    run.semantic.fields.swap(0, 1);
    assert!(expected.verify_completed_with_control(&run, &mut Continue).is_err());
    run.semantic.fields.swap(0, 1);
    run.extraction.map_prompt = "changed-object-prompt";
    assert!(expected.verify_completed_with_control(&run, &mut Continue).is_err());
    run.extraction.map_prompt = crate::batch::extract::grouped_int4::long::object::ROOT_OBJECT_MAP_PROMPT;
    expected.verify_completed_with_control(&run, &mut Continue).unwrap();

    let other_source = SOURCE.replace("Alice", "Elise");
    assert_eq!(other_source.len(), SOURCE.len());
    let other = standard(&other_source, UntrustedEncoding::ByteFallback);
    let other_expected = other.expected.clone();
    let mut other = complete(other, Some("unsupported"));
    other_expected.verify_completed_with_control(&other, &mut Continue).unwrap();
    assert!(other_expected.verify_completed_with_control(&run, &mut Continue).is_err());
    assert_eq!(run.semantic.fields[0].review.as_ref().unwrap().source_span,
        other.semantic.fields[0].review.as_ref().unwrap().source_span);
    assert_eq!(run.semantic.fields[0].review.as_ref().unwrap().planned_model_work,
        other.semantic.fields[0].review.as_ref().unwrap().planned_model_work);
    std::mem::swap(&mut run.semantic.fields[0].review, &mut other.semantic.fields[0].review);
    assert!(expected.verify_completed_with_control(&run, &mut Continue).is_err());
    assert!(other_expected.verify_completed_with_control(&other, &mut Continue).is_err());
}

#[test]
fn cancellation_is_terminal_in_claim_preparation_review_planning_and_final_binding() {
    let fixture = standard(SOURCE, UntrustedEncoding::ByteFallback);
    let requests = review_requests(SOURCE, &fields(&fixture), &fixture.options, &mut Continue).unwrap();
    let expected = fixture.expected.clone();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        assert_eq!(prepare_fields(&fixture.schema, &fixture.extraction, &fixture.expected,
            &fixture.options, &mut Stop(cause)).err().unwrap().cancellation(), Some(cause));
        assert_eq!(prepare_reviews(&fixture.judge, &requests, &fixture.expected,
            &fixture.options, fixture.mapping, &mut Stop(cause)).err().unwrap().cancellation(), Some(cause));
    }
    let run = complete(fixture, Some("unsupported"));
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        assert_eq!(expected.verify_completed_with_control(&run, &mut Stop(cause))
            .err().unwrap().cancellation(), Some(cause));
    }
    struct AfterFirst(usize);
    impl DecodeStepControl for AfterFirst {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0 += 1;
            (self.0 > 1).then_some(DecodeCancellationKind::CostBudget)
        }
    }
    let mut control = AfterFirst(0);
    let mut digest = ControlledDigest::new(b"private-test-only", u64::MAX, &mut control);
    let large = "x".repeat(192 * 1024);
    assert!(matches!(digest.value(&large), Err(Int4SemanticObjectDocumentError::Cancelled(
        DecodeCancellationKind::CostBudget))));
}

#[test]
fn reconstructed_population_work_is_retained_in_complete_review_and_completion_authority() {
    let encoding = UntrustedEncoding::SafeBpeV1(compact_limits());
    let first = prepare(SOURCE, SCHEMA, encoding.clone(), options());
    let prior = first.input_encoding_work().unwrap();
    let original_proof = first.preflight_metadata();
    let original_identities = first.execution_identities().cloned().collect::<Vec<_>>();
    drop(first);
    let repeated = prepare(SOURCE, SCHEMA, encoding, options());
    assert!(repeated.preflight_metadata() == original_proof);
    let repeated = repeated.with_prior_preparation_encoding_work(prior).unwrap();
    let combined = prior.checked_add(prior).unwrap();
    assert_eq!(repeated.input_encoding_work(), Some(combined));
    assert_eq!(repeated.execution_identities().cloned().collect::<Vec<_>>(), original_identities);
    let expected = repeated.preflight_metadata();
    assert!(expected != original_proof);
    let PreparedInt4SemanticObjectDocument { extraction, source, judge, options,
        judge_mapping: mapping, expected: authority, .. } = repeated;
    let count = extraction.preflight_metadata().chunk_count();
    let mut values = vec!["{}".to_owned(); count];
    values[0] = r#"{"amount":7}"#.to_owned();
    values[count - 1] = r#"{"name":"Alice"}"#.to_owned();
    let args = request(SCHEMA, ExtractionBatchGrounding::Structural);
    let (extraction, schema) = object_fixture(extraction, &args, values, &mut Continue).unwrap();
    let run = complete(Composition { source, schema, extraction, judge, options, mapping,
        expected: authority }, Some("unsupported"));
    expected.verify_completed_with_control(&run, &mut Continue).unwrap();
    assert!(original_proof.verify_completed_with_control(&run, &mut Continue).is_err());
    let review_work = run.semantic.fields.iter().filter_map(|field| field.review.as_ref())
        .fold(SafeBpeWork::default(), |total, review|
            total.checked_add(review.input_encoding.as_ref().unwrap().work).unwrap());
    assert_eq!(run.input_encoding.as_ref().unwrap().work, combined.checked_add(review_work).unwrap());
}

#[test]
fn byte_reconstruction_retains_original_encoding_absence_and_rejects_fictitious_compact_work() {
    let prepared = prepare(SOURCE, SCHEMA, UntrustedEncoding::ByteFallback, options());
    let original = prepared.preflight_metadata();
    let repeated = prepared.with_prior_preparation_encoding_work(SafeBpeWork::default()).unwrap();
    assert!(repeated.preflight_metadata() == original);
    assert!(repeated.input_encoding_work().is_none());
    let compact = prepare(SOURCE, SCHEMA, UntrustedEncoding::SafeBpeV1(compact_limits()), options());
    let used = compact.input_encoding_work().unwrap();
    let byte = prepare(SOURCE, SCHEMA, UntrustedEncoding::ByteFallback, options());
    assert!(byte.with_prior_preparation_encoding_work(used).is_err());
}
