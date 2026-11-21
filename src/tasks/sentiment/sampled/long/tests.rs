//! UNRUN model-free preparation, address and private-seal boundary fixtures.
//! Synthetic completions only exercise refusal; they are not native receipts.
use super::*;
use std::sync::OnceLock;
use crate::{grammar::mask::MaskWorkLimits, native_engine::constrained::JsonWorkBudget,
    tokenizer::untrusted::SafeBpeLimits};

pub(super) fn options() -> SampledSentimentOptions {
    SampledSentimentOptions { effective_seed: [37; 32], samples: 2, temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 128, max_justification_chars: 64 }
}
pub(super) fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 1024, max_output_bytes: MIB,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
pub(super) fn limits() -> SampledSentimentMapLimits {
    let mut work = Int4Work::default();
    work.forward_positions = u64::MAX; work.projected_logits = u64::MAX; work.attention_pairs = u64::MAX;
    work.projections.dot_products = u64::MAX; work.projections.multiply_accumulates = u64::MAX;
    SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 1024, max_chunk_bytes: 8,
        max_chunk_tokens: 4096, context_tokens: 4096, reserved_tokens: 127,
        max_chunks: 64, max_tokenizer_calls: 1024 }, max_model_work: work,
        max_retained_result_bytes: 4 * MIB, max_result_bytes: 8 * MIB }
}
fn planner() -> &'static SampledSentimentPlanner {
    static PLANNER: OnceLock<SampledSentimentPlanner> = OnceLock::new();
    PLANNER.get_or_init(|| {
        let compiler = Compiler::pinned(&options(), budget(), SampledSentimentLimits::default()).unwrap();
        let identity = unbound_identity(&compiler);
        SampledSentimentPlanner { compiler, identity, group_width: 32,
            options: options(), ceiling: budget(), limits: SampledSentimentLimits::default() }
    })
}
fn request(text: &str) -> SentimentRequest {
    SentimentRequest { document: text.to_owned(), axes: SentimentAxis::ALL.to_vec(), budget: budget() }
}
fn prepare(text: &str) -> PreparedSampledSentimentMap {
    planner().plan_document_with_control(&request(text), limits(), &mut Continue).unwrap()
}

#[test]
fn complete_partition_and_all_chunk_sample_work_are_prepared_before_native_execution() {
    let source = "é\r\n好 test é\r\n好 test";
    let map = prepare(source);
    assert!(map.chunk_count() > 1);
    assert_eq!(map.sample_count(), map.chunk_count() * 4 * options().samples);
    assert_eq!(map.max_output_tokens(), map.sample_count() as u64 * options().max_sample_tokens as u64);
    let mut work = Int4Work::default(); let mut end = 0; let mut scalars = 0;
    for chunk in &map.chunks {
        assert_eq!(chunk.span.byte_start, end); assert_eq!(chunk.span.scalar_start, scalars);
        let text = &source[chunk.span.byte_start..chunk.span.byte_end];
        assert_eq!(chunk.tokens, text.len());
        assert_eq!(chunk.span.scalar_end - chunk.span.scalar_start, text.chars().count());
        assert!(!(text.ends_with('\r') && source[chunk.span.byte_end..].starts_with('\n')));
        assert!(chunk.plan.required_context() <= map.expected.chunk_limits.context_tokens);
        work = work.checked_add(chunk.plan.planned_work()).unwrap();
        end = chunk.span.byte_end; scalars = chunk.span.scalar_end;
    }
    assert_eq!(end, source.len()); assert_eq!(scalars, source.chars().count());
    assert_eq!(map.planned_work(), work);
    assert_eq!(map.expected.source_span, VerifiedSourceSpan { byte_start: 0, byte_end: source.len(),
        scalar_start: 0, scalar_end: source.chars().count() });
}

#[test]
fn repeated_equal_chunks_get_distinct_addresses_and_axis_order_is_canonical() {
    let source = "abcdefghabcdefgh";
    let original = prepare(source);
    assert_eq!(original.chunk_count(), 2);
    let first = &original.chunks[0].plan; let second = &original.chunks[1].plan;
    assert_eq!(first.identity.prompt_digest, second.identity.prompt_digest);
    assert_ne!(first.identity.decision_policy_digest, second.identity.decision_policy_digest);
    for (a, b) in first.axes.iter().zip(&second.axes) {
        assert_eq!(a.prompt, b.prompt);
        assert_ne!(a.identity.decision_policy_digest, b.identity.decision_policy_digest);
    }
    let mut reverse = request(source); reverse.axes.reverse();
    let reverse = planner().plan_document_with_control(&reverse, limits(), &mut Continue).unwrap();
    assert_eq!(reverse.expected.request_binding, original.expected.request_binding);
    assert_eq!(reverse.planned_work(), original.planned_work());
    let changed = prepare("abcdefghabcdefgi");
    assert_ne!(changed.chunks[0].plan.identity.decision_policy_digest,
        original.chunks[0].plan.identity.decision_policy_digest);
    let single = planner().plan(&request("abcdefgh")).unwrap();
    assert_ne!(single.identity.decision_policy_digest, first.identity.decision_policy_digest);
}

#[test]
fn stable_population_item_ids_bind_the_complete_document_and_every_original_chunk() {
    let source = request("abcdefghabcdefgh");
    let one_shot = prepare(&source.document);
    let first = planner().plan_document_for_item_with_control(&source, limits(), "item-é", &mut Continue).unwrap();
    let second = planner().plan_document_for_item_with_control(&source, limits(), "item-好", &mut Continue).unwrap();
    let replay = planner().plan_document_for_item_with_control(&source, limits(), "item-é", &mut Continue).unwrap();
    assert_eq!(first.expected.request_binding, replay.expected.request_binding);
    assert_ne!(first.expected.request_binding, second.expected.request_binding);
    assert_ne!(first.expected.request_binding, one_shot.expected.request_binding);
    assert_eq!(first.planned_work(), one_shot.planned_work());
    assert_eq!(first.preparation_requirement_bytes(), one_shot.preparation_requirement_bytes());
    for ((a, b), cold) in first.chunks.iter().zip(&second.chunks).zip(&one_shot.chunks) {
        assert_eq!(a.span, b.span); assert_eq!(a.tokens, b.tokens);
        assert_ne!(a.plan.identity.decision_policy_digest, b.plan.identity.decision_policy_digest);
        assert_ne!(a.plan.identity.decision_policy_digest, cold.plan.identity.decision_policy_digest);
        for ((left, right), original) in a.plan.axes.iter().zip(&b.plan.axes).zip(&cold.plan.axes) {
            assert_eq!(left.prompt, right.prompt); assert_eq!(left.prompt, original.prompt);
            assert_ne!(left.identity.decision_policy_digest, right.identity.decision_policy_digest);
            assert_ne!(left.identity.decision_policy_digest, original.identity.decision_policy_digest);
        }
    }
    // The final source chunk belongs to the same complete-document address,
    // so changing it also changes the unchanged first chunk's sample stream.
    let changed = planner().plan_document_for_item_with_control(&request("abcdefghabcdefgi"),
        limits(), "item-é", &mut Continue).unwrap();
    assert_eq!(first.chunks[0].plan.axes[0].prompt, changed.chunks[0].plan.axes[0].prompt);
    assert_ne!(first.chunks[0].plan.axes[0].identity.decision_policy_digest,
        changed.chunks[0].plan.axes[0].identity.decision_policy_digest);
    let mut reordered_axes = source; reordered_axes.axes.reverse();
    let reordered = planner().plan_document_for_item_with_control(&reordered_axes,
        limits(), "item-é", &mut Continue).unwrap();
    assert_eq!(first.expected.request_binding, reordered.expected.request_binding);
}

#[test]
fn population_item_ids_refuse_before_source_encoding_and_keep_one_shot_identity_unchanged() {
    let source = request("abcdefghabcdefgh");
    let baseline = prepare(&source.document);
    for item in [String::new(), "a\n".to_owned(), "a\0b".to_owned(), "x".repeat(257)] {
        let mut work = SafeBpeWork::default();
        assert!(matches!(planner().plan_document_for_item_with_encoding_work(&source, limits(),
            &item, &mut Continue, &mut work), Err(SampledSentimentMapError::Limits)));
        assert_eq!(work, SafeBpeWork::default());
    }
    planner().plan_document_for_item_with_control(&source, limits(), &"x".repeat(256), &mut Continue).unwrap();
    let after = prepare(&source.document);
    assert_eq!(baseline.expected.request_binding, after.expected.request_binding);
    assert_eq!(baseline.chunks[0].plan.identity, after.chunks[0].plan.identity);
}

#[test]
fn every_whole_document_work_axis_refuses_one_unit_short_without_truncation() {
    let map = prepare("abcdefghabcdefgh");
    let source = request("abcdefghabcdefgh");
    for axis in 0..5 {
        let mut short = limits(); short.max_model_work = map.planned_work();
        match axis { 0 => short.max_model_work.forward_positions -= 1,
            1 => short.max_model_work.projected_logits -= 1,
            2 => short.max_model_work.attention_pairs -= 1,
            3 => short.max_model_work.projections.dot_products -= 1,
            _ => short.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(matches!(planner().plan_document_with_control(&source, short, &mut Continue),
            Err(SampledSentimentMapError::Work)));
    }
    let mut short = limits(); short.chunks.max_chunks = 1;
    assert!(matches!(planner().plan_document_with_control(&source, short, &mut Continue),
        Err(SampledSentimentMapError::Chunk(MapReduceError::ChunkBudget))));
    let mut short = limits(); short.chunks.max_input_bytes = source.document.len() - 1;
    assert!(matches!(planner().plan_document_with_control(&source, short, &mut Continue),
        Err(SampledSentimentMapError::Chunk(MapReduceError::InputBudget))));
    let mut short = limits(); short.chunks.context_tokens = 128;
    assert!(planner().plan_document_with_control(&source, short, &mut Continue).is_err());
}

#[test]
fn mask_admission_sums_every_chunk_before_any_forward_and_never_renews_per_chunk() {
    let map = prepare("abcdefghabcdefgh");
    let vocabulary = SampledSentimentVocabulary::pinned().unwrap();
    let work = map.planned_work();
    let mut native = Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: budget().max_kv_bytes, max_total_mask_node_visits: u64::MAX,
        mask_limits: MaskWorkLimits::default() } };
    let sampler = json_sampling_workspace_bytes().unwrap();
    let expected = map.chunks.iter().map(|chunk| chunk.plan.complete_mask_charge(&vocabulary, native, sampler).unwrap()).sum();
    assert_eq!(map.complete_mask_charge(&vocabulary, native, sampler).unwrap(), expected);
    native.json.max_total_mask_node_visits = expected;
    assert_eq!(map.complete_mask_charge(&vocabulary, native, sampler).unwrap(), expected);
    native.json.max_total_mask_node_visits -= 1;
    assert!(map.complete_mask_charge(&vocabulary, native, sampler).is_err());
}

#[test]
fn compact_probes_and_chunk_preparation_share_one_cumulative_work_allowance() {
    let options = options(); let limits = limits();
    let compiler = Compiler::pinned(&options, budget(), SampledSentimentLimits::default()).unwrap();
    let identity = unbound_identity(&compiler);
    let compact = SampledSentimentPlanner { compiler, identity, group_width: 32, options,
        ceiling: budget(), limits: SampledSentimentLimits::default() }.with_input_encoding(
            UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
                max_pair_checks: 1_000_000, max_pair_bytes: 64 * MIB })).unwrap();
    let source = request("testing testing testing testing");
    let mut ledger = SafeBpeWork::default();
    let map = compact.plan_document_with_encoding_work(&source, limits, &mut Continue, &mut ledger).unwrap();
    assert_eq!(map.expected.encoding_work, ledger);
    assert!(ledger.input_bytes() >= source.document.len() * 2);
    assert!(map.expected.encoding_record().is_some());
    let first = ledger;
    let second = compact.plan_document_with_encoding_work(&source, limits, &mut Continue, &mut ledger).unwrap();
    assert_eq!(ledger.checked_sub(first).unwrap(), second.expected.encoding_work);
    assert_eq!(map.expected.request_binding, second.expected.request_binding);
    let mut depleted = ledger;
    while depleted.input_bytes() < 4096 { depleted = depleted.checked_add(depleted).unwrap(); }
    let spent = depleted;
    assert!(compact.plan_document_with_encoding_work(&source, limits, &mut Continue, &mut depleted).is_err());
    assert_eq!(depleted, spent);
}

#[test]
fn all_cancellation_causes_survive_planning_and_streamed_private_hashing() {
    struct Stop { cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.cause) }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let failed = planner().plan_document_with_control(&request("abcdefgh"), limits(), &mut Stop { cause });
        assert_eq!(failed.err().unwrap().cancellation(), Some(cause));
        assert_eq!(binding::chunk_address([1; 32], 0, VerifiedSourceSpan { byte_start: 0,
            byte_end: 8, scalar_start: 0, scalar_end: 8 }, 8, &mut Stop { cause }).unwrap_err().cancellation(), Some(cause));
    }
}

// Synthetic complete data exists only inside tests to exercise private seals.
fn fixture_run(map: &PreparedSampledSentimentMap) -> SampledSentimentMapRun {
    let mut chunks = Vec::new(); let mut model_work = Int4Work::default();
    let mut tokens = 0; let mut masks = 0; let mut retained = 0;
    for chunk in &map.chunks {
        let mut dimensions = Vec::new(); let mut chunk_work = Int4Work::default();
        for axis in &chunk.plan.axes {
            let mut samples = Vec::new();
            for sample_index in 0..options().samples {
                let work = chunk.plan.sample_work(axis, sample_index, 1, false).unwrap();
                chunk_work = chunk_work.checked_add(work).unwrap();
                samples.push(SampledSentimentSample { sample_index, score: 10,
                    justification: "Synthetic boundary fixture".to_owned(), generated_tokens: 1,
                    model_work: work, mask_node_visit_charge: 1 });
            }
            dimensions.push(aggregate(axis.axis, samples).unwrap());
        }
        let count = chunk.plan.sample_count() as u64;
        let run = SampledSentimentRun { schema_version: 1, execution: SAMPLED_SENTIMENT_EXECUTION.to_owned(),
            task_spec: "sentiment-v1".to_owned(), prompt_template_sha256: chunk.plan.identity.template_digest,
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: map.group_width(),
            aggregation: SAMPLED_SENTIMENT_AGGREGATION_VERSION.to_owned(),
            interpretation: "generated_score_variation_not_calibrated_confidence".to_owned(),
            justification_status: "unverified_model_explanations_from_untrusted_document".to_owned(),
            sampler_version: JSON_SAMPLER_VERSION.to_owned(), samples_per_axis: options().samples,
            dimensions, model_work: chunk_work, generated_tokens: count, mask_node_visit_charge: count };
        let result = SampledSentimentMapChunk { chunk_id: chunk.id, source_span: chunk.span, source_tokens: chunk.tokens, run };
        retained += binding::value_bytes(&result, map.limits.max_retained_result_bytes, &mut Continue).unwrap();
        chunks.push(result); model_work = model_work.checked_add(chunk_work).unwrap(); tokens += count; masks += count;
    }
    let aggregate = reduction::aggregate_with_control(&chunks, map.expected.source_span, options().samples, &mut Continue).unwrap();
    let mut run = SampledSentimentMapRun { schema_version: 1, execution: SAMPLED_SENTIMENT_MAP_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: map.group_width(), chunk_profile: CHUNK_PROFILE.to_owned(),
        semantics: SEMANTICS.to_owned(), single_context_equivalence: false, source_span: map.expected.source_span,
        chunk_limits: map.expected.chunk_limits, tokenizer_checks: map.expected.tokenizer_checks, samples_per_axis: options().samples,
        planned_model_work: map.planned_work(), model_work, planned_mask_node_visit_charge: masks,
        mask_node_visit_charge: masks, generated_tokens: tokens, retained_result_bytes: retained, chunks, aggregate,
        input_encoding: map.expected.encoding_record(), request_binding: map.expected.request_binding, result_binding: [0; 32] };
    run.result_binding = binding::result(&run, map.max_result_bytes(), &mut Continue).unwrap(); run
}

#[test]
fn private_result_seals_bind_source_geometry_aggregate_work_and_execution_mode() {
    let map = prepare("abcdefghabcdefgh"); let run = fixture_run(&map);
    map.expected.verify_completed_with_control(&run, false, &mut Continue).unwrap();
    assert!(map.expected.verify_completed_with_control(&run, true, &mut Continue).is_err());
    let other = prepare("abcdefghabcdefgi");
    assert!(other.expected.verify_completed_with_control(&run, false, &mut Continue).is_err());
    for case in 0..6 {
        let mut altered = run.clone();
        match case { 0 => altered.chunks[0].source_span.scalar_end += 1,
            1 => altered.chunks[1].source_tokens += 1,
            2 => altered.aggregate.dimensions[0].mean += 1.0,
            3 => altered.model_work.forward_positions += 1,
            4 => altered.mask_node_visit_charge += 1,
            _ => altered.chunks[0].run.dimensions[0].samples[0].justification.push('!') }
        assert!(map.expected.verify_completed_with_control(&altered, false, &mut Continue).is_err());
    }
    let json = serde_json::to_value(&run).unwrap();
    for field in ["request_binding", "result_binding", "effective_seed", "seed", "document"] {
        assert!(json.get(field).is_none());
    }
}
