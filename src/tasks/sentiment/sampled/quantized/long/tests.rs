//! Model-free complete-document admission and publication fixtures.
//! Synthetic child completions exercise the real task finalizer and map
//! boundary; they are not neural fidelity, model quality or DSR evidence.
use super::*;
use std::sync::OnceLock;
use crate::{grammar::mask::MaskWorkLimits, native_engine::constrained::JsonDecodeOutput,
    tokenizer::untrusted::SafeBpeLimits};
use super::memory::{Int8SampledSentimentMapMemoryLimits,
    MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES};

const MASK_CHARGE: u64 = 7;
fn options() -> SampledSentimentOptions {
    SampledSentimentOptions { effective_seed: [37; 32], samples: 2, temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 128, max_justification_chars: 64 }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 1024, max_output_bytes: MIB,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn limits() -> Int8SampledSentimentMapLimits {
    let mut work = Int8Work::default();
    work.forward_positions = u64::MAX; work.projected_logits = u64::MAX;
    work.attention_pairs = u64::MAX; work.projections.dot_products = u64::MAX;
    work.projections.multiply_accumulates = u64::MAX;
    Int8SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 1024,
        max_chunk_bytes: 8, max_chunk_tokens: 4096, context_tokens: 4096,
        reserved_tokens: 127, max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: work, max_retained_result_bytes: 4 * MIB, max_result_bytes: 8 * MIB }
}
fn planner() -> &'static Int8SampledSentimentPlanner {
    static PLANNER: OnceLock<Int8SampledSentimentPlanner> = OnceLock::new();
    PLANNER.get_or_init(|| Int8SampledSentimentPlanner::for_model(&facts(), options(), budget(),
        SampledSentimentLimits::default()).unwrap())
}
fn request(text: &str) -> SentimentRequest {
    SentimentRequest { document: text.to_owned(), axes: SentimentAxis::ALL.to_vec(), budget: budget() }
}
fn prepare(text: &str) -> PreparedInt8SampledSentimentMap {
    planner().plan_document_with_control(&request(text), limits(), &mut Continue).unwrap()
}
fn native_budget(map: &PreparedInt8SampledSentimentMap) -> Int8JsonBudget {
    let work = map.planned_work();
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: budget().max_kv_bytes,
        max_total_mask_node_visits: map.max_output_tokens() * MASK_CHARGE,
        mask_limits: MaskWorkLimits { max_trie_node_visits: MASK_CHARGE as usize, checkpoint_interval_nodes: 1 } } }
}
fn sampler_bytes() -> u64 { json_sampling_workspace_bytes().unwrap() }

fn synthetic(plan: &PreparedInt8SampledSentiment, scores: &[i16]) -> Int8SampledSentimentRun {
    assert_eq!(scores.len(), plan.inner.options.samples);
    let mut dimensions = Vec::new(); let mut work = Int8Work::default();
    let (mut tokens, mut masks) = (0_u64, 0_u64);
    for axis in &plan.inner.axes {
        let mut samples = Vec::new();
        for (index, score) in scores.iter().enumerate() {
            let json = format!(r#"{{"score":{},"justification":"Fixture explanation."}}"#, score);
            let mut token_ids = planner().compiler.tokenizer.tokenizer()
                .encode_byte_fallback_only(json.as_bytes()).unwrap();
            token_ids.push(plan.inner.decode.eos_token_id);
            let model_work = constrained_int8::planned_work(axis.prompt.len(), token_ids.len()).unwrap();
            let native = Int8JsonRun { schema_version: 1, execution: INT8_JSON_SAMPLED_EXECUTION.to_owned(),
                model_work, output: JsonDecodeOutput { schema_version: 1,
                    numerics_profile: STRICT_INT8_PROFILE.to_owned(),
                    mask_node_visit_charge: token_ids.len() as u64 * MASK_CHARGE,
                    token_ids, json, forward_positions: model_work.forward_positions,
                    projected_logits: model_work.projected_logits } };
            let sample = plan.finalize_sample(axis, index, native).unwrap();
            work = work.checked_add(sample.model_work).unwrap();
            tokens += sample.generated_tokens as u64; masks += sample.mask_node_visit_charge;
            samples.push(sample);
        }
        dimensions.push(aggregate(axis.axis, samples).unwrap());
    }
    Int8SampledSentimentRun { schema_version: 1, execution: INT8_SAMPLED_SENTIMENT_EXECUTION.to_owned(),
        task_spec: "sentiment-v1".to_owned(), prompt_template_sha256: plan.inner.identity.template_digest,
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), aggregation: SAMPLED_SENTIMENT_AGGREGATION_VERSION.to_owned(),
        interpretation: "generated_score_variation_not_calibrated_confidence".to_owned(),
        justification_status: "unverified_model_explanations_from_untrusted_document".to_owned(),
        sampler_version: JSON_SAMPLER_VERSION.to_owned(), samples_per_axis: plan.inner.options.samples,
        dimensions, model_work: work, generated_tokens: tokens, mask_node_visit_charge: masks }
}
fn execute(map: &PreparedInt8SampledSentimentMap) -> Int8SampledSentimentMapRun {
    map.execute_chunks(MASK_CHARGE, native_budget(map), sampler_bytes(), &mut Continue,
        |plan, _, _| Ok(synthetic(plan, &[-20, 20]))).unwrap()
}

#[test]
fn complete_partition_preserves_original_utf8_crlf_and_every_sample_work_axis() {
    let source = "é\r\n好 test é\r\n好 test";
    let map = prepare(source);
    assert!(map.chunk_count() > 1);
    assert_eq!(map.sample_count(), map.chunk_count() * 4 * options().samples);
    assert_eq!(map.max_output_tokens(), map.sample_count() as u64 * options().max_sample_tokens as u64);
    let mut work = Int8Work::default(); let (mut end, mut scalars) = (0, 0);
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
    assert_eq!((end, scalars), (source.len(), source.chars().count()));
    assert_eq!(map.planned_work(), work);
    assert_eq!(map.expected.source_span, VerifiedSourceSpan { byte_start: 0, byte_end: source.len(),
        scalar_start: 0, scalar_end: source.chars().count() });
}

#[test]
fn repeated_chunks_and_population_items_get_independent_replayable_private_addresses() {
    let source = request("abcdefghabcdefgh");
    let original = prepare(&source.document); assert_eq!(original.chunk_count(), 2);
    let (first, second) = (&original.chunks[0].plan.inner, &original.chunks[1].plan.inner);
    assert_eq!(first.identity.prompt_digest, second.identity.prompt_digest);
    assert_ne!(first.identity.decision_policy_digest, second.identity.decision_policy_digest);
    for (left, right) in first.axes.iter().zip(&second.axes) {
        assert_eq!(left.prompt, right.prompt);
        assert_ne!(left.identity.decision_policy_digest, right.identity.decision_policy_digest);
    }
    let a = planner().plan_document_for_item_with_control(&source, limits(), "item-é", &mut Continue).unwrap();
    let b = planner().plan_document_for_item_with_control(&source, limits(), "item-好", &mut Continue).unwrap();
    let replay = planner().plan_document_for_item_with_control(&source, limits(), "item-é", &mut Continue).unwrap();
    assert_eq!(a.expected.request_binding, replay.expected.request_binding);
    assert_ne!(a.expected.request_binding, b.expected.request_binding);
    assert_ne!(a.expected.request_binding, original.expected.request_binding);
    for ((left, right), cold) in a.chunks.iter().zip(&b.chunks).zip(&original.chunks) {
        assert_eq!(left.plan.planned_work(), cold.plan.planned_work());
        for ((x, y), z) in left.plan.inner.axes.iter().zip(&right.plan.inner.axes).zip(&cold.plan.inner.axes) {
            assert_eq!(x.prompt, y.prompt); assert_eq!(x.prompt, z.prompt);
            assert_ne!(x.identity.decision_policy_digest, y.identity.decision_policy_digest);
            assert_ne!(x.identity.decision_policy_digest, z.identity.decision_policy_digest);
        }
    }
    let changed = prepare("abcdefghabcdefgi");
    assert_eq!(first.axes[0].prompt, changed.chunks[0].plan.inner.axes[0].prompt);
    assert_ne!(first.axes[0].identity.decision_policy_digest,
        changed.chunks[0].plan.inner.axes[0].identity.decision_policy_digest);
    let mut reversed = source; reversed.axes.reverse();
    let reordered = planner().plan_document_with_control(&reversed, limits(), &mut Continue).unwrap();
    assert_eq!(original.expected.request_binding, reordered.expected.request_binding);
    assert_eq!(original.planned_work(), reordered.planned_work());
    assert_eq!(original.execution_identity(), reordered.execution_identity());
    for (left, right) in original.chunks.iter().zip(&reordered.chunks) {
        assert_eq!(left.plan.execution_identity(), right.plan.execution_identity());
        assert_eq!(left.plan.planned_work(), right.plan.planned_work());
        for (a, b) in left.plan.inner.axes.iter().zip(&right.plan.inner.axes) {
            assert_eq!(a.identity, b.identity); assert_eq!(a.prompt, b.prompt);
        }
    }
    for item in [String::new(), "a\n".to_owned(), "a\0b".to_owned(), "x".repeat(257)] {
        let mut work = SafeBpeWork::default();
        assert!(matches!(planner().plan_document_for_item_with_encoding_work(&reversed, limits(),
            &item, &mut Continue, &mut work), Err(Int8SampledSentimentMapError::Limits)));
        assert_eq!(work, SafeBpeWork::default());
    }
}

#[test]
fn all_five_document_work_caps_and_complete_source_limits_refuse_before_execution() {
    let source = request("abcdefghabcdefgh"); let map = prepare(&source.document);
    for axis in 0..5 {
        let mut short = limits(); short.max_model_work = map.planned_work();
        match axis { 0 => short.max_model_work.forward_positions -= 1,
            1 => short.max_model_work.projected_logits -= 1,
            2 => short.max_model_work.attention_pairs -= 1,
            3 => short.max_model_work.projections.dot_products -= 1,
            _ => short.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(matches!(planner().plan_document_with_control(&source, short, &mut Continue),
            Err(Int8SampledSentimentMapError::Work)));
    }
    let mut short = limits(); short.chunks.max_chunks = 1;
    assert!(matches!(planner().plan_document_with_control(&source, short, &mut Continue),
        Err(Int8SampledSentimentMapError::Chunk(MapReduceError::ChunkBudget))));
    let mut short = limits(); short.chunks.max_input_bytes = source.document.len() - 1;
    assert!(matches!(planner().plan_document_with_control(&source, short, &mut Continue),
        Err(Int8SampledSentimentMapError::Chunk(MapReduceError::InputBudget))));
    assert!(matches!(planner().plan_document_with_control(&request(""), limits(), &mut Continue),
        Err(Int8SampledSentimentMapError::EmptySource)));
}

#[test]
fn tokenizer_probes_and_final_child_plans_debit_one_cumulative_encoding_allowance() {
    let compact = Int8SampledSentimentPlanner::for_model_with_encoding(&facts(), options(), budget(),
        SampledSentimentLimits::default(), UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1_000_000,
            max_pair_bytes: 64 * MIB })).unwrap();
    let source = request("testing testing testing testing");
    let mut ledger = SafeBpeWork::default();
    let first = compact.plan_document_with_encoding_work(&source, limits(), &mut Continue, &mut ledger).unwrap();
    assert_eq!(first.expected.encoding_work, ledger);
    assert!(ledger.input_bytes() >= source.document.len() * 2);
    let before = ledger;
    let second = compact.plan_document_with_encoding_work(&source, limits(), &mut Continue, &mut ledger).unwrap();
    assert_eq!(ledger.checked_sub(before).unwrap(), second.expected.encoding_work);
    assert_eq!(first.expected.request_binding, second.expected.request_binding);
    for chunk in &first.chunks {
        let text = &source.document[chunk.span.byte_start..chunk.span.byte_end];
        for axis in &chunk.plan.inner.axes {
            let template = compact.compiler.axes.iter().find(|entry| entry.axis == axis.axis).unwrap();
            let start = compact.compiler.global.len() + template.instruction.len();
            let tokens = &axis.prompt[start..start + chunk.tokens];
            assert!(tokens.iter().all(|id| !compact.compiler.controls.ids().contains(id)));
            assert_eq!(compact.compiler.tokenizer.tokenizer().decode_bytes(tokens).unwrap(), text.as_bytes());
        }
    }
    while ledger.input_bytes() < 4096 { ledger = ledger.checked_add(ledger).unwrap(); }
    let spent = ledger;
    assert!(compact.plan_document_with_encoding_work(&source, limits(), &mut Continue, &mut ledger).is_err());
    assert_eq!(ledger, spent);
}

#[test]
fn native_masks_kv_and_sampler_are_admitted_for_all_chunks_before_the_first_callback() {
    let map = prepare("abcdefghabcdefgh");
    let budget = native_budget(&map); let sampler = sampler_bytes();
    for field in 0..11 {
        let mut bad = budget; let mut storage = sampler;
        match field {
            0 => bad.native.max_forward_positions -= 1,
            1 => bad.native.max_attention_pairs -= 1,
            2 => bad.native.max_projection_work.dot_products -= 1,
            3 => bad.native.max_projection_work.multiply_accumulates -= 1,
            4 => bad.json.max_forward_positions -= 1,
            5 => bad.json.max_projected_logits -= 1,
            6 => bad.json.max_total_mask_node_visits -= 1,
            7 => bad.json.max_kv_bytes = map.required_context() as u64 * KV_BYTES_PER_TOKEN as u64 - 1,
            8 => bad.json.mask_limits.max_trie_node_visits = 0,
            9 => bad.json.mask_limits.checkpoint_interval_nodes = 0,
            _ => storage -= 1,
        }
        let mut visits = 0;
        assert!(map.execute_chunks(MASK_CHARGE, bad, storage, &mut Continue, |plan, _, _| {
            visits += 1; Ok(synthetic(plan, &[-20, 20]))
        }).is_err(), "field {field}");
        assert_eq!(visits, 0, "field {field}");
    }
    let vocabulary = SampledSentimentVocabulary::pinned().unwrap();
    assert_eq!(map.complete_mask_charge(&vocabulary, budget, sampler).unwrap(),
        map.max_output_tokens() * MASK_CHARGE);
}

#[test]
fn complete_execution_uses_exact_child_budgets_and_original_scalar_weighted_replicates() {
    let map = prepare("abcdefghé"); assert_eq!(map.chunk_count(), 2);
    let mut visits = 0;
    let run = map.execute_chunks(MASK_CHARGE, native_budget(&map), sampler_bytes(), &mut Continue,
        |plan, child, _| {
            assert_eq!(child.native.max_forward_positions, plan.planned_work().forward_positions);
            assert_eq!(child.native.max_attention_pairs, plan.planned_work().attention_pairs);
            assert_eq!(child.native.max_projection_work, plan.planned_work().projections);
            assert_eq!(child.json.max_projected_logits, plan.planned_work().projected_logits);
            assert_eq!(child.json.max_total_mask_node_visits, plan.max_output_tokens() * MASK_CHARGE);
            let scores = if visits == 0 { [-20, 20] } else { [100, -100] };
            visits += 1; Ok(synthetic(plan, &scores))
        }).unwrap();
    assert_eq!(visits, map.chunk_count()); assert_eq!(run.aggregate.source_scalars, 9);
    assert_eq!(run.source_span.byte_end, 10); assert!(!run.single_context_equivalence);
    for dimension in &run.aggregate.dimensions {
        assert_eq!(dimension.replicates[0].weighted_score_sum, -60);
        assert_eq!(dimension.replicates[1].weighted_score_sum, 60);
        assert_eq!(dimension.replicates[0].score, -60.0 / 9.0);
        assert_eq!(dimension.mean, 0.0);
    }
    assert_eq!(run.mask_node_visit_charge, run.generated_tokens * MASK_CHARGE);
    assert!(run.generated_tokens < map.max_output_tokens());
    assert_eq!(run.planned_mask_node_visit_charge, map.max_output_tokens() * MASK_CHARGE);
    map.preflight_metadata().verify_completed_with_control(&run, &mut Continue).unwrap();
    let serialized = serde_json::to_string(&run).unwrap();
    for private in ["effective_seed", "request_binding", "result_binding", "decision_policy_digest", "group_width"] {
        assert!(!serialized.contains(private), "{private}");
    }
    assert!(serialized.contains(INT8_SAMPLED_SENTIMENT_MAP_EXECUTION));
    assert!(serialized.contains("original_unicode_scalar_count"));
}

#[test]
fn a_bad_late_chunk_or_refusal_never_returns_partial_document_success() {
    let map = prepare("abcdefghabcdefghabcdefgh");
    for fault in 0..7 {
        let mut visits = 0;
        let result = map.execute_chunks(MASK_CHARGE, native_budget(&map), sampler_bytes(), &mut Continue,
            |plan, _, _| {
                visits += 1;
                let mut run = synthetic(plan, &[-20, 20]);
                if visits == 2 { match fault {
                    0 => { run.dimensions.pop(); }
                    1 => { run.dimensions[0].samples.pop(); }
                    2 => run.dimensions[0].mean += 1.0,
                    3 => run.dimensions[0].samples[0].model_work.projections.dot_products -= 1,
                    4 => run.dimensions[0].samples[0].mask_node_visit_charge -= 1,
                    5 => run.dimensions[0].samples[1].sample_index = 0,
                    _ => return Err(Int8SampledSentimentMapError::Admission),
                } }
                Ok(run)
            });
        assert!(result.is_err(), "fault {fault}"); assert_eq!(visits, 2, "fault {fault}");
    }
}

#[test]
fn private_completion_seals_bind_source_geometry_work_and_every_explanation() {
    let map = prepare("abcdefghabcdefgh"); let run = execute(&map);
    let other = prepare("abcdefghabcdefgi");
    assert!(other.expected.verify_completed_with_control(&run, &mut Continue).is_err());
    for field in 0..8 {
        let mut changed = run.clone();
        match field {
            0 => changed.chunks[0].source_span.scalar_end += 1,
            1 => changed.chunks[1].source_tokens += 1,
            2 => changed.aggregate.dimensions[0].mean += 1.0,
            3 => changed.model_work.forward_positions += 1,
            4 => changed.mask_node_visit_charge += 1,
            5 => changed.chunks[0].run.dimensions[0].samples[0].justification.push('!'),
            6 => changed.single_context_equivalence = true,
            _ => changed.planned_model_work.projections.dot_products -= 1,
        }
        assert!(map.expected.verify_completed_with_control(&changed, &mut Continue).is_err(), "field {field}");
    }
}

#[test]
fn retained_and_complete_output_byte_boundaries_are_exact_and_never_truncate() {
    let map = prepare("abcdefghabcdefgh"); let run = execute(&map);
    let bytes = serde_json::to_vec(&run).unwrap().len() as u64;
    let mut exact = limits();
    exact.max_retained_result_bytes = run.retained_result_bytes; exact.max_result_bytes = bytes;
    let exact_map = planner().plan_document_with_control(&request("abcdefghabcdefgh"), exact, &mut Continue).unwrap();
    let exact_run = execute(&exact_map);
    assert_eq!(serde_json::to_vec(&exact_run).unwrap().len() as u64, bytes);
    for field in 0..2 {
        let mut short = exact;
        if field == 0 { short.max_retained_result_bytes -= 1; } else { short.max_result_bytes -= 1; }
        let short = planner().plan_document_with_control(&request("abcdefghabcdefgh"), short, &mut Continue).unwrap();
        assert!(short.execute_chunks(MASK_CHARGE, native_budget(&short), sampler_bytes(), &mut Continue,
            |plan, _, _| Ok(synthetic(plan, &[-20, 20]))).is_err());
    }
}

struct Control { cause: Option<DecodeCancellationKind>, calls: usize, stop_at: usize }
impl Control { fn continuing() -> Self { Self { cause: None, calls: 0, stop_at: usize::MAX } } }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; if self.calls >= self.stop_at { self.cause } else { None }
    }
}
fn causes() -> [DecodeCancellationKind; 11] {
    use DecodeCancellationKind::*;
    [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit]
}

#[test]
fn every_terminal_cause_survives_planning_callback_refusal_and_publication() {
    let map = prepare("abcdefghabcdefgh");
    for cause in causes() {
        let mut stopped = Control { cause: Some(cause), calls: 0, stop_at: 1 };
        let error = planner().plan_document_with_control(&request("abcdefgh"), limits(), &mut stopped).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
        for callback_error in [false, true] {
            let mut control = Control::continuing();
            let error = map.execute_chunks(MASK_CHARGE, native_budget(&map), sampler_bytes(), &mut control,
                |plan, _, control| {
                    control.cause = Some(cause); control.stop_at = 0;
                    if callback_error { Err(Int8SampledSentimentMapError::Admission) }
                    else { Ok(synthetic(plan, &[-20, 20])) }
                }).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause));
        }
        let mut control = Control::continuing();
        let error = map.execute_chunks(MASK_CHARGE, native_budget(&map), sampler_bytes(), &mut control,
            |_, _, control| {
                control.cause = Some(DecodeCancellationKind::Shutdown); control.stop_at = 0;
                Err(Int8SampledSentimentMapError::Cancelled(cause))
            }).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
    }
    let run = execute(&map); let mut counted = Control::continuing();
    map.expected.verify_completed_with_control(&run, &mut counted).unwrap();
    let steps = counted.calls; assert!(steps > 4);
    for at in [1, steps / 2, steps] {
        let mut control = Control { cause: Some(DecodeCancellationKind::Deadline), calls: 0, stop_at: at };
        let error = map.expected.verify_completed_with_control(&run, &mut control).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
}

fn memory() -> Int8SampledSentimentMapMemoryLimits {
    Int8SampledSentimentMapMemoryLimits { context_tokens: 4096,
        preparation_reserve_bytes: MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: MIB, max_axes: 4, max_sampler_bytes: sampler_bytes() }
}
#[test]
fn memory_quote_covers_retained_plans_and_one_shared_encoder_without_double_charging_native_storage() {
    let mut mapping = limits(); mapping.chunks.max_chunks = 1;
    let planning = SampledSentimentLimits::default();
    let one = memory().requirements(budget(), planning, mapping, &options(), &UntrustedEncoding::ByteFallback).unwrap();
    mapping.chunks.max_chunks = 2;
    let two = memory().requirements(budget(), planning, mapping, &options(), &UntrustedEncoding::ByteFallback).unwrap();
    assert_eq!(one.native, two.native); assert_eq!(one.workspace_bytes, two.workspace_bytes);
    assert!(two.input_bytes > one.input_bytes); assert!(two.input_bytes < 2 * one.input_bytes);
    assert_eq!(two.max_output_tokens, 2 * one.max_output_tokens);
    assert_eq!(two.output_reservation_bytes, 4 * mapping.max_result_bytes + 8 * two.max_output_tokens + 4096);
    let map = planner().plan_document_with_control(&request("abcdefghabcdefgh"), mapping, &mut Continue).unwrap();
    assert!(two.input_bytes >= map.preparation_requirement_bytes());
    let compact = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1_000_000, max_pair_bytes: 64 * MIB };
    let quote = memory().requirements(budget(), planning, mapping, &options(), &UntrustedEncoding::SafeBpeV1(compact)).unwrap();
    assert_eq!(quote.input_bytes, two.input_bytes + compact.requirements().unwrap().workspace_bytes);
    assert_eq!(quote.native, two.native); assert_eq!(quote.workspace_bytes, two.workspace_bytes);
    assert_eq!(quote.output_reservation_bytes, two.output_reservation_bytes);
}
#[test]
fn memory_quote_refuses_missing_native_sampler_preparation_and_overflow_bounds() {
    for field in 0..5 {
        let mut selected = memory();
        match field { 0 => selected.preparation_reserve_bytes -= 1,
            1 => selected.allocator_reserve_bytes = 0, 2 => selected.max_sampler_bytes -= 1,
            3 => selected.max_axes = 0, _ => selected.context_tokens = 128 }
        assert!(selected.requirements(budget(), SampledSentimentLimits::default(), limits(),
            &options(), &UntrustedEncoding::ByteFallback).is_err(), "field {field}");
    }
    let mut planning = SampledSentimentLimits::default(); planning.compile.max_schema_bytes = usize::MAX;
    assert!(memory().requirements(budget(), planning, limits(), &options(), &UntrustedEncoding::ByteFallback).is_err());
    let mut task = budget(); task.max_kv_bytes = 1;
    assert!(memory().requirements(task, SampledSentimentLimits::default(), limits(),
        &options(), &UntrustedEncoding::ByteFallback).is_err());
}
