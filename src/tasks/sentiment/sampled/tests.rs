//! Model-free planning and independent-finalizer contracts. Private synthetic
//! native outputs below test refusal/accounting, not model execution or quality.

use super::*;
use std::sync::OnceLock;
use crate::native_engine::constrained::{JsonDecodeError, JsonDecodeOutput, JsonWorkBudget};
use crate::grammar::mask::MaskWorkLimits;

fn options() -> SampledSentimentOptions {
    SampledSentimentOptions { effective_seed: [37; 32], samples: 3, temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 256, max_justification_chars: 64 }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 4096, max_output_bytes: MIB,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn fixture() -> &'static (Compiler, ExecutionIdentity) {
    static FIXTURE: OnceLock<(Compiler, ExecutionIdentity)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let compiler = Compiler::pinned(&options(), budget(), SampledSentimentLimits::default()).unwrap();
        let identity = unbound_identity(&compiler);
        (compiler, identity)
    })
}
fn request() -> SentimentRequest {
    SentimentRequest { document: "The service was reassuring and helpful.".to_owned(),
        axes: SentimentAxis::ALL.to_vec(), budget: budget() }
}
fn prepare(request: &SentimentRequest) -> Result<PreparedSampledSentiment, SampledSentimentError> {
    let (compiler, identity) = fixture();
    compile_request(compiler, identity, 32, &options(), budget(),
        SampledSentimentLimits::default(), request, &mut Continue)
}
fn sample(index: usize, score: i16) -> SampledSentimentSample {
    SampledSentimentSample { sample_index: index, score, justification: "Fixture explanation.".to_owned(),
        generated_tokens: 1, model_work: Int4Work::default(), mask_node_visit_charge: 0 }
}
fn synthetic(plan: &PreparedSampledSentiment, axis: &PreparedAxis, json: &str) -> Int4JsonRun {
    let mut tokens = fixture().0.tokenizer.tokenizer().encode_byte_fallback_only(json.as_bytes()).unwrap();
    tokens.push(plan.decode.eos_token_id);
    let work = Int4Work::for_sequence(plan.group_width, 0,
        axis.prompt.len() + tokens.len() - 1, tokens.len() * NANBEIGE_VOCAB_SIZE).unwrap();
    Int4JsonRun { schema_version: 1, execution: INT4_JSON_SAMPLED_EXECUTION.to_owned(),
        group_width: plan.group_width, model_work: work,
        output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
            token_ids: tokens, json: json.to_owned(), forward_positions: work.forward_positions,
            projected_logits: work.projected_logits, mask_node_visit_charge: 0 } }
}

#[test]
fn options_are_explicit_bounded_and_debug_does_not_publish_seed() {
    let valid = options(); valid.validate().unwrap();
    let mut variants = Vec::new();
    let mut bad = valid.clone(); bad.samples = 0; variants.push(bad);
    let mut bad = valid.clone(); bad.samples = 33; variants.push(bad);
    let mut bad = valid.clone(); bad.max_sample_tokens = 1025; variants.push(bad);
    let mut bad = valid.clone(); bad.max_justification_chars = 4097; variants.push(bad);
    let mut bad = valid.clone(); bad.temperature_milli = 0; variants.push(bad);
    let mut bad = valid.clone(); bad.top_k = Some(0); variants.push(bad);
    let mut bad = valid.clone(); bad.top_p_ppm = 1_000_001; variants.push(bad);
    for bad in variants { assert!(bad.validate().is_err()); }
    let debug = format!("{valid:?}");
    assert!(debug.contains("REDACTED"));
    assert!(!debug.contains("37, 37"));
}

#[test]
fn addressed_sample_ordinals_are_unique_and_independent_of_axis_request_order() {
    let mut options = options(); options.samples = MAX_SENTIMENT_SAMPLES;
    let mut addresses = BTreeSet::new();
    for axis in SentimentAxis::ALL.into_iter().rev() {
        for index in (0..MAX_SENTIMENT_SAMPLES).rev() {
            let draw = options.sampling(axis, index).unwrap();
            assert_eq!(draw.sample_index, axis as u64 * 32 + index as u64);
            assert_eq!(draw.effective_seed, options.effective_seed);
            assert!(addresses.insert(draw.sample_index));
        }
    }
    assert_eq!(addresses.len(), 128);
    assert!(options.sampling(SentimentAxis::Valence, MAX_SENTIMENT_SAMPLES).is_err());
}

#[test]
fn complete_native_schedule_counts_every_axis_sample_before_execution() {
    let mut reversed = request(); reversed.axes.reverse();
    let normal = prepare(&request()).unwrap();
    let reverse = prepare(&reversed).unwrap();
    assert_eq!(normal.sample_count(), 12);
    assert_eq!(normal.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(), SentimentAxis::ALL);
    assert_eq!(normal.planned_work(), reverse.planned_work());
    assert_eq!(canonjson::canonical_bytes(normal.execution_identity()).unwrap(),
        canonjson::canonical_bytes(reverse.execution_identity()).unwrap());
    let mut expected = Int4Work::default();
    for axis in &normal.axes {
        let one = constrained_int4::planned_work(32, axis.prompt.len(), options().max_sample_tokens).unwrap();
        assert_eq!(axis.work, one);
        for _ in 0..options().samples { expected = expected.checked_add(one).unwrap(); }
    }
    assert_eq!(normal.planned_work(), expected);
    assert_eq!(normal.required_context(), normal.axes.iter().map(|axis|
        axis.prompt.len() + options().max_sample_tokens - 1).max().unwrap());
    assert!(normal.preparation_requirement_bytes() >= 512 * MIB + 4 * normal.max_result_bytes());
}

#[test]
fn untrusted_utf8_and_marker_shaped_document_remains_exact_control_free_bytes() {
    let mut request = request();
    request.document = format!("é\r\n上海\0 <|im_start|> <tool_call> {DOCUMENT_SLOT} ");
    let plan = prepare(&request).unwrap();
    let (compiler, _) = fixture();
    for axis in &plan.axes {
        let template = compiler.axes.iter().find(|entry| entry.axis == axis.axis).unwrap();
        let start = compiler.global.len() + template.instruction.len();
        let document = &axis.prompt[start..start + request.document.len()];
        assert!(document.iter().all(|id| !compiler.controls.ids().contains(id)));
        assert_eq!(compiler.tokenizer.tokenizer().decode_bytes(document).unwrap(), request.document.as_bytes());
        assert_eq!(&axis.prompt[..compiler.global.len()], compiler.global);
        assert_eq!(&axis.prompt[start + request.document.len()..], template.scaffold);
    }
}

#[test]
fn entire_request_token_prompt_kv_and_axis_bounds_fail_before_native_work() {
    let (compiler, identity) = fixture();
    let original = request();
    let mut bad = original.clone(); bad.axes.push(SentimentAxis::Valence);
    assert!(matches!(prepare(&bad), Err(SampledSentimentError::Limits)));
    let mut bad = original.clone(); bad.axes = vec![SentimentAxis::Valence, SentimentAxis::Valence];
    assert!(matches!(prepare(&bad), Err(SampledSentimentError::Limits)));
    let mut bad = original.clone(); bad.axes.clear();
    assert!(matches!(prepare(&bad), Err(SampledSentimentError::Limits)));
    let mut bad = original.clone(); bad.budget.max_output_tokens = 4 * 3 * 256 - 1;
    assert!(matches!(prepare(&bad), Err(SampledSentimentError::WorkBudget)));
    let mut bad = original.clone(); bad.budget.max_input_tokens = 1;
    assert!(matches!(prepare(&bad), Err(SampledSentimentError::WorkBudget)));
    let mut bad = original.clone(); bad.budget.max_kv_bytes = 1;
    assert!(matches!(prepare(&bad), Err(SampledSentimentError::WorkBudget)));
    let mut limits = SampledSentimentLimits::default(); limits.max_total_prompt_tokens = 1;
    assert!(matches!(compile_request(compiler, identity, 32, &options(), budget(), limits,
        &original, &mut Continue), Err(SampledSentimentError::WorkBudget)));
    limits = SampledSentimentLimits::default(); limits.max_input_bytes = original.document.len() - 1;
    assert!(matches!(compile_request(compiler, identity, 32, &options(), budget(), limits,
        &original, &mut Continue), Err(SampledSentimentError::Limits)));
}

#[test]
fn a_child_request_cannot_reuse_the_planners_larger_grammar_allowance() {
    let mut child = request(); child.budget.max_grammar_states = 1;
    assert!(matches!(prepare(&child), Err(SampledSentimentError::Schema)));
    let mut child = request(); child.budget.max_output_bytes = 1;
    assert!(matches!(prepare(&child), Err(SampledSentimentError::Schema)));
    let mut child = request(); child.budget.max_output_bytes = 128;
    let prepared = prepare(&child).unwrap();
    assert_eq!(prepared.max_result_bytes(), 128);
    assert_eq!(prepared.program.max_output_bytes(), 128);
}

#[test]
fn sampled_policy_source_and_group_are_bound_to_sealed_admission() {
    let original = prepare(&request()).unwrap();
    original.verify_identity(original.execution_identity()).unwrap();
    let (compiler, identity) = fixture();
    let mut changed = options(); changed.effective_seed[0] ^= 1;
    let different = compile_request(compiler, identity, 32, &changed, budget(),
        SampledSentimentLimits::default(), &request(), &mut Continue).unwrap();
    assert_ne!(original.identity.decision_policy_digest, different.identity.decision_policy_digest);
    assert_ne!(original.axes[0].identity.decision_policy_digest, different.axes[0].identity.decision_policy_digest);
    assert!(matches!(original.verify_identity(different.execution_identity()), Err(SampledSentimentError::Identity)));
    let mut document = request(); document.document.push('!');
    let document = prepare(&document).unwrap();
    assert_ne!(original.identity.prompt_digest, document.identity.prompt_digest);
    let group = compile_request(compiler, identity, 127, &options(), budget(),
        SampledSentimentLimits::default(), &request(), &mut Continue).unwrap();
    assert_ne!(original.identity.decision_policy_digest, group.identity.decision_policy_digest);
    let mut drift = original.identity.clone(); drift.sampler_version = "distribution-only".to_owned();
    assert!(matches!(original.verify_identity(&drift), Err(SampledSentimentError::Identity)));
}

#[test]
fn score_schema_rejects_off_grid_missing_duplicate_and_unbounded_outputs() {
    let prepared = prepare(&request()).unwrap();
    for valid in [r#"{"score":-100,"justification":"low"}"#,
        r#"{"score":0,"justification":"neutral"}"#, r#"{"score":100,"justification":"high"}"#] {
        prepared.program.validate_json(valid).unwrap();
    }
    for invalid in [r#"{"score":101,"justification":"outside"}"#,
        r#"{"score":-101,"justification":"outside"}"#, r#"{"score":0.5,"justification":"fraction"}"#,
        r#"{"score":0}"#, r#"{"score":0,"justification":"ok","extra":1}"#,
        r#"{"score":0,"score":1,"justification":"duplicate"}"#] {
        assert!(prepared.program.validate_json(invalid).is_err(), "{invalid}");
    }
    let excessive = serde_json::json!({"score":0,"justification":"x".repeat(options().max_justification_chars + 1)});
    assert!(prepared.program.validate_json(&excessive.to_string()).is_err());
    let unicode = serde_json::json!({"score":0,"justification":"好".repeat(options().max_justification_chars)});
    prepared.program.validate_json(&unicode.to_string()).unwrap();
}

#[test]
fn finalizer_rechecks_sample_language_eos_and_native_accounting() {
    let prepared = prepare(&request()).unwrap(); let axis = &prepared.axes[0];
    let good = synthetic(&prepared, axis, r#"{"score":45,"justification":"Helpful wording expresses approval."}"#);
    let sample = prepared.finalize_sample(axis, 0, good.clone()).unwrap();
    assert_eq!(sample.score, 45); assert_eq!(sample.sample_index, 0);
    assert_eq!(sample.model_work, good.model_work);
    let mut bad = good.clone(); bad.execution = constrained_int4::INT4_JSON_EXECUTION.to_owned();
    assert!(prepared.finalize_sample(axis, 0, bad).is_err());
    let mut bad = good.clone(); bad.model_work.projections.dot_products += 1;
    assert!(prepared.finalize_sample(axis, 0, bad).is_err());
    let mut bad = good.clone(); bad.output.token_ids.pop();
    assert!(prepared.finalize_sample(axis, 0, bad).is_err());
    let mut bad = good.clone(); bad.output.token_ids[0] = prepared.decode.eos_token_id;
    assert!(prepared.finalize_sample(axis, 0, bad).is_err());
    let mut bad = good.clone(); bad.output.token_ids[0] = NANBEIGE_VOCAB_SIZE as u32;
    assert!(prepared.finalize_sample(axis, 0, bad).is_err());
    assert!(prepared.finalize_sample(axis, prepared.options.samples, good).is_err());
}

#[test]
fn explanations_require_nonempty_text_and_bound_unicode_scalars_independently() {
    let prepared = prepare(&request()).unwrap(); let axis = &prepared.axes[0];
    for text in ["", " \t\r\n", &"x".repeat(options().max_justification_chars + 1)] {
        let json = serde_json::json!({"score":0,"justification":text}).to_string();
        assert!(matches!(prepared.finalize_sample(axis, 0, synthetic(&prepared, axis, &json)),
            Err(SampledSentimentError::NoResult)));
    }
    for json in [r#"{"score":101,"justification":"outside"}"#,
        r#"{"score":0,"justification":"ok","extra":1}"#] {
        assert!(matches!(prepared.finalize_sample(axis, 0, synthetic(&prepared, axis, json)),
            Err(SampledSentimentError::NoResult)));
    }
    let unicode = "好".repeat(options().max_justification_chars);
    let json = serde_json::json!({"score":0,"justification":unicode}).to_string();
    let sample = prepared.finalize_sample(axis, 0, synthetic(&prepared, axis, &json)).unwrap();
    assert_eq!(sample.justification.chars().count(), options().max_justification_chars);
    assert_eq!(sample.justification.len(), 3 * options().max_justification_chars);
}

#[test]
fn canonical_aggregation_uses_population_spread_without_confidence_claims() {
    let input = vec![sample(2, 60), sample(0, -100), sample(3, 100), sample(1, -20)];
    let values = aggregate(SentimentAxis::Valence, input.clone()).unwrap();
    assert_eq!(values.mean, 10.0); assert_eq!(values.median, 20.0);
    assert_eq!((values.minimum, values.maximum), (-100, 100));
    assert!((values.population_stddev - 5900_f64.sqrt()).abs() < 1e-12);
    assert_eq!((values.low_anchor.as_str(), values.high_anchor.as_str()), ("negative", "positive"));
    let mut reverse = input; reverse.reverse();
    assert_eq!(values, aggregate(SentimentAxis::Valence, reverse).unwrap());
    let single = aggregate(SentimentAxis::Arousal, vec![sample(0, -75)]).unwrap();
    assert_eq!((single.mean, single.median, single.population_stddev), (-75.0, -75.0, 0.0));
    assert_eq!((single.low_anchor.as_str(), single.high_anchor.as_str()), ("calm", "activated"));
}

#[test]
fn missing_duplicate_out_of_range_or_overcount_samples_do_not_form_a_result() {
    for invalid in [Vec::new(), vec![sample(1, 0)], vec![sample(0, 0), sample(0, 1)],
        vec![sample(0, 101)], (0..=MAX_SENTIMENT_SAMPLES).map(|index| sample(index, 0)).collect()] {
        assert!(matches!(aggregate(SentimentAxis::Valence, invalid), Err(SampledSentimentError::Accounting)));
    }
}

#[test]
fn complete_serialized_result_including_metadata_obeys_the_exact_byte_cap() {
    let result = aggregate(SentimentAxis::Dominance, vec![sample(0, 12)]).unwrap();
    let size = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    check_output_size(&result, size).unwrap();
    assert!(matches!(check_output_size(&result, size - 1), Err(SampledSentimentError::OutputBudget)));
    assert!(!serde_json::to_string(&result).unwrap().contains("confidence"));
}

#[test]
fn aggregate_mask_and_sampler_work_is_reserved_for_all_samples() {
    let prepared = prepare(&request()).unwrap();
    // A small private trie is sufficient to test arithmetic; public execution
    // can construct only the full pinned vocabulary and checks its width.
    let compiler = &fixture().0;
    let trie = VocabTrie::from_tokenizer(compiler.tokenizer.tokenizer(), 256).unwrap();
    let vocabulary = SampledSentimentVocabulary { oracle: VocabMaskOracle::new(trie),
        controls: compiler.controls.ids().clone() };
    let step = vocabulary.oracle.trie().node_count() as u64;
    let one = step * options().max_sample_tokens as u64;
    let total = one * prepared.sample_count() as u64;
    let mut budget = Int4JsonBudget { native: Int4RunBudget::exact(prepared.work), json: JsonWorkBudget {
        max_forward_positions: prepared.work.forward_positions, max_projected_logits: prepared.work.projected_logits,
        max_kv_bytes: prepared.budget.max_kv_bytes, max_total_mask_node_visits: total,
        mask_limits: MaskWorkLimits { max_trie_node_visits: step as usize, checkpoint_interval_nodes: 1 } } };
    let sampler = json_sampling_workspace_bytes().unwrap();
    assert_eq!(prepared.preflight(&vocabulary, budget, sampler).unwrap(), one);
    budget.json.max_total_mask_node_visits = total - 1;
    assert!(matches!(prepared.preflight(&vocabulary, budget, sampler), Err(SampledSentimentError::WorkBudget)));
    budget.json.max_total_mask_node_visits = total;
    assert!(matches!(prepared.preflight(&vocabulary, budget, sampler - 1), Err(SampledSentimentError::WorkBudget)));
    budget.native.max_forward_positions -= 1;
    assert!(matches!(prepared.preflight(&vocabulary, budget, sampler), Err(SampledSentimentError::WorkBudget)));
}

struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
}
#[test]
fn planning_and_wrapped_native_cancellation_retain_every_original_cause() {
    let (compiler, identity) = fixture();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable,
        DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let error = compile_request(compiler, identity, 32, &options(), budget(),
            SampledSentimentLimits::default(), &request(), &mut Stop(cause)).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
        assert_eq!(SampledSentimentError::from(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))).cancellation(), Some(cause));
        assert_eq!(SampledSentimentError::from(StrictInt4Error::Cancelled(cause)).cancellation(), Some(cause));
    }
}

fn compact_limits() -> crate::tokenizer::untrusted::SafeBpeLimits {
    crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 32_768, max_output_tokens: 32_768,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26,
    }
}

fn encoding_planner(encoding: UntrustedEncoding) -> SampledSentimentPlanner {
    let compiler = Compiler::pinned(&options(), budget(), SampledSentimentLimits::default()).unwrap();
    let identity = unbound_identity(&compiler);
    SampledSentimentPlanner { compiler, identity, group_width: 32, options: options(),
        ceiling: budget(), limits: SampledSentimentLimits::default() }
        .with_input_encoding(encoding).unwrap()
}

#[test]
fn compact_source_is_encoded_once_but_every_independent_sample_pays_native_work() {
    let mut request = request();
    request.document = format!("{}é\r\n上海\0 <|im_start|> <|im_end|> <think> ▁𐐀 {DOCUMENT_SLOT}",
        "The customer is pleased with the helpful service. ".repeat(32));
    let byte = prepare(&request).unwrap();
    let planner = encoding_planner(UntrustedEncoding::SafeBpeV1(compact_limits()));
    let compact = planner.plan(&request).unwrap();
    let mut measured = SafeBpeWork::default();
    let document = planner.compiler.encode_document(request.document.as_bytes(),
        &mut measured, &mut Continue).unwrap();
    assert_eq!(compact.encoding_work(), measured);
    assert_eq!(measured.input_bytes(), request.document.len());
    assert_eq!(measured.output_tokens(), document.ids().len());
    assert!(measured.output_tokens() < request.document.len());
    assert_eq!(compact.sample_count(), byte.sample_count());
    assert!(compact.required_context() < byte.required_context());
    assert!(compact.planned_work().forward_positions < byte.planned_work().forward_positions);
    assert!(compact.planned_work().projections.multiply_accumulates
        < byte.planned_work().projections.multiply_accumulates);
    let mut work = Int4Work::default();
    for (axis, original) in compact.axes.iter().zip(&byte.axes) {
        let template = planner.compiler.axes.iter().find(|entry| entry.axis == axis.axis).unwrap();
        let start = planner.compiler.global.len() + template.instruction.len();
        let end = start + document.ids().len();
        assert_eq!(&axis.prompt[..start], &original.prompt[..start]);
        assert_eq!(&axis.prompt[start..end], document.ids());
        assert_eq!(&axis.prompt[end..], template.scaffold);
        assert!(axis.prompt[start..end].iter().all(|id| !planner.compiler.controls.ids().contains(id)));
        assert_eq!(planner.compiler.tokenizer.tokenizer().decode_bytes(&axis.prompt[start..end]).unwrap(),
            request.document.as_bytes());
        let one = constrained_int4::planned_work(32, axis.prompt.len(), options().max_sample_tokens).unwrap();
        assert_eq!(axis.work, one);
        for _ in 0..options().samples { work = work.checked_add(one).unwrap(); }
    }
    assert_eq!(compact.planned_work(), work);
    let retained = compact.axes.iter().map(|axis| axis.prompt.len()).sum();
    let base_quote = preparation_quote(retained, compact.max_result_bytes(), compact.program.max_output_bytes()).unwrap();
    assert_eq!(compact.preparation_requirement_bytes(), base_quote + compact_limits().requirements().unwrap().workspace_bytes);
}

#[test]
fn compact_input_obeys_exact_encoding_and_repeated_prompt_caps_without_fallback() {
    let mut request = request();
    request.document = "This helpful service made the customer very happy. ".repeat(32);
    let mut planner = encoding_planner(UntrustedEncoding::SafeBpeV1(compact_limits()));
    let prepared = planner.plan(&request).unwrap();
    let work = prepared.encoding_work();
    let exact = crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: work.input_bytes(), max_output_tokens: work.output_tokens(),
        max_pair_checks: work.pair_checks(), max_pair_bytes: work.pair_bytes(),
    };
    let maximum = prepared.axes.iter().map(|axis| axis.prompt.len()).max().unwrap();
    let total = prepared.axes.iter().map(|axis| axis.prompt.len() * options().samples).sum();
    assert!(maximum < request.document.len());
    request.budget.max_input_tokens = maximum as u32;
    planner.limits.max_total_prompt_tokens = total;
    planner = planner.with_input_encoding(UntrustedEncoding::SafeBpeV1(exact)).unwrap();
    let exact_prepared = planner.plan(&request).unwrap();
    assert_eq!(exact_prepared.encoding_work(), work);
    planner.limits.max_total_prompt_tokens -= 1;
    assert!(matches!(planner.plan(&request), Err(SampledSentimentError::WorkBudget)));
    planner.limits.max_total_prompt_tokens = total;
    request.budget.max_input_tokens -= 1;
    assert!(matches!(planner.plan(&request), Err(SampledSentimentError::WorkBudget)));
    request.budget.max_input_tokens += 1;
    for dimension in 0..4 {
        let mut under = exact;
        match dimension {
            0 => under.max_input_bytes -= 1,
            1 => under.max_output_tokens -= 1,
            2 => under.max_pair_checks -= 1,
            _ => under.max_pair_bytes -= 1,
        }
        planner = planner.with_input_encoding(UntrustedEncoding::SafeBpeV1(under)).unwrap();
        assert!(matches!(planner.plan(&request), Err(SampledSentimentError::TokenizerWork)));
    }
    // A byte-only retry would exceed the admitted prompt. Exhaustion never
    // silently changes the selected encoding or renews its allowance.
    planner = planner.with_input_encoding(UntrustedEncoding::ByteFallback).unwrap();
    assert!(matches!(planner.plan(&request), Err(SampledSentimentError::WorkBudget)));
}

#[test]
fn compact_policy_binds_the_sampler_identity_and_byte_default_round_trips_exactly() {
    let request = request();
    let byte = prepare(&request).unwrap();
    let planner = encoding_planner(UntrustedEncoding::ByteFallback);
    let explicit_byte = planner.plan(&request).unwrap();
    assert_eq!(canonjson::canonical_bytes(byte.execution_identity()).unwrap(),
        canonjson::canonical_bytes(explicit_byte.execution_identity()).unwrap());
    assert_eq!(explicit_byte.encoding_work(), SafeBpeWork::default());
    let mut planner = planner.with_input_encoding(UntrustedEncoding::SafeBpeV1(compact_limits())).unwrap();
    let first = planner.plan(&request).unwrap();
    let mut changed = compact_limits(); changed.max_pair_checks -= 1;
    planner = planner.with_input_encoding(UntrustedEncoding::SafeBpeV1(changed)).unwrap();
    let second = planner.plan(&request).unwrap();
    assert_eq!(first.encoding_work(), second.encoding_work());
    assert_ne!(first.execution_identity().template_digest, second.execution_identity().template_digest);
    assert!(matches!(first.verify_identity(second.execution_identity()), Err(SampledSentimentError::Identity)));
    for (left, right) in first.axes.iter().zip(&second.axes) {
        assert_eq!(left.prompt, right.prompt);
        assert_ne!(left.identity, right.identity);
    }
    planner = planner.with_input_encoding(UntrustedEncoding::ByteFallback).unwrap();
    let restored = planner.plan(&request).unwrap();
    assert_eq!(restored.input_encoding(), &UntrustedEncoding::ByteFallback);
    assert_eq!(canonjson::canonical_bytes(byte.execution_identity()).unwrap(),
        canonjson::canonical_bytes(restored.execution_identity()).unwrap());
    assert_eq!(byte.preparation_requirement_bytes(), restored.preparation_requirement_bytes());
    for (old, new) in byte.axes.iter().zip(&restored.axes) { assert_eq!(old.prompt, new.prompt); }
}

#[test]
fn cancellation_inside_compact_encoding_keeps_the_original_cause_and_consumed_work() {
    struct DuringEncoding { calls: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for DuringEncoding {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls == 5).then_some(self.cause)
        }
    }
    let planner = encoding_planner(UntrustedEncoding::SafeBpeV1(compact_limits()));
    let source = "The complete document is independently sampled. ".repeat(32);
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable,
        DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let mut control = DuringEncoding { calls: 0, cause };
        let mut used = SafeBpeWork::default();
        let error = planner.compiler.encode_document(source.as_bytes(), &mut used, &mut control).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
        assert_eq!(control.calls, 5);
        assert_eq!(used.input_bytes(), source.len());
        let mut request = request(); request.document.clone_from(&source);
        let error = planner.plan_with_control(&request, &mut DuringEncoding { calls: 0, cause }).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
    }
}
