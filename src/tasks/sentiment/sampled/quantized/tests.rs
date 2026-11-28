//! Model-free contracts for the real INT8 sampled task. Synthetic native
//! receipts exercise refusal/accounting only, never neural fidelity or quality.

use super::*;
use std::sync::OnceLock;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::constrained::{JsonDecodeError, JsonDecodeOutput},
    tokenizer::untrusted::SafeBpeLimits,
};

const MIB: u64 = 1024 * 1024;
const MASK_CHARGE: u64 = 7;

fn options() -> SampledSentimentOptions {
    SampledSentimentOptions { effective_seed: [37; 32], samples: 3, temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 256, max_justification_chars: 64 }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 4096, max_output_bytes: MIB,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn fixture() -> &'static (Compiler, ExecutionIdentity) {
    static FIXTURE: OnceLock<(Compiler, ExecutionIdentity)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let compiler = Compiler::pinned(&options(), budget(), SampledSentimentLimits::default()).unwrap();
        let identity = bind_model(unbound_identity(&compiler), &facts()).unwrap();
        (compiler, identity)
    })
}
fn request() -> SentimentRequest {
    SentimentRequest { document: "The service was reassuring and helpful.".to_owned(),
        axes: SentimentAxis::ALL.to_vec(), budget: budget() }
}
fn compile(request: &SentimentRequest, options: &SampledSentimentOptions)
    -> Result<PreparedInt8SampledSentiment, Int8SampledSentimentError> {
    let (compiler, identity) = fixture();
    Ok(PreparedInt8SampledSentiment { inner: compile_sampled_request(compiler, identity,
        &Int8SampledSchedule, options, budget(), SampledSentimentLimits::default(),
        request, &mut Continue, &mut SafeBpeWork::default())?, selected_rows: None })
}
fn prepare() -> PreparedInt8SampledSentiment { compile(&request(), &options()).unwrap() }

fn synthetic(plan: &PreparedInt8SampledSentiment, axis: &PreparedAxis<Int8Work>, json: &str) -> Int8JsonRun {
    let mut tokens = fixture().0.tokenizer.tokenizer().encode_byte_fallback_only(json.as_bytes()).unwrap();
    tokens.push(plan.inner.decode.eos_token_id);
    let work = Int8Work::for_sequence(0, axis.prompt.len() + tokens.len() - 1,
        tokens.len() * NANBEIGE_VOCAB_SIZE).unwrap();
    Int8JsonRun { schema_version: 1, execution: INT8_JSON_SAMPLED_EXECUTION.to_owned(), model_work: work,
        output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
            mask_node_visit_charge: tokens.len() as u64 * MASK_CHARGE,
            token_ids: tokens, json: json.to_owned(), forward_positions: work.forward_positions,
            projected_logits: work.projected_logits } }
}
fn completed(plan: &PreparedInt8SampledSentiment) -> Int8SampledSentimentRun {
    let mut dimensions = Vec::new();
    let mut work = Int8Work::default();
    let mut tokens = 0;
    let mut masks = 0;
    for axis in &plan.inner.axes {
        let mut samples = Vec::new();
        for index in 0..plan.inner.options.samples {
            let json = format!(r#"{{"score":{},"justification":"Fixture explanation."}}"#, index as i16 * 20 - 20);
            let sample = plan.finalize_sample(axis, index, synthetic(plan, axis, &json)).unwrap();
            work = work.checked_add(sample.model_work).unwrap();
            tokens += sample.generated_tokens as u64;
            masks += sample.mask_node_visit_charge;
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

#[test]
fn checked_artifact_binding_keeps_actual_model_facts_and_strict_int8_profile() {
    let compiler = &fixture().0;
    let identity = bind_model(unbound_identity(compiler), &facts()).unwrap();
    assert_eq!(identity.source_revision, facts().revision);
    assert_eq!(identity.logical_model_digest.to_hex(), facts().logical_model_sha256);
    assert_eq!(identity.quant_recipe, facts().recipe_id);
    assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
    assert_eq!(identity.backend_semantic_version, STRICT_INT8_EXECUTION);
    assert_eq!(identity.sampler_version, JSON_SAMPLER_VERSION);
    assert_eq!(identity.template_digest, compiler.template_digest);
    assert_eq!(identity.artifact_format, "fnlpq-current-candidate-v1");
    for field in 0..6 {
        let mut changed = facts();
        match field {
            0 => changed.model_id = "other".to_owned(),
            1 => changed.revision = "floating".to_owned(),
            2 => changed.recipe_id.clear(),
            3 => changed.recipe_id = "x".repeat(257),
            4 => changed.source_root_sha256 = "invalid".to_owned(),
            _ => changed.logical_model_sha256 = "invalid".to_owned(),
        }
        assert!(bind_model(unbound_identity(compiler), &changed).is_err(), "field {field}");
    }
    let mut other = facts(); other.logical_model_sha256 = "ef".repeat(32);
    assert!(check_model(&identity, &other).is_err());
}

#[test]
fn independent_int8_schedule_charges_every_axis_sample_and_is_order_invariant() {
    let plan = prepare();
    let mut reversed = request(); reversed.axes.reverse();
    let reordered = compile(&reversed, &options()).unwrap();
    assert_eq!(plan.sample_count(), 12);
    assert_eq!(plan.execution_identity(), reordered.execution_identity());
    assert_eq!(plan.planned_work(), reordered.planned_work());
    assert_eq!(plan.inner.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(), SentimentAxis::ALL);
    let mut expected = Int8Work::default();
    for axis in &plan.inner.axes {
        let one = constrained_int8::planned_work(axis.prompt.len(), options().max_sample_tokens).unwrap();
        assert_eq!(axis.work, one);
        for _ in 0..options().samples { expected = expected.checked_add(one).unwrap(); }
    }
    assert_eq!(plan.planned_work(), expected);
    assert_eq!(plan.required_context(), plan.inner.axes.iter().map(|axis|
        axis.prompt.len() + options().max_sample_tokens - 1).max().unwrap());
    assert_eq!(plan.max_output_tokens(), 12 * 256);
    assert!(plan.preparation_requirement_bytes() >= 512 * MIB + 4 * plan.max_result_bytes());
}

#[test]
fn shared_compiler_retains_int4_identity_bytes_and_exact_prompt_semantics() {
    let (compiler, identity) = fixture();
    let int8 = prepare();
    let int4 = super::super::compile_request(compiler, &unbound_identity(compiler), 32,
        &options(), budget(), SampledSentimentLimits::default(), &request(), &mut Continue).unwrap();
    for (eight, four) in int8.inner.axes.iter().zip(&int4.axes) {
        assert_eq!(eight.axis, four.axis);
        assert_eq!(eight.prompt, four.prompt);
        assert_eq!(eight.identity.taskir_digest, four.identity.taskir_digest);
        assert_eq!(eight.identity.prompt_digest, four.identity.prompt_digest);
        assert_eq!(eight.identity.schema_digest, four.identity.schema_digest);
        assert_eq!(eight.work.forward_positions, four.work.forward_positions);
        assert_eq!(eight.work.projected_logits, four.work.projected_logits);
        assert_eq!(eight.work.attention_pairs, four.work.attention_pairs);
        assert_ne!(eight.identity.decision_policy_digest, four.identity.decision_policy_digest);
        // Reconstruct the original INT4 tuple verbatim: the shared compiler
        // must not add an Option wrapper or otherwise renumber that identity.
        let old_axis = digest(&(super::super::SAMPLED_SENTIMENT_EXECUTION,
            SAMPLED_SENTIMENT_AGGREGATION_VERSION,
            crate::native_engine::constrained_int4::INT4_JSON_SAMPLED_EXECUTION, four.axis,
            options(), 32_usize, request().budget, int4.program.max_output_bytes())).unwrap();
        assert_eq!(four.identity.decision_policy_digest, old_axis);
    }
    let old_bundle = digest(&(super::super::SAMPLED_SENTIMENT_EXECUTION,
        SAMPLED_SENTIMENT_AGGREGATION_VERSION, options(), request().budget, int4.max_result_bytes,
        32_usize, &int4.identity.taskir_digest)).unwrap();
    assert_eq!(int4.identity.decision_policy_digest, old_bundle);
    assert_eq!(int8.execution_identity().template_digest, identity.template_digest);
}

#[test]
fn seed_policy_source_and_exact_model_join_sealed_identity_without_public_seed() {
    let plan = prepare();
    plan.verify_identity(plan.execution_identity()).unwrap();
    let mut changed = options(); changed.effective_seed[0] ^= 1;
    let different = compile(&request(), &changed).unwrap();
    assert_ne!(plan.inner.identity.decision_policy_digest, different.inner.identity.decision_policy_digest);
    assert_ne!(plan.inner.axes[0].identity.decision_policy_digest, different.inner.axes[0].identity.decision_policy_digest);
    assert!(plan.verify_identity(different.execution_identity()).is_err());
    let mut changed = request(); changed.document.push('!');
    assert_ne!(plan.execution_identity().prompt_digest,
        compile(&changed, &options()).unwrap().execution_identity().prompt_digest);
    let mut drift = plan.execution_identity().clone();
    drift.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
    assert!(plan.verify_identity(&drift).is_err());
    let run = completed(&plan);
    let json = canonjson::canonical_bytes(&run).unwrap();
    let text = std::str::from_utf8(&json).unwrap();
    assert!(!text.contains("effective_seed"));
    assert!(!text.contains("group_width"));
    assert!(format!("{:?}", options()).contains("REDACTED"));
}

#[test]
fn addressed_samples_match_the_shared_axis_index_contract() {
    let mut options = options(); options.samples = MAX_SENTIMENT_SAMPLES;
    let mut seen = std::collections::BTreeSet::new();
    for axis in SentimentAxis::ALL.into_iter().rev() {
        for index in (0..MAX_SENTIMENT_SAMPLES).rev() {
            let draw = sampling(&options, axis, index).unwrap();
            assert_eq!(draw.sample_index, axis as u64 * 32 + index as u64);
            assert_eq!(draw.sample_index, options.sampling(axis, index).unwrap().sample_index);
            assert!(seen.insert(draw.sample_index));
        }
    }
    assert_eq!(seen.len(), 128);
    assert!(sampling(&options, SentimentAxis::Valence, MAX_SENTIMENT_SAMPLES).is_err());
}

#[test]
fn compact_source_is_encoded_once_and_preserves_control_free_exact_bytes() {
    let mut request = request();
    request.document = format!("{}é\r\n上海\0 <|im_start|> <|im_end|> <think>",
        "The customer is pleased with the helpful service. ".repeat(16));
    let byte = compile(&request, &options()).unwrap();
    let planner = Int8SampledSentimentPlanner::for_model_with_encoding(&facts(), options(), budget(),
        SampledSentimentLimits::default(), UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 32_768, max_output_tokens: 32_768,
            max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26,
        })).unwrap();
    let mut spent = SafeBpeWork::default();
    let compact = planner.plan_with_encoding_work(&request, &mut Continue, &mut spent).unwrap();
    let mut measured = SafeBpeWork::default();
    let document = planner.compiler.encode_document(request.document.as_bytes(), &mut measured, &mut Continue).unwrap();
    assert_eq!(spent, measured);
    assert_eq!(compact.encoding_work(), measured);
    assert_eq!(measured.input_bytes(), request.document.len());
    assert!(document.ids().len() < request.document.len());
    assert!(compact.planned_work().forward_positions < byte.planned_work().forward_positions);
    assert_eq!(compact.planned_work().projected_logits, byte.planned_work().projected_logits);
    for axis in &compact.inner.axes {
        let template = planner.compiler.axes.iter().find(|entry| entry.axis == axis.axis).unwrap();
        let start = planner.compiler.global.len() + template.instruction.len();
        let tokens = &axis.prompt[start..start + document.ids().len()];
        assert!(tokens.iter().all(|id| !planner.compiler.controls.ids().contains(id)));
        assert_eq!(planner.compiler.tokenizer.tokenizer().decode_bytes(tokens).unwrap(), request.document.as_bytes());
    }
    assert_ne!(compact.execution_identity().template_digest, byte.execution_identity().template_digest);
}

#[test]
fn aggregate_output_context_and_child_grammar_caps_fail_before_native_work() {
    let original = request();
    let mut bad = original.clone(); bad.axes.push(SentimentAxis::Valence);
    assert!(compile(&bad, &options()).is_err());
    let mut bad = original.clone(); bad.budget.max_output_tokens = 12 * 256 - 1;
    assert!(compile(&bad, &options()).is_err());
    let mut bad = original.clone(); bad.budget.max_input_tokens = 1;
    assert!(compile(&bad, &options()).is_err());
    let mut bad = original.clone(); bad.budget.max_kv_bytes = 1;
    assert!(compile(&bad, &options()).is_err());
    let mut bad = original.clone(); bad.budget.max_grammar_states = 1;
    assert!(compile(&bad, &options()).is_err());
    let mut bad = original; bad.budget.max_output_bytes = 1;
    assert!(compile(&bad, &options()).is_err());
}

#[test]
fn every_native_mask_sampler_and_resident_kv_axis_is_admitted_for_the_whole_bundle() {
    let plan = prepare();
    let work = plan.planned_work();
    let one = MASK_CHARGE * options().max_sample_tokens as u64;
    let total = one * plan.sample_count() as u64;
    let budget = Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: plan.task_budget().max_kv_bytes, max_total_mask_node_visits: total,
        mask_limits: MaskWorkLimits { max_trie_node_visits: MASK_CHARGE as usize, checkpoint_interval_nodes: 1 } } };
    let sampler = json_sampling_workspace_bytes().unwrap();
    assert_eq!(plan.preflight_bounds(MASK_CHARGE, budget, sampler).unwrap(), one);
    for field in 0..10 {
        let mut bad = budget;
        match field {
            0 => bad.native.max_forward_positions -= 1,
            1 => bad.native.max_attention_pairs -= 1,
            2 => bad.native.max_projection_work.dot_products -= 1,
            3 => bad.native.max_projection_work.multiply_accumulates -= 1,
            4 => bad.json.max_forward_positions -= 1,
            5 => bad.json.max_projected_logits -= 1,
            6 => bad.json.max_total_mask_node_visits -= 1,
            7 => bad.json.max_kv_bytes = plan.required_context() as u64 * KV_BYTES_PER_TOKEN as u64 - 1,
            8 => bad.json.mask_limits.max_trie_node_visits = 0,
            _ => bad.json.mask_limits.checkpoint_interval_nodes = 0,
        }
        assert!(plan.preflight_bounds(MASK_CHARGE, bad, sampler).is_err(), "field {field}");
    }
    assert!(plan.preflight_bounds(MASK_CHARGE, budget, sampler - 1).is_err());
    assert!(plan.preflight_bounds(0, budget, sampler).is_err());
}

#[test]
fn finalizer_rejects_distribution_greedy_profile_eos_and_native_work_mismatches() {
    let plan = prepare(); let axis = &plan.inner.axes[0];
    let good = synthetic(&plan, axis, r#"{"score":45,"justification":"Helpful wording."}"#);
    assert_eq!(plan.finalize_sample(axis, 0, good.clone()).unwrap().score, 45);
    for field in 0..8 {
        let mut bad = good.clone();
        match field {
            0 => bad.execution = constrained_int8::INT8_JSON_EXECUTION.to_owned(),
            1 => bad.output.numerics_profile = "strict-quantized-v2".to_owned(),
            2 => { bad.output.token_ids.pop(); }
            3 => bad.output.token_ids[0] = plan.inner.decode.eos_token_id,
            4 => bad.output.token_ids[0] = NANBEIGE_VOCAB_SIZE as u32,
            5 => bad.model_work.projections.multiply_accumulates += 1,
            6 => bad.output.forward_positions += 1,
            _ => bad.output.projected_logits -= 1,
        }
        assert!(plan.finalize_sample(axis, 0, bad).is_err(), "field {field}");
    }
    assert!(plan.finalize_sample(axis, options().samples, good).is_err());
    for json in [r#"{"score":101,"justification":"outside"}"#,
        r#"{"score":0.5,"justification":"fraction"}"#,
        r#"{"score":0,"justification":"   "}"#,
        r#"{"score":0,"justification":"ok","extra":1}"#,
        r#"{"score":0,"score":1,"justification":"duplicate"}"#] {
        assert!(plan.finalize_sample(axis, 0, synthetic(&plan, axis, json)).is_err(), "{json}");
    }
}

#[test]
fn complete_bundle_checks_every_sample_charge_and_shared_aggregate() {
    let plan = prepare();
    let good = completed(&plan);
    plan.verify_completed_with_control(&good, MASK_CHARGE, good.mask_node_visit_charge, &mut Continue).unwrap();
    for field in 0..18 {
        let mut bad = good.clone();
        match field {
            0 => { bad.dimensions.pop(); }
            1 => { bad.dimensions[0].samples.pop(); }
            2 => bad.dimensions[0].samples[1].sample_index = 0,
            3 => bad.dimensions[0].mean += 1.0,
            4 => bad.dimensions[0].population_stddev = f64::NAN,
            5 => bad.dimensions[0].samples[0].score = 101,
            6 => bad.dimensions[0].samples[0].justification = "  ".to_owned(),
            7 => bad.model_work.attention_pairs -= 1,
            8 => bad.generated_tokens -= 1,
            9 => bad.mask_node_visit_charge -= 1,
            10 => bad.sampler_version = "distribution".to_owned(),
            11 => bad.prompt_template_sha256 = Sha256Digest::of_bytes(b"other"),
            12 => bad.dimensions[0].axis = SentimentAxis::Arousal,
            13 => bad.dimensions[0].samples[0].model_work.projections.dot_products -= 1,
            14 => bad.dimensions[0].low_anchor.push('!'),
            15 => bad.justification_status.push('!'),
            16 => bad.samples_per_axis -= 1,
            _ => {
                bad.dimensions[0].samples[0].mask_node_visit_charge += 1;
                bad.dimensions[0].samples[1].mask_node_visit_charge -= 1;
            }
        }
        assert!(plan.verify_completed_with_control(&bad, MASK_CHARGE, u64::MAX, &mut Continue).is_err(),
            "field {field}");
    }
}

#[test]
fn complete_result_byte_cap_and_late_cancellation_never_return_partial_success() {
    let mut plan = prepare();
    let run = completed(&plan);
    let size = canonjson::canonical_bytes(&run).unwrap().len() as u64;
    plan.inner.max_result_bytes = size;
    plan.verify_completed_with_control(&run, MASK_CHARGE, run.mask_node_visit_charge, &mut Continue).unwrap();
    plan.inner.max_result_bytes -= 1;
    assert!(matches!(plan.verify_completed_with_control(&run, MASK_CHARGE, run.mask_node_visit_charge, &mut Continue),
        Err(Int8SampledSentimentError::Shared(SampledSentimentError::OutputBudget))));
    plan.inner.max_result_bytes = size;
    let mut stop = StopAfter { remaining: 6, cause: DecodeCancellationKind::Deadline };
    let error = plan.verify_completed_with_control(&run, MASK_CHARGE, run.mask_node_visit_charge, &mut stop)
        .err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}

struct StopAfter { remaining: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for StopAfter {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.remaining = self.remaining.saturating_sub(1);
        (self.remaining == 0).then_some(self.cause)
    }
}

#[test]
fn planning_and_both_native_cancellation_wrappers_preserve_all_eleven_causes() {
    let (compiler, identity) = fixture();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable,
        DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let error = compile_sampled_request(compiler, identity, &Int8SampledSchedule, &options(), budget(),
            SampledSentimentLimits::default(), &request(), &mut StopAfter { remaining: 0, cause },
            &mut SafeBpeWork::default()).err().unwrap();
        assert_eq!(Int8SampledSentimentError::from(error).cancellation(), Some(cause));
        assert_eq!(Int8SampledSentimentError::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause)))
            .cancellation(), Some(cause));
        assert_eq!(Int8SampledSentimentError::from(StrictInt8Error::Cancelled(cause)).cancellation(), Some(cause));
    }
}

mod reuse;
mod selected;
