//! Pinned preparation and admission checks; no model inference is performed.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION, strict_int8::STRICT_INT8_EXECUTION},
    tokenizer::{pinned_controls, untrusted::SafeBpeLimits},
};

const SOURCE: &str = "The quick brown fox jumps over the lazy dog. Alice met Bob in 上海. <think> <|im_start|>";
fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn planner(policy: UntrustedEncoding) -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    SourceTaskPlanner::pinned(pinned_controls::pinned().unwrap().template_controls(),
        tokenizer.eos_token_id().unwrap()).unwrap().with_document_encoding(policy).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn request(kind: BuiltInTask, source: &str) -> SourceTaskRequest {
    match kind {
        BuiltInTask::Ner => SourceTaskRequest::Ner { document: source.to_owned(), options: NerOptions::default(), budget: budget() },
        BuiltInTask::Keyphrases => SourceTaskRequest::Keyphrases { document: source.to_owned(), options: KeyphraseOptions::default(), budget: budget() },
        BuiltInTask::Summarize => SourceTaskRequest::Summarize { document: source.to_owned(), options: SummaryOptions {
            max_bullets: 2, max_bullet_scalars: 64, max_citations_per_bullet: 2, max_quote_scalars: 32 }, budget: budget() },
        BuiltInTask::Answer => SourceTaskRequest::Answer { question: "Who is named in the source?".to_owned(),
            passages: vec![AnswerPassage { id: "p1".to_owned(), text: source.to_owned() }],
            options: AnswerOptions { max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: budget() },
        _ => unreachable!(),
    }
}
#[derive(Clone, Copy, Debug)]
enum Profile { Eager, Int8, Int4 }
fn identity(planner: &SourceTaskPlanner, request: &SourceTaskRequest, profile: Profile) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"source encoding fixture");
    let mut id = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: request.task().spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d, numerics_profile: NumericsProfile::HfBf16Eager,
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None };
    match profile {
        Profile::Eager => {},
        Profile::Int8 => {
            id.quant_recipe = "int8-fixture".to_owned();
            id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
            id.backend_semantic_version = STRICT_INT8_EXECUTION.to_owned();
        }
        Profile::Int4 => {
            id.source_revision = PINNED_REVISION.to_owned();
            id.artifact_format = INT4_SOURCE_FORMAT.to_owned();
            id.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=32");
            id.packing_set_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
                INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 32_usize)).unwrap());
            id.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
            id.backend_semantic_version = STRICT_INT4_EXECUTION.to_owned();
        }
    }
    id
}
fn prepare<C: DecodeStepControl>(p: &SourceTaskPlanner, request: &SourceTaskRequest, profile: Profile,
    limits: SourcePlanningLimits, control: &mut C) -> Result<(ExecutionIdentity, usize), Option<DecodeCancellationKind>> {
    let id = identity(p, request, profile);
    let context = PlanContext::new(&id, budget()).unwrap();
    match profile {
        Profile::Eager => p.plan_with_control(request, &context, limits, control)
            .map(|plan| (plan.execution_identity().clone(), plan.prompt_tokens())).map_err(|error| error.cancellation()),
        Profile::Int8 => p.plan_int8_with_control(request, &context, limits, control)
            .map(|plan| (plan.execution_identity().clone(), plan.prompt_tokens())).map_err(|error| error.cancellation()),
        Profile::Int4 => p.plan_int4_with_control(request, &context, limits, 32, control)
            .map(|plan| (plan.execution_identity().clone(), plan.prompt_tokens())).map_err(|error| error.cancellation()),
    }
}

#[test]
fn all_source_tasks_use_actual_compact_tokens_on_each_native_profile() {
    let byte = planner(UntrustedEncoding::ByteFallback);
    let compact = planner(policy());
    for profile in [Profile::Eager, Profile::Int8, Profile::Int4] {
        for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
            let request = request(kind, SOURCE);
            let (byte_id, byte_tokens) = prepare(&byte, &request, profile, SourcePlanningLimits::default(), &mut PlanningContinue).unwrap();
            let (id, tokens) = prepare(&compact, &request, profile, SourcePlanningLimits::default(), &mut PlanningContinue).unwrap();
            assert!(tokens < byte_tokens, "{profile:?}");
            assert_eq!(id.task_spec, byte_id.task_spec);
            assert_ne!(id.template_digest, byte_id.template_digest);
            let exact = tokens + budget().max_output_tokens as usize;
            assert!(prepare(&compact, &request, profile, SourcePlanningLimits {
                max_context_tokens: exact, ..Default::default() }, &mut PlanningContinue).is_ok());
            assert!(prepare(&compact, &request, profile, SourcePlanningLimits {
                max_context_tokens: exact - 1, ..Default::default() }, &mut PlanningContinue).is_err());
        }
    }
}

#[test]
fn policy_and_work_limits_bind_even_identical_tokens_and_default_identity_is_restored() {
    let byte = planner(UntrustedEncoding::ByteFallback);
    let compact = planner(policy());
    let request = request(BuiltInTask::Ner, "x");
    let (byte_id, _) = prepare(&byte, &request, Profile::Eager, SourcePlanningLimits::default(), &mut PlanningContinue).unwrap();
    let (compact_id, _) = prepare(&compact, &request, Profile::Eager, SourcePlanningLimits::default(), &mut PlanningContinue).unwrap();
    assert_eq!(byte_id.prompt_digest, compact_id.prompt_digest);
    assert_ne!(byte_id.template_digest, compact_id.template_digest);
    let UntrustedEncoding::SafeBpeV1(mut limits) = policy() else { unreachable!() };
    limits.max_pair_checks -= 1;
    let changed = compact.with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    let (changed_id, _) = prepare(&changed, &request, Profile::Eager, SourcePlanningLimits::default(), &mut PlanningContinue).unwrap();
    assert_eq!(changed_id.prompt_digest, compact_id.prompt_digest);
    assert_ne!(changed_id.template_digest, compact_id.template_digest);
    let restored = changed.with_document_encoding(UntrustedEncoding::ByteFallback).unwrap();
    let (restored_id, _) = prepare(&restored, &request, Profile::Eager, SourcePlanningLimits::default(), &mut PlanningContinue).unwrap();
    assert_eq!(restored_id, byte_id);
    limits.max_pair_checks = 0;
    assert!(restored.with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).is_err());
}

struct Stop { calls: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls == 8).then_some(self.cause)
    }
}
#[test]
fn nested_question_encoding_preserves_every_typed_cancellation_on_every_profile() {
    let p = planner(policy());
    let request = request(BuiltInTask::Answer, SOURCE);
    use DecodeCancellationKind::*;
    for profile in [Profile::Eager, Profile::Int8, Profile::Int4] {
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let mut control = Stop { calls: 0, cause };
            assert_eq!(prepare(&p, &request, profile, SourcePlanningLimits::default(), &mut control).err(), Some(Some(cause)));
            assert_eq!(control.calls, 8);
        }
    }
}

#[test]
fn composed_map_capacity_refuses_a_compact_planner_before_chunk_probes() {
    use quantized::long::{Int8SourceMapError, Int8SourceMapLimits};
    use crate::{grammar::mask::MaskWorkLimits, native_engine::strict_int8::Int8Work,
        tasks::mapreduce::{ChunkLimits, ExecutionLimits}};
    let p = planner(policy());
    let request = request(BuiltInTask::Ner, SOURCE);
    let int4 = identity(&p, &request, Profile::Int4);
    assert!(p.int4_map_capacity_with_control(&grouped_int4::long::SourceMapTask::Ner(NerOptions::default()),
        budget(), &PlanContext::new(&int4, budget()).unwrap(), SourcePlanningLimits::default(), 32, &mut PlanningContinue).is_err());
    let int8 = identity(&p, &request, Profile::Int8);
    let task = quantized::long::SourceMapTask::Ner(NerOptions::default());
    let context = PlanContext::new(&int8, budget()).unwrap();
    assert!(p.int8_map_capacity_with_control(&task, budget(), &context, SourcePlanningLimits::default(), &mut PlanningContinue).is_err());
    let mut work = Int8Work::default();
    work.forward_positions = u64::MAX; work.projected_logits = u64::MAX; work.attention_pairs = u64::MAX;
    work.projections.dot_products = u64::MAX; work.projections.multiply_accumulates = u64::MAX;
    let limits = Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 8192,
        max_chunk_bytes: 512, max_chunk_tokens: 512, context_tokens: 8192, reserved_tokens: 1024,
        max_chunks: 64, max_tokenizer_calls: 1024 }, reduction: ExecutionLimits::default(), max_model_work: work,
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64_000 };
    let error = p.plan_int8_map_with_control(SOURCE, &task, budget(), &context,
        SourcePlanningLimits::default(), limits, &mut PlanningContinue).err().unwrap();
    assert!(matches!(error, Int8SourceMapError::Source(_)), "must refuse before the chunk tokenizer");
}

#[test]
fn population_operations_share_static_tables_but_never_share_work_allowances() {
    use crate::tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits};
    let complete = SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 };
    let factory = planner(UntrustedEncoding::ByteFallback).with_map_document_encoding(policy(), complete).unwrap();
    let first = factory.fork_map_document_operation().unwrap();
    let second = factory.fork_map_document_operation().unwrap();
    // A per-record scope must not duplicate the large pinned tokenizer.
    assert!(Arc::ptr_eq(&factory.tokenizer, &first.tokenizer));
    assert!(Arc::ptr_eq(&factory.controls, &first.controls));
    assert_eq!(factory.template_digest(), first.template_digest());
    let document = first.source_encoder().encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
    assert_eq!(document.text(), SOURCE);
    let spent = first.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(spent.input_bytes(), SOURCE.len());
    assert!(spent.output_tokens() < SOURCE.len());
    assert_eq!(second.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
    assert_eq!(factory.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
    first.source_encoder().encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
    assert_eq!(first.map_document_encoding_work().unwrap(), Some(spent.checked_add(spent).unwrap()));
    assert!(std::mem::size_of::<SourceTaskPlanner>() + 2 * std::mem::size_of::<usize>() <= 512);
}

#[test]
fn retained_continuations_cannot_renew_the_original_document_allowance() {
    use crate::tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits};
    let complete = SafeBpeWorkLimits { max_input_bytes: (SOURCE.len() * 2) as u64,
        max_output_tokens: 1 << 20, max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 };
    let owner = planner(UntrustedEncoding::ByteFallback).with_map_document_encoding(policy(), complete).unwrap();
    let retained = owner.retain_map_document_operation().unwrap();
    let separate_document = owner.fork_map_document_operation().unwrap();
    assert!(Arc::ptr_eq(&owner.tokenizer, &retained.tokenizer));
    assert!(Arc::ptr_eq(&owner.controls, &retained.controls));
    assert_eq!(owner.template_digest(), retained.template_digest());
    owner.source_encoder().encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
    let spent = owner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(retained.map_document_encoding_work().unwrap(), Some(spent));
    retained.source_encoder().encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
    let total = spent.checked_add(spent).unwrap();
    assert_eq!(owner.map_document_encoding_work().unwrap(), Some(total));
    assert_eq!(separate_document.map_document_encoding_work().unwrap(), Some(SafeBpeWork::default()));
    assert!(owner.source_encoder().encode_with_control(SOURCE, 8192, 8192, &mut || None).is_err());
    // The continuation owns the exhausted counter even after its original
    // caller and every earlier prepared stage have left scope.
    drop(owner);
    assert!(retained.source_encoder().encode_with_control(SOURCE, 8192, 8192, &mut || None).is_err());
    assert_eq!(retained.map_document_encoding_work().unwrap(), Some(total));
    separate_document.source_encoder().encode_with_control(SOURCE, 8192, 8192, &mut || None).unwrap();
    assert!(planner(policy()).retain_map_document_operation().is_err());
}

#[test]
fn retained_encoding_refuses_reentry_without_deadlock_or_refund() {
    use crate::tokenizer::untrusted::SafeBpeWorkLimits;
    let complete = SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 };
    let owner = planner(UntrustedEncoding::ByteFallback).with_map_document_encoding(policy(), complete).unwrap();
    let retained = owner.retain_map_document_operation().unwrap();
    let mut checked = false;
    let document = owner.source_encoder().encode_with_control(SOURCE, 8192, 8192, &mut || {
        if !checked {
            checked = true;
            assert!(retained.source_encoder().encode_with_control("nested", 8192, 8192, &mut || None).is_err());
        }
        None
    }).unwrap();
    assert!(checked);
    assert_eq!(document.text(), SOURCE);
    let spent = owner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(spent.input_bytes(), SOURCE.len());
    assert_eq!(retained.map_document_encoding_work().unwrap(), Some(spent));
}
