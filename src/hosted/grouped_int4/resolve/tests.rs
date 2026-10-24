//! Host admission, complete-result and ownership contracts. No native model
//! allocation or inference is performed by these fixtures.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    corpus::{native_resolve::NativeResolveLimits, resolve::MentionInput},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind,
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
    validation::grounded_fields::VerifiedSourceSpan,
};

fn config(width: usize) -> Int4ResolveConfig {
    let d = Sha256Digest::of_bytes(b"hosted-int4-resolution-geometry-fixture");
    let context = 8192;
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
        )).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: RESOLVE_VERSION.to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    let mut planning = NativeResolveLimits::default();
    planning.max_context_tokens = context;
    planning.per_head.max_kv_bytes = Int4MemoryRequirement::for_context(context, width).unwrap().kv_bytes;
    Int4ResolveConfig {
        identity, options: ResolveOptions::default(), graph: ResolveLimits::default(),
        scoring: Int4ResolveLimits { planning,
            max_model_work: Int4Work::for_sequence(width, 0, 4096, 1_000_000).unwrap() },
        native: NativeLimits { context_tokens: context, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
                cleanup_reserve_bytes: 65_536 } },
        preparation_reserve_bytes: 16 << 20, graph_reserve_bytes: 16 << 20,
    }
}

fn check(c: &Int4ResolveConfig, width: usize) -> Result<(Int4MemoryRequirement, u64), HostedError> {
    requirements(c, width, c.identity.tokenizer_digest, c.identity.template_digest)
}

#[test]
fn whole_resident_capacity_and_actual_group_are_priced_before_dispatch() {
    for width in [16, 32, 128] {
        let mut c = config(width);
        let (need, scratch) = check(&c, width).unwrap();
        assert_eq!(need, Int4MemoryRequirement::for_context(c.native.context_tokens, width).unwrap());
        assert_eq!(scratch, need.rope_bytes + need.scratch_payload_bound + c.native.allocator_reserve_bytes);
        assert_eq!(need.kv_bytes, c.scoring.planning.per_head.max_kv_bytes);
        c.scoring.planning.per_head.max_kv_bytes -= 1;
        assert!(check(&c, width).is_err());
        let c = config(width);
        assert!(check(&c, if width == 16 { 32 } else { 16 }).is_err());
    }
}

#[test]
fn task_assets_profile_backend_and_group_fields_are_never_repaired() {
    for axis in 0..10 {
        let mut c = config(32);
        match axis {
            0 => c.identity.task_spec = "ner-v1".to_owned(),
            1 => c.identity.numerics_profile = NumericsProfile::HfBf16Eager,
            2 => c.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            3 => c.identity.backend_semantic_version = "other".to_owned(),
            4 => c.identity.kv_dtype = "int8".to_owned(),
            5 => c.identity.thinking_mode = ThinkingMode::Enabled,
            6 => c.identity.tool_mode = ToolMode::Json,
            7 => c.identity.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=16"),
            8 => c.identity.packing_set_digest = Sha256Digest::of_bytes(b"foreign-layout"),
            _ => c.identity.artifact_format = "fnlpq".to_owned(),
        }
        assert!(check(&c, 32).is_err(), "axis {axis}");
    }
    let c = config(32);
    let foreign = Sha256Digest::of_bytes(b"foreign-planner");
    assert!(requirements(&c, 32, foreign, c.identity.template_digest).is_err());
    assert!(requirements(&c, 32, c.identity.tokenizer_digest, foreign).is_err());
}

#[test]
fn context_preparation_graph_allocator_and_cleanup_authority_must_be_explicit() {
    for axis in 0..10 {
        let mut c = config(32);
        match axis {
            0 => c.preparation_reserve_bytes = 0,
            1 => c.graph_reserve_bytes = 0,
            2 => c.native.allocator_reserve_bytes = 0,
            3 => c.native.allocator_reserve_bytes = u64::MAX,
            4 => c.native.run.max_elapsed = Duration::ZERO,
            5 => c.native.run.max_checkpoints = 1,
            6 => c.native.run.cleanup_reserve_bytes = 0,
            7 => c.native.context_tokens -= 1,
            8 => c.native.context_tokens = 0,
            _ => c.scoring.planning.per_head.max_output_bytes = 0,
        }
        assert!(check(&c, 32).is_err(), "axis {axis}");
    }
}

#[test]
fn every_pair_and_order_share_all_five_work_ceilings() {
    let head = Int4Work::for_sequence(32, 0, 64, 4 * crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE).unwrap();
    let total = head.checked_add(head).unwrap().checked_add(head).unwrap().checked_add(head).unwrap();
    check_work(total, total).unwrap();
    for axis in 0..5 {
        let mut cap = total;
        match axis {
            0 => cap.forward_positions -= 1,
            1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1,
            3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1,
        }
        assert!(check_work(total, cap).is_err(), "axis {axis}");
    }
    assert!(check_work(total, head).is_err());
    check_work(Int4Work::default(), Int4Work::default()).unwrap();
}

fn documents() -> Vec<ResolutionDocument> {
    vec![ResolutionDocument { id: "a".to_owned(), text: "é Smith".to_owned(), mentions: vec![MentionInput {
        entity_type: "PERSON".to_owned(), surface: "Smith".to_owned(),
        span: VerifiedSourceSpan { byte_start: 3, byte_end: 8, scalar_start: 2, scalar_end: 7 },
    }] }]
}

#[test]
fn shared_source_capacity_accounting_includes_nested_spare_storage() {
    let mut docs = documents();
    docs.reserve(8); docs[0].id.reserve(17); docs[0].text.reserve(128);
    docs[0].mentions.reserve(9); docs[0].mentions[0].surface.reserve(32); docs[0].mentions[0].entity_type.reserve(64);
    let expected = docs.capacity() * std::mem::size_of::<ResolutionDocument>() + docs[0].id.capacity()
        + docs[0].text.capacity() + docs[0].mentions.capacity() * std::mem::size_of::<MentionInput>()
        + docs[0].mentions[0].surface.capacity() + docs[0].mentions[0].entity_type.capacity();
    let limits = ResolveLimits::default();
    assert_eq!(super::super::super::resolve::input_payload(&docs, limits).unwrap(), expected as u64);
    assert!(super::super::super::resolve::input_payload(&docs,
        ResolveLimits { max_input_bytes: 1, ..limits }).is_err());
    assert!(super::super::super::resolve::input_payload(&docs,
        ResolveLimits { max_mentions: 0, ..limits }).is_err());
    assert_eq!(docs[0].text, "é Smith");
}

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

fn empty_run() -> Int4ResolutionRun {
    let docs = documents();
    let plan = ResolutionPlan::prepare(&docs, ResolveOptions::default(), ResolveLimits::default(), &mut Continue).unwrap();
    assert_eq!(plan.candidate_count(), 0);
    Int4ResolutionRun { schema_version: 1, execution: INT4_RESOLUTION_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE,
        scoring_execution: crate::native_engine::strict_int4::scoring::INT4_SCORING_EXECUTION,
        group_width: 32, model_evaluated: false, head_count: 0,
        planned_model_work: Int4Work::default(), model_work: Int4Work::default(), rewound_positions: 0,
        result: plan.finalize(Vec::new(), &mut Continue).unwrap() }
}

#[test]
fn no_pair_receipt_preserves_singletons_without_inventing_native_work() {
    let result = empty_run();
    check_result(&result, 32, 0, Int4Work::default()).unwrap();
    assert_eq!(result.result.clusters.len(), 1);
    assert!(!result.model_evaluated);
}

#[test]
fn full_result_receipts_refuse_foreign_profile_group_or_partial_work() {
    for axis in 0..10 {
        let mut result = empty_run();
        match axis {
            0 => result.schema_version = 2,
            1 => result.execution = "foreign-executor",
            2 => result.numerics_profile = "strict-quantized-v1",
            3 => result.scoring_execution = "foreign-scoring",
            4 => result.group_width = 16,
            5 => result.model_evaluated = true,
            6 => result.head_count = 1,
            7 => result.planned_model_work.projected_logits = 1,
            8 => result.model_work.projections.dot_products = 1,
            _ => result.rewound_positions = 2,
        }
        assert!(check_result(&result, 32, 0, Int4Work::default()).is_err(), "axis {axis}");
    }
}

#[test]
fn all_cancellation_kinds_survive_fixed_private_diagnostics() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = resolution_error(Int4ResolveError::Native(StrictInt4Error::Cancelled(cause)));
        let HostedError::Int4(Int4HostError::Resolution(inner)) = &error else { panic!("wrong stage"); };
        assert_eq!(inner.cancellation(), Some(cause));
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert!(error.source().is_some());
    }
}

#[test]
fn owned_source_graph_and_guarded_result_cross_the_physical_runtime_boundary() {
    fn send<T: Send + 'static>() {}
    send::<ResolveInput>(); send::<Int4ResolutionRun>(); send::<HostedOutput<Int4ResolutionRun>>();
    let _host = NlpEngine::resolve_int4;
}
