//! Host input, geometry and ownership contracts; no model inference fixtures.
use super::*;
use crate::{
    corpus::native_resolve::NativeResolveLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    validation::grounded_fields::VerifiedSourceSpan,
};
fn config() -> ResolveConfig {
    let d = Sha256Digest::of_bytes(b"hosted-resolution-fixture");
    ResolveConfig {
        identity: ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
            tokenizer_digest: d, template_digest: d, task_spec: RESOLVE_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
            grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
            host_class: None, compiler_identity: None },
        options: ResolveOptions::default(), graph: ResolveLimits::default(),
        scoring: Int8ResolveLimits { planning: NativeResolveLimits::default(),
            max_model_work: Int8Work::for_sequence(0, 4096, 1_000_000).unwrap() },
        native: NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100000, cleanup_reserve_bytes: 65536 } },
        preparation_reserve_bytes: 16 << 20, graph_reserve_bytes: 16 << 20,
    }
}
fn docs() -> Vec<ResolutionDocument> {
    vec![ResolutionDocument { id: "a".to_owned(), text: "Smith".to_owned(), mentions: vec![MentionInput {
        entity_type: "PERSON".to_owned(), surface: "Smith".to_owned(),
        span: VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 },
    }] }]
}
fn check(c: &ResolveConfig, kv: u64) -> Result<(), HostedError> {
    validate(c, c.identity.tokenizer_digest, c.identity.template_digest, kv)
}
#[test]
fn task_profile_backend_thinking_and_tools_are_never_repaired() {
    for axis in 0..7 {
        let mut c = config();
        match axis { 0 => c.identity.task_spec = "ner-v1".to_owned(),
            1 => c.identity.numerics_profile = NumericsProfile::HfBf16Eager,
            2 => c.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            3 => c.identity.backend_semantic_version = "other".to_owned(),
            4 => c.identity.kv_dtype = "int8".to_owned(), 5 => c.identity.thinking_mode = ThinkingMode::Enabled,
            _ => c.identity.tool_mode = ToolMode::Json }
        assert!(matches!(check(&c, 4096), Err(HostedError::ModelIdentity)));
    }
}
#[test]
fn both_pinned_planner_assets_must_match_the_host_identity() {
    let c = config(); let d = c.identity.tokenizer_digest; let other = Sha256Digest::of_bytes(b"other");
    validate(&c, d, d, 4096).unwrap();
    assert!(matches!(validate(&c, other, d, 4096), Err(HostedError::ModelIdentity)));
    assert!(matches!(validate(&c, d, other, 4096), Err(HostedError::ModelIdentity)));
}
#[test]
fn context_and_whole_resident_kv_are_checked_before_planning() {
    let mut c = config(); let cap = c.scoring.planning.per_head.max_kv_bytes;
    check(&c, cap).unwrap(); assert!(check(&c, cap + 1).is_err());
    c.native.context_tokens -= 1; assert!(check(&c, 4096).is_err());
}
#[test]
fn preparation_graph_and_cleanup_authority_must_be_explicit() {
    for axis in 0..5 {
        let mut c = config();
        match axis { 0 => c.preparation_reserve_bytes = 0, 1 => c.graph_reserve_bytes = 0,
            2 => c.native.run.max_elapsed = Duration::ZERO, 3 => c.native.run.max_checkpoints = 1,
            _ => c.native.run.cleanup_reserve_bytes = 0 }
        assert!(check(&c, 4096).is_err());
    }
}
#[test]
fn all_nested_vector_and_string_capacities_are_priced_without_copying() {
    let mut d = docs(); d.reserve(10); d[0].id.reserve(20); d[0].text.reserve(100);
    d[0].mentions.reserve(9); d[0].mentions[0].entity_type.reserve(40); d[0].mentions[0].surface.reserve(80);
    let expected = (d.capacity() * size_of::<ResolutionDocument>() + d[0].id.capacity() + d[0].text.capacity()
        + d[0].mentions.capacity() * size_of::<MentionInput>() + d[0].mentions[0].entity_type.capacity()
        + d[0].mentions[0].surface.capacity()) as u64;
    assert_eq!(input_payload(&d, ResolveLimits::default()).unwrap(), expected);
    assert_eq!(d[0].text, "Smith"); assert_eq!(d[0].mentions[0].surface, "Smith");
}
#[test]
fn empty_spare_storage_is_still_part_of_the_input_claim() {
    let d: Vec<ResolutionDocument> = Vec::with_capacity(32);
    assert_eq!(input_payload(&d, ResolveLimits::default()).unwrap(), (d.capacity() * size_of::<ResolutionDocument>()) as u64);
    let mut d = docs(); d[0].mentions.clear();
    let expected = (d.capacity() * size_of::<ResolutionDocument>() + d[0].id.capacity() + d[0].text.capacity()
        + d[0].mentions.capacity() * size_of::<MentionInput>()) as u64;
    assert_eq!(input_payload(&d, ResolveLimits::default()).unwrap(), expected);
}
#[test]
fn logical_limits_are_not_confused_with_physical_spare_capacity() {
    let mut d = docs(); d[0].text.reserve(10000);
    let logical = d[0].id.len() + d[0].text.len() + d[0].mentions[0].entity_type.len() + d[0].mentions[0].surface.len();
    let mut limits = ResolveLimits { max_input_bytes: logical, ..ResolveLimits::default() };
    assert!(input_payload(&d, limits).unwrap() > logical as u64);
    limits.max_input_bytes -= 1; assert!(input_payload(&d, limits).is_err());
}
#[test]
fn document_and_mention_cardinality_are_bounded_before_nested_scans() {
    let d = docs();
    assert!(input_payload(&d, ResolveLimits { max_documents: 0, ..ResolveLimits::default() }).is_err());
    assert!(input_payload(&d, ResolveLimits { max_mentions: 0, ..ResolveLimits::default() }).is_err());
}
#[test]
fn allocation_and_total_claim_arithmetic_refuse_overflow() {
    assert_eq!(allocation_bytes(0, 123).unwrap(), 0);
    assert_eq!(allocation_bytes(3, 24).unwrap(), 72);
    if usize::BITS == 64 { assert!(allocation_bytes(usize::MAX, 2).is_err()); }
    assert!(sum(&[1, u64::MAX]).is_err());
    let d = docs(); assert!(sum(&[input_payload(&d, ResolveLimits::default()).unwrap(), u64::MAX]).is_err());
}
#[test]
fn typed_resolution_cancellation_is_retained_without_private_diagnostics() {
    let e = HostedError::Resolution(Int8ResolveError::Resolution(ResolveError::Cancelled(DecodeCancellationKind::Deadline)));
    let HostedError::Resolution(ref inner) = e else { unreachable!() };
    assert_eq!(inner.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert!(std::error::Error::source(&e).is_some());
    assert_eq!(format!("{e}"), "hosted native entity resolution failed");
    assert_eq!(format!("{e:?}"), "hosted native entity resolution failed");
}
#[test]
fn owned_source_graph_inputs_and_guarded_results_cross_the_runtime_boundary() {
    fn send<T: Send + 'static>() {}
    send::<ResolveInput>(); send::<Int8ResolutionRun>(); send::<HostedOutput<Int8ResolutionRun>>();
}
