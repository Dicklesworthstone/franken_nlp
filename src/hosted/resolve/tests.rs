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


fn team_prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn pair_prompt_and_worker_capacity_preserve_the_complete_native_geometry() {
    let c = config(); let required = requirements(c.native).unwrap();
    let base = required.rope_bytes + required.scratch_payload_bound + c.native.allocator_reserve_bytes;
    for width in [2, 4, 64] { for prefill in [None, Some(team_prefill(1)), Some(team_prefill(3))] {
        let (scratch, rows) = team_scratch(c.native, required, width, prefill).unwrap();
        assert_eq!(rows, prefill.map_or(1, |p| p.max_batch_rows));
        assert_eq!(scratch, base + prefill.map_or(0, |p| p.validate().unwrap()));
        let children = dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap();
        assert!(children >= dispatch::int8_team_workspace_bytes(width).unwrap());
        assert_eq!(required, requirements(c.native).unwrap());
    } }
}

#[test]
fn invalid_resolution_team_width_or_prompt_capacity_refuses_before_native_work() {
    let c = config(); let required = requirements(c.native).unwrap();
    for width in [0, 1, 65, usize::MAX] {
        assert!(team_scratch(c.native, required, width, None).is_err());
    }
    for prefill in [
        Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_extra_scratch_bytes: team_prefill(3).max_extra_scratch_bytes - 1, ..team_prefill(3) },
    ] {
        assert!(team_scratch(c.native, required, 2, Some(prefill)).is_err());
    }
    let mut native = c.native;
    native.allocator_reserve_bytes = u64::MAX;
    assert!(team_scratch(native, required, 2, Some(team_prefill(3))).is_err());
}

#[test]
fn empty_pair_native_claims_own_no_kv_and_all_team_owners_are_send() {
    let empty = TeamNativeClaims::default();
    assert!(empty.kv.is_none() && empty.scratch.is_none() && empty._kv.is_none() && empty._scratch.is_none());
    fn send<T: Send + 'static>() {}
    send::<TeamAdmission<ResolveInput>>();
    send::<HostedInt8TeamResult<Int8ResolutionRun>>();
    let _sequential = NlpEngine::resolve_int8_with_threads;
    let _prefill = NlpEngine::resolve_int8_layer_major_with_threads;
}

#[test]
#[ignore = "controller-only independently pinned INT8 artifact and fresh resource-host process; no parity claim until executed"]
fn complete_resolution_matches_serial_across_persistent_team_and_prompt_schedules() {
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset, canonjson,
        native_engine::simd::config::process_kernel, tokenizer::pinned_controls};
    fn documents(paired: bool) -> Vec<ResolutionDocument> {
        let mut documents = docs();
        let mut other = docs().pop().unwrap();
        other.id = "b".to_owned();
        if !paired {
            other.text = "Jones".to_owned(); other.mentions[0].surface = "Jones".to_owned();
        }
        documents.push(other); documents
    }
    fn bound_config(model: &ResidentInt8, planner: &ResolutionPlanner) -> ResolveConfig {
        let mut c = config(); let facts = model.artifact_identity();
        c.identity.source_revision = facts.revision.clone();
        c.identity.logical_model_digest = Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap();
        c.identity.artifact_format = "fnlpq-current-candidate-v1".to_owned();
        c.identity.quant_recipe = facts.recipe_id.clone();
        c.identity.packing_set_digest = Sha256Digest::of_bytes(
            &crate::canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap());
        c.identity.template_digest = planner.template_digest();
        c.identity.tokenizer_digest = planner.tokenizer_digest();
        c.scoring.max_model_work = Int8Work::for_sequence(0, 16_384, 100_000_000).unwrap();
        c.preparation_reserve_bytes = 512 << 20;
        c.graph_reserve_bytes = 256 << 20;
        c.native.allocator_reserve_bytes = 64 << 20;
        c.native.run.max_elapsed = Duration::from_secs(3600);
        c.native.run.max_checkpoints = 1_000_000_000;
        c
    }
    fn drained(host: &NlpEngine, baseline: u64) {
        let memory = host.resources().memory_snapshot();
        assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
        assert_eq!(memory.recorded_leaks, 0);
        assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
    }
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_RESOLVE_TEAM_MODEL").expect("admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_RESOLVE_TEAM_LOGICAL_SHA256").expect("independent logical model pin required");
    let source_root = std::env::var("FNLP_INT8_RESOLVE_TEAM_SOURCE_ROOT_SHA256").expect("independent source root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16_u64 << 30,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    let caller = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 << 20).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 },
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let planner = Arc::new(ResolutionPlanner::pinned(controls, eos).unwrap());
    let baseline = host.resources().memory_snapshot().committed_bytes;
    for paired in [false, true] {
        let serial = host.resolve_int8(&model, documents(paired), Arc::clone(&planner),
            bound_config(&model, &planner), CancellationToken::default()).unwrap();
        assert_eq!(serial.result().model_evaluated, paired);
        let expected = canonjson::canonical_bytes(serial.result()).unwrap();
        let work = serial.result().model_work;
        drop(serial); drained(&host, baseline);
        for width in [2, 4] { for schedule in [None, Some(team_prefill(1)), Some(team_prefill(3))] {
            let config = bound_config(&model, &planner);
            let result = match schedule {
                None => host.resolve_int8_with_threads(&model, documents(paired), Arc::clone(&planner),
                    config, process_kernel(), width, CancellationToken::default()),
                Some(prefill) => host.resolve_int8_layer_major_with_threads(&model, documents(paired),
                    Arc::clone(&planner), config, process_kernel(), width, prefill, CancellationToken::default()),
            }.unwrap();
            assert_eq!(canonjson::canonical_bytes(result.result()).unwrap(), expected);
            assert_eq!(result.result().model_work, work);
            assert_eq!(result.result().planned_model_work, work);
            assert_eq!(result.compute().compute_threads, width);
            assert_eq!(result.compute().completed_rows, work.projections.dot_products);
            assert_eq!(result.compute().completed_projections > 0, paired);
            assert_eq!(result.compute().completed_worker_rows > 0, paired);
            assert_eq!(Arc::strong_count(&planner), 1);
            drop(result); drained(&host, baseline);
        } }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let cancelled = CancellationToken::default(); cancelled.cancel(cause);
        let error = host.resolve_int8_layer_major_with_threads(&model, documents(false), Arc::clone(&planner),
            bound_config(&model, &planner), process_kernel(), 4, team_prefill(3), cancelled)
            .err().expect("no-pair snapshot must still observe original cancellation");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        drained(&host, baseline);
    }
    let mut foreign = bound_config(&model, &planner);
    foreign.identity.logical_model_digest = Sha256Digest::of_bytes(b"foreign model");
    assert!(matches!(host.resolve_int8_with_threads(&model, documents(false), Arc::clone(&planner),
        foreign, process_kernel(), 2, CancellationToken::default()), Err(HostedError::ModelIdentity)));
    drained(&host, baseline);
    drop(planner); drop(model); drop(caller); drop(lease); drained(&host, 0);
}
