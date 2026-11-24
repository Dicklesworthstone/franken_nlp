//! Real pinned preparation and genuine empty finalization, without model weights.
use super::*;
use crate::{candidate_cli::resolve::tests::command,
    native_engine::decode::{DecodeStepControl, DecodeCancellationKind}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id:"Nanbeige4.2-3B".to_owned(),revision:"f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id:"model-free-fixture".to_owned(),source_root_sha256:"ab".repeat(32),logical_model_sha256:"cd".repeat(32) }
}
fn request(empty: bool) -> String {
    serde_json::json!({"documents":if empty { vec![] } else { vec![serde_json::json!({"id":"a","text":"é Alice"}),
        serde_json::json!({"id":"b","text":"Alice <tool_call>"})] },
        "options":{"blocking":"ascii_word_overlap","context_scalars":32,"minimum_margin_milli":1000}}).to_string()
}
fn prepare(cmd: &ResolveCommand, empty: bool) -> Result<PreparedInt8EntityCorpus, CandidateError> {
    let (_,limits) = cmd.validate()?; let raw = cmd.discovery.input(cmd,&request(empty))?;
    let (source,_) = super::super::super::source_tasks::planner()?;
    let a = super::super::super::source_tasks::source_identity(&facts(),&source,BuiltInTask::Ner)?;
    let (resolver,b) = super::super::planner(&facts())?;
    let config = cmd.discovery.config(cmd,limits,raw.ner,raw.options);
    entities_int8::prepare_int8_entities(raw.documents,Arc::new(source),a,Arc::new(resolver),b,config,&mut Continue)
        .map_err(|_| CandidateError::Planning)
}
#[test]
fn raw_cli_builds_both_real_planners_and_preflights_every_ner_pass() {
    let cmd = command(&["--discover-entities","--max-ner-tokens","64"]);
    let prepared = prepare(&cmd,false).unwrap();
    assert_eq!(prepared.document_count(),2); assert_eq!(prepared.source_identity().task_spec,"ner-v1");
    assert_eq!(prepared.resolution_identity().task_spec,"resolve-v1");
    assert_eq!(prepared.source_identity().logical_model_digest,prepared.resolution_identity().logical_model_digest);
    assert_eq!(prepared.ner_reserved_work().projected_logits,2*64*166_144);
    cmd.host.admit_plan(prepared.required_ner_context_tokens(),prepared.ner_reserved_work()).unwrap();
    assert!(prepared.finalize_without_model(&mut Continue).is_err());
}
#[test]
fn cli_empty_snapshot_is_zero_work_and_complete_result_checks_reject_substitution() {
    let cmd = command(&["--discover-entities"]); let prepared = prepare(&cmd,true).unwrap(); let expected = Expected::of(&prepared);
    let result = prepared.finalize_without_model(&mut Continue).unwrap(); check_result(&result,expected).unwrap();
    assert_eq!(result.model_work,Int8Work::default()); assert_eq!(result.resolution.head_count,0);
    assert!(check_result(&result,Expected { documents:1,..expected }).is_err());
    let json = crate::canonjson::canonical_string(&result).unwrap();
    assert!(!json.contains("prompt_digest")); assert!(!json.contains("generated_token_ids"));
}
#[test]
fn any_whole_model_axis_can_refuse_all_ner_preflight_before_weight_loading() {
    for flag in ["--max-forward-positions","--max-projected-logits","--max-attention-pairs","--max-dot-products","--max-multiply-accumulates"] {
        let cmd = command(&["--discover-entities",flag,"1"]);
        assert!(prepare(&cmd,false).is_err(),"{flag}");
    }
}

#[test]
fn team_schedule_keeps_original_source_identities_and_ner_reservations() {
    let serial = command(&["--discover-entities", "--max-ner-tokens", "64"]);
    let baseline = prepare(&serial, false).unwrap();
    let mut threaded = command(&["--discover-entities", "--max-ner-tokens", "64",
        "--compute-threads", "4", "--prefill-rows", "3", "--graph-reserve-mib", "256"]);
    threaded.host.memory_mib = 16_384;
    let schedule = threaded.prefill.limits().unwrap().unwrap();
    let prepared = prepare(&threaded, false).unwrap().with_layer_major_prefill(schedule).unwrap();
    assert_eq!(prepared.source_identity(), baseline.source_identity());
    assert_eq!(prepared.resolution_identity(), baseline.resolution_identity());
    assert_eq!(prepared.ner_reserved_work(), baseline.ner_reserved_work());
    assert_eq!(prepared.reserved_mask_visits(), baseline.reserved_mask_visits());
    assert_eq!(prepared.required_ner_context_tokens(), baseline.required_ner_context_tokens());
    assert_eq!(prepared.document_count(), baseline.document_count());
    assert_eq!(prepared.prefill_limits().unwrap().max_batch_rows, schedule.max_batch_rows);
    assert_eq!(prepared.prefill_limits().unwrap().max_extra_scratch_bytes, schedule.max_extra_scratch_bytes);
    let expected = prepare(&serial, true).unwrap().finalize_without_model(&mut Continue).unwrap();
    let actual = prepare(&threaded, true).unwrap().with_layer_major_prefill(schedule).unwrap()
        .finalize_without_model(&mut Continue).unwrap();
    assert_eq!(crate::canonjson::canonical_bytes(&actual).unwrap(),
        crate::canonjson::canonical_bytes(&expected).unwrap());
}

#[test]
#[ignore = "controller-only real INT8 artifact and fresh resource-host process; no parity claim until executed"]
fn complete_entity_results_match_across_workers_prompt_schedules_and_long_documents() {
    use crate::{execution_identity::Sha256Digest, hosted::HostedError,
        native_engine::{simd::config::process_kernel, strict_int8::prefill::Int8PrefillLimits}};
    use crate::corpus::entities_int8::long::prepare_int8_document_entities;
    let path = std::env::var_os("FNLP_INT8_ENTITY_TEAM_MODEL").expect("admitted INT8 artifact required");
    let logical = std::env::var("FNLP_INT8_ENTITY_TEAM_LOGICAL_SHA256").expect("independent logical model pin required");
    let source_root = std::env::var("FNLP_INT8_ENTITY_TEAM_SOURCE_ROOT_SHA256").expect("independent source root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let mut cmd = command(&["--discover-entities", "--chunked", "--max-ner-chunk-bytes", "8",
        "--compute-threads", "4", "--prefill-rows", "3", "--graph-reserve-mib", "256",
        "--max-projected-logits", "1000000000"]);
    cmd.host.model = path.into(); cmd.host.memory_mib = 16_384;
    let (common, limits) = cmd.validate().unwrap();
    let session = Session::new_with_prefill(&common, limits, cmd.prefill.limits().unwrap()).unwrap();
    let facts = session.facts(&common).unwrap();
    assert_eq!(facts.logical_model_sha256, logical); assert_eq!(facts.source_root_sha256, source_root);
    assert_eq!(facts.revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let model = session.load(&common, limits, &facts, CancellationToken::default()).unwrap();
    // The caller retains a separate owner for chunk preparation/witnesses;
    // every hosted call then admits its own transferred plan and graph.
    let lease = session.engine.resources().acquire_lease();
    let graph_preflight = lease.reserve(MemoryClass::JobBuffers, cmd.graph_reserve_mib * MIB)
        .unwrap().commit().unwrap();
    let (source, vocabulary) = crate::candidate_cli::runtime::source_tasks::planner().unwrap();
    let source_identity = crate::candidate_cli::runtime::source_tasks::source_identity(
        &facts, &source, BuiltInTask::Ner).unwrap();
    let (resolver, resolution_identity) = super::super::planner(&facts).unwrap();
    let source = Arc::new(source); let resolver = Arc::new(resolver); let vocabulary = Arc::new(vocabulary);
    let baseline = session.engine.resources().memory_snapshot().committed_bytes;
    let drained = || {
        let memory = session.engine.resources().memory_snapshot();
        assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
        assert_eq!(memory.recorded_leaks, 0);
        assert_eq!(session.engine.resources().outstanding_closure_snapshot().active_closures, 0);
        assert_eq!(Arc::strong_count(&source), 1); assert_eq!(Arc::strong_count(&resolver), 1);
        assert_eq!(Arc::strong_count(&vocabulary), 1);
    };
    let prefill = |rows| Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
    let short = |empty, schedule: Option<Int8PrefillLimits>| {
        let raw = cmd.discovery.input(&cmd, &request(empty)).unwrap();
        let config = cmd.discovery.config(&cmd, limits, raw.ner, raw.options);
        let prepared = entities_int8::prepare_int8_entities(raw.documents, Arc::clone(&source), source_identity.clone(),
            Arc::clone(&resolver), resolution_identity.clone(), config, &mut session.control()).unwrap();
        match schedule { Some(p) => prepared.with_layer_major_prefill(p).unwrap(), None => prepared }
    };
    let long = |empty, schedule: Option<Int8PrefillLimits>| {
        let raw = cmd.discovery.input(&cmd, &request(empty)).unwrap();
        let config = cmd.discovery.config(&cmd, limits, raw.ner, raw.options);
        let config = cmd.discovery.long.configuration(&cmd, config).unwrap();
        let prepared = prepare_int8_document_entities(raw.documents, Arc::clone(&source), source_identity.clone(),
            Arc::clone(&resolver), resolution_identity.clone(), config, &mut session.control()).unwrap();
        if !empty { assert!(prepared.chunk_count() > prepared.document_count()); }
        match schedule { Some(p) => prepared.with_layer_major_prefill(p).unwrap(), None => prepared }
    };
    for empty in [false, true] {
        let result = session.engine.execute_int8_entities(&model, short(empty, None), Arc::clone(&vocabulary),
            session.native(&common).unwrap(), limits.preparation_bytes, cmd.graph_reserve_mib * MIB,
            CancellationToken::default()).unwrap();
        let expected = crate::canonjson::canonical_bytes(result.result()).unwrap();
        let work = result.result().model_work;
        assert_eq!(work == Int8Work::default(), empty); drop(result); drained();
        for width in [1, 2, 4] { for schedule in [None, Some(prefill(1)), Some(prefill(3))] {
            if width == 1 && schedule.is_none() { continue; }
            let prepared = short(empty, schedule); let geometry = Expected::of(&prepared);
            let result = if width == 1 {
                compute::Completion::Serial(session.engine.execute_int8_entities(&model, prepared, Arc::clone(&vocabulary),
                    session.native(&common).unwrap(), limits.preparation_bytes, cmd.graph_reserve_mib * MIB,
                    CancellationToken::default()).unwrap())
            } else {
                compute::Completion::Team(session.engine.execute_int8_entities_with_threads(&model, prepared,
                    Arc::clone(&vocabulary), session.native(&common).unwrap(), limits.preparation_bytes,
                    cmd.graph_reserve_mib * MIB, process_kernel(), width, CancellationToken::default()).unwrap())
            };
            check_result(result.result(), geometry).unwrap();
            assert_eq!(crate::canonjson::canonical_bytes(result.result()).unwrap(), expected);
            assert_eq!(result.result().model_work, work);
            if let Some(receipt) = result.compute() {
                assert_eq!(receipt.compute_threads, width);
                assert_eq!(receipt.completed_rows, work.projections.dot_products);
                assert_eq!(receipt.completed_projections == 0, empty);
            }
            drop(result); drained();
        } }
        let result = session.engine.execute_int8_document_entities(&model, long(empty, None), Arc::clone(&vocabulary),
            session.native(&common).unwrap(), limits.preparation_bytes, cmd.graph_reserve_mib * MIB,
            CancellationToken::default()).unwrap();
        let expected = crate::canonjson::canonical_bytes(result.result()).unwrap();
        let work = result.result().output.model_work;
        assert_eq!(work == Int8Work::default(), empty); drop(result); drained();
        for width in [1, 2, 4] { for schedule in [None, Some(prefill(1)), Some(prefill(3))] {
            if width == 1 && schedule.is_none() { continue; }
            let prepared = long(empty, schedule);
            let result = if width == 1 {
                compute::Completion::Serial(session.engine.execute_int8_document_entities(&model, prepared,
                    Arc::clone(&vocabulary), session.native(&common).unwrap(), limits.preparation_bytes,
                    cmd.graph_reserve_mib * MIB, CancellationToken::default()).unwrap())
            } else {
                compute::Completion::Team(session.engine.execute_int8_document_entities_with_threads(&model, prepared,
                    Arc::clone(&vocabulary), session.native(&common).unwrap(), limits.preparation_bytes,
                    cmd.graph_reserve_mib * MIB, process_kernel(), width, CancellationToken::default()).unwrap())
            };
            assert_eq!(crate::canonjson::canonical_bytes(result.result()).unwrap(), expected);
            assert_eq!(result.result().output.model_work, work);
            if let Some(receipt) = result.compute() {
                assert_eq!(receipt.compute_threads, width);
                assert_eq!(receipt.completed_rows, work.projections.dot_products);
                assert_eq!(receipt.completed_projections == 0, empty);
            }
            drop(result); drained();
        } }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = session.engine.execute_int8_entities_with_threads(&model, short(true, Some(prefill(3))),
            Arc::clone(&vocabulary), session.native(&common).unwrap(), limits.preparation_bytes,
            cmd.graph_reserve_mib * MIB, process_kernel(), 4, cancellation)
            .err().expect("an empty snapshot still obeys original cancellation");
        assert!(matches!(error, HostedError::Stopped { stop, .. } if stop.kind == cause));
        drained();
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = session.engine.execute_int8_document_entities_with_threads(&model, long(true, Some(prefill(3))),
            Arc::clone(&vocabulary), session.native(&common).unwrap(), limits.preparation_bytes,
            cmd.graph_reserve_mib * MIB, process_kernel(), 4, cancellation)
            .err().expect("empty long-document discovery still obeys original cancellation");
        assert!(matches!(error, HostedError::Stopped { stop, .. } if stop.kind == cause));
        drained();
    }
    drop(vocabulary); drop(source); drop(resolver); drop(model); drop(graph_preflight); drop(lease);
    let memory = session.engine.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, limits.preparation_bytes); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
}
