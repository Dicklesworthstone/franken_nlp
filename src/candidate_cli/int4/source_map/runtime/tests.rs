//! Real pinned task/chunk preparation only; no fixture claims native inference.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tokenizer::pinned_controls,
};

fn fixture(kind: Kind, width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig, Option<Int4SummaryLimits>) {
    let width_arg = width.to_string();
    let command = super::super::tests::command(kind,
        &["--max-chunk-bytes", "16", "--max-new-tokens", "16", "--group-width", &width_arg]);
    let Int4Task::SourceMap(options) = command.task else { panic!("wrong source map task"); };
    let limits = options.validate(&command.args).unwrap();
    let budget = options.source.budget(&command.args, limits.common);
    let controls = pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let planner = SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap();
    let d = Sha256Digest::of_bytes(b"metadata-only-INT4-source-map-CLI-fixture");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
    )).unwrap());
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: packing, tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: source::task(kind).spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    let (_, task) = options.request(SOURCE.to_owned(), None, budget, command.args.max_input_bytes).unwrap();
    let config = Int4SourceMapConfig { identity, task, budget, planning: options.source.planning(&command.args),
        mapping: limits.mapping, native: NativeLimits { context_tokens: command.args.context_tokens,
            allocator_reserve_bytes: ALLOCATOR_BYTES, run: RunLimits { max_elapsed: Duration::from_secs(1),
                max_checkpoints: 1000, cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: limits.common.preparation_bytes, reduction_reserve_bytes: limits.reduction_reserve_bytes };
    (planner, config, limits.summary)
}
struct Control { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop_at).then_some(self.cause)
    }
}
fn continuing() -> Control { Control { calls: 0, stop_at: usize::MAX, cause: DecodeCancellationKind::PollQuota } }
const SOURCE: &str = "é\r\n上海 source <tool_call> and another paragraph\r\n";

#[test]
fn all_tasks_preflight_every_original_byte_and_keep_real_grouped_work() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize] {
        let (planner, config, summary) = fixture(kind, 32);
        let expected = preflight(SOURCE, &planner, &config, 32, summary, &mut continuing()).unwrap();
        assert!(expected.chunk_count() > 1);
        assert_eq!((expected.source_span().byte_start, expected.source_span().byte_end), (0, SOURCE.len()));
        assert_eq!((expected.source_span().scalar_start, expected.source_span().scalar_end), (0, SOURCE.chars().count()));
        assert_eq!(expected.group_width(), 32);
        let context = PlanContext::new(&config.identity, config.budget).unwrap();
        let second = planner.plan_int4_map_with_control(SOURCE, &config.task, config.budget, &context,
            config.planning, config.mapping, 32, &mut continuing()).unwrap();
        assert_eq!(second.preflight_metadata(), expected);
        assert_eq!(second.execution_identities().len(), expected.chunk_count());
        assert_eq!(expected.reserved_mask_visits(), config.mapping.mask_visits_per_chunk * expected.chunk_count() as u64);
        for identity in second.execution_identities() {
            assert_eq!(identity.task_spec, source::task(kind).spec().identity());
            assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
        }
        let (planner, config, summary) = fixture(kind, 128);
        let larger = preflight(SOURCE, &planner, &config, 128, summary, &mut continuing()).unwrap();
        assert_eq!(expected.planned_work().forward_positions, larger.planned_work().forward_positions);
        assert_eq!(expected.planned_work().projections.multiply_accumulates, larger.planned_work().projections.multiply_accumulates);
        assert!(expected.planned_work().projections.dot_products > larger.planned_work().projections.dot_products);
    }
}

#[test]
fn whole_document_native_and_mask_allowances_cannot_restart_per_chunk() {
    let (planner, mut config, _) = fixture(Kind::Ner, 32);
    let expected = preflight(SOURCE, &planner, &config, 32, None, &mut continuing()).unwrap();
    let original = config.mapping;
    for axis in 0..6 {
        config.mapping = original;
        config.mapping.max_model_work = expected.planned_work();
        config.mapping.max_mask_visits = expected.reserved_mask_visits();
        match axis {
            0 => config.mapping.max_model_work.forward_positions -= 1,
            1 => config.mapping.max_model_work.projected_logits -= 1,
            2 => config.mapping.max_model_work.attention_pairs -= 1,
            3 => config.mapping.max_model_work.projections.dot_products -= 1,
            4 => config.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => config.mapping.max_mask_visits -= 1,
        }
        assert_eq!(preflight(SOURCE, &planner, &config, 32, None, &mut continuing()), Err(CandidateError::Planning));
    }
    config.mapping.max_model_work = expected.planned_work();
    config.mapping.max_mask_visits = expected.reserved_mask_visits();
    assert_eq!(preflight(SOURCE, &planner, &config, 32, None, &mut continuing()).unwrap(), expected);
}

#[test]
fn summary_final_top_k_never_changes_the_complete_map_population() {
    let (planner, config, summary) = fixture(Kind::Summarize, 32);
    let expected = preflight(SOURCE, &planner, &config, 32, summary, &mut continuing()).unwrap();
    let mut one = summary.unwrap(); one.max_bullets = 1;
    assert_eq!(preflight(SOURCE, &planner, &config, 32, Some(one), &mut continuing()).unwrap(), expected);
    one.aggregation.max_unique_bullets = 0;
    assert_eq!(preflight(SOURCE, &planner, &config, 32, Some(one), &mut continuing()), Err(CandidateError::Planning));
    let (planner, config, _) = fixture(Kind::Ner, 32);
    assert_eq!(preflight(SOURCE, &planner, &config, 32, summary, &mut continuing()), Err(CandidateError::Planning));
}

#[test]
fn source_cannot_be_truncated_or_normalized_to_force_context_fit() {
    let (planner, mut config, summary) = fixture(Kind::Keyphrases, 32);
    assert_eq!(preflight("", &planner, &config, 32, summary, &mut continuing()), Err(CandidateError::Input));
    config.mapping.chunks.max_chunks = 1;
    assert_eq!(preflight(SOURCE, &planner, &config, 32, summary, &mut continuing()), Err(CandidateError::Planning));
    config.mapping.chunks.max_chunks = 64;
    let source = "é\r\n                 上海";
    let expected = preflight(source, &planner, &config, 32, summary, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1);
    assert_eq!(expected.source_span().byte_end, source.len());
    assert_eq!(expected.source_span().scalar_end, source.chars().count());
    config.mapping.chunks.max_input_bytes = source.len() - 1;
    assert_eq!(preflight(source, &planner, &config, 32, summary, &mut continuing()), Err(CandidateError::Planning));
}

#[test]
fn caller_cancellation_survives_the_last_preflight_checkpoint() {
    let (planner, config, summary) = fixture(Kind::Summarize, 32);
    let mut complete = continuing();
    preflight(SOURCE, &planner, &config, 32, summary, &mut complete).unwrap();
    assert!(complete.calls > 2);
    for stop_at in [1, 2, complete.calls] {
        let mut control = Control { stop_at, ..continuing() };
        assert_eq!(preflight(SOURCE, &planner, &config, 32, summary, &mut control), Err(CandidateError::Execution));
    }
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        let mut control = Control { calls: 0, stop_at: 1, cause };
        assert_eq!(preflight(SOURCE, &planner, &config, 32, summary, &mut control), Err(CandidateError::Timeout));
        let context = PlanContext::new(&config.identity, config.budget).unwrap();
        let error = planner.plan_int4_map_with_control(SOURCE, &config.task, config.budget, &context,
            config.planning, config.mapping, 32, &mut Control { calls: 0, stop_at: 1, cause }).err().unwrap();
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::SourceMap(error))), CandidateError::Timeout);
    }
    complete.stop_at = complete.calls + 1;
    assert_eq!(preflight(SOURCE, &planner, &config, 32, summary, &mut complete), Err(CandidateError::Execution));
}

#[test]
fn full_host_workspace_matches_cli_for_maps_and_final_summary() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize] {
        let (_, config, _) = fixture(kind, 128);
        for chunks in [1, config.mapping.chunks.max_chunks] {
            assert_eq!(config.workspace_bytes(chunks).unwrap(),
                workspace_bytes(config.budget, config.mapping, chunks, config.reduction_reserve_bytes).unwrap());
        }
        let command = super::super::tests::command(kind, &[]);
        let Int4Task::SourceMap(options) = command.task else { panic!("wrong task"); };
        let limits = options.validate(&command.args).unwrap();
        // Chunk byte/output-token choices may alter preparation or output
        // bytes, so compare the exact matching budget as well as maximum count.
        let mut quoted = config;
        quoted.budget = options.source.budget(&command.args, limits.common);
        quoted.mapping = limits.mapping;
        assert_eq!(quoted.workspace_bytes(limits.mapping.chunks.max_chunks).unwrap(), limits.workspace_bytes);
    }
}

#[test]
fn complete_publication_keeps_provenance_and_original_coordinates_and_refuses_oversize() {
    let output = serde_json::json!({"fixture_only":true,"source_span":{"byte_start":0,"byte_end":17,
        "scalar_start":0,"scalar_end":11},"mapped_chunks":2,"bullets":[
            {"text":"Untrusted <tool_call>","evidence":[{"chunk_id":1,"quote":"上海","byte_start":8,"byte_end":14}]}]});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-source-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "summarize",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["output"], output);
    assert_eq!(value["evidence"], "non_authoritative");
    assert_eq!(value["task"], "summarize");
    for absent in ["source_dir", "source_manifest", "options", "snapshot_path", "prompt_digest"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush failure")) }
    }
    let mut failed = FlushFailure(Vec::new());
    assert_eq!(publish(&response, 4096, &mut failed), Err(CandidateError::Output));
    assert!(!failed.0.is_empty(), "transport failure is not a successful task response");
}

fn question_fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceQuestion>, MapLimits) {
    let (planner, base, _) = fixture(Kind::Ner, width);
    let width_arg = width.to_string();
    let command = super::super::tests::command(Kind::Answer,
        &["--question", "private-question.txt", "--max-chunk-bytes", "16",
            "--max-new-tokens", "16", "--group-width", &width_arg]);
    let Int4Task::SourceMap(options) = command.task else { panic!("wrong QA dispatcher"); };
    let limits = options.validate(&command.args).unwrap();
    let task = options.question.as_ref().unwrap().task("What happened in 上海?".to_owned(), None).unwrap();
    let mut identity = base.identity;
    identity.task_spec = source::task(Kind::Answer).spec().identity();
    (planner, configuration(identity, task, &command.args, &options, limits, Duration::from_secs(1)), limits)
}

#[test]
fn document_qa_preflight_matches_every_fully_compiled_chunk_and_original_coordinate() {
    let (planner, config, _) = question_fixture(32);
    let expected = question_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1);
    assert_eq!(expected.source_span().byte_end, SOURCE.len());
    assert_eq!(expected.source_span().scalar_end, SOURCE.chars().count());
    assert_eq!(expected.native_chunks() + expected.whitespace_chunks(), expected.chunk_count());
    let context = PlanContext::new(&config.identity, config.budget).unwrap();
    let compiled = planner.plan_int4_question_with_control(SOURCE, &config.task, config.budget,
        &context, config.planning, config.mapping, 32, &mut continuing()).unwrap();
    assert_eq!(compiled.preflight_metadata(), expected);
    assert_eq!(compiled.execution_identities().len(), expected.native_chunks());
    for identity in compiled.execution_identities() {
        assert_eq!(identity.task_spec, source::task(Kind::Answer).spec().identity());
        assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
    }
    let (planner, wider, _) = question_fixture(128);
    let wide = question_preflight(SOURCE, &planner, &wider, 128, &mut continuing()).unwrap();
    assert_eq!(expected.planned_work().forward_positions, wide.planned_work().forward_positions);
    assert_eq!(expected.planned_work().projections.multiply_accumulates,
        wide.planned_work().projections.multiply_accumulates);
    assert!(expected.planned_work().projections.dot_products > wide.planned_work().projections.dot_products);
}

#[test]
fn document_qa_complete_work_and_mask_caps_cannot_restart_for_another_passage() {
    let (planner, mut config, _) = question_fixture(32);
    let expected = question_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    let original = config.mapping;
    for axis in 0..6 {
        config.mapping = original;
        config.mapping.max_model_work = expected.planned_work();
        config.mapping.max_mask_visits = expected.reserved_mask_visits();
        match axis {
            0 => config.mapping.max_model_work.forward_positions -= 1,
            1 => config.mapping.max_model_work.projected_logits -= 1,
            2 => config.mapping.max_model_work.attention_pairs -= 1,
            3 => config.mapping.max_model_work.projections.dot_products -= 1,
            4 => config.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => config.mapping.max_mask_visits -= 1,
        }
        assert_eq!(question_preflight(SOURCE, &planner, &config, 32, &mut continuing()),
            Err(CandidateError::Planning));
    }
    config.mapping.max_model_work = expected.planned_work();
    config.mapping.max_mask_visits = expected.reserved_mask_visits();
    assert_eq!(question_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap(), expected);
}

#[test]
fn document_qa_refuses_empty_source_and_preserves_typed_cancellation() {
    let (planner, config, _) = question_fixture(32);
    assert_eq!(question_preflight("", &planner, &config, 32, &mut continuing()), Err(CandidateError::Input));
    assert!(question_preflight(" \t\r\n", &planner, &config, 32, &mut continuing()).is_err());
    let mut complete = continuing();
    question_preflight(SOURCE, &planner, &config, 32, &mut complete).unwrap();
    for stop_at in [1, complete.calls] {
        assert_eq!(question_preflight(SOURCE, &planner, &config, 32,
            &mut Control { stop_at, ..continuing() }), Err(CandidateError::Execution));
    }
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(question_preflight(SOURCE, &planner, &config, 32,
            &mut Control { calls: 0, stop_at: 1, cause }), Err(CandidateError::Timeout));
    }
}

#[test]
fn qa_host_workspace_prices_both_original_passage_and_lifted_document_evidence() {
    let (_, config, limits) = question_fixture(128);
    for chunks in [1, config.mapping.chunks.max_chunks] {
        assert_eq!(config.question_workspace_bytes(chunks).unwrap(), question::workspace_bytes(
            config.budget, config.mapping, chunks, config.reduction_reserve_bytes,
            config.task.verification).unwrap());
        assert!(config.question_workspace_bytes(chunks).unwrap() > config.workspace_bytes(chunks).unwrap());
    }
    assert_eq!(config.question_workspace_bytes(config.mapping.chunks.max_chunks).unwrap(), limits.workspace_bytes);
}

#[test]
fn document_qa_publication_exports_neither_private_question_nor_question_path() {
    let output = serde_json::json!({"fixture_only":true,"source_span":{"byte_start":0,"byte_end":23,
        "scalar_start":0,"scalar_end":19},"chunks":[{"chunk_id":0,"answer":"上海",
        "evidence":[{"byte_start":7,"byte_end":13,"scalar_start":7,"scalar_end":9}]}]});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-document-qa-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "answer",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-document-qa-v1");
    assert_eq!(value["output"], output);
    for absent in ["question", "question_path", "options", "source_dir", "source_manifest", "prompt_digest"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

fn synthesis_fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceSummarySynthesis>) {
    let (planner, config, _) = fixture(Kind::Summarize, width);
    let SourceMapTask::Summarize(options) = config.task else { panic!("wrong task"); };
    let request = SourceSummarySynthesis { map_options: options, synthesis_options: options,
        limits: crate::tasks::source_planning::grouped_int4::long::summary::synthesis::SummarySynthesisLimits {
            max_evidence_segments: 256, max_evidence_bytes: 65536,
            verification: GroundingBudget::default() } };
    (planner, Int4SourceMapConfig { identity: config.identity, task: request, budget: config.budget,
        planning: config.planning, mapping: config.mapping, native: config.native,
        preparation_reserve_bytes: config.preparation_reserve_bytes,
        reduction_reserve_bytes: config.reduction_reserve_bytes })
}

#[test]
fn neural_summary_preflight_retains_complete_source_and_reserves_the_final_pass() {
    let (planner, config) = synthesis_fixture(32);
    let expected = summary_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1);
    assert_eq!(expected.source_span().byte_end, SOURCE.len());
    assert_eq!(expected.source_span().scalar_end, SOURCE.chars().count());
    assert_eq!(expected.group_width(), 32);
    let maps = expected.discovery();
    let final_pass = expected.synthesis_reserved_work();
    let total = expected.reserved_model_work().unwrap();
    assert_eq!(total.forward_positions, maps.planned_work().forward_positions + final_pass.forward_positions);
    assert_eq!(total.projected_logits, maps.planned_work().projected_logits + final_pass.projected_logits);
    assert_eq!(total.attention_pairs, maps.planned_work().attention_pairs + final_pass.attention_pairs);
    assert_eq!(total.projections.dot_products, maps.planned_work().projections.dot_products + final_pass.projections.dot_products);
    assert_eq!(total.projections.multiply_accumulates,
        maps.planned_work().projections.multiply_accumulates + final_pass.projections.multiply_accumulates);
    assert_eq!(expected.reserved_mask_visits().unwrap(),
        maps.reserved_mask_visits() + config.mapping.mask_visits_per_chunk);
    assert!(final_pass.forward_positions > 0);
    assert!(final_pass.projections.dot_products > 0);
    let again = summary_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    assert_eq!(expected, again);
}

#[test]
fn every_native_work_axis_and_mask_limit_admits_discovery_plus_synthesis_once() {
    let (planner, mut config) = synthesis_fixture(32);
    let expected = summary_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    let work = expected.reserved_model_work().unwrap();
    let masks = expected.reserved_mask_visits().unwrap();
    for axis in 0..6 {
        config.mapping.max_model_work = work;
        config.mapping.max_mask_visits = masks;
        match axis {
            0 => config.mapping.max_model_work.forward_positions -= 1,
            1 => config.mapping.max_model_work.projected_logits -= 1,
            2 => config.mapping.max_model_work.attention_pairs -= 1,
            3 => config.mapping.max_model_work.projections.dot_products -= 1,
            4 => config.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => config.mapping.max_mask_visits -= 1,
        }
        assert_eq!(summary_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()),
            Err(CandidateError::Planning));
    }
    config.mapping.max_model_work = work;
    config.mapping.max_mask_visits = masks;
    assert_eq!(summary_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap(), expected);
}

#[test]
fn final_pass_group_dots_and_host_workspace_match_the_actual_selected_profile() {
    let mut previous = None;
    for width in [16, 32, 128] {
        let (planner, config) = synthesis_fixture(width);
        let expected = summary_synthesis_preflight(SOURCE, &planner, &config, width, &mut continuing()).unwrap();
        let final_pass = expected.synthesis_reserved_work();
        if let Some(old) = previous {
            let old: Int4Work = old;
            assert!(old.projections.dot_products > final_pass.projections.dot_products);
            assert_eq!(old.forward_positions, final_pass.forward_positions);
            assert_eq!(old.projections.multiply_accumulates, final_pass.projections.multiply_accumulates);
        }
        previous = Some(final_pass);
        for chunks in [1, config.mapping.chunks.max_chunks] {
            assert_eq!(config.summary_synthesis_workspace_bytes(chunks).unwrap(),
                synthesis::workspace_bytes(config.budget, config.mapping, chunks,
                    config.reduction_reserve_bytes, config.task.limits).unwrap());
            assert!(config.summary_synthesis_workspace_bytes(chunks).unwrap() > config.workspace_bytes(chunks).unwrap());
        }
    }
}

#[test]
fn synthesis_preflight_rejects_empty_or_truncated_source_and_preserves_cancellation() {
    let (planner, mut config) = synthesis_fixture(32);
    assert_eq!(summary_synthesis_preflight("", &planner, &config, 32, &mut continuing()), Err(CandidateError::Input));
    config.mapping.chunks.max_chunks = 1;
    assert_eq!(summary_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()), Err(CandidateError::Planning));
    config.mapping.chunks.max_chunks = 64;
    for (cause, error) in [(DecodeCancellationKind::Deadline, CandidateError::Timeout),
        (DecodeCancellationKind::Timeout, CandidateError::Timeout),
        (DecodeCancellationKind::PollQuota, CandidateError::Execution),
        (DecodeCancellationKind::ResourceUnavailable, CandidateError::Execution)] {
        let mut control = Control { calls: 0, stop_at: 1, cause };
        assert_eq!(summary_synthesis_preflight(SOURCE, &planner, &config, 32, &mut control), Err(error));
    }
}

#[test]
fn synthesized_summary_publication_has_its_own_protocol_and_no_private_inputs() {
    let output = serde_json::json!({"fixture_only":true,"bullets":[{"text":"cited claim",
        "citations":[{"quote":"上海","spans":[{"byte_start":4,"byte_end":10,
            "scalar_start":3,"scalar_end":5}]}]}],"semantic_support":"not_assessed"});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-summary-synthesis-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "summarize",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-summary-synthesis-v1");
    assert_eq!(value["output"], output);
    for absent in ["document", "source", "options", "source_dir", "source_manifest", "prompt_digest"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

fn question_synthesis_fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceQuestionSynthesis>) {
    let (planner, config, _) = question_fixture(width);
    let request = SourceQuestionSynthesis { question: config.task,
        limits: crate::tasks::source_planning::grouped_int4::long::question::synthesis::QuestionSynthesisLimits {
            max_evidence_passages: 32, max_evidence_bytes: 16384 } };
    (planner, Int4SourceMapConfig { identity: config.identity, task: request, budget: config.budget,
        planning: config.planning, mapping: config.mapping, native: config.native,
        preparation_reserve_bytes: config.preparation_reserve_bytes,
        reduction_reserve_bytes: config.reduction_reserve_bytes })
}

#[test]
fn answer_synthesis_preflight_keeps_the_complete_question_population_and_final_reserve() {
    let (planner, config) = question_synthesis_fixture(32);
    let expected = question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    let discovery = expected.discovery();
    assert!(discovery.chunk_count() > 1);
    assert_eq!(discovery.source_span().byte_end, SOURCE.len());
    assert_eq!(discovery.source_span().scalar_end, SOURCE.chars().count());
    assert_eq!(discovery.group_width(), 32);
    assert!(discovery.native_chunks() > 0);
    let final_pass = expected.synthesis_reserved_work();
    let total = expected.reserved_model_work().unwrap();
    let maps = discovery.planned_work();
    assert_eq!(total.forward_positions, maps.forward_positions + final_pass.forward_positions);
    assert_eq!(total.projected_logits, maps.projected_logits + final_pass.projected_logits);
    assert_eq!(total.attention_pairs, maps.attention_pairs + final_pass.attention_pairs);
    assert_eq!(total.projections.dot_products, maps.projections.dot_products + final_pass.projections.dot_products);
    assert_eq!(total.projections.multiply_accumulates,
        maps.projections.multiply_accumulates + final_pass.projections.multiply_accumulates);
    assert_eq!(expected.reserved_mask_visits().unwrap(),
        discovery.reserved_mask_visits() + config.mapping.mask_visits_per_chunk);
    assert_eq!(question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap(), expected);
}

#[test]
fn complete_qa_and_final_answer_cannot_each_spend_the_full_work_ceiling() {
    let (planner, mut config) = question_synthesis_fixture(32);
    let expected = question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    let work = expected.reserved_model_work().unwrap();
    let masks = expected.reserved_mask_visits().unwrap();
    for axis in 0..6 {
        config.mapping.max_model_work = work;
        config.mapping.max_mask_visits = masks;
        match axis {
            0 => config.mapping.max_model_work.forward_positions -= 1,
            1 => config.mapping.max_model_work.projected_logits -= 1,
            2 => config.mapping.max_model_work.attention_pairs -= 1,
            3 => config.mapping.max_model_work.projections.dot_products -= 1,
            4 => config.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => config.mapping.max_mask_visits -= 1,
        }
        assert_eq!(question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()),
            Err(CandidateError::Planning));
    }
    config.mapping.max_model_work = work;
    config.mapping.max_mask_visits = masks;
    assert_eq!(question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap(), expected);
}

#[test]
fn question_synthesis_group_work_and_host_workspace_use_exact_finite_quotes() {
    let mut previous = None;
    for width in [16, 32, 128] {
        let (planner, config) = question_synthesis_fixture(width);
        let expected = question_synthesis_preflight(SOURCE, &planner, &config, width, &mut continuing()).unwrap();
        let final_pass = expected.synthesis_reserved_work();
        if let Some(old) = previous {
            let old: Int4Work = old;
            assert!(old.projections.dot_products > final_pass.projections.dot_products);
            assert_eq!(old.projections.multiply_accumulates, final_pass.projections.multiply_accumulates);
            assert_eq!(old.forward_positions, final_pass.forward_positions);
        }
        previous = Some(final_pass);
        for chunks in [1, config.mapping.chunks.max_chunks] {
            assert_eq!(config.question_synthesis_workspace_bytes(chunks).unwrap(),
                question::synthesis::workspace_bytes(config.budget, config.mapping, chunks,
                    config.reduction_reserve_bytes, config.task.question.verification, config.task.limits).unwrap());
            assert!(config.question_synthesis_workspace_bytes(chunks).unwrap() > config.workspace_bytes(chunks).unwrap());
        }
    }
}

#[test]
fn question_synthesis_empty_input_overflow_and_cancellation_never_become_abstention() {
    let (planner, mut config) = question_synthesis_fixture(32);
    assert_eq!(question_synthesis_preflight("", &planner, &config, 32, &mut continuing()), Err(CandidateError::Input));
    config.mapping.chunks.max_chunks = 1;
    assert_eq!(question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()), Err(CandidateError::Planning));
    config.mapping.chunks.max_chunks = 64;
    for (cause, error) in [(DecodeCancellationKind::Deadline, CandidateError::Timeout),
        (DecodeCancellationKind::Timeout, CandidateError::Timeout),
        (DecodeCancellationKind::PollQuota, CandidateError::Execution),
        (DecodeCancellationKind::ResourceUnavailable, CandidateError::Execution)] {
        let mut control = Control { calls: 0, stop_at: 1, cause };
        assert_eq!(question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut control), Err(error));
    }
}

#[test]
fn synthesized_answer_protocol_stages_complete_output_without_private_question_or_paths() {
    let output = serde_json::json!({"fixture_only":true,"synthesis":{"status":"answered","answer":"上海",
        "citations":[{"quote":"上海","spans":[{"byte_start":4,"byte_end":10,
            "scalar_start":3,"scalar_end":5}]}]},"calibration":"uncalibrated"});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-answer-synthesis-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "answer",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-answer-synthesis-v1");
    assert_eq!(value["output"], output);
    for absent in ["question", "question_path", "source", "options", "source_dir", "source_manifest", "prompt_digest"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

#[test]
fn expanded_answer_evidence_cap_reaches_cli_preflight_host_and_actual_source_planner() {
    let (planner, base) = question_synthesis_fixture(32);
    let mut command = super::super::tests::command(Kind::Answer,
        &["--question", "q", "--synthesize-answer", "--max-synthesis-passages", "128",
            "--max-new-tokens", "16", "--group-width", "32"]);
    // The explicit population cap is 128; this 64-passage source exceeds the
    // old 32 limit while fitting the native 8192-context admission ceiling.
    // The canonical manifest itself consumes byte-before-token authority.
    command.args.context_tokens = 8192;
    command.args.memory_mib = 16384;
    let Int4Task::SourceMap(options) = command.task else { panic!("wrong task"); };
    let limits = options.validate(&command.args).unwrap();
    let question = options.question.as_ref().unwrap();
    let task = question.synthesis_request(question.task("What happened?".to_owned(), None).unwrap(),
        options.planning(&command.args)).unwrap();
    let config = configuration(base.identity, task, &command.args, &options, limits, Duration::from_secs(1));
    assert_eq!(config.planning.max_passages, 128);
    assert_eq!(config.task.limits.max_evidence_passages, 128);
    assert_eq!(config.question_synthesis_workspace_bytes(config.mapping.chunks.max_chunks).unwrap(), limits.workspace_bytes);
    question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    let passages = (0..64).map(|index| crate::tasks::answer::AnswerPassage {
        id: format!("e{index}"), text: "fact".to_owned(),
    }).collect();
    let request = SourceTaskRequest::Answer { question: "What happened?".to_owned(), passages,
        options: config.task.question.options, budget: config.budget };
    let context = PlanContext::new(&config.identity, config.budget).unwrap();
    let prepared = planner.plan_int4_with_control(&request, &context, config.planning, 32, &mut continuing()).unwrap();
    assert_eq!(prepared.group_width(), 32);
    let mut ordinary = config.planning; ordinary.max_passages = shared::MAX_PASSAGES;
    assert!(planner.plan_int4_with_control(&request, &context, ordinary, 32, &mut continuing()).is_err());
}

#[test]
fn hierarchy_preflight_reserves_every_allowed_pass_before_discovery() {
    let (planner, config) = synthesis_fixture(32);
    let limits = SummaryHierarchyLimits::default();
    let single = summary_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    let expected = summary_hierarchy_preflight(SOURCE, &planner, &config, limits, 32, &mut continuing()).unwrap();
    assert_eq!(expected.group_width(), 32);
    assert_eq!(expected.source_span(), single.source_span());
    assert_eq!(expected.chunk_count(), single.chunk_count());
    assert_eq!(expected.limits(), limits);
    let maps = single.discovery().planned_work();
    let pass = single.synthesis_reserved_work();
    let count = limits.max_passes as u64;
    let total = expected.reserved_model_work();
    assert_eq!(total.forward_positions, maps.forward_positions + count * pass.forward_positions);
    assert_eq!(total.projected_logits, maps.projected_logits + count * pass.projected_logits);
    assert_eq!(total.attention_pairs, maps.attention_pairs + count * pass.attention_pairs);
    assert_eq!(total.projections.dot_products, maps.projections.dot_products + count * pass.projections.dot_products);
    assert_eq!(total.projections.multiply_accumulates,
        maps.projections.multiply_accumulates + count * pass.projections.multiply_accumulates);
    assert_eq!(expected.reserved_mask_visits(),
        single.discovery().reserved_mask_visits() + count * config.mapping.mask_visits_per_chunk);
}

#[test]
fn one_allowed_hierarchy_pass_reserves_the_same_native_work_as_explicit_single_synthesis() {
    let (planner, config) = synthesis_fixture(128);
    let hierarchy = SummaryHierarchyLimits { max_passes: 1, max_levels: 1, ..SummaryHierarchyLimits::default() };
    let single = summary_synthesis_preflight(SOURCE, &planner, &config, 128, &mut continuing()).unwrap();
    let grouped = summary_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 128, &mut continuing()).unwrap();
    assert_eq!(grouped.reserved_model_work(), single.reserved_model_work().unwrap());
    assert_eq!(grouped.reserved_mask_visits(), single.reserved_mask_visits().unwrap());
    assert_eq!(grouped.chunk_count(), single.chunk_count());
    assert_eq!(grouped.source_span(), single.source_span());
    assert_eq!(grouped.limits().max_levels, 1);
}

#[test]
fn hierarchy_cannot_renew_any_work_axis_or_masks_for_a_later_level() {
    let (planner, mut config) = synthesis_fixture(32);
    let hierarchy = SummaryHierarchyLimits { max_passes: 3, ..SummaryHierarchyLimits::default() };
    let expected = summary_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()).unwrap();
    let work = expected.reserved_model_work();
    let masks = expected.reserved_mask_visits();
    for axis in 0..6 {
        config.mapping.max_model_work = work;
        config.mapping.max_mask_visits = masks;
        match axis {
            0 => config.mapping.max_model_work.forward_positions -= 1,
            1 => config.mapping.max_model_work.projected_logits -= 1,
            2 => config.mapping.max_model_work.attention_pairs -= 1,
            3 => config.mapping.max_model_work.projections.dot_products -= 1,
            4 => config.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => config.mapping.max_mask_visits -= 1,
        }
        assert_eq!(summary_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()),
            Err(CandidateError::Planning));
    }
    config.mapping.max_model_work = work; config.mapping.max_mask_visits = masks;
    assert_eq!(summary_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()).unwrap(), expected);
}

#[test]
fn hierarchy_host_workspace_and_actual_group_dot_reservations_match_cli() {
    let hierarchy = SummaryHierarchyLimits::default();
    let mut previous = None;
    for width in [16, 32, 128] {
        let (planner, config) = synthesis_fixture(width);
        let expected = summary_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, width, &mut continuing()).unwrap();
        let total = expected.reserved_model_work();
        if let Some(old) = previous {
            let old: Int4Work = old;
            assert!(old.projections.dot_products > total.projections.dot_products);
            assert_eq!(old.projections.multiply_accumulates, total.projections.multiply_accumulates);
            assert_eq!(old.forward_positions, total.forward_positions);
        }
        previous = Some(total);
        for chunks in [1, config.mapping.chunks.max_chunks] {
            assert_eq!(config.summary_hierarchy_workspace_bytes(chunks, hierarchy).unwrap(),
                synthesis::hierarchy::workspace_bytes(config.budget, config.mapping, chunks,
                    config.reduction_reserve_bytes, config.task.limits, hierarchy).unwrap());
            assert!(config.summary_hierarchy_workspace_bytes(chunks, hierarchy).unwrap()
                > config.summary_synthesis_workspace_bytes(chunks).unwrap());
        }
    }
}

#[test]
fn default_work_caps_fit_sixty_four_max_context_maps_and_sixteen_hierarchy_passes() {
    for width in [16, 32, 128] {
        let width_arg = width.to_string();
        let command = super::super::tests::command(Kind::Summarize,
            &["--synthesize-summary", "--hierarchical-summary", "--group-width", &width_arg,
                "--max-total-mask-node-visits", "160000000000"]);
        let Int4Task::SourceMap(options) = command.task else { panic!("wrong task"); };
        let limits = options.validate(&command.args).unwrap();
        let passes = (limits.mapping.chunks.max_chunks + limits.hierarchy.unwrap().max_passes) as u64;
        assert_eq!(passes, 80);
        let per_pass = crate::native_engine::constrained_int4::planned_work(width,
            command.args.context_tokens - command.args.max_new_tokens, command.args.max_new_tokens).unwrap();
        let cap = limits.mapping.max_model_work;
        assert!(passes * per_pass.forward_positions <= cap.forward_positions);
        assert!(passes * per_pass.projected_logits <= cap.projected_logits);
        assert!(passes * per_pass.attention_pairs <= cap.attention_pairs);
        assert!(passes * per_pass.projections.dot_products <= cap.projections.dot_products);
        assert!(passes * per_pass.projections.multiply_accumulates <= cap.projections.multiply_accumulates);
        assert_eq!(passes * limits.mapping.mask_visits_per_chunk, limits.mapping.max_mask_visits);
    }
}

#[test]
fn hierarchy_preflight_refuses_empty_source_invalid_limits_and_cancellation() {
    let (planner, config) = synthesis_fixture(32);
    let hierarchy = SummaryHierarchyLimits::default();
    assert_eq!(summary_hierarchy_preflight("", &planner, &config, hierarchy, 32, &mut continuing()), Err(CandidateError::Input));
    let mut invalid = hierarchy; invalid.max_passes = 0;
    assert_eq!(summary_hierarchy_preflight(SOURCE, &planner, &config, invalid, 32, &mut continuing()), Err(CandidateError::Planning));
    for (cause, error) in [(DecodeCancellationKind::Deadline, CandidateError::Timeout),
        (DecodeCancellationKind::Timeout, CandidateError::Timeout),
        (DecodeCancellationKind::PollQuota, CandidateError::Execution),
        (DecodeCancellationKind::ResourceUnavailable, CandidateError::Execution)] {
        let mut control = Control { calls: 0, stop_at: 1, cause };
        assert_eq!(summary_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut control), Err(error));
    }
}

#[test]
fn hierarchy_publication_retains_final_pass_selection_and_all_pass_receipts() {
    let output = serde_json::json!({"fixture_only":true,"final_pass":1,"passes":[
        {"level":0,"group":0,"bullets":[{"text":"intermediate"}]},
        {"level":1,"group":0,"bullets":[{"text":"final"}]}],
        "semantics":"lossy-quote-selection-every-level-not-full-document-equivalence-v1"});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-summary-hierarchy-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "summarize",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-summary-hierarchy-v1");
    assert_eq!(value["output"], output);
    for absent in ["source", "source_dir", "source_manifest", "options", "prompt_digest"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

#[test]
fn answer_hierarchy_preflight_reserves_every_permitted_pass_with_complete_discovery() {
    let (planner, config) = question_synthesis_fixture(32);
    let hierarchy = QuestionHierarchyLimits::default();
    let single = question_synthesis_preflight(SOURCE, &planner, &config, 32, &mut continuing()).unwrap();
    let expected = question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()).unwrap();
    let discovery = expected.discovery();
    assert_eq!(discovery, single.discovery());
    assert_eq!(discovery.source_span().byte_end, SOURCE.len());
    assert_eq!(discovery.source_span().scalar_end, SOURCE.chars().count());
    assert_eq!(expected.group_width(), 32); assert_eq!(expected.limits(), hierarchy);
    let maps = discovery.planned_work(); let pass = single.synthesis_reserved_work();
    let count = hierarchy.max_passes as u64; let total = expected.reserved_model_work();
    assert_eq!(total.forward_positions, maps.forward_positions + count * pass.forward_positions);
    assert_eq!(total.projected_logits, maps.projected_logits + count * pass.projected_logits);
    assert_eq!(total.attention_pairs, maps.attention_pairs + count * pass.attention_pairs);
    assert_eq!(total.projections.dot_products, maps.projections.dot_products + count * pass.projections.dot_products);
    assert_eq!(total.projections.multiply_accumulates,
        maps.projections.multiply_accumulates + count * pass.projections.multiply_accumulates);
    assert_eq!(expected.reserved_mask_visits(),
        discovery.reserved_mask_visits() + count * config.mapping.mask_visits_per_chunk);
    assert_eq!(question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()).unwrap(), expected);
}

#[test]
fn one_answer_hierarchy_pass_keeps_the_single_synthesis_native_reservation() {
    let (planner, config) = question_synthesis_fixture(128);
    let hierarchy = QuestionHierarchyLimits { max_passes: 1, max_levels: 1, ..QuestionHierarchyLimits::default() };
    let single = question_synthesis_preflight(SOURCE, &planner, &config, 128, &mut continuing()).unwrap();
    let grouped = question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 128, &mut continuing()).unwrap();
    assert_eq!(grouped.reserved_model_work(), single.reserved_model_work().unwrap());
    assert_eq!(grouped.reserved_mask_visits(), single.reserved_mask_visits().unwrap());
    assert_eq!(grouped.discovery(), single.discovery());
    assert_eq!(grouped.limits().max_levels, 1);
}

#[test]
fn later_answer_levels_cannot_renew_any_native_work_axis_or_mask_allowance() {
    let (planner, mut config) = question_synthesis_fixture(32);
    let hierarchy = QuestionHierarchyLimits { max_passes: 3, ..QuestionHierarchyLimits::default() };
    let expected = question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()).unwrap();
    let work = expected.reserved_model_work(); let masks = expected.reserved_mask_visits();
    for axis in 0..6 {
        config.mapping.max_model_work = work; config.mapping.max_mask_visits = masks;
        match axis {
            0 => config.mapping.max_model_work.forward_positions -= 1,
            1 => config.mapping.max_model_work.projected_logits -= 1,
            2 => config.mapping.max_model_work.attention_pairs -= 1,
            3 => config.mapping.max_model_work.projections.dot_products -= 1,
            4 => config.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => config.mapping.max_mask_visits -= 1,
        }
        assert_eq!(question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()),
            Err(CandidateError::Planning));
    }
    config.mapping.max_model_work = work; config.mapping.max_mask_visits = masks;
    assert_eq!(question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()).unwrap(), expected);
}

#[test]
fn answer_hierarchy_host_workspace_and_actual_group_reservations_match_the_cli() {
    let hierarchy = QuestionHierarchyLimits::default(); let mut previous = None;
    for width in [16, 32, 128] {
        let (planner, config) = question_synthesis_fixture(width);
        let expected = question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, width, &mut continuing()).unwrap();
        let total = expected.reserved_model_work();
        if let Some(old) = previous {
            let old: Int4Work = old;
            assert!(old.projections.dot_products > total.projections.dot_products);
            assert_eq!(old.projections.multiply_accumulates, total.projections.multiply_accumulates);
            assert_eq!(old.forward_positions, total.forward_positions);
        }
        previous = Some(total);
        for chunks in [1, config.mapping.chunks.max_chunks] {
            assert_eq!(config.question_hierarchy_workspace_bytes(chunks, hierarchy).unwrap(),
                question::synthesis::hierarchy::workspace_bytes(config.budget, config.mapping, chunks,
                    config.reduction_reserve_bytes, config.task.question.verification, config.task.limits, hierarchy).unwrap());
            assert!(config.question_hierarchy_workspace_bytes(chunks, hierarchy).unwrap()
                > config.question_synthesis_workspace_bytes(chunks).unwrap());
        }
    }
}

#[test]
fn documented_answer_hierarchy_limits_cover_all_maps_and_permitted_passes() {
    for width in [16, 32, 128] {
        let width_arg = width.to_string();
        let command = super::super::tests::command(Kind::Answer,
            &["--question", "q", "--synthesize-answer", "--hierarchical-answer", "--group-width", &width_arg,
                "--max-synthesis-passages", "256", "--max-synthesis-evidence-bytes", "65536",
                "--max-total-mask-node-visits", "160000000000"]);
        let Int4Task::SourceMap(options) = command.task else { panic!("wrong task"); };
        let limits = options.validate(&command.args).unwrap();
        let passes = (limits.mapping.chunks.max_chunks + limits.question_hierarchy.unwrap().max_passes) as u64;
        assert_eq!(passes, 80); assert_eq!(options.planning(&command.args).max_passages, 256);
        let pass = crate::native_engine::constrained_int4::planned_work(width,
            command.args.context_tokens - command.args.max_new_tokens, command.args.max_new_tokens).unwrap();
        let cap = limits.mapping.max_model_work;
        assert!(passes * pass.forward_positions <= cap.forward_positions);
        assert!(passes * pass.projected_logits <= cap.projected_logits);
        assert!(passes * pass.attention_pairs <= cap.attention_pairs);
        assert!(passes * pass.projections.dot_products <= cap.projections.dot_products);
        assert!(passes * pass.projections.multiply_accumulates <= cap.projections.multiply_accumulates);
        assert_eq!(passes * limits.mapping.mask_visits_per_chunk, limits.mapping.max_mask_visits);
    }
}

#[test]
fn answer_hierarchy_invalid_source_limits_and_cancellation_never_publish_abstention() {
    let (planner, mut config) = question_synthesis_fixture(32);
    let hierarchy = QuestionHierarchyLimits::default();
    assert_eq!(question_hierarchy_preflight("", &planner, &config, hierarchy, 32, &mut continuing()), Err(CandidateError::Input));
    config.mapping.chunks.max_chunks = 1;
    assert_eq!(question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut continuing()), Err(CandidateError::Planning));
    config.mapping.chunks.max_chunks = 64;
    let mut invalid = hierarchy; invalid.max_passes = 0;
    assert_eq!(question_hierarchy_preflight(SOURCE, &planner, &config, invalid, 32, &mut continuing()), Err(CandidateError::Planning));
    for (cause, error) in [(DecodeCancellationKind::Deadline, CandidateError::Timeout),
        (DecodeCancellationKind::Timeout, CandidateError::Timeout),
        (DecodeCancellationKind::PollQuota, CandidateError::Execution),
        (DecodeCancellationKind::ResourceUnavailable, CandidateError::Execution)] {
        let mut control = Control { calls: 0, stop_at: 1, cause };
        assert_eq!(question_hierarchy_preflight(SOURCE, &planner, &config, hierarchy, 32, &mut control), Err(error));
    }
}

#[test]
fn answer_hierarchy_publication_keeps_final_index_receipts_and_private_input_boundaries() {
    let output = serde_json::json!({"fixture_only":true,"final_pass":1,"passes":[
        {"level":0,"group":0,"answer":{"status":"answered","answer":"intermediate"}},
        {"level":1,"group":0,"answer":{"status":"answered","answer":"final"}}],
        "calibration":"uncalibrated","semantic_support":"not_assessed"});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-answer-hierarchy-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "answer",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-answer-hierarchy-v1");
    assert_eq!(value["output"], output);
    for absent in ["question", "question_path", "source", "source_dir", "source_manifest", "options", "prompt_digest"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
}
