//! Configuration/publication fixtures, independent of model availability.
use super::*;
use crate::candidate_cli::int4::source_map::tests::{command, SYNTHESIS_MODES};

fn verify_quote<T>(config: Int4SourceMapConfig<T>, limits: MapLimits, width: usize) {
    check_synthesis_schedule(&config, limits, width).unwrap();
    let schedule = limits.hierarchy_cohorts.unwrap();
    assert_eq!(limits.hierarchy_preparation_bytes,
        (schedule.max_groups - 1) as u64 * config.preparation_reserve_bytes);
    assert_eq!(config.budget.max_kv_bytes, limits.common.native.kv_bytes);
    for axis in 0..3 {
        let mut stale = limits;
        match axis {
            0 => stale.cohort.as_mut().unwrap().native.kv_bytes -= 1,
            1 => stale.cohort.as_mut().unwrap().workspace_bytes -= 1,
            _ => stale.hierarchy_preparation_bytes -= 1,
        }
        assert_eq!(check_synthesis_schedule(&config, stale, width), Err(CandidateError::Memory));
    }
    let mut stale = limits;
    stale.hierarchy_cohorts.as_mut().unwrap().discovery_cohorts = !schedule.discovery_cohorts;
    assert_eq!(check_synthesis_schedule(&config, stale, width), Err(CandidateError::Memory));
}

#[test]
fn both_hierarchies_recheck_exact_native_decoder_and_preparation_quotes() {
    for width in [32, 128] {
        for chunks in [None, Some("2"), Some("8")] {
            for (kind, mode) in SYNTHESIS_MODES {
                if !mode.iter().any(|flag| flag.starts_with("--hierarchical-")) { continue; }
                let width_arg = width.to_string(); let mut flags = mode.to_vec();
                flags.extend(["--group-width", &width_arg, "--cohort-hierarchy-groups", "4",
                    "--max-chunk-bytes", "16", "--max-new-tokens", "16"]);
                if let Some(chunks) = chunks { flags.extend(["--cohort-chunks", chunks]); }
                let mut command = command(kind, &flags); command.args.memory_mib = 262_144;
                let Int4Task::SourceMap(options) = command.task else { panic!("wrong hierarchy task"); };
                let limits = options.validate(&command.args).unwrap();
                if kind == Kind::Summarize {
                    let (_, base) = synthesis_fixture(width);
                    let task = options.summary.unwrap().synthesis.request(base.task.map_options,
                        options.planning(&command.args)).unwrap();
                    let config = configuration(base.identity, task, &command.args, &options, limits, Duration::from_secs(1));
                    assert_eq!(config.summary_hierarchy_workspace_bytes(limits.mapping.chunks.max_chunks,
                        limits.hierarchy.unwrap()).unwrap(), limits.workspace_bytes);
                    verify_quote(config, limits, width);
                } else {
                    let (_, base) = question_synthesis_fixture(width);
                    let question = options.question.as_ref().unwrap();
                    let task = question.synthesis_request(question.task("  Private exact question?\r\n".to_owned(), None).unwrap(),
                        options.planning(&command.args)).unwrap();
                    let config = configuration(base.identity, task, &command.args, &options, limits, Duration::from_secs(1));
                    assert_eq!(config.task.question.question, "  Private exact question?\r\n");
                    assert_eq!(config.question_hierarchy_workspace_bytes(limits.mapping.chunks.max_chunks,
                        limits.question_hierarchy.unwrap()).unwrap(), limits.workspace_bytes);
                    verify_quote(config, limits, width);
                }
            }
        }
    }
}

#[test]
fn independent_phase_metadata_preserves_the_original_hierarchy_receipt() {
    let document = serde_json::json!({"fixture_only":true,"passes":[],"final_pass":null,
        "semantic_support":"not_assessed"});
    let discovery = Int4SourceMapCohortStats::default();
    let hierarchy = Int4HierarchyCohortStats::default();
    for (task, protocol) in [("summarize", "fnlp-candidate-int4-summary-hierarchy-v1"),
        ("answer", "fnlp-candidate-int4-answer-hierarchy-v1")] {
        let provenance = Provenance { protocol, schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
            model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
            quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
        let serial = serde_json::to_value(Response { provenance: &provenance, output: &document }).unwrap();
        for chunks in [None, Some(2)] {
            let schedule = Int4HierarchyCohortSchedule { max_groups: 4, discovery_cohorts: chunks.is_some() };
            let physical_discovery = chunks.map(|_| &discovery);
            let mut bytes = Vec::new();
            publish_hierarchy_cohorts(&provenance, &document, physical_discovery, &hierarchy,
                chunks, schedule, None, 4096, &mut bytes, &|| Ok(())).unwrap();
            let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let ledger = value.as_object_mut().unwrap().remove("hierarchy_cohort_schedule").unwrap();
            assert_eq!(ledger["execution"], "within_level_constrained_hierarchy_cohorts");
            assert_eq!(ledger["declared_max_groups"], 4);
            assert_eq!(ledger["physical"], serde_json::to_value(hierarchy).unwrap());
            let discovery_ledger = value.as_object_mut().unwrap().remove("cohort_schedule");
            assert_eq!(discovery_ledger.is_some(), chunks.is_some());
            if let Some(ledger) = discovery_ledger {
                assert_eq!(ledger["execution"], "cross_chunk_constrained_discovery_cohorts");
                assert_eq!(ledger["declared_max_chunks"], 2);
                assert_eq!(ledger["physical"], serde_json::to_value(discovery).unwrap());
            }
            assert_eq!(value, serial);
            let mut refused = Vec::new();
            assert_eq!(publish_hierarchy_cohorts(&provenance, &document, physical_discovery, &hierarchy,
                chunks, schedule, None, 1, &mut refused, &|| Ok(())), Err(CandidateError::Output));
            assert!(refused.is_empty());
            assert_eq!(publish_hierarchy_cohorts(&provenance, &document, physical_discovery, &hierarchy,
                chunks, Int4HierarchyCohortSchedule { discovery_cohorts: !schedule.discovery_cohorts, ..schedule },
                None, 4096, &mut refused, &|| Ok(())), Err(CandidateError::Execution));
            assert!(refused.is_empty());
            assert_eq!(publish_hierarchy_cohorts(&provenance, &document, physical_discovery, &hierarchy,
                chunks, schedule, None, 4096, &mut refused, &|| Err(CandidateError::Timeout)), Err(CandidateError::Timeout));
            assert!(refused.is_empty());
        }
    }
}

#[test]
fn compact_neural_publication_reports_retained_work_across_every_schedule() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};
    let (planner, _, _) = fixture(Kind::Summarize, 32);
    let planner = planner.with_map_document_encoding(UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
    }), SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }).unwrap();
    let operation = planner.retain_map_document_operation().unwrap();
    for _ in 0..2 {
        operation.source_encoder().encode_with_control(SOURCE, 4096, 4096, &mut || None).unwrap();
    }
    drop(operation);
    let work = planner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(work.input_bytes(), 2 * SOURCE.len());
    let report = encoding::report(&planner).unwrap();
    let encoded = serde_json::to_value(report.as_ref().unwrap()).unwrap();
    let document = serde_json::json!({"fixture_only":true,"semantic_support":"not_assessed"});
    let discovery = Int4SourceMapCohortStats::default();
    let hierarchy = Int4HierarchyCohortStats::default();
    let compute = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1",
        compute_threads: 2, completed_projections: 3, completed_rows: 8, completed_worker_rows: 4 };
    for (task, protocol) in [
        ("summarize", "fnlp-candidate-int4-summary-synthesis-v1"),
        ("summarize", "fnlp-candidate-int4-summary-hierarchy-v1"),
        ("answer", "fnlp-candidate-int4-answer-synthesis-v1"),
        ("answer", "fnlp-candidate-int4-answer-hierarchy-v1"),
    ] {
        let provenance = Provenance { protocol, schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
            model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
            quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
        let expected = serde_json::to_value(Response { provenance: &provenance, output: &document }).unwrap();
        for route in 0..5 {
            for (max_bytes, cancelled) in [(8192, false), (1, false), (8192, true)] {
                let check = || if cancelled { Err(CandidateError::Timeout) } else { Ok(()) };
                let mut bytes = Vec::new();
                let result = match route {
                    0 => publish_map(&provenance, &document, None, &planner, max_bytes, &mut bytes, &check),
                    1 => publish_discovery_cohort(&provenance, &document, &discovery, 2,
                        report.as_ref(), max_bytes, &mut bytes, &check),
                    2 | 3 => publish_hierarchy_cohorts(&provenance, &document,
                        (route == 3).then_some(&discovery), &hierarchy, (route == 3).then_some(2),
                        Int4HierarchyCohortSchedule { max_groups: 2, discovery_cohorts: route == 3 },
                        report.as_ref(), max_bytes, &mut bytes, &check),
                    _ => publish_team_map(&provenance, &document, &compute, &planner,
                        max_bytes, &mut bytes, &check),
                };
                if cancelled || max_bytes == 1 {
                    assert_eq!(result, Err(if cancelled { CandidateError::Timeout } else { CandidateError::Output }));
                    assert!(bytes.is_empty());
                    continue;
                }
                result.unwrap();
                let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                let fields = value.as_object_mut().unwrap();
                assert_eq!(fields.remove("input_encoding").unwrap(), encoded);
                assert_eq!(fields.remove("cohort_schedule").is_some(), route == 1 || route == 3);
                assert_eq!(fields.remove("hierarchy_cohort_schedule").is_some(), route == 2 || route == 3);
                assert_eq!(fields.remove("compute").is_some(), route == 4);
                assert_eq!(value, expected);
                assert!(!String::from_utf8(bytes).unwrap().contains(SOURCE));
            }
        }
    }
    assert_eq!(planner.map_document_encoding_work().unwrap(), Some(work));
}

#[test]
fn runtime_revalidates_hierarchy_width_before_reading_private_inputs() {
    struct NoRead;
    impl std::io::Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("invalid hierarchy reached input"); }
    }
    for (kind, mode) in SYNTHESIS_MODES {
        if !mode.iter().any(|flag| flag.starts_with("--hierarchical-")) { continue; }
        let mut flags = mode.to_vec(); flags.extend(["--cohort-hierarchy-groups", "4"]);
        let mut command = command(kind, &flags); command.args.memory_mib = 262_144;
        let Int4Task::SourceMap(mut options) = command.task else { panic!("wrong hierarchy task"); };
        let admitted = options.validate(&command.args).unwrap();
        options.cohort_hierarchy_groups = Some(0);
        let mut bytes = Vec::new();
        assert_eq!(execute(command.args, options, admitted, &mut NoRead, &mut bytes), Err(CandidateError::Arguments));
        assert!(bytes.is_empty());
    }
}
