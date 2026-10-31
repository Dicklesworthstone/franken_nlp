//! Durable grouping keeps the complete original document as the only item.
use super::*;

#[test]
fn both_document_hierarchies_accept_independent_schedules_on_start_and_resume() {
    for &(task, mode, name) in MODES {
        if !mode.iter().any(|flag| flag.starts_with("--hierarchical-")) { continue; }
        let (serial, args) = command("start", task, mode); let original = serial.args.validate(&args).unwrap();
        let fixed = lifetime(&args, original);
        for operation in ["start", "resume"] {
            for chunks in [None, Some("2"), Some("8")] {
                let mut flags = mode.to_vec(); flags.extend(["--cohort-hierarchy-groups", "4"]);
                if let Some(chunks) = chunks { flags.extend(["--cohort-chunks", chunks]); }
                let (c, args) = command(operation, task, &flags); let map = c.args.validate(&args).unwrap();
                assert_eq!(c.mode.name(), operation); assert_eq!(c.args.mode_name(map), name);
                let schedule = map.hierarchy_cohorts.unwrap();
                assert_eq!(schedule.max_groups, 4); assert_eq!(schedule.discovery_cohorts, chunks.is_some());
                assert_eq!(map.cohort_chunks, chunks.map(|width| width.parse::<usize>().unwrap()));
                assert_eq!(map.mapping.reduction.map_batch_chunks, chunks.map_or(1, |width| width.parse::<usize>().unwrap()));
                assert_eq!(map.mapping.chunks, original.mapping.chunks);
                assert_eq!(map.mapping.max_model_work, original.mapping.max_model_work);
                assert_eq!(map.mapping.max_mask_visits, original.mapping.max_mask_visits);
                assert_eq!(map.mapping.verification, original.mapping.verification);
                assert_eq!(map.hierarchy, original.hierarchy); assert_eq!(map.question_hierarchy, original.question_hierarchy);
                assert_eq!(map.synthesis, original.synthesis); assert_eq!(map.question_synthesis, original.question_synthesis);
                assert_eq!(map.max_result_bytes, original.max_result_bytes);
                assert_eq!(map.workspace_bytes, original.workspace_bytes);
                assert_eq!(c.args.map.source.budget(&args, map.common), serial.args.map.source.budget(&args, original.common));
                let admitted = c.args.lifetime_limits(&args, map, fixed).unwrap();
                assert_eq!(admitted.map.hierarchy_cohorts, Some(schedule));
                assert_eq!(admitted.map.hierarchy_preparation_bytes, 3 * original.common.preparation_bytes);
                assert_eq!(admitted.buffer_bytes, serial.args.lifetime_limits(&args, original, fixed).unwrap().buffer_bytes);
                assert_eq!(admitted.map.common.minimum_memory_bytes, map.common.minimum_memory_bytes + admitted.buffer_bytes);
            }
        }
    }
}

#[test]
fn durable_group_peak_retains_native_plans_population_and_semantic_output_without_wrapper() {
    for &(task, mode, _) in MODES {
        if !mode.iter().any(|flag| flag.starts_with("--hierarchical-")) { continue; }
        for chunks in [None, Some("8")] {
            let mut flags = mode.to_vec(); flags.extend(["--cohort-hierarchy-groups", "4"]);
            if let Some(chunks) = chunks { flags.extend(["--cohort-chunks", chunks]); }
            let (c, mut args) = command("resume", task, &flags);
            let map = c.args.validate(&args).unwrap(); let common = map.common;
            let budget = c.args.map.source.budget(&args, common); let quote = map.cohort.unwrap();
            let passes = map.hierarchy.map(|h| h.max_passes).or(map.question_hierarchy.map(|h| h.max_passes)).unwrap();
            let transcripts = passes + if task == "summarize" { map.mapping.chunks.max_chunks } else { 0 };
            let output = map.max_result_bytes as u64 * 4
                + transcripts as u64 * u64::from(budget.max_output_tokens) * 8 + 4096;
            let question = if task == "answer" { crate::candidate_cli::source::OPTIONS_BYTES as u64 + 1 } else { 0 };
            let input = args.max_input_bytes as u64 + 1 + question + common.preparation_bytes;
            let execution = quote.native.kv_bytes + quote.native.rope_bytes + quote.native.scratch_payload_bound
                + ALLOCATOR_BYTES + quote.workspace_bytes + input + map.workspace_bytes
                + 3 * common.preparation_bytes + output;
            let loading = common.source_preparation_bytes + common.load.scratch_payload_bytes;
            let expected = common.preparation_bytes + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                + loading.max(execution) + CLEANUP_BYTES;
            assert_eq!(common.minimum_memory_bytes, expected);
            let fixed = lifetime(&args, map); let all = c.args.lifetime_limits(&args, map, fixed).unwrap();
            assert_eq!(all.map.common.minimum_memory_bytes, expected + all.buffer_bytes);
            let mut tight = map; tight.common.memory_bytes = all.map.common.minimum_memory_bytes - 1;
            assert_eq!(c.args.lifetime_limits(&args, tight, fixed).unwrap_err().code, "job_population_memory");
            tight.common.memory_bytes += 1; c.args.lifetime_limits(&args, tight, fixed).unwrap();
            args.memory_mib = expected.div_ceil(MIB); c.args.validate(&args).unwrap();
            args.memory_mib -= 1;
            assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
        }
    }
}

#[test]
fn durable_hierarchy_schedule_refuses_invalid_or_nonhierarchical_requests_before_io() {
    for &(task, mode, _) in MODES {
        let hierarchical = mode.iter().any(|flag| flag.starts_with("--hierarchical-"));
        for operation in ["start", "resume"] {
            let mut words = argv(operation, task); words.extend(mode.iter().map(|flag| (*flag).to_owned()));
            words.extend(["--cohort-hierarchy-groups", "4"].into_iter().map(str::to_owned));
            assert_eq!(parse(words).is_ok(), hierarchical);
            if !hierarchical { continue; }
            for invalid in ["0", "17", "65"] {
                let mut flags = mode.to_vec(); flags.extend(["--cohort-hierarchy-groups", invalid]);
                let (c, args) = command(operation, task, &flags);
                assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
            }
        }
    }
    for definition in source_job::definitions() {
        for operation in ["start", "resume"] {
            let mut words = vec![definition.get_name().to_owned(), operation.to_owned()];
            words.extend(["--source-dir", "sources", "--source-manifest", "manifest.json", "--memory-mib", "65536",
                "--job-dir", "private/job", "--key-file", "private/key", "--limits", "limits.json",
                "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"].into_iter().map(str::to_owned));
            assert!(definition.clone().try_get_matches_from(words.clone()).is_ok());
            words.extend(["--cohort-hierarchy-groups", "2"].into_iter().map(str::to_owned));
            assert!(definition.clone().try_get_matches_from(words).is_err());
        }
    }
}
