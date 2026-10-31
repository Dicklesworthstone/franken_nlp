//! Static contracts only; these fixtures do not certify native model parity.
use super::*;

const MODES: [(Kind, &[&str]); 2] = [
    (Kind::Summarize, &["--synthesize-summary", "--hierarchical-summary"]),
    (Kind::Answer, &["--question", "private-question.txt", "--synthesize-answer", "--hierarchical-answer"]),
];

#[test]
fn hierarchy_grouping_is_exposed_only_on_explicit_document_hierarchies() {
    fn visit(command: &clap::Command, ancestors: &mut Vec<String>, found: &mut usize) {
        if command.get_arguments().any(|arg| arg.get_id() == "cohort_hierarchy_groups") {
            let name = command.get_name();
            let map = matches!(name, "summarize-map" | "answer-map");
            let job = matches!(name, "summarize" | "answer") && ancestors.len() >= 2
                && matches!(ancestors[ancestors.len() - 1].as_str(), "start" | "resume")
                && ancestors[ancestors.len() - 2] == "document-job";
            assert!(map || job, "hierarchy grouping escaped into {ancestors:?}/{name}");
            *found += 1;
        }
        ancestors.push(command.get_name().to_owned());
        for child in command.get_subcommands() { visit(child, ancestors, found); }
        ancestors.pop();
    }
    let mut found = 0;
    visit(&crate::candidate_cli::definition(), &mut Vec::new(), &mut found);
    assert_eq!(found, 6, "two map routes and two tasks for both start and resume");
    for (kind, mode) in MODES {
        let name = if kind == Kind::Summarize { "summarize-map" } else { "answer-map" };
        let command = definitions().into_iter().find(|c| c.get_name() == name).unwrap();
        let base = [name, "--source-dir", "source", "--source-manifest", "manifest", "--memory-mib", "262144"];
        for hierarchical in [false, true] {
            let mut words = base.to_vec();
            words.extend(mode.iter().copied().filter(|flag| hierarchical || !flag.starts_with("--hierarchical-")));
            words.extend(["--cohort-hierarchy-groups", "2"]);
            assert_eq!(command.clone().try_get_matches_from(words.clone()).is_ok(), hierarchical);
            if hierarchical {
                words.extend(["--prefill-rows", "2"]);
                assert!(command.clone().try_get_matches_from(words).is_err());
            }
        }
    }
}

#[test]
fn independent_group_width_preserves_task_authority_and_prices_phase_peak() {
    for (kind, mode) in MODES {
        let (original, mut args) = arguments(kind, mode); args.memory_mib = 262_144;
        let serial = original.validate(&args).unwrap();
        let budget = original.source.budget(&args, serial.common);
        assert!(serial.hierarchy_cohorts.is_none()); assert_eq!(serial.hierarchy_preparation_bytes, 0);
        for chunks in [None, Some(1), Some(8)] {
            for groups in [1, 4, 8] {
                let mut options = original.clone();
                options.cohort_chunks = chunks; options.cohort_hierarchy_groups = Some(groups);
                let limits = options.validate(&args).unwrap();
                let schedule = Int4HierarchyCohortSchedule { max_groups: groups, discovery_cohorts: chunks.is_some() };
                assert_eq!(limits.hierarchy_cohorts, Some(schedule));
                assert_eq!(limits.cohort_chunks, chunks);
                assert_eq!(limits.mapping.reduction.map_batch_chunks, chunks.unwrap_or(1));
                assert_eq!(limits.mapping.chunks, serial.mapping.chunks);
                assert_eq!(limits.mapping.max_model_work, serial.mapping.max_model_work);
                assert_eq!(limits.mapping.max_mask_visits, serial.mapping.max_mask_visits);
                assert_eq!(limits.mapping.mask_visits_per_chunk, serial.mapping.mask_visits_per_chunk);
                assert_eq!(limits.mapping.verification, serial.mapping.verification);
                assert_eq!(limits.hierarchy, serial.hierarchy);
                assert_eq!(limits.question_hierarchy, serial.question_hierarchy);
                assert_eq!(limits.synthesis, serial.synthesis);
                assert_eq!(limits.question_synthesis, serial.question_synthesis);
                assert_eq!(limits.common.native, serial.common.native);
                assert_eq!(options.source.budget(&args, limits.common), budget);
                assert_eq!(limits.max_result_bytes, serial.max_result_bytes);
                assert_eq!(limits.workspace_bytes, serial.workspace_bytes);
                assert_eq!(limits.hierarchy_preparation_bytes, (groups - 1) as u64 * serial.common.preparation_bytes);
                let quote = limits.cohort.unwrap();
                assert_eq!(quote.native.kv_bytes, groups.max(chunks.unwrap_or(1)) as u64 * serial.common.native.kv_bytes,
                    "sequential phases share a peak reservation, never summed KV populations");
                let mut discovery = options.clone(); discovery.cohort_hierarchy_groups = None;
                let expected = schedule.requirements(discovery.validate(&args).unwrap().cohort,
                    args.context_tokens, args.group_width, budget).unwrap();
                assert_eq!(quote, expected);
            }
        }
    }
}

#[test]
fn hierarchy_memory_covers_all_phase_owners_before_source_or_snapshot_io() {
    for (kind, mode) in MODES {
        for chunks in [None, Some(2), Some(8)] {
            for storage in 0..3 {
                let (mut options, mut args) = arguments(kind, mode);
                args.memory_mib = 262_144;
                options.cohort_hierarchy_groups = Some(4); options.cohort_chunks = chunks;
                if storage == 1 {
                    args.source_dir = None; args.source_manifest = None;
                    args.snapshot = Some("candidate.int4".into()); args.expected_model_sha256 = Some("ab".repeat(32));
                } else if storage == 2 { args.save_snapshot = Some("new-candidate.int4".into()); }
                let limits = options.validate(&args).unwrap(); let common = limits.common;
                let budget = options.source.budget(&args, common); let quote = limits.cohort.unwrap();
                let passes = limits.hierarchy.map(|h| h.max_passes).or(limits.question_hierarchy.map(|h| h.max_passes)).unwrap();
                let transcripts = passes + if kind == Kind::Summarize { limits.mapping.chunks.max_chunks } else { 0 };
                let output = (limits.max_result_bytes as u64 + 4096) * 4
                    + transcripts as u64 * u64::from(budget.max_output_tokens) * 8 + 4096;
                let private_question = if kind == Kind::Answer { question::QUESTION_BYTES as u64 + 1 } else { 0 };
                let input = args.max_input_bytes as u64 + 1 + private_question + common.preparation_bytes;
                let execution = quote.native.kv_bytes + quote.native.rope_bytes + quote.native.scratch_payload_bound
                    + ALLOCATOR_BYTES + quote.workspace_bytes + input + limits.workspace_bytes
                    + 3 * common.preparation_bytes + output;
                let loading = if storage == 1 { common.snapshot_need.scratch_bytes }
                    else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
                let saving = if storage == 2 { common.snapshot_need.scratch_bytes } else { 0 };
                let expected = common.preparation_bytes + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                    + loading.max(saving).max(execution) + CLEANUP_BYTES;
                assert_eq!(common.minimum_memory_bytes, expected);
                let durable = options.validate_document_job(&args, false).unwrap();
                assert_eq!(durable.workspace_bytes, limits.workspace_bytes);
                assert_eq!(durable.hierarchy_preparation_bytes, limits.hierarchy_preparation_bytes);
                assert_eq!(durable.common.minimum_memory_bytes,
                    common.preparation_bytes + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                    + loading.max(saving).max(execution - 4096 * 4) + CLEANUP_BYTES);
                args.memory_mib = expected.div_ceil(MIB); options.validate(&args).unwrap();
                args.memory_mib -= 1;
                assert_eq!(options.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
            }
        }
    }
}

#[test]
fn invalid_width_or_nonhierarchical_programmatic_schedule_refuses_before_io() {
    for (kind, mode) in MODES {
        for groups in [0, 17, 65, usize::MAX] {
            let (mut options, mut args) = arguments(kind, mode); args.memory_mib = 262_144;
            options.cohort_hierarchy_groups = Some(groups);
            assert_eq!(options.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
        }
    }
    for (kind, mode) in [(Kind::Ner, &[][..]), (Kind::Keyphrases, &[][..]), (Kind::Summarize, &[][..]),
        (Kind::Answer, &["--question", "private-question.txt"][..])].into_iter().chain(SYNTHESIS_MODES) {
        if mode.iter().any(|flag| flag.starts_with("--hierarchical-")) { continue; }
        let (mut options, mut args) = arguments(kind, mode); args.memory_mib = 262_144;
        options.cohort_hierarchy_groups = Some(1);
        assert_eq!(options.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
}
