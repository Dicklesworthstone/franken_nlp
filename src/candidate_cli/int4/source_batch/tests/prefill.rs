//! Task-wide packed scheduling admission; no model or private input IO.
use super::*;

#[test]
fn every_source_stream_prices_packed_scratch_without_widening_its_task() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        let (base, mut base_args) = arguments(name, &["--cohort-documents", "3"]);
        base_args.memory_mib = 262_144;
        let ordinary = base.validate(&base_args).unwrap();
        for rows in [1, 2, 64] {
            let cap = rows.to_string();
            let (batch, mut args) = arguments(name, &["--cohort-documents", "3", "--prefill-rows", &cap]);
            args.memory_mib = 262_144;
            let admitted = batch.validate(&args).unwrap();
            let prefill = admitted.common.prefill.unwrap();
            assert_eq!(prefill.max_batch_rows, rows);
            assert_eq!(admitted.common.native, ordinary.common.native);
            assert_eq!(admitted.native.max_model_work, ordinary.native.max_model_work);
            assert_eq!(admitted.native.masks.per_mask, ordinary.native.masks.per_mask);
            assert_eq!(admitted.native.masks.max_visits_per_item, ordinary.native.masks.max_visits_per_item);
            assert_eq!(admitted.native.masks.max_visits_per_run, ordinary.native.masks.max_visits_per_run);
            assert_eq!(admitted.transport, ordinary.transport);
            assert_eq!(batch.source.budget(&args, admitted.common), base.source.budget(&base_args, ordinary.common));
            let memory = Int4SourceCohortMemoryLimits { max_documents: 3, context_tokens: args.context_tokens,
                preparation_reserve_bytes: admitted.preparation_reserve_bytes, io_reserve_bytes: admitted.io_bytes,
                allocator_reserve_bytes: ALLOCATOR_BYTES };
            let reference = memory.requirements(args.group_width, admitted.native, admitted.transport).unwrap();
            let packed = memory.requirements_packed(args.group_width, admitted.native, admitted.transport, prefill).unwrap();
            assert_eq!(packed.workspace_bytes, reference.workspace_bytes + prefill.validate().unwrap());
            let execution = packed.native.kv_bytes + packed.workspace_bytes + packed.input_bytes + packed.output_reservation_bytes;
            assert_eq!(admitted.common.minimum_memory_bytes, grouped_memory_floor(&args, admitted.common, execution).unwrap());
            args.memory_mib = admitted.common.minimum_memory_bytes.div_ceil(MIB);
            batch.validate(&args).unwrap(); args.memory_mib -= 1;
            assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
        }
    }
}

#[test]
fn each_source_task_admits_independent_prompt_morsels_with_one_fixed_cpu_team() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        for selected in [None, Some("7")] {
            let mut flags = vec!["--max-requests", "8"];
            if let Some(cap) = selected { flags.extend(["--max-selected-rows", cap]); }
            let (base, mut args) = arguments(name, &flags);
            args.memory_mib = 262_144;
            for snapshot in [false, true] {
                if snapshot {
                    args.source_dir = None; args.source_manifest = None;
                    args.snapshot = Some(PathBuf::from("private.snapshot"));
                    args.expected_model_sha256 = Some("ab".repeat(32));
                }
                let ordinary = base.validate(&args).unwrap();
                for (rows, threads) in [(1, 1), (3, 2), (64, 4), (3, 64)] {
                    let mut scheduled = args.clone();
                    let rows_text = rows.to_string(); let threads_text = threads.to_string();
                    let mut scheduled_flags = flags.clone();
                    scheduled_flags.extend(["--prefill-rows", &rows_text, "--compute-threads", &threads_text]);
                    let (batch, parsed) = arguments(name, &scheduled_flags);
                    scheduled.policy = parsed.policy;
                    scheduled.compute = parsed.compute;
                    let admitted = batch.validate(&scheduled).unwrap();
                    let prefill = admitted.common.prefill.unwrap();
                    let workers = if threads == 1 { 0 } else {
                        crate::native_engine::strict_int4::projection::int4_prefill_team_workspace_bytes(
                            threads, scheduled.group_width, rows).unwrap()
                    };
                    assert_eq!(admitted.cohort_documents, None);
                    assert_eq!(admitted.common.compute.workspace_bytes, workers);
                    let extra = prefill.validate().unwrap() + workers;
                    if snapshot {
                        assert_eq!(admitted.common.minimum_memory_bytes, ordinary.common.minimum_memory_bytes + extra);
                    } else {
                        assert!(admitted.common.minimum_memory_bytes >= ordinary.common.minimum_memory_bytes);
                        assert!(admitted.common.minimum_memory_bytes <= ordinary.common.minimum_memory_bytes + extra);
                    }
                    assert_eq!(admitted.common.native, ordinary.common.native);
                    assert_eq!(admitted.common.load, ordinary.common.load);
                    assert_eq!(admitted.native.max_model_work, ordinary.native.max_model_work);
                    assert_eq!(admitted.native.masks.max_visits_per_run, ordinary.native.masks.max_visits_per_run);
                    assert_eq!(admitted.transport, ordinary.transport);
                    assert_eq!(admitted.selected_rows, ordinary.selected_rows);
                    assert_eq!(batch.source.budget(&scheduled, admitted.common), base.source.budget(&args, ordinary.common));
                    scheduled.memory_mib = admitted.common.minimum_memory_bytes.div_ceil(MIB);
                    batch.validate(&scheduled).unwrap(); scheduled.memory_mib -= 1;
                    assert_eq!(batch.execute_owned(scheduled, NoRead, NoWrite), Err(CandidateError::Arguments));
                }
            }
        }
    }
}

#[test]
fn every_source_stream_refuses_unbounded_or_unsupported_packing_before_io() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", name, "--source-dir", "source", "--source-manifest", "manifest",
            "--memory-mib", "8192", "--prefill-rows", "1"]).is_ok());
        for rows in [0, 65, usize::MAX] {
            for cohort in [false, true] {
                let text = rows.to_string(); let mut flags = vec!["--prefill-rows", &text];
                if cohort { flags.extend(["--cohort-documents", "3"]); }
                let (batch, args) = arguments(name, &flags);
                assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
            }
        }
        for selected in [None, Some(7)] {
            let (mut batch, args) = arguments(name, &["--cohort-documents", "3", "--prefill-rows", "2", "--compute-threads", "2"]);
            batch.stream.max_selected_rows = selected;
            assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
        }
    }
}
