use super::*;
use crate::tokenizer::untrusted::UntrustedEncoding;

#[test]
fn all_source_streams_price_one_compact_encoder_for_serial_team_and_packed_cohorts() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        for schedule in [
            &[][..], &["--compute-threads", "3"][..],
            &["--cohort-documents", "3"][..],
            &["--cohort-documents", "3", "--prefill-rows", "2"][..],
        ] {
            for selection in [None, Some("7")] {
                let mut flags = schedule.to_vec();
                if let Some(cap) = selection { flags.extend(["--max-selected-rows", cap]); }
                let (byte_batch, mut byte_args) = arguments(name, &flags);
                byte_args.memory_mib = 262_144;
                let byte = byte_batch.validate(&byte_args).unwrap();
                flags.extend(["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "123456",
                    "--max-tokenizer-pair-bytes", "2345678"]);
                let (batch, mut args) = arguments(name, &flags);
                args.memory_mib = 262_144;
                let admitted = batch.validate(&args).unwrap();
                let policy = args.input_encoding.policy(args.max_input_bytes, admitted.common.max_prompt_tokens).unwrap();
                let UntrustedEncoding::SafeBpeV1(limits) = &policy else { panic!("compact policy was lost"); };
                assert_eq!(limits.max_pair_checks, 123456);
                assert_eq!(limits.max_pair_bytes, 2345678);
                assert_eq!(limits.max_input_bytes, args.max_input_bytes);
                assert_eq!(limits.max_output_tokens, admitted.common.max_prompt_tokens);
                assert_eq!(admitted.encoding_workspace_bytes, limits.requirements().unwrap().workspace_bytes);
                assert_eq!(byte.encoding_workspace_bytes, 0);
                assert_eq!(admitted.common.native, byte.common.native);
                assert_eq!(admitted.transport, byte.transport);
                assert_eq!(admitted.native.max_model_work, byte.native.max_model_work);
                assert_eq!(admitted.native.masks.max_visits_per_run, byte.native.masks.max_visits_per_run);
                assert_eq!(admitted.selected_rows, byte.selected_rows);
                assert_eq!(batch.source.budget(&args, admitted.common), byte_batch.source.budget(&byte_args, byte.common));
                if let Some(documents) = admitted.cohort_documents {
                    let quote = Int4SourceCohortMemoryLimits { max_documents: documents,
                        context_tokens: args.context_tokens, preparation_reserve_bytes: admitted.preparation_reserve_bytes,
                        io_reserve_bytes: admitted.io_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES };
                    let baseline = quote.requirements_for_execution(args.group_width, admitted.native, admitted.transport,
                        admitted.selected_rows, args.prefill().unwrap()).unwrap();
                    let compact = quote.requirements_for_execution_with_document_encoding(args.group_width,
                        admitted.native, admitted.transport, admitted.selected_rows, args.prefill().unwrap(), &policy).unwrap();
                    assert_eq!(compact.input_bytes - baseline.input_bytes, admitted.encoding_workspace_bytes);
                    assert_eq!(compact.workspace_bytes, baseline.workspace_bytes);
                    assert_eq!(compact.output_reservation_bytes, baseline.output_reservation_bytes);
                    let peak = compact.native.kv_bytes + compact.workspace_bytes + compact.input_bytes
                        + compact.output_reservation_bytes;
                    assert_eq!(admitted.common.minimum_memory_bytes, grouped_memory_floor(&args, admitted.common, peak).unwrap());
                } else {
                    let common = batch.source.validate_batch_with_prefill(&args).unwrap();
                    let t = admitted.transport;
                    let corpus = 8 * t.max_line_bytes as u64 + 4 * t.max_output_line_bytes as u64
                        + 128 * t.max_epoch_ids as u64 + t.max_epoch_id_bytes as u64
                        + admitted.preparation_reserve_bytes + admitted.io_bytes + admitted.encoding_workspace_bytes;
                    assert_eq!(admitted.common.minimum_memory_bytes, common.minimum_memory_bytes + corpus);
                }
                args.memory_mib = admitted.common.minimum_memory_bytes.div_ceil(MIB);
                batch.validate(&args).unwrap();
                args.memory_mib -= 1;
                assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
            }
        }
    }
}

#[test]
fn source_streams_reject_invalid_compact_policies_before_defaults_model_and_stream_io() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        for limits in [
            &["--max-tokenizer-pair-checks", "0"][..], &["--max-tokenizer-pair-bytes", "0"][..],
            &["--max-tokenizer-pair-bytes", "18446744073709551615"][..],
            &["--max-input-bytes", "1048576"][..],
        ] {
            let mut flags = vec!["--input-encoding", "safe-bpe-v1", "--defaults", "private-defaults.json"];
            flags.extend_from_slice(limits);
            let (batch, args) = arguments(name, &flags);
            assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
        }
        let (batch, args) = arguments(name, &["--input-encoding", "safe-bpe-v1",
            "--max-input-bytes", "1048576", "--preparation-mib", "1024"]);
        batch.validate(&args).unwrap();
        for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
            let matches = crate::candidate_cli::definition().try_get_matches_from([
                "candidate", "source-int4", name, "--source-dir", "source", "--source-manifest", "manifest.json",
                "--memory-mib", "8192", flag, "100",
            ]).unwrap();
            assert!(CandidateCommand::from_matches(&matches).is_err());
        }
    }
}

#[test]
fn compact_source_stream_snapshot_and_source_paths_share_work_and_input_authority() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        let (batch, mut args) = arguments(name, &["--input-encoding", "safe-bpe-v1"]);
        let source = batch.validate(&args).unwrap();
        args.source_dir = None; args.source_manifest = None;
        args.snapshot = Some(PathBuf::from("private.snapshot"));
        args.expected_model_sha256 = Some("ab".repeat(32));
        let snapshot = batch.validate(&args).unwrap();
        assert!(snapshot.common.snapshot_pin.is_some());
        assert_eq!(source.encoding_workspace_bytes, snapshot.encoding_workspace_bytes);
        assert_eq!(source.transport, snapshot.transport);
        assert_eq!(source.native.max_model_work, snapshot.native.max_model_work);
    }
}
