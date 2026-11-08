use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};

#[test]
fn compact_extraction_stream_admits_schema_and_source_for_every_execution_schedule() {
    for schedule in [
        &[][..], &["--compute-threads", "3"][..],
        &["--cohort-documents", "3"][..],
        &["--cohort-documents", "3", "--prefill-rows", "2"][..],
    ] {
        for selection in [None, Some("7")] {
            let mut flags = schedule.to_vec();
            if let Some(cap) = selection { flags.extend(["--max-selected-rows", cap]); }
            let (byte_batch, mut byte_args) = arguments(&flags);
            byte_args.memory_mib = 262_144;
            let byte = byte_batch.validate(&byte_args).unwrap();
            flags.extend(["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "123456",
                "--max-tokenizer-pair-bytes", "2345678"]);
            let (batch, mut args) = arguments(&flags);
            args.memory_mib = 262_144;
            let admitted = batch.validate(&args).unwrap();
            let policy = batch.extraction.encoding_policy(&args, admitted.common.max_prompt_tokens).unwrap();
            let UntrustedEncoding::SafeBpeV1(limits) = &policy else { panic!("compact policy was lost"); };
            assert_eq!(limits.max_pair_checks, 123456);
            assert_eq!(limits.max_pair_bytes, 2345678);
            assert_eq!(limits.max_input_bytes,
                (args.max_input_bytes + crate::candidate_cli::extract::SCHEMA_BYTES).min(SafeBpeLimits::MAX_INPUT_BYTES));
            assert_eq!(limits.max_output_tokens, admitted.common.max_prompt_tokens);
            assert_eq!(admitted.encoding_workspace_bytes, limits.requirements().unwrap().workspace_bytes);
            assert_eq!(byte.encoding_workspace_bytes, 0);
            assert_eq!(admitted.common.native, byte.common.native);
            assert_eq!(admitted.transport, byte.transport);
            assert_eq!(admitted.native.max_model_work, byte.native.max_model_work);
            assert_eq!(admitted.native.masks.max_visits_per_run, byte.native.masks.max_visits_per_run);
            assert_eq!(admitted.selected_rows, byte.selected_rows);
            assert_eq!(batch.extraction.budget(&args, admitted.common), byte_batch.extraction.budget(&byte_args, byte.common));
            if let Some(documents) = admitted.cohort_documents {
                let quote = Int4ExtractionCohortMemoryLimits { max_documents: documents,
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
                let common = batch.extraction.validate_batch_with_prefill(&args).unwrap();
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

#[test]
fn extraction_stream_dispatcher_preserves_compact_policy_and_pinned_snapshot_selection() {
    let pin = "ab".repeat(32);
    for encoding in ["byte-fallback", "safe-bpe-v1"] {
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", "extract-batch", "--snapshot", "private.snapshot",
            "--expected-model-sha256", &pin, "--memory-mib", "8192", "--schema", "private-schema.json",
            "--input-encoding", encoding,
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong dispatcher"); };
        let Int4Task::ExtractBatch(batch) = command.task else { panic!("wrong task"); };
        let quote = batch.validate(&command.args).unwrap();
        assert!(quote.common.snapshot_pin.is_some());
        assert_eq!(quote.encoding_workspace_bytes > 0, encoding == "safe-bpe-v1");
    }
}

#[test]
fn extraction_stream_invalid_tokenizer_caps_and_preparation_refuse_before_private_io() {
    for limits in [
        &["--max-tokenizer-pair-checks", "0"][..], &["--max-tokenizer-pair-bytes", "0"][..],
        &["--max-tokenizer-pair-checks", "18446744073709551615"][..],
        &["--max-input-bytes", "1048576"][..],
    ] {
        let mut flags = vec!["--input-encoding", "safe-bpe-v1"];
        flags.extend_from_slice(limits);
        let (batch, args) = arguments(&flags);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
    let (batch, args) = arguments(&["--input-encoding", "safe-bpe-v1",
        "--max-input-bytes", "1048576", "--preparation-mib", "1024"]);
    batch.validate(&args).unwrap();
    for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        assert!(input_encoding::InputEncodingArgs::from_matches(&matches(&[flag, "100"])).is_err());
    }
}
