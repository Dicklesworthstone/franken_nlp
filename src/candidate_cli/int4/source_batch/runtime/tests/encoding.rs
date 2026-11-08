use super::*;
use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};

#[test]
fn compact_source_policy_is_reported_on_results_flush_and_empty_completion_with_exact_spans() {
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 1024, max_pair_checks: 123456, max_pair_bytes: 2345678,
    });
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let mut provenance = Provenance {
            protocol: "fnlp-candidate-int4-source-batch-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
            group_width: 128, integer_kernel: "s_scalar_i32", task, snapshot: None,
            cohort_schedule: None, max_selected_rows: Some(7), compute_schedule: None, input_encoding: None,
        };
        let byte = serde_json::to_value(&provenance).unwrap();
        provenance.input_encoding = Some(&encoding);
        let mut compact = serde_json::to_value(&provenance).unwrap();
        assert_eq!(compact.as_object_mut().unwrap().remove("input_encoding"), Some(serde_json::to_value(&encoding).unwrap()));
        assert_eq!(compact, byte);
        for records in [
            "{\"event\":\"doc\",\"fixture_text\":\"上海\",\"byte_start\":4,\"scalar_start\":3,\"fixture_decimal\":12345678901234567890123456789012345678}\n{\"event\":\"flush\"}\n{\"event\":\"run_complete\"}\n",
            "{\"event\":\"run_complete\",\"summary\":{\"succeeded\":0}}\n",
        ] {
            let mut output = Vec::new();
            {
                let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 16384).unwrap();
                for frame in records.split_inclusive('\n') {
                    writer.write_all(frame.as_bytes()).unwrap(); writer.flush().unwrap();
                }
            }
            let output = std::str::from_utf8(&output).unwrap();
            assert_eq!(output.lines().count(), records.lines().count());
            for (line, original) in output.lines().zip(records.lines()) {
                assert!(line.ends_with(&format!("\"record\":{original}}}")));
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                assert_eq!(value["input_encoding"], serde_json::to_value(&encoding).unwrap());
                assert_eq!(value["task"], task);
            }
        }
    }
}

#[test]
fn owned_source_stream_rechecks_compact_policy_before_defaults_or_runtime_io() {
    for name in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        for schedule in [&[][..], &["--compute-threads", "3"][..],
            &["--cohort-documents", "2", "--prefill-rows", "1"][..]] {
            let mut flags = schedule.to_vec();
            flags.extend(["--input-encoding", "safe-bpe-v1", "--defaults", "private-defaults.json"]);
            let command = super::super::super::tests::command(name, &flags);
            let Int4Command { task: Int4Task::SourceBatch(batch), mut args, batch: None, job: None } = command
                else { panic!("wrong task"); };
            args.memory_mib = 262_144;
            let quote = batch.validate(&args).unwrap();
            let invalid = input_encoding::definition(clap::Command::new("fixture")).try_get_matches_from([
                "fixture", "--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "0",
            ]).unwrap();
            args.input_encoding = input_encoding::InputEncodingArgs::from_matches(&invalid).unwrap();
            assert_eq!(execute(args, batch, quote, NoIo, NoIo), Err(CandidateError::Arguments));
        }
    }
}
