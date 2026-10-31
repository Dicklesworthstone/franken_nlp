use super::*;
use crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE;

fn matches(extra: &[&str]) -> clap::ArgMatches {
    let mut argv = vec!["extract-batch", "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192"];
    if !extra.contains(&"--schema") { argv.extend_from_slice(&["--schema", "schema.json"]); }
    argv.extend_from_slice(extra);
    definition().try_get_matches_from(argv).unwrap()
}
fn arguments(extra: &[&str]) -> (BatchArgs, Int4Args) {
    let matches = matches(extra);
    (BatchArgs::from_matches(&matches).unwrap(), Int4Args::from_arg_matches(&matches).unwrap())
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused corpus reached private input"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused corpus published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused corpus flushed output"); }
}

#[test]
fn explicit_batch_route_uses_shared_schema_and_owned_streaming() {
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "extract-batch", "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192", "--schema", "schema.json",
    ]).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong dispatcher"); };
    assert!(matches!(&command.task, Int4Task::ExtractBatch(_)));
    assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn source_and_independently_pinned_snapshot_share_the_same_schema_policy() {
    let (batch, args) = arguments(&[]);
    let source = batch.validate(&args).unwrap();
    assert!(source.common.snapshot_pin.is_none());
    assert!(source.common.prefill.is_none());
    assert_eq!(args.max_new_tokens, 512);
    assert_eq!(source.common.max_prompt_tokens + args.max_new_tokens - 1, args.context_tokens);
    let digest = "ab".repeat(32);
    let matches = definition().try_get_matches_from(["extract-batch",
        "--snapshot", "local.snapshot", "--expected-model-sha256", &digest,
        "--memory-mib", "8192", "--schema", "schema.json"]).unwrap();
    let batch = BatchArgs::from_matches(&matches).unwrap();
    let args = Int4Args::from_arg_matches(&matches).unwrap();
    let snapshot = batch.validate(&args).unwrap();
    assert!(snapshot.common.snapshot_pin.is_some());
    assert_eq!(source.native.max_model_work, snapshot.native.max_model_work);
    assert_eq!(source.transport, snapshot.transport);
}

#[test]
fn explicit_cohorts_preserve_per_document_task_identity_and_whole_run_work() {
    let (serial, mut args) = arguments(&["--max-requests", "8", "--max-result-bytes", "65536"]);
    args.memory_mib = 262_144;
    let serial_limits = serial.validate(&args).unwrap();
    let serial_budget = serial.extraction.budget(&args, serial_limits.common);
    for documents in [1, 2, 8, 64] {
        let mut grouped = serial.clone(); grouped.stream.cohort_documents = Some(documents);
        let limits = grouped.validate(&args).unwrap();
        assert_eq!(limits.cohort_documents, Some(documents));
        assert_eq!(limits.common.native, serial_limits.common.native);
        assert_eq!(grouped.extraction.budget(&args, limits.common), serial_budget);
        assert_eq!(limits.native.max_model_work, serial_limits.native.max_model_work);
        assert_eq!(limits.native.masks.max_visits_per_item, serial_limits.native.masks.max_visits_per_item);
        assert_eq!(limits.native.masks.max_visits_per_run, serial_limits.native.masks.max_visits_per_run);
        assert_eq!(limits.native.masks.per_mask.max_trie_node_visits,
            serial_limits.native.masks.per_mask.max_trie_node_visits);
        assert_eq!(limits.transport, serial_limits.transport);
    }
}

#[test]
fn cohort_bounds_and_complete_output_peak_refuse_before_io() {
    for documents in [0, 65, usize::MAX, 64] {
        let (mut batch, mut args) = arguments(&[]);
        args.memory_mib = 262_144; batch.stream.cohort_documents = Some(documents);
        // The default one-MiB item output plus its frame cannot fit 64 rows
        // under the native aggregate ceiling. Explicit windows never shrink.
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
    let (batch, mut args) = arguments(&["--cohort-documents", "4"]);
    args.memory_mib = 262_144;
    let quote = batch.validate(&args).unwrap();
    args.memory_mib = quote.common.minimum_memory_bytes.div_ceil(MIB);
    batch.validate(&args).unwrap();
    args.memory_mib -= 1;
    assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn cli_and_host_share_the_complete_window_plus_lookahead_memory_quote() {
    let (batch, mut args) = arguments(&["--cohort-documents", "3"]);
    args.memory_mib = 262_144;
    let limits = batch.validate(&args).unwrap();
    let quote = Int4ExtractionCohortMemoryLimits { max_documents: 3, context_tokens: args.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes, io_reserve_bytes: limits.io_bytes,
        allocator_reserve_bytes: ALLOCATOR_BYTES,
    }.requirements(args.group_width, limits.native, limits.transport).unwrap();
    assert_eq!(quote.native.kv_bytes, 3 * limits.common.native.kv_bytes);
    assert!(quote.input_bytes >= 4 * limits.preparation_reserve_bytes);
    let execution = quote.native.kv_bytes + quote.workspace_bytes + quote.input_bytes + quote.output_reservation_bytes;
    assert_eq!(limits.common.minimum_memory_bytes, grouped_memory_floor(&args, limits.common, execution).unwrap());
}

#[test]
fn explicit_generation_and_prefill_controls_do_not_leak_into_extraction_batch() {
    for flags in [vec!["--max-output-bytes", "65536"], vec!["--prefill-rows", "1"],
        vec!["--min-new-tokens", "0"], vec!["--logprobs"], vec!["--logit-bias", "42=0"]] {
        assert!(BatchArgs::from_matches(&matches(&flags)).is_err());
    }
    let (batch, mut args) = arguments(&[]);
    args.seed = Some("ab".repeat(32));
    assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn unsupported_limits_refuse_before_input_schema_model_or_output_io() {
    for flags in [vec!["--max-requests", "0"], vec!["--max-requests", "100001"],
        vec!["--max-total-input-bytes", "0"], vec!["--max-total-input-bytes", "17179869185"],
        vec!["--max-total-output-bytes", "0"], vec!["--max-total-output-bytes", "17179869185"],
        vec!["--max-line-bytes", "0"], vec!["--max-line-bytes", "65535"],
        vec!["--max-line-bytes", "4194305"], vec!["--max-new-tokens", "0"],
        vec!["--max-mask-node-visits", "1"], vec!["--preparation-mib", "1"],
        vec!["--max-checkpoints", "1"], vec!["--schema", "-"]] {
        let (batch, args) = arguments(&flags);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
}

#[test]
fn whole_run_work_scales_independent_documents_without_growing_the_resident_engine() {
    let (mut batch, args) = arguments(&["--max-requests", "8"]);
    let a = batch.validate(&args).unwrap();
    batch.stream.max_requests = 16;
    let b = batch.validate(&args).unwrap();
    assert_eq!(b.native.max_model_work, scale_work(a.native.max_model_work, 2).unwrap());
    assert_eq!(b.native.masks.max_visits_per_run, 2 * a.native.masks.max_visits_per_run);
    assert_eq!(a.native.masks.max_visits_per_item, b.native.masks.max_visits_per_item);
    assert_eq!(a.common.native.kv_bytes, b.common.native.kv_bytes);
    assert_eq!(a.common.minimum_memory_bytes, b.common.minimum_memory_bytes);
    let w = a.native.max_model_work;
    assert_eq!(w.forward_positions, 8 * args.context_tokens as u64);
    assert_eq!(w.projected_logits, 8 * args.max_new_tokens as u64 * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(a.transport.max_work.forward_positions, w.forward_positions);
    assert_eq!(a.transport.max_work.projected_logits, w.projected_logits);
    assert_eq!(a.transport.max_input_bytes, batch.stream.max_total_input_bytes);
    assert_eq!(a.transport.max_document_bytes, args.max_input_bytes);
}

#[test]
fn group_partition_changes_dot_budget_without_changing_tokens_or_multiply_accumulates() {
    let (batch, mut args) = arguments(&[]);
    args.group_width = 16;
    let small = batch.validate(&args).unwrap().native.max_model_work;
    args.group_width = 128;
    let large = batch.validate(&args).unwrap().native.max_model_work;
    assert_eq!(small.forward_positions, large.forward_positions);
    assert_eq!(small.projected_logits, large.projected_logits);
    assert_eq!(small.projections.multiply_accumulates, large.projections.multiply_accumulates);
    assert!(small.projections.dot_products > large.projections.dot_products);
}

#[test]
fn checked_scaling_rejects_overflow_on_every_native_work_axis() {
    let one = constrained_int4::planned_work(128, 1, 1).unwrap();
    assert!(scale_work(one, 0).is_err());
    for axis in 0..5 {
        let mut work = one;
        match axis { 0 => work.forward_positions = u64::MAX, 1 => work.projected_logits = u64::MAX,
            2 => work.attention_pairs = u64::MAX, 3 => work.projections.dot_products = u64::MAX,
            _ => work.projections.multiply_accumulates = u64::MAX }
        assert!(scale_work(work, 2).is_err());
    }
    assert!(times(u64::MAX, 2).is_err());
}

#[test]
fn total_output_prices_every_wrapper_and_keeps_terminal_capacity() {
    let (mut batch, args) = arguments(&["--max-requests", "1"]);
    let framing = (batch.stream.max_requests + 4) * FRAME_ALLOWANCE;
    for bytes in [framing - 1, framing, framing + 4095] {
        batch.stream.max_total_output_bytes = bytes;
        assert!(batch.validate(&args).is_err());
    }
    batch.stream.max_total_output_bytes = framing + 4096;
    let limits = batch.validate(&args).unwrap();
    assert_eq!(limits.transport.max_output_bytes, 4096);
    assert_eq!(limits.transport.max_output_bytes + framing, limits.output_bytes);
    assert_eq!(limits.transport.max_output_line_bytes, limits.common.result_bytes + 4096);
}

#[test]
fn process_floor_prices_outer_preparation_and_full_hosted_corpus_staging_together() {
    let (batch, mut args) = arguments(&[]);
    let single = batch.extraction.validate(&args).unwrap();
    let limits = batch.validate(&args).unwrap();
    let t = limits.transport;
    let host = t.max_line_bytes as u64 * 8 + t.max_output_line_bytes as u64 * 4
        + t.max_epoch_ids as u64 * 128 + t.max_epoch_id_bytes as u64
        + limits.preparation_reserve_bytes + limits.io_bytes;
    assert_eq!(limits.preparation_reserve_bytes, single.preparation_bytes);
    assert_eq!(limits.common.minimum_memory_bytes, single.minimum_memory_bytes + host);
    assert!(host > single.preparation_bytes);
    args.memory_mib = (limits.common.minimum_memory_bytes - 1) / MIB;
    assert!(batch.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(batch.validate(&args).is_ok());
}

#[test]
fn shared_schema_preserves_exact_decimals_and_source_policy_before_model_access() {
    let (batch, args) = arguments(&[]);
    let ceiling = batch.extraction.budget(&args, batch.validate(&args).unwrap().common);
    let schema = r#"{"type":"number","const":12345678901234567890123456789012345678}"#;
    let defaults = batch.load_defaults(schema.to_owned(), ceiling).unwrap();
    assert_eq!(defaults.schema, schema);
    assert_eq!(defaults.grounding, ExtractionBatchGrounding::Structural);
    assert_eq!(defaults.budget, ceiling);
    let (grounded, _) = arguments(&["--source-membership"]);
    let schema = r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#;
    let defaults = grounded.load_defaults(schema.to_owned(), ceiling).unwrap();
    assert_eq!(defaults.schema, schema);
    assert_eq!(defaults.grounding, ExtractionBatchGrounding::SourceMembership);
    assert!(batch.load_defaults(schema.to_owned(), ceiling).is_err());
}

#[test]
fn shared_schema_refuses_bad_shape_duplicates_and_structural_unsupported_keywords() {
    let (batch, args) = arguments(&[]);
    let ceiling = batch.extraction.budget(&args, batch.validate(&args).unwrap().common);
    for schema in ["", "null", "[]", r#"{"type":"boolean","type":"string"}"#,
        r#"{"$ref":"remote"}"#, r#"{"type":"string","pattern":".*"}"#] {
        assert!(batch.load_defaults(schema.to_owned(), ceiling).is_err());
    }
    let (grounded, _) = arguments(&["--source-membership"]);
    assert!(grounded.load_defaults(r#"{"type":"string","type":"boolean"}"#.to_owned(), ceiling).is_err());
    assert!(grounded.load_defaults(" ".repeat(crate::candidate_cli::extract::SCHEMA_BYTES + 1), ceiling).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn missing_runtime_refuses_before_reading_schema_stream_or_model() {
    let (batch, args) = arguments(&[]);
    assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Unavailable));
}
