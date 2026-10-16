use super::*;
use wire::Content;
use std::io::{BufReader, Cursor};

fn parsed(extra: &[&str]) -> (BatchArgs, Int4Args) {
    let mut argv = vec!["text-batch", "--source-dir", "local-source", "--source-manifest", "manifest.json",
        "--memory-mib", "65536", "--max-load-mib", "65536", "--preparation-mib", "4096"];
    argv.extend_from_slice(extra);
    let matches = definition().try_get_matches_from(argv).unwrap();
    (BatchArgs::from_arg_matches(&matches).unwrap(), Int4Args::from_arg_matches(&matches).unwrap())
}
fn read(options: BatchArgs, text: &[u8], cap: usize) -> Result<Vec<Record>, CandidateError> {
    options.read(&mut BufReader::with_capacity(2, Cursor::new(text)), cap, || Ok(()))
}
#[test]
fn bounded_records_keep_ids_sample_addresses_and_original_utf8() {
    let (options, _) = parsed(&[]);
    let rows = read(options, "{\"id\":\"a\",\"sample_index\":7,\"prompt\":\" é\\n上海 \"}\r\n{\"id\":\"b\",\"prompt\":\"two\"}".as_bytes(), 4096).unwrap();
    assert_eq!(rows.len(), 2); assert_eq!(rows[0].id, "a"); assert_eq!(rows[0].sample_index, 7);
    let Content::Generate(text) = &rows[0].content else { panic!("wrong content") };
    assert_eq!(text, " é\n上海 "); assert_eq!(rows[1].sample_index, 0);
}
#[test]
fn empty_duplicate_ambiguous_and_extra_records_are_not_partial_batches() {
    let (mut options, _) = parsed(&[]); options.max_records = 1;
    for text in ["", "\n", "{\"id\":\"a\",\"prompt\":\"x\",\"seed\":\"bad\"}",
        "{\"id\":\"a\",\"id\":\"b\",\"prompt\":\"x\"}",
        "{\"id\":\"a\",\"prompt\":\"x\"}\n{\"id\":\"b\",\"prompt\":\"y\"}\n"] {
        assert!(read(options, text.as_bytes(), 4096).is_err());
    }
    options.max_records = 2;
    assert!(read(options, b"{\"id\":\"a\",\"prompt\":\"x\"}\n{\"id\":\"a\",\"sample_index\":1,\"prompt\":\"y\"}", 4096).is_err());
    assert!(read(options, &[0xff], 4096).is_err());
}
#[test]
fn line_and_whole_input_limits_include_framing_and_eof_is_not_an_extra_record() {
    let (mut options, _) = parsed(&[]); options.max_records = 1;
    let line = b"{\"id\":\"a\",\"prompt\":\"x\"}\r\n";
    options.max_total_input_bytes = line.len() as u64;
    assert_eq!(read(options, line, line.len()).unwrap().len(), 1);
    assert!(read(options, line, line.len() - 1).is_err());
    options.max_total_input_bytes -= 1;
    assert!(read(options, line, line.len()).is_err());
}
#[test]
fn chat_reuses_strict_transcript_parser_and_cancellation_checks_between_records() {
    let (options, _) = parsed(&["--task", "chat"]);
    assert!(read(options, b"{\"id\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", 4096).is_ok());
    assert!(read(options, b"{\"id\":\"a\",\"messages\":[{\"role\":\"assistant\",\"content\":\"x\"}]}", 4096).is_err());
    let (options, _) = parsed(&[]); let mut calls = 0;
    let mut reader = Cursor::new(b"{\"id\":\"a\",\"prompt\":\"x\"}\n{\"id\":\"b\",\"prompt\":\"y\"}\n");
    assert!(matches!(options.read(&mut reader, 4096, || { calls += 1;
        if calls == 2 { Err(CandidateError::Timeout) } else { Ok(()) } }), Err(CandidateError::Timeout)));
}
#[test]
fn full_declared_cohort_not_one_row_determines_memory_work_and_output_admission() {
    let (mut options, mut args) = parsed(&[]);
    let mut last = 0;
    for rows in [1, 2, 8, 64] {
        options.max_records = rows;
        let quote = options.validate(&args).unwrap();
        let native = Int4MemoryRequirement::for_cohort(&vec![args.context_tokens; rows], args.group_width).unwrap();
        assert_eq!(quote.generation.max_native_payload_bytes, native.kv_bytes + native.rope_bytes + native.scratch_payload_bound);
        assert_eq!(quote.generation.max_sampler_payload_bytes, rows as u64 * SAMPLER_BYTES);
        assert_eq!(quote.result_bytes, rows as u64 * quote.common.result_bytes as u64 + 16_384);
        assert!(quote.common.minimum_memory_bytes >= last); last = quote.common.minimum_memory_bytes;
        assert_eq!(quote.generation.native.max_forward_positions, (rows * args.context_tokens) as u64);
        let exact_mib = last.div_ceil(MIB); args.memory_mib = exact_mib;
        assert!(options.validate(&args).is_ok());
        args.memory_mib = exact_mib - 1; assert!(options.validate(&args).is_err());
        args.memory_mib = 65536;
    }
}
#[test]
fn invalid_width_prefill_or_undersized_preparation_refuses_before_reading() {
    let (mut options, mut args) = parsed(&[]);
    for rows in [0, 65, usize::MAX] { options.max_records = rows; assert!(options.validate(&args).is_err()); }
    options.max_records = 64; args.preparation_mib = 128; assert!(options.validate(&args).is_err());
    for width in ["0", "65"] {
        let (options, args) = parsed(&["--prefill-rows", width]); assert!(options.validate(&args).is_err());
    }
    let (options, mut args) = parsed(&[]); args.max_input_bytes = usize::MAX; assert!(options.validate(&args).is_err());
}
#[test]
fn snapshot_source_exclusivity_and_independent_content_pins_survive_batch_parsing() {
    let valid = ["text-batch", "--snapshot", "model.int4snap", "--expected-model-sha256",
        "abababababababababababababababababababababababababababababababab", "--memory-mib", "8192"];
    let matches = definition().try_get_matches_from(valid).unwrap();
    let options = BatchArgs::from_arg_matches(&matches).unwrap();
    let args = Int4Args::from_arg_matches(&matches).unwrap();
    assert!(options.validate(&args).unwrap().common.snapshot_pin.is_some());
    assert!(definition().try_get_matches_from(["text-batch", "--snapshot", "file", "--memory-mib", "8192"]).is_err());
}

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused batch read input"); }
}
#[test]
fn root_dispatch_exposes_batch_without_replacing_single_or_int8_tasks() {
    for task in ["generate", "chat"] {
        let matches = crate::candidate_cli::definition().try_get_matches_from(["candidate", "source-int4", "text-batch",
            "--task", task, "--source-dir", "source", "--source-manifest", "manifest", "--memory-mib", "8192"]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong route") };
        assert!(command.batch.is_some());
        assert_eq!(command.task, Int4Task::Text(if task == "chat" { Task::Chat } else { Task::Generate }));
    }
    assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", "text-batch",
        "--model", "model.fnlpq", "--memory-mib", "8192"]).is_ok());
}
#[test]
fn programmatic_task_substitution_and_impossible_cohort_refuse_before_io() {
    let (batch, args) = parsed(&[]);
    let command = Int4Command { task: Int4Task::Text(Task::Chat), args, batch: Some(Int4BatchCommand::Fixed(batch)) };
    assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    let (mut batch, args) = parsed(&[]); batch.max_records = 65;
    assert_eq!(batch.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn missing_runtime_refuses_before_any_input_or_output() {
    for flags in [vec![], vec!["--prefill-rows", "4"]] {
        let (batch, args) = parsed(&flags); let mut out = Vec::new();
        assert_eq!(batch.execute(args, &mut NoRead, &mut out), Err(CandidateError::Unavailable));
        assert!(out.is_empty());
    }
}
#[test]
fn packed_width_prices_one_total_token_workspace_without_changing_semantic_options() {
    for rows in [1, 4, 64] {
        let row = rows.to_string(); let (base, args) = parsed(&["--max-records", &row]);
        let ordinary = base.validate(&args).unwrap(); let options = args.options(166_101).unwrap();
        for width in [1, 4, 16, 64] {
            let width_text = width.to_string();
            let (batch, args) = parsed(&["--max-records", &row, "--prefill-rows", &width_text]);
            let packed = batch.validate(&args).unwrap();
            let prefill = packed.common.prefill.unwrap();
            assert_eq!(prefill.max_batch_rows, width);
            assert_eq!(packed.generation.max_native_payload_bytes,
                ordinary.generation.max_native_payload_bytes + prefill.validate().unwrap());
            assert_eq!(packed.generation.max_sampler_payload_bytes, ordinary.generation.max_sampler_payload_bytes);
            assert_eq!(packed.generation.max_output_payload_bytes, ordinary.generation.max_output_payload_bytes);
            assert_eq!(packed.generation.native.max_forward_positions, ordinary.generation.native.max_forward_positions);
            assert_eq!(packed.publication_bytes, ordinary.publication_bytes);
            assert!(args.options(166_101).unwrap() == options);
        }
    }
}
#[cfg(feature = "asupersync-runtime")]
#[test]
fn stale_packed_quote_is_rejected_before_input_even_when_load_phase_dominates_memory() {
    for axis in 0..3 {
        let (batch, args) = parsed(&["--prefill-rows", "4"]);
        let mut quote = batch.validate(&args).unwrap();
        match axis { 0 => quote.common.prefill = None,
            1 => quote.common.prefill.as_mut().unwrap().max_batch_rows = 1,
            _ => quote.common.prefill.as_mut().unwrap().max_extra_scratch_bytes -= 1 }
        assert_eq!(super::runtime::execute(args, batch, quote, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
}
