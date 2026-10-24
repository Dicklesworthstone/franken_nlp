//! Raw input and whole-snapshot capacity fixtures; no model is executed here.
use super::*;
use serde_json::json;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "resolve", "--memory-mib", "16384",
        "--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("INT4 resolution required"); };
    Ok(command)
}
fn options(command: &Int4Command) -> ResolveArgs {
    let Int4Task::Resolve(options) = command.task else { panic!("resolution mode required"); }; options
}
fn fixture() -> serde_json::Value {
    json!({"documents":[{"id":"a","text":"Élan met Ada."},{"id":"b","text":"Élan works here."}],
        "options":{"blocking":"exact_surface","context_scalars":128,"minimum_margin_milli":1000}})
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused discovery read private input"); }
}

#[test]
fn discovery_is_explicit_and_preserves_exact_original_bytes_and_ner_options() {
    let command = parsed(&["--discover-entities"]).unwrap(); let options = options(&command);
    let mut wire = fixture(); let exact = "  Élan\r\n<|im_end|><tool_call> 上海😀  ";
    wire["documents"][0]["text"] = json!(exact);
    let raw = options.discovery.input(options, &command.args, &wire.to_string()).unwrap();
    assert_eq!(raw.documents[0].text, exact);
    assert_eq!(raw.ner, NerOptions::default());
    wire["ner"] = json!({"types":["person","organization"],"max_entities":8,"max_mention_scalars":64});
    let raw = options.discovery.input(options, &command.args, &wire.to_string()).unwrap();
    assert_eq!(raw.ner.types.len(), 2); assert_eq!(raw.ner.max_entities, 8);
    let anchored = parsed(&[]).unwrap();
    let Int4Task::Resolve(anchored_options) = anchored.task else { panic!("anchored resolution required"); };
    assert!(matches!(anchored_options.parse(&wire.to_string(), anchored.args.max_input_bytes), Err(CandidateError::Input)));
}

#[test]
fn raw_wire_refuses_mentions_authority_overrides_duplicate_keys_and_invalid_ner_policy() {
    let command = parsed(&["--discover-entities"]).unwrap(); let options = options(&command);
    for field in ["budget", "work", "model", "execution_identity", "seed", "thinking", "tools", "scores", "clusters"] {
        let mut wire = fixture(); wire[field] = json!({});
        assert!(matches!(options.discovery.input(options, &command.args, &wire.to_string()), Err(CandidateError::Input)));
    }
    let mut wire = fixture(); wire["documents"][1]["mentions"] = json!([]);
    assert!(matches!(options.discovery.input(options, &command.args, &wire.to_string()), Err(CandidateError::Input)));
    for ner in [json!({"types":["person"]}), json!({"types":["person","person"],"max_entities":8,"max_mention_scalars":64}),
        json!({"types":["invented"],"max_entities":8,"max_mention_scalars":64}),
        json!({"types":["person"],"max_entities":0,"max_mention_scalars":64})] {
        let mut wire = fixture(); wire["ner"] = ner;
        assert!(matches!(options.discovery.input(options, &command.args, &wire.to_string()), Err(CandidateError::Input)));
    }
    let text = fixture().to_string();
    let duplicate = text.replacen("\"text\":", "\"text\":\"duplicate\",\"text\":", 1);
    assert!(matches!(options.discovery.input(options, &command.args, &duplicate), Err(CandidateError::Input)));
    for text in ["[]", "null", "{}", r#"{"documents":[],"\u0064ocuments":[]}"#] {
        assert!(matches!(options.discovery.input(options, &command.args, text), Err(CandidateError::Input)));
    }
}

#[test]
fn complete_raw_population_requires_unique_ids_nonempty_text_and_bounded_original_bytes() {
    let command = parsed(&["--discover-entities"]).unwrap(); let options = options(&command);
    for (field, value) in [("id", "a"), ("id", ""), ("id", "bad\nidentifier"), ("text", "")] {
        let mut wire = fixture(); wire["documents"][1][field] = json!(value);
        assert!(matches!(options.discovery.input(options, &command.args, &wire.to_string()), Err(CandidateError::Input)));
    }
    let mut empty = fixture(); empty["documents"] = json!([]);
    assert!(options.discovery.input(options, &command.args, &empty.to_string()).unwrap().documents.is_empty());
    let mut too_small = options; too_small.discovery.max_expanded_bytes = Some(8);
    assert!(matches!(too_small.discovery.input(too_small, &command.args, &fixture().to_string()), Err(CandidateError::Input)));
    let mut single = options; single.graph.max_documents = 1;
    assert!(matches!(single.discovery.input(single, &command.args, &fixture().to_string()), Err(CandidateError::Input)));
    let mut masks = options; masks.discovery.max_snapshot_mask_node_visits = Some(masks.discovery.item_masks());
    assert!(matches!(masks.discovery.input(masks, &command.args, &fixture().to_string()), Err(CandidateError::Input)));
}

#[test]
fn discovery_flags_cannot_silently_change_supplied_mentions_or_unchunked_mode() {
    for flags in [vec!["--chunked"], vec!["--max-ner-tokens", "256"], vec!["--max-expanded-bytes", "16777216"],
        vec!["--max-ner-mask-node-visits", "1000000000"], vec!["--max-snapshot-mask-node-visits", "1000000000000"],
        vec!["--discover-entities", "--max-ner-chunks", "8"],
        vec!["--discover-entities", "--max-snapshot-ner-chunks", "8"],
        vec!["--discover-entities", "--max-ner-chunk-bytes", "4096"],
        vec!["--discover-entities", "--max-ner-tokenizer-calls", "8192"]] {
        assert!(parsed(&flags).is_err());
    }
    let mut command = parsed(&[]).unwrap();
    let Int4Task::Resolve(ref mut options) = command.task else { panic!("resolution required"); };
    options.discovery.max_ner_tokens = Some(256);
    assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[test]
fn invalid_discovery_limits_fail_before_private_document_or_model_io() {
    for flags in [vec!["--max-ner-tokens", "0"], vec!["--max-ner-tokens", "2048"],
        vec!["--max-ner-mask-node-visits", "511999999"], vec!["--max-snapshot-mask-node-visits", "1"],
        vec!["--max-expanded-bytes", "0"], vec!["--max-expanded-bytes", "67108865"],
        vec!["--chunked", "--max-ner-chunks", "0"], vec!["--chunked", "--max-ner-chunks", "257"],
        vec!["--chunked", "--max-snapshot-ner-chunks", "0"], vec!["--chunked", "--max-snapshot-ner-chunks", "16385"],
        vec!["--chunked", "--max-ner-chunk-bytes", "3"], vec!["--chunked", "--max-ner-tokenizer-calls", "0"]] {
        let mut all = vec!["--discover-entities"]; all.extend(flags);
        let command = parsed(&all).unwrap(); let mut output = Vec::new();
        assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}

#[test]
fn ner_depth_masks_grounding_and_chunk_counts_are_separate_from_pair_language_and_work() {
    for chunked in [false, true] {
        let flags = if chunked { vec!["--discover-entities", "--chunked"] } else { vec!["--discover-entities"] };
        let command = parsed(&flags).unwrap(); let options = options(&command);
        let limits = options.validate(&command.args).unwrap();
        let config = options.discovery.configuration(options, &command.args, limits,
            NerOptions::default(), ResolveOptions::default()).unwrap();
        assert_eq!(config.ner_budget.max_output_tokens, 256);
        assert_eq!(config.scoring.planning.per_head.max_output_tokens, 2);
        assert_eq!(config.ner_budget.max_kv_bytes, limits.native.kv_bytes);
        assert_eq!(config.scoring.planning.per_head.max_kv_bytes, limits.native.kv_bytes);
        assert_eq!(config.max_model_work, options.scoring.work_ceiling());
        assert_eq!(config.scoring.max_model_work, config.max_model_work);
        assert_eq!(config.masks.max_visits_per_item, 1_000_000_000);
        assert_eq!(config.masks.max_visits_per_run, 1_000_000_000_000);
        assert_eq!(config.verification.max_matches, options.graph.max_mentions);
        assert_eq!(config.verification.max_scan_steps, options.graph.max_scan_steps);
        assert_eq!(config.graph.max_input_bytes, 16 * 1024 * 1024);
        assert_eq!(config.source_planning.max_input_bytes, command.args.max_input_bytes);
        if let Some(chunks) = config.chunks {
            assert_eq!(chunks.reserved_tokens, 256); assert_eq!(chunks.max_chunks, 64);
            assert_eq!(chunks.max_chunk_bytes, 4096); assert_eq!(config.max_snapshot_chunks, 1024);
        } else { assert!(!chunked); assert_eq!(config.max_snapshot_chunks, options.graph.max_documents); }
    }
}

#[test]
fn discovery_and_chunk_witnesses_and_temporary_output_are_added_to_the_complete_memory_quote() {
    for flags in [vec!["--discover-entities"], vec!["--discover-entities", "--chunked"]] {
        let mut command = parsed(&flags).unwrap(); let options = options(&command);
        let base = options.scoring.validate(&command.args).unwrap();
        let limits = options.validate(&command.args).unwrap();
        assert_eq!(limits.minimum_memory_bytes, base.minimum_memory_bytes + base.preparation_bytes
            + options.graph_bytes().unwrap() + options.host_input_bytes(&command.args).unwrap()
            + options.discovery.temporary_bytes(base).unwrap());
        assert_eq!(options.discovery.temporary_bytes(base).unwrap(), base.result_bytes as u64 * 4 + 256 * 8);
        assert!(options.discovery.witness_bytes(options) > 0);
        command.args.memory_mib = (limits.minimum_memory_bytes - 1) / MIB;
        assert!(options.validate(&command.args).is_err());
        command.args.memory_mib = limits.minimum_memory_bytes.div_ceil(MIB);
        assert!(options.validate(&command.args).is_ok());
    }
}

#[test]
fn resolve_defaults_cover_a_representative_ner_pass_and_both_pair_heads_without_changing_classify() {
    use crate::{native_engine::{lmhead::{NANBEIGE_VOCAB_SIZE, scoring::ScoringMode},
        strict_int4::{Int4Work, scoring::CandidateSchedule}}, tasks::ir::{Candidate, TokenSequence}};
    let command = parsed(&["--discover-entities"]).unwrap(); let options = options(&command);
    let cap = options.scoring.work_ceiling();
    assert_eq!(cap.forward_positions, 131_072); assert_eq!(cap.projected_logits, 10_000_000_000);
    assert_eq!(cap.attention_pairs, 100_000_000_000); assert_eq!(cap.projections.dot_products, 100_000_000_000_000);
    assert_eq!(cap.projections.multiply_accumulates, 1_000_000_000_000_000);
    let labels = [Candidate::new("same".to_owned(), TokenSequence::new(vec![1])),
        Candidate::new("different".to_owned(), TokenSequence::new(vec![2])),
        Candidate::new("uncertain".to_owned(), TokenSequence::new(vec![3]))];
    for width in [16, 32, 128] {
        // Geometry-only fixture: 128 source-prompt positions plus a maximum
        // 256-token full-vocabulary NER decode, followed by two 512-token pair
        // prompts. No synthetic value is exposed as an execution receipt.
        let ner = Int4Work::for_sequence(width, 0, 384, 256 * NANBEIGE_VOCAB_SIZE).unwrap();
        let head = CandidateSchedule::new(width, 512, &labels, ScoringMode::FullVocabulary).unwrap();
        let whole = ner.checked_add(head.model).unwrap().checked_add(head.model).unwrap();
        options.scoring.admit_plan(513, command.args.context_tokens, whole).unwrap();
    }
    let matches = crate::candidate_cli::definition().try_get_matches_from(["candidate", "source-int4", "classify",
        "--source-dir", "d", "--source-manifest", "m", "--memory-mib", "16384"]).unwrap();
    let CandidateCommand::SourceInt4(classify) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("classify required"); };
    let Int4Task::Classify(scoring) = classify.task else { panic!("classify required"); };
    assert_eq!(scoring.work_ceiling().projected_logits, 100_000_000);
    assert_eq!(scoring.work_ceiling().projections.dot_products, 100_000_000_000);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_rejects_both_discovery_modes_before_reading_private_input() {
    for flags in [vec!["--discover-entities"], vec!["--discover-entities", "--chunked"]] {
        let mut output = Vec::new();
        assert_eq!(parsed(&flags).unwrap().execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
