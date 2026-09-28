//! CLI mode, resource and input contracts without weights or executable receipts.
use super::*;
use crate::candidate_cli::resolve::tests::command;

#[test]
fn chunking_requires_discovery_and_all_chunk_knobs_require_chunking() {
    let root = || crate::candidate_cli::definition();
    assert!(root().try_get_matches_from(["candidate", "resolve", "--model", "local", "--memory-mib", "8192", "--chunked"]).is_err());
    for flag in ["--max-ner-chunks", "--max-snapshot-ner-chunks", "--max-ner-chunk-bytes", "--max-ner-tokenizer-calls"] {
        assert!(root().try_get_matches_from(["candidate", "resolve", "--model", "local", "--memory-mib", "8192",
            "--discover-entities", flag, "16"]).is_err());
    }
    command(&[]).validate().unwrap();
    command(&["--discover-entities"]).validate().unwrap();
    command(&["--discover-entities", "--chunked"]).validate().unwrap();
}
#[test]
fn native_and_mask_authority_is_shared_not_multiplied_by_chunk_count() {
    let cmd = command(&["--discover-entities", "--chunked", "--max-ner-tokens", "64"]);
    let (_, limits) = cmd.validate().unwrap();
    let base = cmd.discovery.config(&cmd, limits, NerOptions::default(), ResolveOptions::default());
    let work = base.max_model_work; let masks = base.masks.max_visits_per_run;
    let c = cmd.discovery.long.configuration(&cmd, base).unwrap();
    assert_eq!(c.entities.max_model_work, work); assert_eq!(c.entities.scoring.max_model_work, work);
    assert_eq!(c.entities.masks.max_visits_per_run, masks);
    assert_eq!(c.entities.ner_budget.max_output_tokens, 64);
    assert_eq!(c.chunks.reserved_tokens, 64); // The actual planner adds exact scaffold capacity.
    assert_eq!(c.chunks.max_chunks, 64); assert_eq!(c.max_snapshot_chunks, 1024);
    assert_eq!(c.entities.graph.max_documents, cmd.graph().max_documents);
    assert_eq!(c.entities.graph.max_mentions, cmd.graph().max_mentions);
}
#[test]
fn finite_chunk_limits_and_graph_witness_headroom_are_enforced_before_io() {
    for args in [vec!["--max-ner-chunks", "0"], vec!["--max-ner-chunks", "257"],
        vec!["--max-snapshot-ner-chunks", "16385"], vec!["--max-ner-chunk-bytes", "3"],
        vec!["--max-ner-tokenizer-calls", "0"], vec!["--max-ner-tokenizer-calls", "1000001"]] {
        let mut flags = vec!["--discover-entities", "--chunked"]; flags.extend(args);
        assert!(command(&flags).validate().is_err());
    }
    let baseline = command(&["--discover-entities"]);
    let chunked = command(&["--discover-entities", "--chunked"]);
    assert_eq!(chunked.discovery.extra_graph_bytes(2048).unwrap() - baseline.discovery.extra_graph_bytes(2048).unwrap(), 1024 * 1024);
    assert!(command(&["--discover-entities", "--chunked", "--max-snapshot-ner-chunks", "16384"]).validate().is_err());
    command(&["--discover-entities", "--chunked", "--max-snapshot-ner-chunks", "16384", "--graph-reserve-mib", "128"]).validate().unwrap();
}
fn raw(text: &str) -> String {
    serde_json::json!({"documents":[{"id":"opaque","text":text}],
        "options":{"blocking":"ascii_word_overlap","context_scalars":32,"minimum_margin_milli":1000}}).to_string()
}
#[test]
fn original_unicode_and_ids_survive_without_accepting_per_document_chunk_overrides() {
    let cmd = command(&["--discover-entities", "--chunked"]);
    let text = "  é Alice\r\n上海 <tool_call>";
    let parsed = cmd.discovery.input(&cmd, &raw(text)).unwrap();
    assert_eq!(parsed.documents[0].text, text); assert_eq!(parsed.documents[0].id, "opaque");
    let mut changed: serde_json::Value = serde_json::from_str(&raw(text)).unwrap();
    for name in ["chunks", "mentions", "execution_identity", "scores"] {
        changed["documents"][0][name] = serde_json::json!([]);
        assert!(cmd.discovery.input(&cmd, &changed.to_string()).is_err());
        changed["documents"][0].as_object_mut().unwrap().remove(name);
    }
}
#[test]
fn empty_documents_are_explicitly_refused_only_in_chunked_mode() {
    let short = command(&["--discover-entities"]);
    short.discovery.input(&short, &raw("")).unwrap();
    let chunked = command(&["--discover-entities", "--chunked"]);
    assert!(chunked.discovery.input(&chunked, &raw("")).is_err());
    let mut empty: serde_json::Value = serde_json::from_str(&raw("")).unwrap(); empty["documents"] = serde_json::json!([]);
    chunked.discovery.input(&chunked, &empty.to_string()).unwrap();
}
#[test]
fn manual_configuration_cannot_silently_ignore_chunk_switches() {
    let mut cmd = command(&[]); cmd.discovery.long.chunked = true;
    assert!(cmd.validate().is_err());
    let mut cmd = command(&["--discover-entities"]); cmd.discovery.long.max_ner_chunks = Some(4);
    assert!(cmd.validate().is_err());
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn feature_disabled_chunked_resolution_refuses_before_source_and_model_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("must not read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("must not write") }
        fn flush(&mut self) -> io::Result<()> { panic!("must not flush") }
    }
    let cmd = command(&["--discover-entities", "--chunked"]);
    assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
}
